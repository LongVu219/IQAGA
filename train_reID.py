from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import time
from torch.nn import init
from modules import datasets
from modules import models
from modules import loss
from modules.trainers_IQ import PreTrainer
from modules.evaluators import Evaluator
from modules.datasets.data import IterLoader
from modules.datasets.data import transforms as T
from modules.datasets.data.sampler import RandomMultipleGallerySampler
from modules.datasets.data.preprocessor import Preprocessor
from modules.utils.logger import Logger
from modules.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from modules.utils.lr_scheduler import WarmupMultiStepLR

logger = Logger()
start_epoch = best_mAP = 0

def get_data(name, data_dir, height, width, batch_size, workers, num_instances, iters=200, 
            use_syn_cam=False, use_syn_pose=False, **kwargs):
    root = osp.join(data_dir)#osp.join(data_dir, name)

    dataset = datasets.create(name, root, use_syn_cam=use_syn_cam, use_syn_pose=use_syn_pose)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             #T.RandomErasing(),
             normalizer
         ])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                        transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_source, num_classes, train_loader_source, test_loader_source = \
        get_data(args.dataset_source, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, args.num_instances, iters, 
                 use_syn_cam=args.use_syn_cam, use_syn_pose=args.use_syn_pose)

    dataset_target, _, train_loader_target, test_loader_target = \
        get_data(args.dataset_target, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, 0, iters)

    # Create model
    model = models.create(args.arch, num_features=args.features, dropout=args.dropout)
    lossfunc = None
    if args.head == "":
        header = nn.Linear(model.out_planes, num_classes, bias=False)
        init.normal_(header.weight, std=0.001)
        lossfunc = loss.CrossEntropyLabelSmooth(num_classes, epsilon = 1e-3).cuda()
    else: 
        header = loss.create(args.head, model.out_planes, num_classes, args.lm, args.ls)
    logger.log(f"Embeding size: {model.out_planes}" )
    model.cuda(), header.cuda()
    model = nn.DataParallel(model)
    header = nn.DataParallel(header)
    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['best_mAP']
        logger.log("=> Start epoch {}  best mAP {:.1%}"
              .format(start_epoch, best_mAP))
        

    # Evaluator
    evaluator = Evaluator(model, logger=logger)
    if args.evaluate:
        logger.log("Test on source domain:")
        evaluator.evaluate(test_loader_source, dataset_source.query, dataset_source.gallery, cmc_flag=True, rerank=args.rerank)
        logger.log("Test on target domain:")
        evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True, rerank=args.rerank)
        return

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad: continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    params += [{"params": header.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    # optimizer_he = torch.optim.Adam([{'params': header.parameters()}],lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    # Trainer
    assert args.arch.find("bb") > -1, "backbone must be type `[name]bb`"
    trainer = PreTrainer(model, header, lossfunc, num_classes, margin=args.margin, 
                        batch_mean=args.bm, batch_std=args.bs, batch_const=args.bc, use_IQA=args.iqa)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        train_loader_source.new_epoch()
        train_loader_target.new_epoch()

        trainer.train(epoch, train_loader_source, optimizer,
                    train_iters=len(train_loader_source), print_freq=args.print_freq, logger=logger, target_loader_source=train_loader_target)
        lr_scheduler.step()
        # lr_scheduler_he.step()

        with torch.no_grad():
            torch.cuda.empty_cache()
            if (epoch > 3 or epoch==0) and ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):

                mAP = evaluator.evaluate(test_loader_source, dataset_source.query, dataset_source.gallery, cmc_flag=False)

                is_best = mAP > best_mAP
                best_mAP = max(mAP, best_mAP)
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_mAP': best_mAP,
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

                logger.log('\n * Finished epoch {:3d}  source mAP: {:5.1%}  best: {:5.1%}{}\n'.
                    format(epoch, mAP, best_mAP, ' *' if is_best else ''))
                # if is_best:
                #     logger.log("Test new best on target data:")
                #     evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False, rerank=False)

    logger.log("Final Test on target domain:")
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    copy_state_dict(checkpoint['state_dict'], model)
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True, rerank=args.rerank)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-training on the source domain")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='market1501|dukemtmc|msmt17')
    parser.add_argument('-dt', '--dataset-target', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.names(), help='backbone name')
    parser.add_argument('-e', '--head', type=str, default='', choices=loss.names(), help="Header of model, the default will use Softmax loss")
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70], help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',help="evaluation only")
    parser.add_argument('--eval-step', type=int, default=40)
    parser.add_argument('--rerank', action='store_true',help="evaluation only")
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--iters', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=25)
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    print("working_dir: ", working_dir)
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'datasets'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument("-pt", "--pretrained", type=str, default="", help="path to pretrained model file")

    # IQA
    parser.add_argument('--bm', type=float, default=20.0, help='batch mean initialization for IQA')
    parser.add_argument('--bs', type=float, default=100.0, help='batch std initialization for IQA')
    parser.add_argument('--bc', type=float, default=1., help='batch const initialization for IQA')
    # Margin loss
    parser.add_argument('--lm', type=float, default=0.4, help='loss margin if u use margin-based loss function')
    parser.add_argument('--ls', type=float, default=64, help=' loss scale value if u use margin-based loss function')
    # Synthetic Dataset
    parser.add_argument('--use-syn-cam', action='store_true', help='use CamStyle-Synthetic Image')
    parser.add_argument('--use-syn-pose', action='store_true', help='use Pose-Synthetic Image')
    #toggle
    parser.add_argument('--iqa', action='store_true', help='use Image quality assessment as Weight of loss')

    main()