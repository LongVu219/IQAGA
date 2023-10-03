from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image

from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy, KLDivLoss
from .utils.meters import AverageMeter
from .utils.memory import clean_cuda


#Pretrainer only for real image
class PreTrainer(object):

    def __init__(self, model, header, loss, num_classes, margin=0.0, ce_epsilon=0.1, batch_mean=20, batch_std=100, batch_const=1., use_IQA=False, **kwargs):
        super(PreTrainer, self).__init__()
        self.model = model
        self.header = header
        self.classification_loss = loss
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon = ce_epsilon).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin, skip_mean=True).cuda()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

        self.batch_mean = batch_mean
        self.batch_std = batch_std
        self.batch_const = batch_const
        self.t_alpha   = 0.99
        self.eps       = 1e-6
        self.h         = 0.333
        self.max_std   = batch_std

        self.use_IQA   = use_IQA
        self.num_classes = num_classes
        self.ce_eps = ce_epsilon
        

    def train(self, epoch, data_loader_source, optimizer, train_iters=200, print_freq=1, logger=None, **kwargs):
        self.model.train()
        self.header.train()

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            s_inputs, targets = self._parse_data(source_inputs)
            embd, embd_norm = self.model(s_inputs)
            #Tinh loss
            loss_weight = 1
            if self.classification_loss is None: 
                logits = self.header(embd, targets) 
            else:
                logits = self.header(embd_norm)

            if self.use_IQA:
                # update batchmean batchstd
                with torch.no_grad():
                    safe_norms = torch.clip(torch.norm(embd.clone(), dim=1, keepdim=True), min=0.001, max=self.max_std).clone().detach()
                    mean = safe_norms.mean().detach()
                    std = safe_norms.std().detach()
                    self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
                    self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std
                    #Image Quality Indicator based on AdaFace
                    margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
                    margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
                    loss_weight = torch.clip(margin_scaler, -1, 1).reshape(-1) * self.batch_const  + 1

            #triplet loss
            loss_tr = self.criterion_triple(embd, embd, targets)
            #softmax-loss
            log_preds = self.logsoftmax(logits)
            targets_tmp = torch.zeros_like(log_preds).scatter_(1, targets.unsqueeze(1), 1)
            targets_tmp = (1 - self.ce_eps) * targets_tmp + self.ce_eps / self.num_classes
            loss_ce = (- targets_tmp * log_preds).sum(1)            
            # print(self.criterion_ce(logits, targets))
        
            loss = loss_weight * (loss_ce + loss_tr)  #could improve it?
            loss = loss.mean()
            losses_ce.update(loss_ce.mean().item())
            losses_tr.update(loss_tr.mean().item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5, norm_type=2)
            optimizer.step()

            if ((i + 1) % print_freq == 0):
                logger.traininglog(epoch=epoch, i=i+1, iters=train_iters,
                    avgloss=loss, 
                    loss_ce=f"{losses_ce.val:.3f}({losses_ce.avg:.3f})",
                    loss_tr=f"{losses_tr.val:.3f}({losses_tr.avg:.3f})")

    def _parse_data(self, inputs):
        imgs, _, pids, _, _= inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets
    