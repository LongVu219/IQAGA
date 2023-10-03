from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image

from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy
from .loss import KLDivLoss, CrossEntropyLabelSmoothFilterNoise, AALS, PGLR, UET, RegLoss, UET2
from .loss import PartAveragedTripletLoss, CenterTripletLoss
from .utils.meters import AverageMeter
from .utils.memory import clean_cuda


#Pretrainer only for real image
class PreTrainer(object):

    def __init__(self, model, num_classes, margin=0.0, ce_epsilon=0.1, model_type="resnet", **kwargs):
        self.__typemodel ={
                "resnet" : self._forward_loss,
                "resnetbpart": self._forward_loss_2
                }
        super(PreTrainer, self).__init__()
        print("Normal Pretrainer")
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon = ce_epsilon).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        try: self.__forward = self.__typemodel[model_type]
        except:                 
            print(model_type)
            raise ImportError(name="Not support that type of backbone")

    def train(self, epoch, data_loader_source, optimizer, 
                train_iters=200, print_freq=1, logger=None, **kwargs):
        self.model.train()

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            s_inputs, targets = self._parse_data(source_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            loss_ce, loss_tr, prec1 = self.__forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr #could improve it?
            
            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i + 1) % print_freq == 0):
                logger.traininglog(epoch=epoch, i=i+1, iters=train_iters,
                    avgloss=loss, 
                    loss_ce=f"{losses_ce.val:.3f}({losses_ce.avg:.3f})",
                    loss_tr=f"{losses_tr.val:.3f}({losses_tr.avg:.3f})",
                    prec=f"{precisions.val:.2%}({precisions.avg:.2%})")

    def _parse_data(self, inputs):
        imgs, _, pids, _, _= inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward_loss(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

    def _forward_loss_2(self, s_features, s_outputs, targets):
        [x, part_up, part_down], [prob, prob_part_up, prob_part_down] = s_features, s_outputs
        loss_ce = self.criterion_ce(prob, targets) + self.criterion_ce(prob_part_up, targets) + self.criterion_ce(prob_part_down, targets)
        
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(x, x, targets) + self.criterion_triple(part_up, part_up, targets) + self.criterion_triple(part_down, part_down, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(x, targets) + self.criterion_triple(part_up, targets) + self.criterion_triple(part_down, targets)
        prec, = accuracy(prob.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

#Pretrainer with both fake and real images
class PreTrainerwSynImgs(object):

    def __init__(self, model, num_classes, margin=0.0, ce_epsilon=0.1, model_type="resnet", lam=1., **kwargs):
        self.__typemodel ={
                "resnet" : self._forward_loss,
                "resnetbpart": self._forward_loss_2
                }
        super(PreTrainerwSynImgs, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon = ce_epsilon).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.lamda = lam
        try: self.__forward = self.__typemodel[model_type]
        except:                 
            print(model_type)
            raise ImportError(name="Not support that type of backbone")

    def train(self, epoch, data_loader_source, optimizer, 
                train_iters=200, print_freq=1, logger=None, **kwargs):
        self.model.train()

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            s_inputs, targets, isreal = self._parse_data(source_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            loss_ce, loss_tr, prec1 = self.__forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr #could improve it?
            # print(loss.shape, isreal.shape)
            # if not isreal:
            #     loss *= self.lamda
            
            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i + 1) % print_freq == 0):
                logger.traininglog(epoch=epoch, i=i+1, iters=train_iters,
                    avgloss=loss, 
                    loss_ce=f"{losses_ce.val:.3f}({losses_ce.avg:.3f})",
                    loss_tr=f"{losses_tr.val:.3f}({losses_tr.avg:.3f})",
                    prec=f"{precisions.val:.2%}({precisions.avg:.2%})")

    def _parse_data(self, inputs):
        imgs, _, pids, _, isreal= inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, isreal

    def _forward_loss(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

    def _forward_loss_2(self, s_features, s_outputs, targets):
        [x, part_up, part_down], [prob, prob_part_up, prob_part_down] = s_features, s_outputs
        loss_ce = self.criterion_ce(prob, targets) + self.criterion_ce(prob_part_up, targets) + self.criterion_ce(prob_part_down, targets)
        
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(x, x, targets) + self.criterion_triple(part_up, part_up, targets) + self.criterion_triple(part_down, part_down, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(x, targets) + self.criterion_triple(part_up, targets) + self.criterion_triple(part_down, targets)
        prec, = accuracy(prob.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec


#Pretrainer only for real image
class PreTrainerwDIM(object):

    def __init__(self, model, num_classes, margin=0.0, ce_epsilon=0.1, model_type="resnet", disnet=None, lam=1, **kwargs):
        self.__typemodel ={
                "resnet" : self._forward_loss,
                "resnetbpart": self._forward_loss_2
                }
        super(PreTrainerwDIM, self).__init__()
        print("init_PretrainerwDIM")
        self.model = model
        self.disnet = disnet
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon = ce_epsilon).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        try: self.__forward = self.__typemodel[model_type]
        except:                 
            print(model_type)
            raise ImportError(name="Not support that type of backbone")
        self.ALoss = nn.MSELoss().cuda()
        self.lamda = lam

    def train(self, epoch, data_loader_source, optimizer,  optimizerD, target_loader_source,
                train_iters=200, print_freq=1, logger=None, **kwargs):
        self.model.train()
        self.disnet.train()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_limD = AverageMeter()
        losses_limM = AverageMeter()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            s_inputs, targets = self._parse_data(source_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            loss_ce, loss_tr, prec1 = self.__forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr #could improve it?
            
            t_inputs, _, _, _, _ = target_loader_source.next()
            t_features, _ = self.model(t_inputs.cuda())   
            #train model with netD
            loss_lim = 0
            if epoch >= 1:
                D_duke = self.disnet(t_features)
                D_market = self.disnet(s_features)
                loss_lim = (self.ALoss(D_duke, torch.ones_like(D_duke)/2.) + self.ALoss(D_market, torch.ones_like(D_market))/2.)
                loss += loss_lim * self.lamda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # train discriminator
            D_duke = self.disnet(t_features.detach())
            D_market = self.disnet(s_features.detach())
            d_loss = self.ALoss(D_market, torch.ones_like(D_market)) + self.ALoss(D_duke, torch.zeros_like(D_duke))
            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            losses_limM.update(loss_lim.item() if loss_lim > 0 else 0)
            losses_limD.update(d_loss.item() if d_loss > 0 else 0)

            if ((i + 1) % print_freq == 0):
                logger.traininglog(epoch=epoch, i=i+1, iters=train_iters,
                    avgloss=loss, 
                    loss_ce=f"{losses_ce.val:.3f}({losses_ce.avg:.3f})",
                    loss_tr=f"{losses_tr.val:.3f}({losses_tr.avg:.3f})",
                    loss_G=f"{losses_limM.val:.3f}({losses_limM.avg:.3f})",
                    loss_D=f"{losses_limD.val:.3f}({losses_limD.avg:.3f})")

    def _parse_data(self, inputs):
        imgs, _, pids, _, _= inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward_loss(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

    def _forward_loss_2(self, s_features, s_outputs, targets):
        [x, part_up, part_down], [prob, prob_part_up, prob_part_down] = s_features, s_outputs
        loss_ce = self.criterion_ce(prob, targets) + self.criterion_ce(prob_part_up, targets) + self.criterion_ce(prob_part_down, targets)
        
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(x, x, targets) + self.criterion_triple(part_up, part_up, targets) + self.criterion_triple(part_down, part_down, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(x, targets) + self.criterion_triple(part_up, targets) + self.criterion_triple(part_down, targets)
        prec, = accuracy(prob.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec


#Pretrainer with both fake and real images + DIM
class PreTrainerwSynImgsandLIM(object):

    def __init__(self, model, num_classes, margin=0.0, ce_epsilon=0.1, model_type="resnet", lam=1., disnet=None, **kwargs):
        self.__typemodel ={
                "resnet" : self._forward_loss,
                "resnetbpart": self._forward_loss_2
                }
        super(PreTrainerwSynImgsandLIM, self).__init__()
        self.model = model
        self.disnet = disnet
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon = ce_epsilon).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()

        self.ALoss = nn.MSELoss().cuda()
        self.lamda = lam
        try: self.__forward = self.__typemodel[model_type]
        except:                 
            print(model_type)
            raise ImportError(name="Not support that type of backbone")

    def train(self, epoch, data_loader_source, optimizer, optimizerD, target_loader_source,
                train_iters=200, print_freq=1, logger=None, **kwargs):
        self.model.train()
        self.disnet.train()

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_limD = AverageMeter()
        losses_limM = AverageMeter()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            s_inputs, targets, _ = self._parse_data(source_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            loss_ce, loss_tr, prec1 = self.__forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr #could improve it?


            t_inputs, _, _, _, _ = target_loader_source.next()
            t_features, _ = self.model(t_inputs.cuda())            
            #train model with netD
            loss_lim = 0
            if epoch >= 2:
                D_duke = self.disnet(t_features)
                D_market = self.disnet(s_features)
                loss_lim = (self.ALoss(D_duke, torch.ones_like(D_duke)/2.) + self.ALoss(D_market, torch.ones_like(D_market))/2.)
                loss += loss_lim * self.lamda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # train discriminator
            D_duke = self.disnet(t_features.detach())
            D_market = self.disnet(s_features.detach())
            d_loss = self.ALoss(D_market, torch.ones_like(D_market)) + self.ALoss(D_duke, torch.zeros_like(D_duke))
            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()



            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            losses_limM.update(loss_lim.item() if loss_lim > 0 else 0)
            losses_limD.update(d_loss.item() if d_loss > 0 else 0)
            

            if ((i + 1) % print_freq == 0):
                logger.traininglog(epoch=epoch, i=i+1, iters=train_iters,
                    avgloss=loss, 
                    loss_ce=f"{losses_ce.val:.3f}({losses_ce.avg:.3f})",
                    loss_tr=f"{losses_tr.val:.3f}({losses_tr.avg:.3f})",
                    loss_G=f"{losses_limM.val:.3f}({losses_limM.avg:.3f})",
                    loss_D=f"{losses_limD.val:.3f}({losses_limD.avg:.3f})")

    def _parse_data(self, inputs):
        imgs, _, pids, _, isreal= inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, isreal

    def _forward_loss(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

    def _forward_loss_2(self, s_features, s_outputs, targets):
        [x, part_up, part_down], [prob, prob_part_up, prob_part_down] = s_features, s_outputs
        loss_ce = self.criterion_ce(prob, targets) + self.criterion_ce(prob_part_up, targets) + self.criterion_ce(prob_part_down, targets)
        
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(x, x, targets) + self.criterion_triple(part_up, part_up, targets) + self.criterion_triple(part_down, part_down, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(x, targets) + self.criterion_triple(part_up, targets) + self.criterion_triple(part_down, targets)
        prec, = accuracy(prob.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec



#Pretrainer with both fake and real images + DIM
class PreTrainerMwSynImg(object):

    def __init__(self, model, num_classes, margin=0.0, ce_epsilon=0.1, ratio=[1,1], disnet=None, lam=0.01, **kwargs):
        super(PreTrainerMwSynImg, self).__init__()
        self.model = model
        self.disnet = disnet
        self.islim  = not self.disnet is None
        self.ratio = ratio[0] // ratio[1] + 1 if ratio[0] // ratio[1] > 0 else 1
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon = ce_epsilon).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.ALoss = nn.MSELoss().cuda()
        self.lamda = lam

    def train(self, epoch, real_loader_source,fake_loader_source,  optimizer, 
                train_iters=200, print_freq=1, logger=None, forDisNet=None, **kwargs):
        self.model.train()
        train_loader_target, optimizerD = None, None
        if self.islim:
            self.disnet.train()
            train_loader_target, optimizerD = forDisNet

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_limD = AverageMeter()
        losses_limM = AverageMeter()


        flag = True
        for i in range(train_iters):
            if flag and (i+1) % self.ratio == 0:
                source_inputs = fake_loader_source.next()
                i -= 1
                flag=False
            else:
                flag = True
                source_inputs = real_loader_source.next()

            s_inputs, targets, _ = self._parse_data(source_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            loss_ce, loss_tr, prec1 = self._forward_loss(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr #could improve it?
            
            if self.islim:
                t_inputs, _, _, _, _ = train_loader_target.next()
                t_features, _ = self.model(t_inputs.cuda())   

            #train model with netD
            loss_lim = 0
            if epoch >= 1 and self.islim:
                D_duke = self.disnet(t_features)
                D_market = self.disnet(s_features)
                loss_lim = (self.ALoss(D_duke, torch.ones_like(D_duke)/2.) + self.ALoss(D_market, torch.ones_like(D_market))/2.)
                loss += loss_lim * self.lamda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # train discriminator
            d_loss=0
            if self.islim:
                D_duke = self.disnet(t_features.detach())
                D_market = self.disnet(s_features.detach())
                d_loss = self.ALoss(D_market, torch.ones_like(D_market)) + self.ALoss(D_duke, torch.zeros_like(D_duke))
                optimizerD.zero_grad()
                d_loss.backward()
                optimizerD.step()

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            losses_limM.update(loss_lim.item() if loss_lim > 0 else 0)
            losses_limD.update(d_loss.item() if d_loss > 0 else 0)

            

            if ((i + 1) % print_freq == 0):
                logger.traininglog(epoch=epoch, i=i+1, iters=train_iters,
                    avgloss=loss, 
                    loss_ce=f"{losses_ce.val:.3f}({losses_ce.avg:.3f})",
                    loss_tr=f"{losses_tr.val:.3f}({losses_tr.avg:.3f})",
                    loss_G=f"{losses_limM.val:.3f}({losses_limM.avg:.3f})",
                    loss_D=f"{losses_limD.val:.3f}({losses_limD.avg:.3f})")


    def _parse_data(self, inputs):
        imgs, _, pids, _, isreal= inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, isreal

    def _forward_loss(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec


class HardPreTrainerMwSynImg(object):

    def __init__(self, model, num_classes, margin=0.0, ce_epsilon=0.1, ratio=[1,1], disnet=None, lam=0.01, **kwargs):
        super(HardPreTrainerMwSynImg, self).__init__()
        self.model = model
        self.disnet = disnet
        self.islim  = not self.disnet is None
        self.ratio = ratio[0] // ratio[1] + 1 if ratio[0] // ratio[1] > 0 else 1
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon = ce_epsilon).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.ALoss = nn.MSELoss().cuda()
        self.lamda = lam

    def train(self, epoch, real_loader_source,fake_loader_source,  optimizer, 
                train_iters=200, print_freq=1, logger=None, forDisNet=None, **kwargs):
        self.model.train()
        train_loader_target, optimizerD = None, None
        if self.islim:
            self.disnet.train()
            train_loader_target, optimizerD = forDisNet

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_limD = AverageMeter()
        losses_limM = AverageMeter()


        flag = True
        for i in range(train_iters):
            if flag and (i+1) % self.ratio == 0:
                source_inputs = fake_loader_source.next()
                i -= 1
                flag=False
            else:
                flag = True
                source_inputs = real_loader_source.next()

            s_inputs, targets, _ = self._parse_data(source_inputs)
            [s_features, s_features_part], [prob, prob_part] = self.model(s_inputs)
            loss_ce1, loss_tr1, _ = self._forward_loss(s_features, prob, targets)
            loss_ce2, loss_tr2, _ = self._forward_loss(s_features_part, prob_part, targets)
            loss = loss_ce1 + loss_tr1 #could improve it?
            loss += 1 * (loss_ce2 + loss_tr2) #could improve it?
            
            if self.islim:
                t_inputs, _, _, _, _ = train_loader_target.next()
                [t_features, _], _ = self.model(t_inputs.cuda())   

            #train model with netD
            loss_lim = 0
            if epoch >= 1 and self.islim:
                D_duke = self.disnet(t_features)
                D_market = self.disnet(s_features)
                loss_lim = (self.ALoss(D_duke, torch.ones_like(D_duke)/2.) + self.ALoss(D_market, torch.ones_like(D_market))/2.)
                loss += loss_lim * self.lamda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # train discriminator
            d_loss=0
            if self.islim:
                D_duke = self.disnet(t_features.detach())
                D_market = self.disnet(s_features.detach())
                d_loss = self.ALoss(D_market, torch.ones_like(D_market)) + self.ALoss(D_duke, torch.zeros_like(D_duke))
                optimizerD.zero_grad()
                d_loss.backward()
                optimizerD.step()

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            losses_limM.update(loss_lim.item() if loss_lim > 0 else 0)
            losses_limD.update(d_loss.item() if d_loss > 0 else 0)

            

            if ((i + 1) % print_freq == 0):
                logger.traininglog(epoch=epoch, i=i+1, iters=train_iters,
                    avgloss=loss, 
                    loss_ce=f"{losses_ce.val:.3f}({losses_ce.avg:.3f})",
                    loss_tr=f"{losses_tr.val:.3f}({losses_tr.avg:.3f})",
                    loss_G=f"{losses_limM.val:.3f}({losses_limM.avg:.3f})",
                    loss_D=f"{losses_limD.val:.3f}({losses_limD.avg:.3f})")


    def _parse_data(self, inputs):
        imgs, _, pids, _, isreal= inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, isreal

    def _forward_loss(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec


#Finetune Trainer
class FTTrainer(object):
    def __init__(self, model, num_cluster_list, npart=2, 
                    ce_epsilon=0.1, 
                    cross_agreements=None, 
                    uetal=0.4, uetbe=0.2, graph=None,
                    logger=None, **kwargs) :
        super(FTTrainer, self).__init__()
        self.model = model
        self.num_clusters = num_cluster_list

        #pplr loss
        self.num_part = npart
        self.ca_scores = cross_agreements
        self.aff_score = graph

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster_list, epsilon = ce_epsilon).cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.).cuda() #0.3 | 0.1 | 0.2 |0.4
        self.criterion_reg = RegLoss()

        self.criterion_dauet = UET(alpha=uetal, beta=uetbe).cuda()

        self.logger = logger

    def train(self, epoch, data_loader_target, optimizer, 
            ce_weights=(1, 1, 1), tri_weights=(1, 1, 1), reg_weights=(0,0),
            prefeatures=None, allabels=None,
            print_freq=1, train_iters=200, **kwargs):
        self.model.train()

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_reg = AverageMeter()

        gtw, utw, btw    = tri_weights # global | upper part | bottom part branch tri weight 
        gcew, ucew, bcew = ce_weights
        grw, prw         = reg_weights

        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            # process inputs
            inputs,targets, ca, afa = self._parse_data(target_inputs)
            # forward
            [x, part], _, [prob, prob_part] = self.model(inputs, finetune = True)
            prob = prob[:,:self.num_clusters]
            prob_part = prob_part[:,:self.num_clusters,:]
            del inputs
        #####################################################
        #                           LOSSES                  #

        #ID Loss global 
            neighbours = self._infer(prefeatures) #Could improve it? new way to get neighbours
            loss_ce, refined_targets = self.criterion_dauet(logits_g=prob,logits_p=prob_part, 
                        neighbours=neighbours,
                        targets=targets, aff_score=afa, ca_score=ca)
        #ID Loss part   
            loss_upce = self.criterion_ce(prob_part[:, :, 0], targets) if ucew > 0 else 0
            loss_bpce = self.criterion_ce(prob_part[:, :, 1], targets) if bcew > 0 else 0

        #Tri loss
            gloss_tri = self.criterion_tri(x, x, targets)
            uploss_tri = self.criterion_tri(part[:, :, 0], part[:, :, 0], targets)
            bploss_tri = self.criterion_tri(part[:, :, 1], part[:, :, 1], targets)
               

        #Regulation
            # gloss_reg = self.criterion_reg(prob, afa, ensembled_preds)
            
        #ToTal loss
            loss_tri= gtw * gloss_tri + utw * uploss_tri + btw * bploss_tri
            loss_ce = gcew * loss_ce + ucew * loss_upce + bcew * loss_bpce
            # loss_reg = grw * gloss_reg

            loss =  loss_ce  + loss_tri
                   
            #optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tri.item())
            # losses_reg.update(0 if loss_reg == 0 else loss_reg.item())

            if (i + 1) % print_freq == 0:
                self.logger.traininglog(epoch=epoch+1, i=i+1, iters=train_iters,
                    Lce=f"{losses_ce.avg:.2f}",
                    Ltr=f"{losses_tr.avg:.2f}",
                    # Lreg=f"{losses_reg.avg:.2f}",
                    avgloss=loss, 
                    )

    def _parse_data(self, inputs):
        imgs_1, _, pids, _, (_, newidx) = inputs #img, fname, pid, camid, (old_idx, new_idx)
        inputs_1 = imgs_1.cuda()
        targets = pids.cuda()
        ca = self.ca_scores[newidx].cuda()
        afa = self.aff_score[newidx] if not self.aff_score is None else None
        return inputs_1, targets, ca, afa
    
    def _infer(self, idxs,**kwargs):
        b = idxs.shape[0] // 2
        # features = idxs #N x num_feature
        features_1, features_2 = idxs[:b], idxs[b:]
        weights = self.model.module.classifier.weight.data[:self.num_clusters]  #num_featurexClass
        logits_1 = features_1.cuda() @ weights.T  #N/2 x C
        del features_1
        logits_2 = features_2.cuda() @ weights.T  #N/2 x C 
        del features_2
        logits = torch.cat([logits_1, logits_2])       
        return logits.detach().cpu()

#Finetune Trainer
class FTTrainerwCGL(object):
    def __init__(self, model, model_ema=None, num_cluster_list=None,  ce_epsilon=0.1, tri_margin=0.3, cetr_margin=1,
                    uetal=0.4, cent_uncertainty=None, centp_uncertainty=None, alphas=None,
                    logger=None, **kwargs) :
        super(FTTrainerwCGL, self).__init__()
        self.model = model
        self.model_ema = model_ema
        self.num_clusters = num_cluster_list
        self.cent_uncertainty = cent_uncertainty
        self.centp_uncertainty = centp_uncertainty
        self.alpha_score = alphas

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster_list, epsilon = ce_epsilon).cuda()
        self.criterion_stri = SoftTripletLoss(margin=0).cuda() #0.3 | 0.5
        self.criterion_tri = TripletLoss(margin=tri_margin).cuda() #0.3 |0.5 
        self.criterion_centri = CenterTripletLoss(margin=cetr_margin, num_classes=2048).cuda() #0.3 | 0.1 | 0.2 |0.5 | 1 | 5 
        self.criterion_tri_p = PartAveragedTripletLoss(margin=tri_margin).cuda() #0.3 | 0.1 | 0.2 |0.4
        self.criterion_dauet = UET2(alpha=uetal).cuda()

        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
        self.criterion_ce_soft = KLDivLoss().cuda()

        self.logger = logger

    def train(self, epoch, data_loader_target, optimizer, 
            ce_weights=(1, 1), tri_weights=(1, 1), reg_weights=0.01, ema_weights=(0.5, 0.8),
            print_freq=1, train_iters=200,  **kwargs):
        self.model.train()
        if not self.model_ema is None: self.model_ema.train()

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_reg = AverageMeter()
        losses_ce_ema = AverageMeter()
        losses_tr_ema = AverageMeter()
        losses_total = AverageMeter()

        gtw, ptw    = tri_weights 
        gcew, pcew = ce_weights
        regw = reg_weights

        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            inputs, targets, gcentroidW, pcentroidW, alphas  = self._parse_data(target_inputs)
            # masks  = self.saliency_map(inputs) 
            # inputs = sefl.adaptive_erasing(inputs, masks).cuda()
            if self.model_ema is None:
                [x, part], [prob, prob_part] = self.model(inputs[0].cuda())
                prob = prob[:,:self.num_clusters]
                prob_part = prob_part[:,:self.num_clusters]
            else:
                # B = inputs[0].shape[0] // 2
                # inputs1, inputs2, inpute1, inpute2 = inputs[0][:B], inputs[0][B:], inputs[1][:B], inputs[1][B:]
                # [x, _, prob, prob_part], [x_ema, _, prob_ema, prob_part_ema] = self._infer_ema((inputs1, inpute1))
                # [x1, _, prob1, prob_part1], [x_ema1, _, prob_ema1, prob_part_ema1] = self._infer_ema((inputs2, inpute2))
                # x = torch.cat([x, x1])
                # # part = torch.cat([part, part1])
                # prob = torch.cat([prob, prob1])
                # prob_part = torch.cat([prob_part, prob_part1])
                # x_ema = torch.cat([x_ema, x_ema1]) #EMA
                # prob_ema = torch.cat([prob_ema, prob_ema1])
                # prob_part_ema = torch.cat([prob_part_ema, prob_part_ema1])


                [x, _], [prob, prob_part] = self.model(inputs[0].cuda())
                [x_ema, _], [prob_ema, prob_part_ema] = self.model_ema(inputs[1].cuda())
                prob = prob[:,:self.num_clusters]
                prob_part = prob_part[:,:self.num_clusters]
                prob_ema = prob_ema[:,:self.num_clusters]
                prob_part_ema = prob_part_ema[:,:self.num_clusters]

            
            # del inputs

            
        #####################################################
        #                           LOSSES                  #

        #ID Loss global 
            gloss_ce    = self.criterion_dauet(logits=prob, targets=targets, aff_score=gcentroidW, alphas=alphas)
            ploss_ce    = self.criterion_dauet(logits=prob_part, targets=targets, aff_score=pcentroidW, alphas=alphas) if pcew > 0 else 0

        #Tri loss
            gloss_tri  = self.criterion_stri(x, x, targets) if gtw > 0 else 0
            # gloss_tri  = self.criterion_centri(x, targets) if gtw > 0 else 0
            ploss_tri  = 0
            # if ptw > 0:
            #     # part = part.transpose(2,1)
            #     # ploss_tri = self.criterion_tri_p(part, targets)
            #     ploss_tri = self.criterion_stri(part, part, targets)
               
        #Regulation
            # loss_reg = self.criterion_centri(inputs=prob, targets=targets) if regw > 0 else 0
            loss_reg =  0
            loss_tri= gtw * gloss_tri + ptw * ploss_tri
            loss_ce = gcew * gloss_ce + pcew * ploss_ce

        #ema loss:
            loss = loss_ce_soft = loss_tri_soft = 0.
            if not self.model_ema is None:
                loss_ce_soft = self.criterion_ce_soft(prob, prob_ema) + \
                              (self.criterion_ce_soft(prob_part, prob_part_ema)  if pcew > 0 else 0) #improve it?
                # loss_tri_soft = self.criterion_tri_soft(x, x_ema, targets) + \
                #             (self.criterion_tri_soft(part, part_ema, targets) if ptw > 0 else 0) 
                loss_tri_soft = self.criterion_tri_soft(x, x_ema, targets)
                
                a, b = ema_weights
                loss =  (1-a) * loss_ce  + (1-b) * loss_tri + a * loss_ce_soft + b * loss_tri_soft + loss_reg 
            else:
                loss =  loss_ce  + loss_tri + loss_reg
            
                   
            #optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not self.model_ema is None: self._update_ema_variables(self.model, self.model_ema, 0.999, epoch*len(data_loader_target)+i)

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tri.item())
            losses_ce_ema.update(0 if loss_ce_soft == 0 else loss_ce_soft.item())
            losses_tr_ema.update(0 if loss_tri_soft == 0 else loss_tri_soft.item())

            losses_total.update(loss.item())
            losses_reg.update(0 if loss_reg == 0 else loss_reg.item())
            
            if (i + 1) % print_freq == 0:
                self.logger.traininglog(epoch=epoch+1, i=i+1, iters=train_iters,
                    Lce=f"{losses_ce.avg:.2f}",
                    Ltr=f"{losses_tr.avg:.2f}",
                    Lreg=f"{losses_reg.avg:.2f}",
                    Lce_ema=f"{losses_ce_ema.avg:.2f}",
                    Ltr_ema=f"{losses_tr_ema.avg:.2f}",
                    avgloss=losses_total.avg, 
                    )
    def _infer_ema(self, inputs):
        [x, part], [prob, prob_part] = self.model(inputs[0].cuda())
        [x_ema, _], [prob_ema, prob_part_ema] = self.model_ema(inputs[1].cuda())
        prob_ema = prob_ema[:,:self.num_clusters]
        prob_part_ema = prob_part_ema[:,:self.num_clusters]

        prob = prob[:,:self.num_clusters]
        prob_part = prob_part[:,:self.num_clusters]
        return [x, part, prob, prob_part], [x_ema, None, prob_ema, prob_part_ema]

    def _parse_data(self, inputs):
        imgs, _, pids, _, (_, newidx) = inputs #img, fname, pid, camid, (old_idx, new_idx)
        targets = pids.cuda()
        aga = self.cent_uncertainty[newidx].cuda()
        apa = self.centp_uncertainty[newidx].cuda() if not self.centp_uncertainty is None else None
        alpha = self.alpha_score[newidx].cuda() if not self.alpha_score is None else None
        if self.model_ema is None:
            return (imgs, None) , targets, aga, apa, alpha
        else:
            return imgs , targets, aga, apa, alpha

    def _update_ema_variables(self, model, ema_model, alpha=0.99, global_step=0):
        # alpha = min(1 - 1 / (global_step * 0.01 + 1), alpha)
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    # Define saliency map function
    def saliency_map(self, inputs):
        # Convert PIL image to PyTorch tensor
        inputs.requires_grad = True
        
        # Forward pass through model to obtain logits
        _, [logits, _] = self.model(inputs.cuda())
        
        # Compute gradients of logits w.r.t. input image
        gradients = torch.autograd.grad(logits.max(), inputs)[0]
        
        # Compute pixel-wise saliency map as L2 norm of gradients
        saliency_map = F.normalize(gradients.pow(2).sum(dim=0).sqrt(), p=2)
        
        # Convert saliency map to NumPy array and resize to match input image size
        saliency_map = saliency_map.detach().cpu().numpy()
        
        return saliency_map

    def adaptive_erasing(self, image, saliency_map, saliency_threshold=0.5, erasing_prob=0.5, erasing_scale=(0.02, 0.2)):
        if random.uniform(0, 1) >= erasing_prob:
            return img
        
        # Threshold saliency map to obtain binary mask
        mask = np.zeros_like(saliency_map)
        mask[saliency_map > saliency_threshold] = 1
        
        # Compute size and location of erasing rectangle based on saliency map
        b, h, w = mask.shape
        mask_indices = np.argwhere(mask == 1)
        ymin, ymax = mask_indices[:, 0].min(), mask_indices[:, 0].max()
        xmin, xmax = mask_indices[:, 1].min(), mask_indices[:, 1].max()
        erasing_h = int(np.random.uniform(erasing_scale[0], erasing_scale[1]) * h)
        erasing_w = int(np.random.uniform(erasing_scale[0], erasing_scale[1]) * w)
        erasing_ymin = max(ymin - int((erasing_h - ymax + ymin) * np.random.rand()), 0)
        erasing_ymax = min(ymax + int((erasing_h - ymax + ymin) * np.random.rand()), h)
        erasing_xmin = max(xmin - int((erasing_w - xmax + xmin) * np.random.rand()), 0)
        erasing_xmax = min(xmax + int((erasing_w - xmax + xmin) * np.random.rand()), w)
        
        ######
        mean=(0.4914, 0.4822, 0.4465)
        image[0, erasing_ymin:erasing_ymax, erasing_xmin:erasing_xmax] = mean[0]
        image[1, erasing_ymin:erasing_ymax, erasing_xmin:erasing_xmax] = mean[1]
        image[2, erasing_ymin:erasing_ymax, erasing_xmin:erasing_xmax] = mean[2]
        
        return image



#Finetune Trainer
class FTTrainerwCGL2(object):
    def __init__(self, model, model_ema=None, num_cluster_list=None,  ce_epsilon=0.1, tri_margin=0.3, cetr_margin=1,
                    uetal=0.4, cent_uncertainty=None, alphas=None, num_parts=2,
                    logger=None, **kwargs) :
        super(FTTrainerwCGL2, self).__init__()
        self.model = model
        self.model_ema = model_ema
        self.num_parts = num_parts
        self.num_clusters = num_cluster_list
        self.cent_uncertainty = cent_uncertainty
        # self.centp_uncertainty = centp_uncertainty
        self.alpha_score = alphas

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster_list, epsilon = ce_epsilon).cuda()
        self.criterion_stri = SoftTripletLoss(margin=0).cuda() #0.3 | 0.5
        # self.criterion_tri = TripletLoss(margin=tri_margin).cuda() #0.3 |0.5 
        # self.criterion_centri = CenterTripletLoss(margin=cetr_margin, num_classes=2048).cuda() #0.3 | 0.1 | 0.2 |0.5 | 1 | 5 
        # self.criterion_tri_p = PartAveragedTripletLoss(margin=tri_margin).cuda() #0.3 | 0.1 | 0.2 |0.4
        self.criterion_dauet = UET2(alpha=uetal).cuda()

        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
        self.criterion_ce_soft = KLDivLoss().cuda()

        self.logger = logger

    def train(self, epoch, data_loader_target, optimizer, 
            tri_weights=(1 , 1), reg_weights=0.01, ema_weights=(0.5, 0.8),
            print_freq=1, train_iters=200,  **kwargs):
        self.model.train()
        if not self.model_ema is None: self.model_ema.train()

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_reg = AverageMeter()
        losses_ce_ema = AverageMeter()
        losses_tr_ema = AverageMeter()
        losses_total = AverageMeter()

        ptw    = tri_weights 
        assert len(ptw) == self.num_parts
        regw = reg_weights

        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            inputs, targets, gcentroidW, alphas  = self._parse_data(target_inputs)
            # masks  = self.saliency_map(inputs) 
            # inputs = sefl.adaptive_erasing(inputs, masks).cuda()

            # B = inputs[0].shape[0] // 2
            # inputs1, inputs2, inpute1, inpute2 = inputs[0][:B], inputs[0][B:], inputs[1][:B], inputs[1][B:]
            # [x, part, prob], [x_ema, x_part_ema, prob_ema] = self._infer_ema((inputs1, inpute1))
            # [x1, part1, prob1], [x_ema1, x_part_ema1, prob_ema1] = self._infer_ema((inputs2, inpute2))
            # x = torch.cat([x, x1])
            # part = torch.cat([part, part1])
            # prob = torch.cat([prob, prob1])
            # x_ema = torch.cat([x_ema, x_ema1])
            # prob_ema = torch.cat([prob_ema, prob_ema1])
            # x_part_ema = torch.cat([x_part_ema, x_part_ema1])


            [x, part], [prob, _] = self.model(inputs[0].cuda())
            [x_ema, part_ema], [prob_ema, _] = self.model_ema(inputs[1].cuda())
            prob = prob[:,:self.num_clusters[0]]
            prob_ema = prob_ema[:,:self.num_clusters[0]]


            
            # del inputs

            
        #####################################################
        #                           LOSSES                  #

        #ID Loss global 
            gloss_ce    = self.criterion_dauet(logits=prob, targets=targets[0], aff_score=gcentroidW, alphas=alphas)

        #Tri loss
            gloss_tri  = self.criterion_stri(x, x, targets[0]) if gtw > 0 else 0
            ploss_tri  = 0
            for i in range(self.num_parts):
                if ptw[i] > 0:
                    ploss_tri += self.criterion_stri(part[:,:, i], part[:, :, i], targets[1][i]) * ptw[i]
               
        #Regulation
            # loss_reg = self.criterion_centri(inputs=prob, targets=targets) if regw > 0 else 0
            loss_reg = 0
            loss_tri= gloss_tri + ploss_tri
            loss_ce = gloss_ce 

        #ema loss:
            loss_ce_soft = self.criterion_ce_soft(prob, prob_ema) 
            loss_tri_soft = self.criterion_tri_soft(x, x_ema, targets)
            # for i in range(self.num_parts):
            #     if ptw[i] > 0:
            #         loss_tri_soft += self.criterion_tri_soft(part[:,:, i], part_ema[:, :, [i]], targets[1][i]) * ptw[i]
            
            a, b = ema_weights
            loss =  (1-a) * loss_ce  + (1-b) * loss_tri + a * loss_ce_soft + b * loss_tri_soft #+ loss_reg 

                   
            #optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not self.model_ema is None: self._update_ema_variables(self.model, self.model_ema, 0.99, epoch*len(data_loader_target)+i)

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tri.item())
            losses_ce_ema.update(0 if loss_ce_soft == 0 else loss_ce_soft.item())
            losses_tr_ema.update(0 if loss_tri_soft == 0 else loss_tri_soft.item())

            losses_total.update(loss.item())
            losses_reg.update(0 if loss_reg == 0 else loss_reg.item())
            
            if (i + 1) % print_freq == 0:
                self.logger.traininglog(epoch=epoch+1, i=i+1, iters=train_iters,
                    Lce=f"{losses_ce.avg:.2f}",
                    Ltr=f"{losses_tr.avg:.2f}",
                    Lreg=f"{losses_reg.avg:.2f}",
                    Lce_ema=f"{losses_ce_ema.avg:.2f}",
                    Ltr_ema=f"{losses_tr_ema.avg:.2f}",
                    avgloss=losses_total.avg, 
                    )
    def _infer_ema(self, inputs):
        [x, part], [prob, _] = self.model(inputs[0].cuda())
        [x_ema, part_ema], [prob_ema,_] = self.model_ema(inputs[1].cuda())
        prob_ema = prob_ema[:,:self.num_clusters[0]]
        prob = prob[:,:self.num_clusters[0]]
        return [x, part, prob], [x_ema, part_ema, prob_ema]

    def _parse_data(self, inputs):
        imgs, _, gpids, ppids, (_, newidx) = inputs #img, fname, gpids, ppids, (old_idx, new_idx)
        targets = pids.cuda()
        ptargets = [x.cuda() for x in ppids]
        aga = self.cent_uncertainty[newidx].cuda()
        alpha = self.alpha_score[newidx].cuda() if not self.alpha_score is None else None

        return imgs , (targets, ptargets), aga, alpha

    def _update_ema_variables(self, model, ema_model, alpha=0.99, global_step=0):
        # alpha = min(1 - 1 / (global_step * 0.01 + 1), alpha)
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


#Finetune Trainer
class HardFTTrainerwCGL(object):
    def __init__(self, model, model_ema=None, use_2branch=False, num_cluster_list=None,  ce_epsilon=0.1, tri_margin=0.3, cetr_margin=1,
                    uetal=0.4, cent_uncertainty=None, centp_uncertainty=None, alphas=None,
                    logger=None, **kwargs) :
        super(HardFTTrainerwCGL, self).__init__()
        self.model = model
        self.model_ema = model_ema
        self.num_clusters = num_cluster_list
        self.cent_uncertainty = cent_uncertainty
        self.centp_uncertainty = centp_uncertainty
        self.alpha_score = alphas
        self.use_2branch = use_2branch

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster_list, epsilon = ce_epsilon).cuda()
        self.criterion_stri = SoftTripletLoss(margin=0).cuda() #0.3 | 0.5
        self.criterion_tri = TripletLoss(margin=tri_margin).cuda() #0.3 |0.5 
        self.criterion_centri = CenterTripletLoss(margin=cetr_margin, num_classes=2048).cuda() #0.3 | 0.1 | 0.2 |0.5 | 1 | 5 
        self.criterion_tri_p = PartAveragedTripletLoss(margin=tri_margin).cuda() #0.3 | 0.1 | 0.2 |0.4
        self.criterion_dauet = UET2(alpha=uetal).cuda()

        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
        self.criterion_ce_soft = KLDivLoss().cuda()

        self.logger = logger

    def train(self, epoch, data_loader_target, optimizer, 
            ce_weights=(1, 1), tri_weights=(1, 1), reg_weights=0.01, ema_weights=(0.5, 0.8),
            print_freq=1, train_iters=200,  **kwargs):
        if self.use_2branch:
            self.train_use_2branch(epoch, data_loader_target, optimizer, ce_weights, tri_weights, reg_weights, ema_weights, print_freq, train_iters)
        else: 
            self.train_not_use_2branch(epoch, data_loader_target, optimizer, ce_weights, tri_weights, reg_weights, ema_weights, print_freq, train_iters)
    
    def train_not_use_2branch(self, epoch, data_loader_target, optimizer, 
            ce_weights=(1, 1), tri_weights=(1, 1), reg_weights=0.01, ema_weights=(0.5, 0.8),
            print_freq=1, train_iters=200):
        self.model.train()
        if not self.model_ema is None: self.model_ema.train()

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_reg = AverageMeter()
        losses_ce_ema = AverageMeter()
        losses_tr_ema = AverageMeter()
        losses_total = AverageMeter()

        gtw, ptw    = tri_weights # global | upper part | bottom part branch tri weight 
        gcew, pcew = ce_weights
        regw = reg_weights

        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            inputs, targets, gcentroidW, pcentroidW, alphas  = self._parse_data(target_inputs)
            # masks  = self.saliency_map(inputs) 
            # inputs = sefl.adaptive_erasing(inputs, masks).cuda()
            if self.model_ema is None:
                [x, part], [prob, prob_part] = self.model(inputs[0].cuda())
            else:
                [x, part], [prob, prob_part] = self.model(inputs[0].cuda())
                [x_ema, part_ema], [prob_ema, prob_part_ema] = self.model_ema(inputs[1].cuda())
                prob_ema = prob_ema[:,:self.num_clusters]
                prob_part_ema = prob_part_ema[:,:self.num_clusters]
            
            del inputs
            prob = prob[:,:self.num_clusters]
            prob_part = prob_part[:,:self.num_clusters]
            
        #####################################################
        #                           LOSSES                  #

        #ID Loss global 
            gloss_ce    = self.criterion_dauet(logits=prob, targets=targets, aff_score=gcentroidW, alphas=alphas)
            ploss_ce    = self.criterion_dauet(logits=prob_part, targets=targets, aff_score=pcentroidW, alphas=alphas) if pcew > 0 else 0

        #Tri loss
            gloss_tri  = self.criterion_stri(x, x, targets) if gtw > 0 else 0
            ploss_tri  = 0
            if ptw > 0:
                # part = part.transpose(2,1)
                # ploss_tri = self.criterion_tri_p(part, targets)
                ploss_tri = self.criterion_stri(part, part, targets)
               
        #Regulation
            loss_reg = self.criterion_centri(inputs=prob, targets=targets) if regw > 0 else 0
            loss_tri= gtw * gloss_tri + ptw * ploss_tri
            loss_ce = gcew * gloss_ce + pcew * ploss_ce

        #ema loss:
            loss = loss_ce_soft = loss_tri_soft = 0.
            if not self.model_ema is None:
                loss_ce_soft = self.criterion_ce_soft(prob, prob_ema) + \
                              (self.criterion_ce_soft(prob_part, prob_part_ema)  if pcew > 0 else 0)
                loss_tri_soft = self.criterion_tri_soft(x, x_ema, targets) + \
                              (self.criterion_tri_soft(part, part_ema, targets) if ptw > 0 else 0) 
                
                a, b = ema_weights
                loss =  (1-a) * loss_ce  + (1-b) * loss_tri + a * loss_ce_soft + b * loss_tri_soft + loss_reg 
            else:
                loss =  loss_ce  + loss_tri + loss_reg
                   
            #optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not self.model_ema is None: self._update_ema_variables(self.model, self.model_ema, 0.999, epoch*len(data_loader_target)+i)

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tri.item())
            losses_ce_ema.update(0 if loss_ce_soft == 0 else loss_ce_soft.item())
            losses_tr_ema.update(0 if loss_tri_soft == 0 else loss_tri_soft.item())

            losses_total.update(loss.item())
            losses_reg.update(0 if loss_reg == 0 else loss_reg.item())

            if (i + 1) % print_freq == 0:
                self.logger.traininglog(epoch=epoch+1, i=i+1, iters=train_iters,
                    Lce=f"{losses_ce.avg:.2f}",
                    Ltr=f"{losses_tr.avg:.2f}",
                    Lreg=f"{losses_reg.avg:.2f}",
                    Lce_ema=f"{losses_ce_ema.avg:.2f}",
                    Ltr_ema=f"{losses_tr_ema.avg:.2f}",
                    avgloss=losses_total.avg, 
                    )

    def train_use_2branch(self, epoch, data_loader_target, optimizer, ce_weights, tri_weights, reg_weights, ema_weights,  print_freq, train_iters):
        self.model.train()
        if not self.model_ema is None: self.model_ema.train()

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_reg = AverageMeter()
        losses_ce_ema = AverageMeter()
        losses_tr_ema = AverageMeter()
        losses_total = AverageMeter()

        gtw, ptw    = tri_weights # global | upper part | bottom part branch tri weight 
        gcew, pcew = ce_weights
        regw = reg_weights

        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            inputs, targets, gcentroidW, pcentroidW, alphas  = self._parse_data(target_inputs)
            # masks  = self.saliency_map(inputs) 
            # inputs = sefl.adaptive_erasing(inputs, masks).cuda()
            if self.model_ema is None:
                [x, part1, part2], [prob, prob_part, prob_part2] = self.model(inputs[0].cuda())
            else:
                [x, part1, part2], [prob, prob_part, prob_part2] = self.model(inputs[0].cuda())
                [x_ema, part_ema, part_ema2], [prob_ema, prob_part_ema, prob_part_ema2] = self.model_ema(inputs[1].cuda())
                prob_ema = prob_ema[:,:self.num_clusters]
                prob_part_ema = prob_part_ema[:,:self.num_clusters]
                prob_part_ema2 = prob_part_ema2[:,:self.num_clusters]
            
            del inputs
            prob = prob[:,:self.num_clusters]
            prob_part1 = prob_part[:,:self.num_clusters]
            prob_part2 = prob_part2[:,:self.num_clusters]
            
        #####################################################
        #                           LOSSES                  #

        #ID Loss global 
            gloss_ce    = self.criterion_dauet(logits=prob, targets=targets, aff_score=gcentroidW, alphas=alphas)
            ploss_ce    = (self.criterion_dauet(logits=prob_part1, targets=targets, aff_score=pcentroidW[0], alphas=alphas) \
                            + self.criterion_dauet(logits=prob_part2, targets=targets, aff_score=pcentroidW[1], alphas=alphas))

        #Tri loss
            gloss_tri  = self.criterion_stri(x, x, targets) if gtw > 0 else 0
            # gloss_tri  = self.criterion_centri(x, targets) if gtw > 0 else 0
            ploss_tri  = 0
            if ptw > 0:
                # part = part.transpose(2,1)
                # ploss_tri = self.criterion_tri_p(part, targets)
                ploss_tri = self.criterion_stri(part2, part2, targets) + self.criterion_stri(part1, part1, targets)
               
        #Regulation
            loss_reg = self.criterion_centri(inputs=prob, targets=targets) if regw > 0 else 0
            loss_tri= gtw * gloss_tri + ptw * ploss_tri
            loss_ce = gcew * gloss_ce + pcew * ploss_ce

        #ema loss:
            loss = loss_ce_soft = loss_tri_soft = 0.
            if not self.model_ema is None:
                loss_ce_soft = self.criterion_ce_soft(prob, prob_ema) + \
                              (self.criterion_ce_soft(prob_part, prob_part_ema)  if pcew > 0 else 0)
                loss_tri_soft = self.criterion_tri_soft(x, x_ema, targets) + \
                              (self.criterion_tri_soft(part, part_ema, targets) if ptw > 0 else 0) 
                
                a, b = ema_weights
                loss =  (1-a) * loss_ce  + (1-b) * loss_tri + a * loss_ce_soft + b * loss_tri_soft + loss_reg 
            else:
                loss =  loss_ce  + loss_tri + loss_reg
                   
            #optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not self.model_ema is None: self._update_ema_variables(self.model, self.model_ema, 0.999, epoch*len(data_loader_target)+i)

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tri.item())
            losses_ce_ema.update(0 if loss_ce_soft == 0 else loss_ce_soft.item())
            losses_tr_ema.update(0 if loss_tri_soft == 0 else loss_tri_soft.item())

            losses_total.update(loss.item())
            losses_reg.update(0 if loss_reg == 0 else loss_reg.item())

            if (i + 1) % print_freq == 0:
                self.logger.traininglog(epoch=epoch+1, i=i+1, iters=train_iters,
                    Lce=f"{losses_ce.avg:.2f}",
                    Ltr=f"{losses_tr.avg:.2f}",
                    Lreg=f"{losses_reg.avg:.2f}",
                    Lce_ema=f"{losses_ce_ema.avg:.2f}",
                    Ltr_ema=f"{losses_tr_ema.avg:.2f}",
                    avgloss=losses_total.avg, 
                    )


    def _parse_data(self, inputs):
        imgs, _, pids, _, (_, newidx) = inputs #img, fname, pid, camid, (old_idx, new_idx)
        targets = pids.cuda()
        aga = self.cent_uncertainty[newidx].cuda()
        apa = self.centp_uncertainty[0][newidx].cuda(), self.centp_uncertainty[1][newidx].cuda() if self.use_2branch else  self.centp_uncertainty[newidx].cuda()
        alpha = self.alpha_score[newidx].cuda() if not self.alpha_score is None else None
        if self.model_ema is None:
            return (imgs, None) , targets, aga, apa, alpha
        else:
            return imgs , targets, aga, apa, alpha

    def _update_ema_variables(self, model, ema_model, alpha=0.99, global_step=0):
        alpha = min(1 - 1 / (global_step * 0.01 + 1), alpha)
        # alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
