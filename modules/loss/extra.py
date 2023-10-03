"""
Yoonki Cho, Woo Jae Kim, Seunghoon Hong, Sung-Eui Yoon. 28 Mar 2022
url:https://arxiv.org/abs/2203.14675
github:https://github.com/yoonkicho/pplr
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AALS(nn.Module):
    """ Agreement-aware label smoothing """
    def __init__(self):
        super(AALS, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, logits, targets, ca):
        log_preds = self.logsoftmax(logits)  # B * C
        targets = torch.zeros_like(log_preds).scatter_(1, targets.unsqueeze(1), 1)
        uni = (torch.ones_like(log_preds) / log_preds.size(-1)).cuda()

        loss_ce = (- targets * log_preds).sum(1)
        loss_kld = F.kl_div(log_preds, uni, reduction='none').sum(1)
        loss = (ca * loss_ce + (1-ca) * loss_kld).mean()
        return loss


class PGLR(nn.Module):
    """ Part-guided label refinement """
    def __init__(self, lam=0.5):
        super(PGLR, self).__init__()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.lam = lam

    def forward(self, logits_g, logits_p, targets, ca):
        targets = torch.zeros_like(logits_g).scatter_(1, targets.unsqueeze(1), 1)
        w = torch.softmax(ca, dim=1)  # B * P
        w = torch.unsqueeze(w, 1)  # B * 1 * P
        preds_p = self.softmax(logits_p)  # B * numClusters * numParts
        ensembled_preds = (preds_p * w).sum(2).detach()  # B * class_num
        refined_targets = self.lam * targets + (1-self.lam) * ensembled_preds

        log_preds_g = self.logsoftmax(logits_g)
        loss = (-refined_targets * log_preds_g).sum(1).mean()
        return loss



"""
combine part score and neighbour score to refine pseudo label.
"""
class UET(nn.Module):
    def __init__(self, alpha=0.3, beta=0.2):
        super(UET, self).__init__()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits_g, logits_p, neighbours, targets, aff_score, ca_score):
        #logits : BxC    C: num_classes
        #neighbour : NxC   N: num_samples
        #target  : Bx1
        #aff_score: BxN
        #ca_score : BxP    P:num_parts
        
        targets = torch.zeros_like(logits_g).scatter_(1, targets.unsqueeze(1), 1)
        B, (N, C) = aff_score.shape[0], neighbours.shape
        wa = torch.unsqueeze(aff_score, 1) #B * 1 * N
        wa1, wa2, wa3, wa4 = wa[:32], wa[32:64], wa[64:96], wa[96:] #Trick for mini sample
        preds_n = self.softmax(neighbours)  # N * numClusters
        preds_n = preds_n.T.expand(32, C, N).cuda() #(B,C,N)
        # preds_n1, preds_n2 = preds_n[:B//2], preds_n[B//2:]
        ensembled_preds_n1 = (preds_n * wa1.cuda()).sum(2).cpu() #B/2 * class_num
        ensembled_preds_n2 = (preds_n * wa2.cuda()).sum(2).cpu() #B/2 * class_num
        ensembled_preds_n3 = (preds_n * wa3.cuda()).sum(2).cpu() #B/2 * class_num
        ensembled_preds_n4 = (preds_n * wa4.cuda()).sum(2).cpu() #B/2 * class_num

        del preds_n
        ensembled_preds_n = torch.cat([ensembled_preds_n1,
                                       ensembled_preds_n2,
                                       ensembled_preds_n3,
                                       ensembled_preds_n4]).detach().cuda()

        wc = torch.softmax(ca_score, dim=1)  # B * numParts
        wc = torch.unsqueeze(wc, 1)  # B * 1 * numParts
        preds_p = self.softmax(logits_p)  # B * numClusters * numParts
        ensembled_preds_p = (preds_p * wc).sum(2).detach()  # B * class_num
        
        refined_targets = (1-self.alpha -self.beta) * targets + self.alpha * ensembled_preds_p + self.beta * ensembled_preds_n
        log_preds_g = self.logsoftmax(logits_g)
        loss = (-refined_targets * log_preds_g).sum(1).mean()
        return loss, refined_targets


"""
combine  neighbour centroid score to refine pseudo label.
"""
class UET2(nn.Module):
    def __init__(self, alpha=0.3):
        super(UET2, self).__init__()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.alpha = alpha


    def forward(self, logits, targets, aff_score=None, alphas=None, **kwargs):
        #logits : BxC    C: num_classes
        #neighbour : NxC   N: num_samples
        #target  : Bx1
        #aff_score: BxN
        #ca_score : BxP    P:num_parts
        targets = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        
        if alphas is None:
            refined_targets = (1-self.alpha) * targets + self.alpha * aff_score
        else:
            refined_targets = (1-alphas) * targets + alphas * aff_score
        log_preds = self.logsoftmax(logits)
        loss = (-refined_targets * log_preds).sum(1)
        loss = loss.mean()
        return loss

# Regulation Loss
class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.kl = F.kl_div

    def forward(self, logits_g, afa, e_preds):
        # B, (N, C) = afa.shape[0], neighbours.shape

        glog_preds = self.logsoftmax(logits_g)
        wa = torch.where(afa > 0, afa / afa, afa * 0) # --> [0 0 1 1 0] BxN
        wa = torch.unsqueeze(afa, 1).cpu() #B * 1 * N
        # print(ensembled_preds.shape) B x C x N
        ensembled_preds_n = torch.log(ensembled_preds) * wa #ensembled_preds from softmax - > logsoftmax
        ensembled_preds_n = ensembled_preds_n.mean(2).detach().cuda()  # B * class_num
        
        klloss = self.kl(glog_preds, ensembled_preds_n, reduction='none').sum(1).mean()
        return klloss