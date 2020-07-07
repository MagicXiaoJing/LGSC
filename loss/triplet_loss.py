import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

class TripletLoss_Baidu(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=0.3, special_label=None, hard_factor=0.0):
        self.margin = margin
        self.special_label = special_label
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist = euclidean_dist(global_feat, global_feat)

        ap_dist = dist.reshape(dist.shape[0], dist.shape[1], 1)
        an_dist = dist.reshape(dist.shape[0], 1, dist.shape[1])

        batchsize = labels.shape[0]
        # loss(i,j,k) = dis(i,j)-dis(i,k)+margin
        loss = ap_dist.expand(ap_dist.shape[0], ap_dist.shape[1], batchsize) + self.margin - an_dist.expand(an_dist.shape[0], an_dist.shape[1]*batchsize, an_dist.shape[2])

        indice_equal = torch.diag(torch.ones(batchsize)).cuda()
        indice_not_equal = 1.0 - indice_equal

        labels_use = 1-labels
        # broad_matrix(i,j) = label(i)+label(j)
        broad_matrix = labels_use.reshape(-1,1).expand(batchsize, batchsize) + \
                       labels_use.reshape(1,-1).expand(batchsize, batchsize)

        pp = broad_matrix==0
        pp = (indice_not_equal * pp).reshape(pp.shape[0], pp.shape[1], 1)

        pn = broad_matrix==1
        pn = (indice_not_equal * pn).reshape(1, pn.shape[0], pn.shape[1])

        # apn(i,j,k) = pp(i,j)*pn(j,k) if apn(i,j,k)=1 i==j j!=k
        apn = pp.expand(pp.shape[0], pp.shape[1], batchsize) * pn.expand(batchsize, pn.shape[1], pn.shape[2])

        ap_smaller_an = ap_dist.expand(ap_dist.shape[0], ap_dist.shape[1], batchsize) <  \
                        an_dist.expand(an_dist.shape[0], an_dist.shape[1] * batchsize,an_dist.shape[2])
        apmargin_larger_an = (ap_dist.expand(ap_dist.shape[0], ap_dist.shape[1], batchsize) + self.margin) > \
                        an_dist.expand(an_dist.shape[0], an_dist.shape[1] * batchsize, an_dist.shape[2])
        apn_1 = ap_smaller_an * apmargin_larger_an
        apn = apn #* apn_1

        loss = (loss * apn).clamp(0)
        loss_g = loss[loss!=0]
        if loss_g.shape[0] == 0:
            return loss.mean(), loss.mean()
        else:
            return loss_g.mean(), loss_g.mean()


