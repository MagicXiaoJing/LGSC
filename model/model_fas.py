import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import random
from loss.triplet_loss import TripletLoss_Baidu
from model.resnet import ResNet
from model.unet_res import U_Net


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class model_fas_classifier(nn.Module):
    def __init__(self):
        super(model_fas_classifier, self).__init__()
        self.backbone_img = ResNet("resnet18", -1, use_pretrain=False)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(512, 2)
        self.classifier.apply(weights_init_classifier)

    def forward(self, img, mask, labels=None):
        output = self.backbone_img(img+mask)[-1]#img*(1-mask.abs()))[-1]#img+mask)[-1]
        output = self.global_pool(output)
        output = output.view(output.shape[0], output.shape[1])
        output = self.dropout(output)
        score = self.classifier(output)
        if self.training:
            return score, labels
        else:
            return score

class model_fas(nn.Module):
    def __init__(self, cfg):
        super(model_fas, self).__init__()
        self.cfg = cfg
        self.backbone = U_Net()
        self.aux_classifier = model_fas_classifier()
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, labels=None):
        backbone = self.backbone(x)
        if self.training:
            triplet_feat_list = []
            for feat in backbone[:-1]:
                feat_pool = self.global_pool(feat)
                feat_pool = feat_pool.view(feat_pool.shape[0], feat_pool.shape[1])
                triplet_feat_list.append(feat_pool)
            feat_pool = self.global_pool(backbone[-1].abs())
            feat_pool = feat_pool.view(feat_pool.shape[0], feat_pool.shape[1])
            triplet_feat_list.append(feat_pool)

            labels_mask = labels
            score_classifier, labels_mask = self.aux_classifier(x, backbone[-1], labels_mask)
            return triplet_feat_list, backbone[-1], score_classifier, labels, labels_mask, x
        else:
            score_classifier = self.aux_classifier(x, backbone)
            return backbone, F.softmax(score_classifier)

class Model_Losses(object):

    def __init__(self):
        super(Model_Losses, self).__init__()
        self.triplet_loss = TripletLoss_Baidu(0.3, special_label=1)
        self.loss_reg = nn.L1Loss() # nn.SmoothL1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()#label_smooth_loss()#nn.CrossEntropyLoss()


    def forward(self, features_all, meters):
        triplet_feat_list, mask_img, score_classifier, label_class, labels_mask, img = features_all

        loss_dict = {}

        for idx, triplet_feature in enumerate(triplet_feat_list[:-1]):
            # triplet_loss = self.triplet_loss(triplet_feature, label_class, normalize_feature=False)[0]
            triplet_loss = self.triplet_loss(triplet_feature, label_class, normalize_feature=True)[0]

            if triplet_loss is None:
                continue

            loss_dict.update({'triplet_loss_%d' % idx: triplet_loss})

        aux_classifier_loss = self.cross_entropy(score_classifier, labels_mask)
        loss_dict.update({'aux_classifier_loss': aux_classifier_loss*5})

        nonattack_index = label_class == 1
        mask_img_nonattack = mask_img[nonattack_index]
        if mask_img_nonattack.shape[0]!=0:
            target_img_nonattack = torch.zeros(mask_img_nonattack.shape).cuda()
            loss_mask = self.loss_reg(mask_img_nonattack, target_img_nonattack)
            loss_dict.update({'loss_mask': 5*loss_mask})


        losses = sum(loss for loss in loss_dict.values())
        meters.update(loss=losses, **loss_dict)

        dict_curr = {}


        mask_img_mean = mask_img.abs().view(mask_img.shape[0],-1).mean(dim=1)
        mask_label = mask_img_mean<0.01
        acc_mask_all = (mask_label == label_class).float()
        acc_mask_all_tp = acc_mask_all[nonattack_index]
        acc_mask_all_tn = acc_mask_all[~nonattack_index]
        dict_curr['acc_mask'] = acc_mask_all.mean()
        if acc_mask_all_tp.shape[0]>0:
            acc_tp = acc_mask_all_tp.mean()
            dict_curr['acc_mask_true_positive'] = acc_tp
        if acc_mask_all_tn.shape[0]>0:
            acc_tn = acc_mask_all_tn.mean()
            dict_curr['acc_mask_true_negative'] = acc_tn


        acc_all = (score_classifier.max(1)[1] == labels_mask).float()
        acc_all_tp = acc_all[nonattack_index]
        acc_all_tn = acc_all[~nonattack_index]
        acc = acc_all.mean()
        dict_curr['acc'] = acc


        if acc_all_tp.shape[0]>0:
            acc_tp = acc_all_tp.mean()
            dict_curr['acc_true_positive_%d'%idx] = acc_tp
        if acc_all_tn.shape[0]>0:
            acc_tn = acc_all_tn.mean()
            dict_curr['acc_true_negative_%d'%idx] = acc_tn

        meters.update(**dict_curr)

        return losses



def make_model_fas(cfg):
    model = model_fas(cfg)
    return model


def make_model_fas_loss():
    model_loss = Model_Losses()
    return model_loss