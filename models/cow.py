"""
FSS via CoW
Extended from ADNet code by Hansen et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Res101Encoder
from models.modules import MLP, Decoder, Supp_Decoder
import numpy as np
import random
import cv2
from boundary_loss import BoundaryLoss


class FewShotSeg(nn.Module):
    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()
        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.criterion_b = BoundaryLoss(theta0=3, theta=5)
        self.criterion_MSE = nn.MSELoss()
        self.fg_num = 100
        self.bg_num = 600
        self.mlp1 = MLP(256, self.fg_num)
        self.mlp2 = MLP(256, self.bg_num)
        self.decoder1 = Decoder(self.fg_num)
        self.decoder2 = Decoder(self.bg_num)
        self.supp_decoder = Supp_Decoder()

    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W]
            supp_mask: foreground masks for support images
                way x shot x [B x H x W]
            qry_imgs: query images
                N x [B x 3 x H x W]
            train: whether to train model or not
        """
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)

        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        supp_fts = supp_fts[0]

        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        qry_fts = qry_fts[0]

        ###### Get threshold ######
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        ssp_loss = torch.zeros(1).to(self.device)
        aux_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(supp_bs):
            ###### Extract prototypes ######
            supp_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]
            fg_prototypes = self.getPrototype(supp_fts_)
            if supp_mask[epi, 0, 0].sum() == 0:
                ###### Get query predictions ######
                qry_pred = torch.stack(
                    [self.getPred(qry_fts[epi], fg_prototypes[way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)
                preds = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)
                preds = torch.cat((1.0 - preds, preds), dim=1)
                outputs.append(preds)
                if train:
                    align_loss_epi, b_loss_epi = self.alignLoss(supp_fts[epi], qry_fts[epi], preds, supp_mask[epi])
                    align_loss += align_loss_epi
                    b_loss += b_loss_epi
            else:
                supp_pred = self.getSelfPred(supp_fts[[epi], 0, 0], fg_prototypes[0])
                supp_pred = F.interpolate(supp_pred, size=img_size, mode='bilinear', align_corners=True)
                supp_pred = torch.cat((1.0 - supp_pred, supp_pred), dim=1)
                self_pred = supp_pred
                supp_pred = torch.argmax(supp_pred, dim=1, keepdim=True).squeeze(1)

                fg_pts = [[self.get_fg_pts(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], supp_pred)
                           for shot in range(self.n_shots)] for way in range(self.n_ways)]
                fg_pts = self.get_all_prototypes(fg_pts)

                bg_pts = [[self.get_bg_pts(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], supp_pred)
                           for shot in range(self.n_shots)] for way in range(self.n_ways)]
                bg_pts = self.get_all_prototypes(bg_pts)

                ###### Get query predictions ######
                fg_sim = torch.stack(
                    [self.get_fg_sim(qry_fts[epi], fg_pts[way]) for way in range(self.n_ways)], dim=1).squeeze(0)
                bg_sim = torch.stack(
                    [self.get_bg_sim(qry_fts[epi], bg_pts[way]) for way in range(self.n_ways)], dim=1).squeeze(0)

                fg_pred = F.interpolate(fg_sim, size=img_size, mode='bilinear', align_corners=True)
                bg_pred = F.interpolate(bg_sim, size=img_size, mode='bilinear', align_corners=True)
                preds = torch.cat([bg_pred, fg_pred], dim=1)
                preds = torch.softmax(preds, dim=1)

                outputs.append(preds)
                if train:
                    align_loss_epi, aux_loss_epi, b_loss_epi, ssp_loss_epi = self.align_aux_Loss(supp_fts[epi], qry_fts[epi], supp_mask[epi], preds,
                                                                                                 fg_pts, bg_pts, self_pred)
                    align_loss += align_loss_epi
                    aux_loss += aux_loss_epi
                    b_loss += b_loss_epi
                    ssp_loss += ssp_loss_epi

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])

        return output, align_loss / supp_bs, aux_loss / supp_bs, b_loss / supp_bs, ssp_loss / supp_bs

    def getPred(self, fts, prototype, thresh):
        """
        Args:
            fts: (1, 512, 64, 64)
            prototype: (1, 512)
            thresh: (1, 1)
        """
        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getSelfPred(self, supp_fts, supp_vec):
        """
        Args:
            supp_fts: 1 x 512 x 64 x 64
            supp_vec: 1 x 512
        """
        supp_vec = supp_vec[..., None, None].expand(-1, -1, supp_fts.shape[-2], supp_fts.shape[-1])
        supp_pred = torch.cat([supp_fts, supp_vec, supp_vec], dim=1)
        supp_pred = self.supp_decoder(supp_pred)

        return supp_pred

    def getFeatures(self, fts, mask):
        """
        Args:
            fts: (1, 512, 64, 64)
            mask: (1, 256, 256)
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Args:
            fg_fts: way x shot x [1 x 512]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots
                         for way in fg_fts]

        return fg_prototypes

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        """
        Args:
            supp_fts: (way, shot, 512, 64, 64)
            qry_fts: (N, 512, 64, 64)
            pred: (1, 2, H, W)
            fore_mask: (way, shot, 256, 256)
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            for shot in range(n_shots):
                qry_fts_ = [[self.getFeatures(qry_fts, pred_mask[way + 1])]]
                fg_prototypes = self.getPrototype(qry_fts_)

                # Get predictions
                supp_pred = self.getPred(supp_fts[way, [shot]], fg_prototypes[way], self.thresh_pred[way])
                supp_pred = F.interpolate(supp_pred[None, ...], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                pred_ups = torch.cat((1.0 - supp_pred, supp_pred), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss, b_loss

    def align_aux_Loss(self, supp_fts, qry_fts, fore_mask, pred, sup_fg_pts, sup_bg_pts, self_pred):
        """
        Args:
            supp_fts: (way, shot, 512, 64, 64)
            qry_fts: (N, 512, 64, 64)
            fore_mask: (way, shot, 256, 256)
            pred: (1, 2, 256, 256)
            sup_fg_pts: way x [102 x 512]
            sup_bg_pts: way x [602 x 512]
            self_pred: (1, 2, 256, 256)
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        loss_aux = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        ssp_loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                qry_fts_ = [[self.getFeatures(qry_fts, pred_mask[way + 1])]]
                fg_prototypes = self.getPrototype(qry_fts_)
                qry_pred = self.getSelfPred(qry_fts, fg_prototypes[0])
                qry_pred = F.interpolate(qry_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                qry_pred = torch.cat((1.0 - qry_pred, qry_pred), dim=1)
                qry_pred = torch.argmax(qry_pred, dim=1, keepdim=True).squeeze(1)

                fg_pts_ = [[self.get_fg_pts(qry_fts, pred_mask[way + 1], qry_pred)]]
                fg_pts_ = self.get_all_prototypes(fg_pts_)
                bg_pts_ = [[self.get_bg_pts(qry_fts, pred_mask[way + 1], qry_pred)]]
                bg_pts_ = self.get_all_prototypes(bg_pts_)

                loss_aux += self.get_aux_loss(sup_fg_pts[way], fg_pts_[way], sup_bg_pts[way], bg_pts_[way])

                # Get predictions
                supp_pred = self.get_fg_sim(supp_fts[way, [shot]], fg_pts_[way])
                bg_pred_ = self.get_bg_sim(supp_fts[way, [shot]], bg_pts_[way])
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                bg_pred_ = F.interpolate(bg_pred_, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                # Combine predictions
                preds = torch.cat([bg_pred_, supp_pred], dim=1)
                preds = torch.softmax(preds, dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(preds, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

                ssp_log_prob = torch.log(torch.clamp(self_pred, eps, 1 - eps))
                ssp_loss += self.criterion(ssp_log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss, loss_aux, b_loss, ssp_loss

    def get_fg_pts(self, features, mask, pred_mask):
        """
        Args:
        features: (1, 512, 64, 64)
        mask: (1, 256, 256)
        pred_mask: (1, 256, 256)
        """
        features_trans = F.interpolate(features, size=mask.shape[-2:], mode='bilinear', align_corners=True)

        ie_mask = mask.squeeze(0) - torch.tensor(cv2.erode(mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8), iterations=2)).to(self.device)
        ie_mask = ie_mask.unsqueeze(0)
        ie_prototype = torch.sum(features_trans * ie_mask[None, ...], dim=(-2, -1)) \
                       / (ie_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        origin_prototype = torch.sum(features_trans * mask[None, ...], dim=(-2, -1)) \
                           / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        add_mask = (pred_mask.float() + mask).long()
        mask1 = torch.zeros_like(mask)
        mask2 = torch.zeros_like(mask)
        mask1[add_mask == 2] = 1
        mask2[add_mask == 1] = 1
        mask1[mask == 0] = 0
        mask2[mask == 0] = 0

        fg_fts = self.get_fg_fts(features_trans, mask)
        fg_prototypes = self.mlp1(fg_fts.view(512, 256 * 256)).permute(1, 0)

        if torch.sum(mask2[mask2 == 1]) > 0:
            hard_fg = self.get_random_pts(features_trans, mask2, 50)
            k = random.sample(range(len(fg_prototypes)), 50)
            fg_prototypes = torch.cat([fg_prototypes[k], hard_fg], dim=0)

        fg_prototypes = torch.cat([fg_prototypes, origin_prototype, ie_prototype], dim=0)

        return fg_prototypes

    def get_bg_pts(self, features, mask, pred_mask):
        """
        Args:
            features: (1, 512, 64, 64)
            mask: (1, 256, 256)
            pred_mask: (1, 256, 256)
        """
        bg_mask = 1 - mask
        features_trans = F.interpolate(features, size=bg_mask.shape[-2:], mode='bilinear', align_corners=True)

        oe_mask = torch.tensor(cv2.dilate(mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8), iterations=2)).to(self.device) - mask.squeeze(0)
        oe_mask = oe_mask.unsqueeze(0)
        oe_prototype = torch.sum(features_trans * oe_mask[None, ...], dim=(-2, -1)) \
                       / (oe_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        origin_prototype = torch.sum(features_trans * bg_mask[None, ...], dim=(-2, -1)) \
                           / (bg_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        add_mask = (pred_mask.float() + mask).long()
        mask1 = torch.zeros_like(mask)
        mask2 = torch.zeros_like(mask)
        mask1[add_mask == 0] = 1
        mask2[add_mask == 1] = 1
        mask1[bg_mask == 0] = 0
        mask2[bg_mask == 0] = 0

        bg_fts = self.get_fg_fts(features_trans, bg_mask)
        bg_prototypes = self.mlp2(bg_fts.view(512, 256 * 256)).permute(1, 0)

        if torch.sum(mask2[mask2 == 1]) > 0:
            hard_bg = self.get_random_pts(features_trans, mask2, 100)
            k = random.sample(range(len(bg_prototypes)), 500)
            bg_prototypes = torch.cat([bg_prototypes[k], hard_bg], dim=0)

        bg_prototypes = torch.cat([bg_prototypes, origin_prototype, oe_prototype], dim=0)

        return bg_prototypes

    def get_random_pts(self, features_trans, mask, n_prototype):
        """
        Args:
            features_trans: (1, 512, 256, 256)
            mask: (1, 256, 256)
            n_prototype: int
        """
        features_trans = features_trans.squeeze(0)
        features_trans = features_trans.permute(1, 2, 0)
        features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
                                             features_trans.shape[-1])
        mask = mask.squeeze(0).view(-1)
        features_trans = features_trans[mask == 1]
        if len(features_trans) >= n_prototype:
            k = random.sample(range(len(features_trans)), n_prototype)
            prototypes = features_trans[k]
        else:
            if len(features_trans) == 0:
                prototypes = torch.zeros(n_prototype, 512).to(self.device)
            else:
                r = n_prototype // len(features_trans)
                k = random.sample(range(len(features_trans)), (n_prototype - len(features_trans)) % len(features_trans))
                prototypes = torch.cat([features_trans for _ in range(r)], dim=0)
                prototypes = torch.cat([features_trans[k], prototypes], dim=0)

        return prototypes

    def get_fg_fts(self, fts, mask):
        """
        Args:
            fts: (1, 512, 256, 256)
            mask: (1, 256, 256)
        """
        _, c, h, w = fts.shape
        # select masked fg features
        fg_fts = fts * mask[None, ...]
        bg_fts = torch.ones_like(fts) * mask[None, ...]
        mask_ = mask.view(-1)
        n_pts = len(mask_) - len(mask_[mask_ == 1])
        select_pts = self.get_random_pts(fts, mask, n_pts)
        index = bg_fts == 0
        fg_fts[index] = select_pts.permute(1, 0).reshape(512*n_pts)

        return fg_fts

    def get_all_prototypes(self, fg_fts):
        """
        Args:
            fg_fts: way x shot x [all x 512]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        prototypes = [sum([shot for shot in way]) / n_shots for way in fg_fts]

        return prototypes

    def get_fg_sim(self, fts, prototypes):
        """
        Args:
            fts: (1, 512, 64, 64)
            prototypes: (102, 512)
        """
        fts_ = fts.permute(0, 2, 3, 1)
        fts_ = F.normalize(fts_, dim=-1)
        pts_ = F.normalize(prototypes, dim=-1)
        fg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(0, 3, 1, 2)
        fg_sim = self.decoder1(fg_sim)

        return fg_sim

    def get_bg_sim(self, fts, prototypes):
        """
        Args:
            fts: (1, 512, 64, 64)
            prototypes: (602, 512)
        """
        fts_ = fts.permute(0, 2, 3, 1)
        fts_ = F.normalize(fts_, dim=-1)
        pts_ = F.normalize(prototypes, dim=-1)
        bg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(0, 3, 1, 2)
        bg_sim = self.decoder2(bg_sim)

        return bg_sim

    def get_aux_loss(self, sup_fg_pts, qry_fg_pts, sup_bg_pts, qry_bg_pts):
        """
        Args:
            sup_fg_pts: (102, 512)
            qry_fg_pts: (102, 512)
            sup_bg_pts: (602, 512)
            qry_bg_pts: (602, 512)
        """
        d1 = torch.mean(sup_fg_pts, dim=0, keepdim=True)
        d2 = torch.mean(qry_fg_pts, dim=0, keepdim=True)
        b1 = torch.mean(sup_bg_pts, dim=0, keepdim=True)
        b2 = torch.mean(qry_bg_pts, dim=0, keepdim=True)

        d1 = F.normalize(d1, dim=-1)
        d2 = F.normalize(d2, dim=-1)
        b1 = F.normalize(b1, dim=-1)
        b2 = F.normalize(b2, dim=-1)

        fg_intra = torch.matmul(d1, d2.transpose(0, 1)).squeeze(0).squeeze(0)
        bg_intra = torch.matmul(b1, b2.transpose(0, 1)).squeeze(0).squeeze(0)
        intra_loss = 2 - fg_intra - bg_intra

        zero = torch.zeros(1).squeeze(0)
        sup_inter = torch.matmul(d1, b1.transpose(0, 1))
        qry_inter = torch.matmul(d2, b2.transpose(0, 1))
        inter_loss = torch.max(zero, torch.mean(sup_inter)) + torch.max(zero, torch.mean(qry_inter))

        return intra_loss + inter_loss
