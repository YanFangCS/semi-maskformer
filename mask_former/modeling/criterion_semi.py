# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import torch
from torch._C import device
import torch.nn.functional as F
from torch import nn, set_deterministic
import numpy as np

from detectron2.utils import comm
from detectron2.utils.comm import get_world_size

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


class SetCriterion_Semi(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = 0.0
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        """use maxpooling instead of bipart matching"""
        src_logits = F.max_pool2d(src_logits, kernel_size = (100, 1))
    
        src_logits = src_logits.squeeze() # [4, 22]

        gt_classes = torch.as_tensor([F.one_hot(t["labels"], num_classes = 22).sum(dim=0).tolist() for t in targets])
        gt_classes = gt_classes.to(dtype=torch.float, device=src_logits.device)

        loss_ce = F.binary_cross_entropy_with_logits(src_logits, gt_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    """
    def loss_image_labels(self, outputs, targets, indices, num_masks):
        assert "pred_imgs" in outputs
        src_logits = outputs["pred_imgs"] #[4, 20] 
    
        # src_logits shape [4, 20]
        # we need a src_logits of shape [4, 20] , 20 represents categories excluded background
        targets_classes = torch.zeros(src_logits.shape, device=src_logits.device)
        cate_list = [t["categories"] for t in targets]
        for t, c in zip(targets_classes, cate_list):
            if len(c) == 0:
                continue
            t[c-1] = 1
        targets_classes.to(torch.int64)
        # bcewithlogits need input logits and target logits have the same shape
        loss_bce = F.binary_cross_entropy_with_logits(src_logits, targets_classes)
        loss = {"loss_bce": loss_bce}
        return loss
    """

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        batch_size = len(outputs)
        labeled_size = int(batch_size / 2)
        
        src_idx1, src_idx2 = [], []
        for i, p in zip(src_idx[0], src_idx[1]):
            if i < labeled_size:
                src_idx1.append(i.item())
                src_idx2.append(p.item())
        src_idx = (torch.tensor(src_idx1), torch.tensor(src_idx2))

        tgt_idx1, tgt_idx2 = [], []
        for i, p in zip(tgt_idx[0], tgt_idx[1]):
            if i < labeled_size:
                tgt_idx1.append(i.item())
                tgt_idx2.append(p.item())
        tgt_idx = (torch.tensor(tgt_idx1), torch.tensor(tgt_idx2))

        src_masks = outputs["pred_masks"] 
        src_masks = src_masks[src_idx] 
        masks = [t["masks"] for t in targets]

        #src_masks = src_masks[:2]
        #masks = masks[:2]
        # in target_masks,the gt_masks of weakly supervisied image are all 255
        # in each batch, the last half images are weakly supervisied image

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        #if comm.is_main_process():
        #    print("src_masks shape", src_masks.shape)
        #    print("target_masks shape", target_masks.shape)

        # upsample predictions to the target size
        src_masks = F.interpolate(
            src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss(src_masks, target_masks, num_masks),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        # loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        batch_size = len(outputs)
        labeled_size = int(batch_size / 2)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # strongly-supervised 
        #num_masks = sum(len(t["labels"] for t in targets))
        # semi-
        num_masks = sum(len(t["labels"]) for t in targets[:labeled_size])
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses