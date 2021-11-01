# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""

"""
this is a fixed-matching version contrary to bipart matching
when using fixed-matching instead of bipart matching, ensure the number of queries is equal to categories
"""
import torch
from torch._C import device
import torch.nn.functional as F
from torch import nn, set_deterministic
import numpy as np

from detectron2.utils import comm
from detectron2.utils.comm import get_world_size

from mask_former.modeling.matcher import batch_dice_loss

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


class SetCriterion_Fixed(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, unlabeled_ratio):
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
        self.unlabeled_ratio = unlabeled_ratio
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)


    def loss_labels(self, outputs, targets, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        
        # target_classes [4, 21]
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype = torch.int64, device=src_logits.device
        )

        # next, fill the category class value at corresponding cordinate
        # targets [4] each element in targets is a dict consists of 
        # {"labels", "masks"} 
    
        idx = torch.cat([torch.full_like(t["labels"], i) for i, t in enumerate(targets)]), torch.cat([t["labels"] for t in targets])
        target_classes_o = torch.cat([t["labels"] for t in targets])

        target_classes[idx] = target_classes_o

        batch_size = len(targets)
        unlabled_size = int(batch_size * self.unlabeled_ratio)
        labeled_size = batch_size - unlabled_size
        # labeled_size = int(batch_size / 2)

        src_logits_labeled = src_logits[:labeled_size]
        src_logits_unlabel = src_logits[labeled_size:]
        target_classes_labeled = target_classes[:labeled_size]
        target_classes_unlabel = target_classes[labeled_size:]
        

        loss_ce = F.cross_entropy(src_logits_labeled.transpose(1,2), target_classes_labeled, self.empty_weight) + 0.1 * (
                F.cross_entropy(src_logits_unlabel.transpose(1,2) , target_classes_unlabel, self.empty_weight))
        # loss_ce = F.cross_entropy(src_logits.transpose(1,2), target_classes, self.empty_weight)

        # loss_ce = 1.0 * F.cross_entropy(src_logits_labeled.transpose(1, 2), target_classes_labeled, self.empty_weight) + 0.1 * (
        #           F.cross_entropy(src_logits_unlabel.transpose(1, 2), target_classes_unlabel, self.empty_weight) )
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        batch_size = len(targets)
        unlabeled_size = int(batch_size * self.unlabeled_ratio)
        labeled_size = batch_size - unlabeled_size
        # labeled_size = int(batch_size / 2)
        
        # just make 
        # when using fixed matching, each query just learn to the corresponding category's representation
        targets_labeled = targets[:labeled_size]
        idx = torch.cat([torch.full_like(t["labels"], i) for i, t in enumerate(targets_labeled)]), torch.cat([t["labels"] for t in targets_labeled])

        src_masks = outputs["pred_masks"][:labeled_size] 
 
        src_masks = src_masks[idx] 
        masks = [t["masks"] for t in targets_labeled]

        # in target_masks,the gt_masks of weakly supervisied image are all 255
        # in each batch, the last half images are weakly supervisied image

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks = torch.cat(masks, dim=0)
        # target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        
        target_masks = target_masks.to(src_masks)

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

    def get_loss(self, loss, outputs, targets, num_masks):
        # loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        batch_size = len(targets)
        unlabeled_size = int(batch_size * self.unlabeled_ratio)
        labeled_size = batch_size - unlabeled_size
        # labeled_size = int(batch_size / 2)

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # strongly-supervised 
        # num_masks = sum(len(t["labels"] for t in targets))
        # semi-supervised 
        num_masks = sum([len(t["labels"]) for t in targets[:labeled_size]])
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
