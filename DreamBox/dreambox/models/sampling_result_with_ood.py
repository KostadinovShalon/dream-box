# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from mmdet.models.task_modules import SamplingResult
from .assign_result_with_ood import AssignResultWithOOD


class SamplingResultWithOOD(SamplingResult):
    """Bbox sampling result.

    Args:
        pos_inds (Tensor): Indices of positive samples.
        neg_inds (Tensor): Indices of negative samples.
        priors (Tensor): The priors can be anchors or points,
            or the bboxes predicted by the previous stage.
        gt_bboxes (Tensor): Ground truth of bboxes.
        assign_result (:obj:`AssignResult`): Assigning results.
        gt_flags (Tensor): The Ground truth flags.
        avg_factor_with_neg (bool):  If True, ``avg_factor`` equal to
            the number of total priors; Otherwise, it is the number of
            positive priors. Defaults to True.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.models.task_modules.samplers.sampling_result import *  # NOQA
        >>> self = SamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <SamplingResult({
            'neg_inds': tensor([1,  2,  3,  5,  6,  7,  8,
                                9, 10, 11, 12, 13]),
            'neg_priors': torch.Size([12, 4]),
            'num_gts': 1,
            'num_neg': 12,
            'num_pos': 1,
            'avg_factor': 13,
            'pos_assigned_gt_inds': tensor([0]),
            'pos_inds': tensor([0]),
            'pos_is_gt': tensor([1], dtype=torch.uint8),
            'pos_priors': torch.Size([1, 4])
        })>
    """

    def __init__(self,
                 pos_inds: Tensor,
                 neg_inds: Tensor,
                 priors: Tensor,
                 gt_bboxes: Tensor,
                 assign_result: AssignResultWithOOD,
                 gt_flags: Tensor,
                 avg_factor_with_neg: bool = True) -> None:
        super(SamplingResultWithOOD, self).__init__(
            pos_inds, neg_inds, priors, gt_bboxes, assign_result, gt_flags,
            avg_factor_with_neg)
        self.pos_gt_ood_labels = assign_result.ood_labels[pos_inds]

    @classmethod
    def random(cls, rng=None, **kwargs):
        raise NotImplementedError('random sampling result is not supported')
