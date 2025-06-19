import torch
from torch import Tensor

from mmdet.models.task_modules import AssignResult


class AssignResultWithOOD(AssignResult):
    def __init__(self, ood_labels: Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ood_labels = ood_labels

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'max_overlaps': self.max_overlaps,
            'labels': self.labels,
            'ood_labels': self.ood_labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    def __nice__(self):
        """str: a "nice" summary string describing this assign result"""
        parts = []
        parts.append(f'num_gts={self.num_gts!r}')
        if self.gt_inds is None:
            parts.append(f'gt_inds={self.gt_inds!r}')
        else:
            parts.append(f'gt_inds.shape={tuple(self.gt_inds.shape)!r}')
        if self.max_overlaps is None:
            parts.append(f'max_overlaps={self.max_overlaps!r}')
        else:
            parts.append('max_overlaps.shape='
                         f'{tuple(self.max_overlaps.shape)!r}')
        if self.labels is None:
            parts.append(f'labels={self.labels!r}')
        else:
            parts.append(f'labels.shape={tuple(self.labels.shape)!r}')
        if self.ood_labels is None:
            parts.append(f'ood_labels={self.ood_labels!r}')
        else:
            parts.append(f'ood_labels.shape={tuple(self.ood_labels.shape)!r}')
        return ', '.join(parts)

    @classmethod
    def random(cls, **kwargs):
        """Create random AssignResult for tests or debugging.

        Args:
            num_preds: number of predicted boxes
            num_gts: number of true boxes
            p_ignore (float): probability of a predicted box assigned to an
                ignored truth
            p_assigned (float): probability of a predicted box not being
                assigned
            p_use_label (float | bool): with labels or not
            rng (None | int | numpy.random.RandomState): seed or state

        Returns:
            :obj:`AssignResult`: Randomly generated assign results.

        Example:
            >>> from mmdet.models.task_modules.assigners.assign_result import *  # NOQA
            >>> self = AssignResult.random()
            >>> print(self.info)
        """
        from mmdet.models.task_modules.samplers.sampling_result import ensure_rng
        rng = ensure_rng(kwargs.get('rng', None))

        num_gts = kwargs.get('num_gts', None)
        num_preds = kwargs.get('num_preds', None)
        p_ignore = kwargs.get('p_ignore', 0.3)
        p_assigned = kwargs.get('p_assigned', 0.7)
        num_classes = kwargs.get('num_classes', 3)

        if num_gts is None:
            num_gts = rng.randint(0, 8)
        if num_preds is None:
            num_preds = rng.randint(0, 16)

        if num_gts == 0:
            max_overlaps = torch.zeros(num_preds, dtype=torch.float32)
            gt_inds = torch.zeros(num_preds, dtype=torch.int64)
            labels = torch.zeros(num_preds, dtype=torch.int64)
            ood_labels = torch.zeros(num_preds, dtype=torch.int64)

        else:
            import numpy as np

            # Create an overlap for each predicted box
            max_overlaps = torch.from_numpy(rng.rand(num_preds))

            # Construct gt_inds for each predicted box
            is_assigned = torch.from_numpy(rng.rand(num_preds) < p_assigned)
            # maximum number of assignments constraints
            n_assigned = min(num_preds, min(num_gts, is_assigned.sum()))

            assigned_idxs = np.where(is_assigned)[0]
            rng.shuffle(assigned_idxs)
            assigned_idxs = assigned_idxs[0:n_assigned]
            assigned_idxs.sort()

            is_assigned[:] = 0
            is_assigned[assigned_idxs] = True

            is_ignore = torch.from_numpy(
                rng.rand(num_preds) < p_ignore) & is_assigned

            gt_inds = torch.zeros(num_preds, dtype=torch.int64)

            true_idxs = np.arange(num_gts)
            rng.shuffle(true_idxs)
            true_idxs = torch.from_numpy(true_idxs)
            gt_inds[is_assigned] = true_idxs[:n_assigned].long()

            gt_inds = torch.from_numpy(
                rng.randint(1, num_gts + 1, size=num_preds))
            gt_inds[is_ignore] = -1
            gt_inds[~is_assigned] = 0
            max_overlaps[~is_assigned] = 0

            if num_classes == 0:
                labels = torch.zeros(num_preds, dtype=torch.int64)
                ood_labels = torch.zeros(num_preds, dtype=torch.int64)
            else:
                labels = torch.from_numpy(
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    rng.randint(0, num_classes, size=num_preds))
                labels[~is_assigned] = 0
                ood_labels = torch.from_numpy(
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    rng.randint(0, 2, size=num_preds))
                ood_labels[~is_assigned] = 0

        self = cls(num_gts=num_gts, gt_inds=gt_inds, max_overlaps=max_overlaps, labels=labels, ood_labels=ood_labels)
        return self

    def add_gt_(self, gt_labels, gt_ood_labels=None):
        """Add ground truth as assigned results.

        Args:
            gt_labels (torch.Tensor): Labels of gt boxes
        """
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])

        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])

        self.labels = torch.cat([gt_labels, self.labels])
        if gt_ood_labels is not None:
            self.ood_labels = torch.cat([gt_ood_labels, self.ood_labels])
        else:
            self.ood_labels = torch.cat([torch.zeros_like(gt_labels), self.ood_labels])