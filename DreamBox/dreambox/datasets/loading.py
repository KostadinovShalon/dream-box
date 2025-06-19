import numpy as np

from mmdet.datasets.transforms import LoadAnnotations
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadAnnotationsWithOOD(LoadAnnotations):
    def __init__(self, with_ood_labels=True, **kwargs):
        super().__init__(**kwargs)
        self.with_ood_labels = with_ood_labels

    def _load_bbox_ood_labels(self, results: dict) -> None:
        """Private function to load OOD labels (binary).

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded pseudo labels.
        """
        ood_labels = []
        for instance in results.get('instances', []):
            ood_labels.append(instance['ood'])
        results['gt_ood_labels'] = np.array(
            ood_labels, dtype=np.int8)

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        if self.with_ood_labels:
            self._load_bbox_ood_labels(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_ood_labels={self.with_ood_labels}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str
