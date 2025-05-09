import numpy as np
import torch

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_FocalLoss


class nnUNetTrainerDiceFocalLoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50  # ⬅️ Training limit modificato

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_FocalLoss(
                {},
                {
                    'batch_dice': self.configuration_manager.batch_dice,
                    'do_bg': True,
                    'smooth': 1e-5,
                    'ddp': self.is_ddp
                },
                use_ignore_label=self.label_manager.ignore_label is not None
            )
        else:
            raise NotImplementedError("DC_and_FocalLoss è pensato solo per il region-based learning")

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)
            loss.focal = torch.compile(loss.focal)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
