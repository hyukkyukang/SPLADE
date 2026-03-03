"""Progress bar helpers for SPLADE training scripts."""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.utilities.types import STEP_OUTPUT


class StepAwareRichProgressBar(RichProgressBar):
    """Use optimizer-step progress when training is configured with max_steps."""

    def _is_step_based_training(self) -> bool:
        max_steps: int = int(self.trainer.max_steps)
        return self.trainer.max_epochs == -1 and max_steps > 0

    @property
    def total_train_batches(self) -> int | float:
        if self._is_step_based_training():
            return int(self.trainer.max_steps)
        return super().total_train_batches

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self._is_step_based_training():
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
            return

        if not self.is_disabled and self.train_progress_bar_id is None:
            # This can happen when resuming from a mid-epoch checkpoint.
            self._initialize_train_progress_bar_id()
        self._update(self.train_progress_bar_id, trainer.global_step)
        self._update_metrics(trainer, pl_module)
        self.refresh()
