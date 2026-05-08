"""Progress bar helpers for SPLADE training scripts."""

from __future__ import annotations

import time
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.utilities.types import STEP_OUTPUT


class StepAwareRichProgressBar(RichProgressBar):
    """Use optimizer-step progress when training is configured with max_steps."""

    def __init__(self, refresh_rate: float = 0.1) -> None:
        # RichProgressBar expects an integer batch refresh interval, but this repo's
        # config treats refresh_rate as updates-per-second. We handle throttling
        # ourselves for step-based training and keep Lightning's default behavior
        # elsewhere.
        super().__init__(refresh_rate=1)
        self._updates_per_second: float = max(float(refresh_rate), 0.0)
        self._last_rendered_global_step: int = -1
        self._last_render_time: float = 0.0

    def _should_render_step(self, current_step: int, trainer: pl.Trainer) -> bool:
        if current_step == self._last_rendered_global_step:
            return False
        max_steps: int = int(trainer.max_steps)
        if max_steps > 0 and current_step >= max_steps:
            return True
        if self._updates_per_second <= 0.0:
            return True
        now: float = time.monotonic()
        min_interval_seconds: float = 1.0 / self._updates_per_second
        if (
            self._last_render_time > 0.0
            and (now - self._last_render_time) < min_interval_seconds
        ):
            return False
        return True

    def _render_step_progress(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *,
        current_step: int,
    ) -> None:
        if self.progress is not None and self.is_enabled and self.train_progress_bar_id is not None:
            # Avoid RichProgressBar._update(), which refreshes immediately. We want
            # to update the progress state, then update metrics, then refresh once.
            self.progress.update(
                self.train_progress_bar_id,
                completed=current_step,
                visible=True,
            )
        self._update_metrics(trainer, pl_module)
        self.refresh()
        self._last_rendered_global_step = current_step
        self._last_render_time = time.monotonic()

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
        current_step: int = int(trainer.global_step)
        if not self._should_render_step(current_step, trainer):
            return
        self._render_step_progress(
            trainer,
            pl_module,
            current_step=current_step,
        )
