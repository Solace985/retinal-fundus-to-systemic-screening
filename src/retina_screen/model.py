"""
model.py -- Multi-task head for the retinal screening pipeline.

Owns: MultiTaskHead architecture, task-head construction from TASK_REGISTRY,
MC-dropout helper.

Must not contain: concrete adapter imports, backbone loading, dataset-specific
conditionals, optimizer/training loop, evaluation metrics, or paper logic.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn

from retina_screen.tasks import TASK_REGISTRY, TaskType

logger = logging.getLogger(__name__)


class MultiTaskHead(nn.Module):
    """Multi-task head that consumes embeddings and produces per-task logits.

    Architecture::

        input: embedding (+ optional metadata concatenation)
            ↓
        trunk: Linear→LayerNorm→GELU→Dropout → Linear→LayerNorm→GELU→Dropout
            ↓
        per-task heads: Linear(hidden, 64)→GELU→Dropout→Linear(64, output_size)

    Output shapes:
    - BINARY / REGRESSION tasks: (batch_size,)
    - ORDINAL tasks:             (batch_size, num_classes)

    Does not load pretrained backbone weights. Backbone extraction is handled
    by embeddings.py.
    """

    def __init__(
        self,
        embedding_dim: int,
        task_names: Sequence[str],
        metadata_dim: int = 0,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_paired_eye_attention: bool = False,
    ) -> None:
        super().__init__()

        if use_paired_eye_attention:
            raise NotImplementedError(
                "use_paired_eye_attention=True is not implemented in Stage 5. "
                "Set use_paired_eye_attention=False and configure it via model config."
            )

        self._task_names = list(task_names)
        self._metadata_dim = metadata_dim

        # Validate all task names and definitions before building any modules.
        for tn in task_names:
            if tn not in TASK_REGISTRY:
                raise KeyError(
                    f"Task {tn!r} not found in TASK_REGISTRY. "
                    f"Register it in tasks.py first. "
                    f"Available: {sorted(TASK_REGISTRY)}"
                )
            task = TASK_REGISTRY[tn]
            if task.task_type == TaskType.ORDINAL and task.num_classes is None:
                raise ValueError(
                    f"Ordinal task {tn!r} has num_classes=None in TASK_REGISTRY."
                )

        input_dim = embedding_dim + metadata_dim

        # Fusion trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-task heads
        self.task_heads = nn.ModuleDict()
        for tn in task_names:
            task = TASK_REGISTRY[tn]
            if task.task_type in (TaskType.BINARY, TaskType.REGRESSION):
                output_size = 1
            else:
                output_size = task.num_classes  # type: ignore[assignment]
            self.task_heads[tn] = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_size),
            )

    def forward(
        self,
        embedding: torch.Tensor,
        metadata: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass; returns per-task logits.

        Parameters
        ----------
        embedding:
            (batch_size, embedding_dim) float tensor.
        metadata:
            (batch_size, metadata_dim) float tensor, or None if metadata_dim==0.

        Returns
        -------
        dict[task_name, Tensor]
            Binary/regression → (B,); ordinal → (B, num_classes).
        """
        if self._metadata_dim > 0:
            if metadata is None:
                raise ValueError(
                    f"MultiTaskHead has metadata_dim={self._metadata_dim} but "
                    f"metadata=None was passed. Provide the metadata tensor."
                )
            if metadata.shape[-1] != self._metadata_dim:
                raise ValueError(
                    f"Expected metadata of dim {self._metadata_dim}, "
                    f"got {metadata.shape[-1]}."
                )
            x = torch.cat([embedding, metadata], dim=-1)
        else:
            if metadata is not None:
                logger.debug(
                    "metadata_dim=0 but a metadata tensor was supplied; ignoring."
                )
            x = embedding

        h = self.trunk(x)

        outputs: dict[str, torch.Tensor] = {}
        for tn in self._task_names:
            task = TASK_REGISTRY[tn]
            out = self.task_heads[tn](h)
            if task.task_type in (TaskType.BINARY, TaskType.REGRESSION):
                out = out.squeeze(-1)  # (B, 1) → (B,)
            outputs[tn] = out

        return outputs

    @property
    def task_names(self) -> list[str]:
        return list(self._task_names)


def activate_mc_dropout(model: nn.Module) -> None:
    """Activate dropout layers only, keeping BatchNorm in eval mode.

    MC-dropout helper for uncertainty estimation.  Never calls model.train()
    to avoid accidentally putting BatchNorm into training mode.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
