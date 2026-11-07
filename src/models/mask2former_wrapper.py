from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation


class Mask2FormerWrapper(torch.nn.Module):

    def __init__(
        self,
        checkpoint: Optional[str],
        num_labels: int,
        pretrained: bool = False,
    ):
        super().__init__()
        self.is_mask2former = True
        self.num_labels = num_labels
        self.checkpoint = checkpoint
        self.pretrained = pretrained

        
        if self.pretrained and checkpoint is not None:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                checkpoint,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )
        elif checkpoint is not None:
            config = Mask2FormerConfig.from_pretrained(
                checkpoint,
                num_labels=num_labels,
            )
            self.model = Mask2FormerForUniversalSegmentation(config)
        else:
            config = Mask2FormerConfig(num_labels=num_labels)
            self.model = Mask2FormerForUniversalSegmentation(config)

    def _build_targets(
        self, target: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Convert semantic segmentation masks into the target format expected by Mask2Former.
        Background (class 0) is not represented as a foreground mask.
        """
        mask_labels = []
        class_labels = []

        for mask in target:
            mask = mask.squeeze(0)
            masks = []
            classes = []

            unique_classes = torch.unique(mask)

            for class_id in unique_classes:
                class_id = int(class_id)

                binary_mask = mask == class_id
                if binary_mask.any():
                    masks.append(binary_mask)
                    classes.append(class_id)

            if masks:
                masks_tensor = torch.stack(masks).float()
                classes_tensor = torch.tensor(
                    classes, dtype=torch.long, device=mask.device
                )
            else:
                masks_tensor = torch.empty(
                    (0, mask.shape[-2], mask.shape[-1]),
                    dtype=torch.float32,
                    device=mask.device,
                )
                classes_tensor = torch.empty(
                    (0,), dtype=torch.long, device=mask.device
                )

            mask_labels.append(masks_tensor)
            class_labels.append(classes_tensor)

        return mask_labels, class_labels

    def _semantic_scores(
        self, outputs, target_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        Convert Mask2Former query predictions into semantic segmentation scores.
        Returns Tensor of shape [B, C, H, W].
        """
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        # Mask2Former has one additional "no object" class.
        class_probs = torch.softmax(class_queries_logits, dim=-1)
        class_probs = class_probs[..., : self.num_labels]

        # Convert mask logits into probabilities.
        mask_probs = torch.sigmoid(masks_queries_logits)

        # Combine class probabilities and mask probabilities.
        # [B, Q, C] x [B, Q, H, W] -> [B, C, H, W]
        semantic_scores = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

        # Restore original input resolution.
        semantic_scores = F.interpolate(
            semantic_scores,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return semantic_scores

    def forward(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            images: Input tensor [B, C, H, W].
            masks: Semantic masks [B, 1, H, W].
        Returns:
            semantic_scores: Tensor [B, num_classes, H, W].
            loss: Mask2Former native loss when masks are provided.
        """
        target_size = (images.shape[-2], images.shape[-1])
        mask_labels, class_labels = None, None

        if masks is not None:
            mask_labels, class_labels = self._build_targets(masks)

        outputs = self.model(
            pixel_values=images,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        semantic_scores = self._semantic_scores(outputs, target_size)
        loss = outputs.loss

        return semantic_scores, loss