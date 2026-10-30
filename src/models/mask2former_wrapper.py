import torch

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


class Mask2FormerWrapper(torch.nn.Module):

    def __init__(
        self,
        checkpoint: str,
        num_labels: int,
    ):
        super().__init__()

        self.is_mask2former = True
        self.num_labels = num_labels

        self.processor = AutoImageProcessor.from_pretrained(
            checkpoint,
        )

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    def _build_targets(self, target):

        mask_labels = []
        class_labels = []

        for mask in target:

            masks = []
            classes = []

            unique_classes = torch.unique(mask)

            for class_id in unique_classes:

                class_id = int(class_id)

                # Background
                if class_id == 0:
                    continue

                binary_mask = mask == class_id

                if binary_mask.any():
                    masks.append(binary_mask)
                    classes.append(class_id)

            if masks:

                masks = torch.stack(masks).float()

                classes = torch.tensor(
                    classes,
                    dtype=torch.long,
                    device=mask.device,
                )

            else:

                masks = torch.empty(
                    (
                        0,
                        mask.shape[-2],
                        mask.shape[-1],
                    ),
                    dtype=torch.float32,
                    device=mask.device,
                )

                classes = torch.empty(
                    (0,),
                    dtype=torch.long,
                    device=mask.device,
                )

            mask_labels.append(masks)
            class_labels.append(classes)

        return mask_labels, class_labels

    def forward(self, x, target=None):

        # =========================================================
        # TRAINING
        # =========================================================

        if self.training:

            mask_labels, class_labels = self._build_targets(target)

            outputs = self.model(
                pixel_values=x,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )

            return outputs

        # =========================================================
        # VALIDATION / TEST
        # =========================================================

        outputs = self.model(
            pixel_values=x,
        )

        predictions = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[x.shape[-2:]] * x.shape[0],
        )

        # [B, H, W]
        predictions = torch.stack(predictions)

        # [B, C, H, W]
        logits = torch.nn.functional.one_hot(
            predictions.long(),
            num_classes=self.num_labels,
        ).permute(0, 3, 1, 2).float()

        return logits