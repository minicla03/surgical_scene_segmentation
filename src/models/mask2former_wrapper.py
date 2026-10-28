import torch
from transformers import Mask2FormerForUniversalSegmentation


class Mask2FormerWrapper(torch.nn.Module):

    def __init__(
        self,
        checkpoint: str,
        num_labels: int,
    ):
        super().__init__()

        self.is_mask2former = True

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x, target=None):

        if self.training:

            mask_labels = []
            class_labels = []

            for mask in target:
                masks = []
                classes = []

                unique_classes = torch.unique(mask)

                for class_id in unique_classes:

                    class_id = int(class_id)

                    # Ignore background
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
                        (0, mask.shape[-2], mask.shape[-1]),
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

            outputs = self.model(
                pixel_values=x,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )

            return outputs

        # Validation / test
        outputs = self.model(
            pixel_values=x,
        )

        predictions = self.model.post_process_semantic_segmentation(
            outputs,
            target_sizes=[x.shape[-2:]] * x.shape[0],
        )

        predictions = torch.stack(predictions)

        return predictions