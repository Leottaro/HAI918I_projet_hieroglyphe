from typing import Self

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class Mobilenetv2(nn.Module):
    # build the mobilenetv2 backbone and one projection head per attribute
    def __init__(self, num_attributes: int, embedding_dim: int, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)
        feature_dim: int = self.backbone.classifier[-1].in_features # dimension = 1280
        # remove the original classifier so the backbone returns features
        self.backbone.classifier = nn.Identity()
        # self.backbone.requires_grad_(False)
        # self.backbone.eval()

        # each head maps the same backbone features to one attribute vector
        self.heads = nn.ModuleList(
            nn.Linear(feature_dim, embedding_dim) for _ in range(num_attributes)
        )

    # switch the backbone and heads to train or eval mode
    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        # self.backbone.eval()
        return self

    # get one normalized vector per attribute for each image
    def forward(self, images: Tensor) -> Tensor:
        features: Tensor = self.backbone(images)
        attribu_vectors: list[Tensor] = [
            F.normalize(head(features), dim=1) for head in self.heads
        ]
        return torch.stack(attribu_vectors, dim=1)  # shape is (batch, attributes, embedding_dim)

def attribute_contrastive_loss(
    vectors: Tensor, attribute_ids: Tensor, margin: float = 0.25,
    *, attribute_indices: list[int] | None = None,
) -> tuple[Tensor | None, dict[int, Tensor]]:
    """calculate the contrastive loss for the batch and each attribute"""
    batch_size: int = vectors.size(0)

    # generate each image pair once and skip self-pairs
    left, right = torch.triu_indices(
        batch_size, batch_size, offset=1, device=vectors.device
    )

    losses: dict[int, Tensor] = {}
    if attribute_indices is None:
        attribute_indices = list(range(attribute_ids.size(1)))

    for index in attribute_indices:
        labels: Tensor = attribute_ids[:, index]
        # both images in the pair need a label for this attribute
        known: Tensor = (labels[left] >= 0) & (labels[right] >= 0)
        if not known.any():
            continue

        i, j = left[known], right[known]

        same: Tensor = labels[i] == labels[j]

        # get cosine similarity from the dot product of normalized vectors
        cosine: Tensor = (vectors[i, index] * vectors[j, index]).sum(dim=1) 
        cosine = cosine.clamp(-1.0, 1.0)

        # for same labels, push cosine toward 1
        # for different labels, penalize cosine above the margin
        pair_loss: Tensor = torch.where(same, 1.0 - cosine, F.relu(cosine - margin))

        losses[index] = pair_loss.mean()
    # average over the attributes with valid pairs
    batch_loss = torch.stack(tuple(losses.values())).mean() if losses else None
    return batch_loss, losses
