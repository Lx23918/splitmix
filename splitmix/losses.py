import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftClipLoss(nn.Module):
    def __init__(self, top_k: int = 7, cos_batch: int = 256) -> None:
        super().__init__()
        self.top_k = top_k
        self.cos_batch = cos_batch

    def _pairwise_similarity(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        size = features.size(0)
        similarity = torch.zeros(size, size, device=features.device, dtype=features.dtype)
        for i in range(0, size, self.cos_batch):
            block_i = features[i : i + self.cos_batch]
            for j in range(0, size, self.cos_batch):
                block_j = features[j : j + self.cos_batch]
                similarity[i : i + block_i.size(0), j : j + block_j.size(0)] = F.cosine_similarity(
                    block_i.unsqueeze(1),
                    block_j.unsqueeze(0),
                    dim=2,
                )
        return similarity

    @staticmethod
    def _class_mask(img_index: torch.Tensor) -> torch.Tensor:
        return (img_index.unsqueeze(0) == img_index.unsqueeze(1)).to(torch.int32)

    def forward(
        self,
        eeg_features: torch.Tensor,
        target_features: torch.Tensor,
        logit_scale: torch.Tensor | float,
        img_index: torch.Tensor,
    ) -> torch.Tensor:
        eeg_features = F.normalize(eeg_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)
        scale = logit_scale if torch.is_tensor(logit_scale) else torch.tensor(logit_scale, device=eeg_features.device)
        scale = scale.to(torch.float32).exp().to(eeg_features.dtype)
        logits_eeg = scale * eeg_features @ target_features.T
        logits_target = scale * target_features @ eeg_features.T
        with torch.no_grad():
            similarity = self._pairwise_similarity(target_features)
            if self.top_k is None or self.top_k <= 0:
                mask_sim = torch.ones_like(similarity)
            else:
                k = min(self.top_k, similarity.size(0))
                scores, indices = torch.topk(similarity.clone().fill_diagonal_(0), k=k, dim=1, sorted=False)
                del scores
                mask_sim = torch.zeros_like(similarity)
                mask_sim.scatter_(1, indices, 1)
                mask_sim.fill_diagonal_(1)
            labels = similarity * mask_sim * self._class_mask(img_index)
            labels = labels / labels.sum(dim=1, keepdim=True).clamp_min(1e-8)
            labels = labels.to(dtype=logits_eeg.dtype, device=logits_eeg.device)
        loss = (F.cross_entropy(logits_eeg, labels) + F.cross_entropy(logits_target, labels)) / 2.0
        return loss
