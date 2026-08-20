from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from .losses import SoftClipLoss
from .metrics import compute_retrieval_metrics
from .model import ProjectionHead, SplitMixBackbone


class SplitMixModule(LightningModule):
    def __init__(
        self,
        num_channels: int = 63,
        sequence_length: int = 250,
        token_dim: int = 40,
        projection_dim: int = 1024,
        use_cotar: bool = True,
        cotar_type: str = "mean",
        top_k: int = 7,
        cos_batch: int = 256,
        consistency_weight: float = 0.2,
        consistency_branch: str = "all",
        image_loss_weight: float = 1.0,
        text_loss_weight: float = 0.2,
        depth_loss_weight: float = 0.2,
        image_text_distill_weight: float = 0.0,
        image_depth_distill_weight: float = 0.0,
        detach_aux_teachers: bool = True,
        split_half_target_weight: float = 0.0,
        split_half_target_view: str = "a",
        distill_warmup_epochs: int = 0,
        auxiliary_warmup_epochs: int = 0,
        compile: bool = False,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 0.0,
        scheduler_factor: float = 0.1,
        scheduler_patience: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.backbone = SplitMixBackbone(
            num_channels=num_channels,
            sequence_length=sequence_length,
            token_dim=token_dim,
            use_cotar=use_cotar,
            cotar_type=cotar_type,
        )
        if compile and hasattr(torch, "compile"):
            self.backbone = torch.compile(self.backbone)

        eeg_dim = self.backbone.output_dim
        self.eeg_image_head = ProjectionHead(eeg_dim, projection_dim)
        self.eeg_text_head = ProjectionHead(eeg_dim, projection_dim)
        self.eeg_depth_head = ProjectionHead(eeg_dim, projection_dim)

        self.image_head = ProjectionHead(1024, projection_dim)
        self.text_head = ProjectionHead(1024, projection_dim)
        self.depth_head = ProjectionHead(1024, projection_dim)

        self.loss_fn = SoftClipLoss(top_k=top_k, cos_batch=cos_batch)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=False)

        valid_branches = {"image", "text", "depth", "all"}
        if consistency_branch not in valid_branches:
            raise ValueError(f"Unknown consistency_branch: {consistency_branch}")
        valid_views = {"a", "mean"}
        if split_half_target_view not in valid_views:
            raise ValueError(f"Unknown split_half_target_view: {split_half_target_view}")

    def _project_target_features(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        depth_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_out = F.normalize(self.image_head(image_features), dim=1)
        text_out = F.normalize(self.text_head(text_features), dim=1)
        depth_out = F.normalize(self.depth_head(depth_features), dim=1)
        return image_out, text_out, depth_out

    def _encode_eeg_embeddings(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.backbone(eeg)

    def _encode_eeg_feature_dict(self, eeg: torch.Tensor, branches: tuple[str, ...] = ("image", "text", "depth")):
        embedding = self._encode_eeg_embeddings(eeg)
        feature_dict: dict[str, torch.Tensor] = {}
        if "image" in branches:
            feature_dict["image"] = F.normalize(self.eeg_image_head(embedding), dim=1)
        if "text" in branches:
            feature_dict["text"] = F.normalize(self.eeg_text_head(embedding), dim=1)
        if "depth" in branches:
            feature_dict["depth"] = F.normalize(self.eeg_depth_head(embedding), dim=1)
        return feature_dict

    def _compute_losses_and_features(
        self,
        eeg: torch.Tensor,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        depth_features: torch.Tensor,
        img_index: torch.Tensor,
    ):
        eeg_feature_dict = self._encode_eeg_feature_dict(eeg)
        image_targets, text_targets, depth_targets = self._project_target_features(
            image_features,
            text_features,
            depth_features,
        )
        loss_image = self.loss_fn(eeg_feature_dict["image"], image_targets, self.logit_scale, img_index)
        loss_text = self.loss_fn(eeg_feature_dict["text"], text_targets, self.logit_scale, img_index)
        loss_depth = self.loss_fn(eeg_feature_dict["depth"], depth_targets, self.logit_scale, img_index)
        return (
            loss_image,
            loss_text,
            loss_depth,
            eeg_feature_dict["image"],
            eeg_feature_dict["text"],
            eeg_feature_dict["depth"],
            image_targets,
            text_targets,
            depth_targets,
        )

    def _auxiliary_scale(self) -> float:
        if self.hparams.auxiliary_warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(self.current_epoch) / float(self.hparams.auxiliary_warmup_epochs))

    def _active_loss_weights(self) -> tuple[float, float, float]:
        aux_scale = self._auxiliary_scale()
        return (
            self.hparams.image_loss_weight,
            self.hparams.text_loss_weight * aux_scale,
            self.hparams.depth_loss_weight * aux_scale,
        )

    def _active_distill_weights(self) -> tuple[float, float]:
        if self.current_epoch < self.hparams.distill_warmup_epochs:
            return 0.0, 0.0
        aux_scale = self._auxiliary_scale()
        return (
            self.hparams.image_text_distill_weight * aux_scale,
            self.hparams.image_depth_distill_weight * aux_scale,
        )

    def _weighted_total_loss(self, loss_image: torch.Tensor, loss_text: torch.Tensor, loss_depth: torch.Tensor) -> torch.Tensor:
        weight_image, weight_text, weight_depth = self._active_loss_weights()
        return weight_image * loss_image + weight_text * loss_text + weight_depth * loss_depth

    def _compute_distill_loss(
        self,
        eeg_image: torch.Tensor,
        eeg_text: torch.Tensor,
        eeg_depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = eeg_image.new_zeros(())
        distill_text = zero
        distill_depth = zero
        if self.hparams.image_text_distill_weight > 0:
            teacher = eeg_text.detach() if self.hparams.detach_aux_teachers else eeg_text
            distill_text = 1.0 - F.cosine_similarity(eeg_image, teacher, dim=1).mean()
        if self.hparams.image_depth_distill_weight > 0:
            teacher = eeg_depth.detach() if self.hparams.detach_aux_teachers else eeg_depth
            distill_depth = 1.0 - F.cosine_similarity(eeg_image, teacher, dim=1).mean()
        return distill_text, distill_depth

    def training_step(self, batch, batch_idx: int):
        eeg, image_features, text_features, depth_features, img_index = batch

        if isinstance(eeg, (tuple, list)) and len(eeg) == 2:
            eeg_a, eeg_b = eeg
            eeg_target = 0.5 * (eeg_a + eeg_b) if self.hparams.split_half_target_view == "mean" else eeg_a
            (
                loss_image,
                loss_text,
                loss_depth,
                eeg_image_target,
                eeg_text_target,
                eeg_depth_target,
                _,
                _,
                _,
            ) = self._compute_losses_and_features(eeg_target, image_features, text_features, depth_features, img_index)

            eeg_b_features = None
            if self.hparams.split_half_target_weight > 0:
                (
                    loss_image_b,
                    loss_text_b,
                    loss_depth_b,
                    eeg_image_b,
                    eeg_text_b,
                    eeg_depth_b,
                    _,
                    _,
                    _,
                ) = self._compute_losses_and_features(eeg_b, image_features, text_features, depth_features, img_index)
                alpha = self.hparams.split_half_target_weight
                denom = 1.0 + alpha
                loss_image = (loss_image + alpha * loss_image_b) / denom
                loss_text = (loss_text + alpha * loss_text_b) / denom
                loss_depth = (loss_depth + alpha * loss_depth_b) / denom
                eeg_b_features = {"image": eeg_image_b, "text": eeg_text_b, "depth": eeg_depth_b}

            branches = ("image", "text", "depth") if self.hparams.consistency_branch == "all" else (self.hparams.consistency_branch,)
            if self.hparams.split_half_target_view == "mean":
                eeg_a_features = self._encode_eeg_feature_dict(eeg_a, branches=branches)
            else:
                eeg_a_features = {}
                if "image" in branches:
                    eeg_a_features["image"] = eeg_image_target
                if "text" in branches:
                    eeg_a_features["text"] = eeg_text_target
                if "depth" in branches:
                    eeg_a_features["depth"] = eeg_depth_target
            if eeg_b_features is None:
                eeg_b_features = self._encode_eeg_feature_dict(eeg_b, branches=branches)

            zero = loss_image.new_zeros(())
            consistency_image = zero
            consistency_text = zero
            consistency_depth = zero
            if "image" in branches:
                consistency_image = 1.0 - F.cosine_similarity(eeg_a_features["image"], eeg_b_features["image"], dim=1).mean()
                loss_image = loss_image + self.hparams.consistency_weight * consistency_image
            if "text" in branches:
                consistency_text = 1.0 - F.cosine_similarity(eeg_a_features["text"], eeg_b_features["text"], dim=1).mean()
                loss_text = loss_text + self.hparams.consistency_weight * consistency_text
            if "depth" in branches:
                consistency_depth = 1.0 - F.cosine_similarity(eeg_a_features["depth"], eeg_b_features["depth"], dim=1).mean()
                loss_depth = loss_depth + self.hparams.consistency_weight * consistency_depth

            distill_text, distill_depth = self._compute_distill_loss(
                eeg_image_target,
                eeg_text_target,
                eeg_depth_target,
            )
            self.log("train_consistency_image", consistency_image, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_consistency_text", consistency_text, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_consistency_depth", consistency_depth, on_step=False, on_epoch=True, prog_bar=False)
        else:
            (
                loss_image,
                loss_text,
                loss_depth,
                eeg_image_target,
                eeg_text_target,
                eeg_depth_target,
                _,
                _,
                _,
            ) = self._compute_losses_and_features(eeg, image_features, text_features, depth_features, img_index)
            distill_text, distill_depth = self._compute_distill_loss(
                eeg_image_target,
                eeg_text_target,
                eeg_depth_target,
            )

        distill_weight_text, distill_weight_depth = self._active_distill_weights()
        loss_image = loss_image + distill_weight_text * distill_text + distill_weight_depth * distill_depth
        loss_total = self._weighted_total_loss(loss_image, loss_text, loss_depth)

        self.log("train_loss_total", loss_total, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_image", loss_image, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_text", loss_text, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_depth", loss_depth, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_distill_text", distill_text, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_distill_depth", distill_depth, on_step=False, on_epoch=True, prog_bar=False)
        return loss_total

    def _shared_eval(self, batch, prefix: str):
        eeg, image_features, text_features, depth_features, img_index = batch
        (
            loss_image,
            loss_text,
            loss_depth,
            eeg_image,
            eeg_text,
            eeg_depth,
            image_targets,
            text_targets,
            depth_targets,
        ) = self._compute_losses_and_features(eeg, image_features, text_features, depth_features, img_index)
        loss_total = self._weighted_total_loss(loss_image, loss_text, loss_depth)
        metrics = compute_retrieval_metrics(
            eeg_image_features=eeg_image,
            eeg_text_features=eeg_text,
            eeg_depth_features=eeg_depth,
            image_features=image_targets,
            text_features=text_targets,
            depth_features=depth_targets,
        )

        self.log(f"{prefix}_loss_total", loss_total, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_loss_image", loss_image, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{prefix}_loss_text", loss_text, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{prefix}_loss_depth", loss_depth, on_step=False, on_epoch=True, prog_bar=False)
        for key, value in metrics.items():
            self.log(f"{prefix}_{key}", value, on_step=False, on_epoch=True, prog_bar=key == "top1_image")

    def validation_step(self, batch, batch_idx: int):
        self._shared_eval(batch, "val")

    def test_step(self, batch, batch_idx: int):
        self._shared_eval(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss_total",
                "interval": "epoch",
                "frequency": 1,
            },
        }
