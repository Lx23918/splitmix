from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

from splitmix import SplitMixDataModule, SplitMixModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--test-subject", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.subject is not None:
        cfg.data.subject = args.subject
    if args.test_subject is not None:
        cfg.data.test_subject = args.test_subject
    if args.seed is not None:
        cfg.seed = args.seed

    subject = str(cfg.data.subject)
    test_subject = str(cfg.data.test_subject) if cfg.data.test_subject is not None else subject
    output_dir = Path(cfg.output_root) / cfg.experiment_name / f"eval_{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    L.seed_everything(int(cfg.seed), workers=True)

    datamodule = SplitMixDataModule(
        eeg_data_path=str(cfg.data.eeg_data_path),
        image_root=str(cfg.data.image_root),
        feature_path=str(cfg.data.feature_path),
        subject=subject,
        test_subject=test_subject,
        batch_size=int(cfg.data.batch_size),
        batch_size_eval=int(cfg.data.batch_size_eval),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        time_window=tuple(cfg.data.time_window),
        use_split_half_train=bool(cfg.data.use_split_half_train),
        no_leakage=bool(cfg.data.no_leakage),
        low_trial_bias=bool(cfg.data.low_trial_bias),
        selected_channels_mode=str(cfg.data.selected_channels_mode),
        eeg_aug=str(cfg.data.eeg_aug),
        image_feature_source=str(cfg.data.image_feature_source),
        missing_aug_fallback=str(cfg.data.missing_aug_fallback),
    )

    model = SplitMixModule(
        num_channels=int(cfg.model.num_channels),
        sequence_length=int(cfg.model.sequence_length),
        token_dim=int(cfg.model.token_dim),
        projection_dim=int(cfg.model.projection_dim),
        use_cotar=bool(cfg.model.use_cotar),
        cotar_type=str(cfg.model.cotar_type),
        top_k=int(cfg.model.top_k),
        cos_batch=int(cfg.model.cos_batch),
        consistency_weight=float(cfg.model.consistency_weight),
        consistency_branch=str(cfg.model.consistency_branch),
        image_loss_weight=float(cfg.model.image_loss_weight),
        text_loss_weight=float(cfg.model.text_loss_weight),
        depth_loss_weight=float(cfg.model.depth_loss_weight),
        image_text_distill_weight=float(cfg.model.image_text_distill_weight),
        image_depth_distill_weight=float(cfg.model.image_depth_distill_weight),
        detach_aux_teachers=bool(cfg.model.detach_aux_teachers),
        split_half_target_weight=float(cfg.model.split_half_target_weight),
        split_half_target_view=str(cfg.model.split_half_target_view),
        distill_warmup_epochs=int(cfg.model.distill_warmup_epochs),
        auxiliary_warmup_epochs=int(cfg.model.auxiliary_warmup_epochs),
        compile=bool(cfg.model.compile),
        learning_rate=float(cfg.optimizer.lr),
        weight_decay=float(cfg.optimizer.weight_decay),
        scheduler_factor=float(cfg.scheduler.factor),
        scheduler_patience=int(cfg.scheduler.patience),
    )

    trainer = L.Trainer(
        default_root_dir=str(output_dir),
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=CSVLogger(save_dir=str(output_dir), name="logs"),
    )

    test_metrics = trainer.test(model=model, datamodule=datamodule, ckpt_path=args.checkpoint)
    payload = {
        "subject": subject,
        "test_subject": test_subject,
        "seed": int(cfg.seed),
        "checkpoint": args.checkpoint,
        "metrics": test_metrics[0] if test_metrics else {},
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
