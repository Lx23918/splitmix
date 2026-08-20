from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EEGFeatureDataset(Dataset):
    posterior_17_channels = (
        "P7",
        "P5",
        "P3",
        "P1",
        "PZ",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO3",
        "POZ",
        "PO4",
        "PO8",
        "O1",
        "OZ",
        "O2",
    )

    def __init__(
        self,
        eeg_data_path: str,
        image_root: str,
        feature_path: str,
        subject: str,
        train: bool,
        time_window: tuple[float, float] = (0.0, 1.0),
        selected_channels_mode: str = "all",
        eeg_aug: str = "none",
        image_feature_source: str = "original",
        missing_aug_fallback: str = "original",
    ) -> None:
        self.eeg_data_path = Path(eeg_data_path)
        self.image_root = Path(image_root)
        self.feature_path = Path(feature_path)
        self.subject = subject
        self.train = train
        self.time_window = time_window
        self.selected_channels_mode = selected_channels_mode.lower()
        self.eeg_aug = eeg_aug.lower()
        self.image_feature_source = image_feature_source.lower()
        self.missing_aug_fallback = missing_aug_fallback.lower()
        self.times: torch.Tensor | None = None
        self.ch_names: list[str] | None = None

        self.image_dir = self._resolve_image_dir()
        (
            self.data,
            self.image_items,
            self.sample_img_indices,
            self.subject_labels,
        ) = self._load_subject()
        self.data = self._select_channels(self.data)
        self.data = self._extract_time_window(self.data)
        self.data = self._apply_eeg_aug(self.data)
        (
            self.image_feature_dict,
            self.text_feature_dict,
            self.depth_feature_dict,
            self.aug_image_feature_dict,
        ) = self._load_feature_dicts()

    def _resolve_image_dir(self) -> Path:
        relative = Path("image_set") / ("training_images" if self.train else "test_images")
        candidate = self.image_root / relative
        if candidate.is_dir():
            return candidate
        fallback = self.image_root / ("training_images" if self.train else "test_images")
        if fallback.is_dir():
            return fallback
        raise FileNotFoundError(f"Image directory not found under {self.image_root}")

    @staticmethod
    def _torch_load(path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @staticmethod
    def _to_list(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _build_image_lookup(self) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for folder in sorted(path for path in self.image_dir.iterdir() if path.is_dir()):
            for image_path in sorted(folder.iterdir()):
                if image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    lookup.setdefault(image_path.name, str(image_path))
        return lookup

    def _normalize_image_path(self, value, image_lookup: dict[str, str]) -> str:
        if value is None:
            return ""
        image_path = str(value)
        if os.path.exists(image_path):
            return image_path
        basename = os.path.basename(image_path)
        if basename in image_lookup:
            return image_lookup[basename]
        candidate = self.image_dir / image_path
        if candidate.exists():
            return str(candidate)
        return image_path

    def _expand_metadata(self, values, target_length: int, repeat_factor: int = 1, default=None):
        values = self._to_list(values)
        if values is None:
            values = [default] * target_length
        if len(values) == target_length:
            return values
        if repeat_factor > 1 and len(values) * repeat_factor == target_length:
            return [item for item in values for _ in range(repeat_factor)]
        if len(values) > 0 and target_length % len(values) == 0:
            factor = target_length // len(values)
            return [item for item in values for _ in range(factor)]
        if len(values) == 1:
            return values * target_length
        raise ValueError(f"Cannot align metadata of length {len(values)} to {target_length}")

    def _reduce_metadata(self, values, target_length: int):
        values = self._to_list(values)
        if values is None:
            return None
        if len(values) == target_length:
            return values
        if len(values) > target_length and len(values) % target_length == 0:
            block = len(values) // target_length
            return [values[i * block] for i in range(target_length)]
        if len(values) == 1:
            return values * target_length
        raise ValueError(f"Cannot reduce metadata of length {len(values)} to {target_length}")

    @staticmethod
    def _flatten_eeg(eeg: torch.Tensor, metadata_length: int | None = None) -> tuple[torch.Tensor, int]:
        if eeg.dim() == 3:
            return eeg, 1
        if eeg.dim() < 3:
            raise ValueError(f"Unexpected EEG shape: {tuple(eeg.shape)}")
        leading = int(np.prod(eeg.shape[:-2]))
        repeat_factor = 1
        if metadata_length is not None and metadata_length > 0 and leading % metadata_length == 0:
            repeat_factor = leading // metadata_length
        return eeg.reshape(leading, *eeg.shape[-2:]), repeat_factor

    def _align_times(self, times, eeg_length: int) -> torch.Tensor:
        times = torch.as_tensor(times).detach()
        if times.numel() == eeg_length:
            return times
        if times.numel() < eeg_length:
            raise ValueError(f"Times length {times.numel()} is shorter than EEG length {eeg_length}")
        zero_indices = (times == 0).nonzero(as_tuple=False).flatten()
        if zero_indices.numel() > 0:
            start = int(zero_indices[0].item())
            if start + eeg_length <= times.numel():
                return times[start : start + eeg_length]
        return times[-eeg_length:]

    def _build_unique_images(self, image_paths: list[str]) -> tuple[list[str], torch.Tensor]:
        mapping: dict[str, int] = {}
        unique_images: list[str] = []
        sample_indices: list[int] = []
        for image_path in image_paths:
            if image_path not in mapping:
                mapping[image_path] = len(unique_images)
                unique_images.append(image_path)
            sample_indices.append(mapping[image_path])
        return unique_images, torch.as_tensor(sample_indices, dtype=torch.long)

    def _load_pt_subject(self, file_path: Path) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
        loaded = self._torch_load(file_path)
        if not isinstance(loaded, dict) or "eeg" not in loaded:
            raise ValueError(f"Invalid PT dataset file: {file_path}")

        image_lookup = self._build_image_lookup()
        eeg_raw = torch.as_tensor(loaded["eeg"]).float().detach()
        raw_labels = self._to_list(loaded.get("label"))
        raw_images = loaded.get("img", loaded.get("image"))
        times = loaded.get("times")
        ch_names = loaded.get("ch_names")
        if times is None or ch_names is None:
            raise KeyError(f"Missing times or ch_names in {file_path}")

        if self.train:
            eeg, repeat_factor = self._flatten_eeg(eeg_raw, len(raw_labels) if raw_labels is not None else None)
            image_paths = self._expand_metadata(raw_images, eeg.shape[0], repeat_factor=repeat_factor, default="")
        else:
            if eeg_raw.dim() == 3:
                eeg = eeg_raw
            elif eeg_raw.dim() >= 4:
                eeg = eeg_raw.reshape(eeg_raw.shape[0], -1, *eeg_raw.shape[-2:]).mean(dim=1)
            else:
                raise ValueError(f"Unexpected test EEG shape: {tuple(eeg_raw.shape)}")
            image_paths = self._reduce_metadata(raw_images, eeg.shape[0])
            if image_paths is None:
                image_paths = [""] * eeg.shape[0]

        image_paths = [self._normalize_image_path(value, image_lookup) for value in image_paths]
        self.times = self._align_times(times, eeg.shape[-1])
        self.ch_names = [str(name) for name in ch_names]
        unique_images, sample_img_indices = self._build_unique_images(image_paths)
        subject_id = int(self.subject.split("-")[-1])
        subject_labels = torch.full((eeg.shape[0],), subject_id, dtype=torch.long)
        return eeg, unique_images, sample_img_indices, subject_labels

    def _load_legacy_subject(self) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
        if self.train:
            file_path = self.eeg_data_path / self.subject / "preprocessed_eeg_training.npy"
            raw = np.load(file_path, allow_pickle=True)
            eeg = torch.from_numpy(raw["preprocessed_eeg_data"]).float().detach()
            detach_index = 0 if "MVNN" in str(self.eeg_data_path) else (torch.from_numpy(raw["times"]) == 0).nonzero().item()
            self.times = torch.from_numpy(raw["times"]).detach()[detach_index:]
            if "MVNN" in str(self.eeg_data_path):
                self.times = self.times + 0.2
            self.ch_names = [str(name) for name in raw["ch_names"]]
            image_items = self._list_all_images()
            num_classes = len(image_items)
            samples_per_class = 10
            views_per_sample = 4
            data_blocks = []
            for class_index in range(num_classes):
                start = class_index * samples_per_class
                data_blocks.append(eeg[start : start + samples_per_class])
            data_tensor = torch.cat(data_blocks, dim=0).view(-1, *data_blocks[0].shape[2:])
            sample_img_indices = torch.as_tensor(
                [(index % (num_classes * samples_per_class * views_per_sample)) // views_per_sample for index in range(data_tensor.shape[0])],
                dtype=torch.long,
            )
            subject_id = int(self.subject.split("-")[-1])
            subject_labels = torch.full((data_tensor.shape[0],), subject_id, dtype=torch.long).repeat_interleave(1)
            return data_tensor, image_items, sample_img_indices, subject_labels

        file_path = self.eeg_data_path / self.subject / "preprocessed_eeg_test.npy"
        raw = dict(np.load(file_path, allow_pickle=True).items())
        eeg = torch.from_numpy(raw["preprocessed_eeg_data"]).float().detach()
        detach_index = 0 if "MVNN" in str(self.eeg_data_path) else (torch.from_numpy(raw["times"]) == 0).nonzero().item()
        self.times = torch.from_numpy(raw["times"]).detach()[detach_index:]
        if "MVNN" in str(self.eeg_data_path):
            self.times = self.times + 0.2
        self.ch_names = [str(name) for name in raw["ch_names"]]
        image_items = self._list_all_images()
        num_classes = len(image_items)
        samples_per_class = 1
        data_blocks = []
        for class_index in range(num_classes):
            start = class_index * samples_per_class
            sample = eeg[start : start + samples_per_class]
            sample = torch.mean(sample.squeeze(0), 0)
            data_blocks.append(sample)
        data_tensor = torch.cat(data_blocks, dim=0).view(-1, *data_blocks[0].shape)
        sample_img_indices = torch.arange(num_classes, dtype=torch.long)
        subject_id = int(self.subject.split("-")[-1])
        subject_labels = torch.full((data_tensor.shape[0],), subject_id, dtype=torch.long)
        return data_tensor, image_items, sample_img_indices, subject_labels

    def _list_all_images(self) -> list[str]:
        image_paths: list[str] = []
        for folder in sorted(path for path in self.image_dir.iterdir() if path.is_dir()):
            for image_path in sorted(folder.iterdir()):
                if image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    image_paths.append(str(image_path))
        return image_paths

    def _load_subject(self) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
        pt_name = "train.pt" if self.train else "test.pt"
        pt_path = self.eeg_data_path / self.subject / pt_name
        if pt_path.exists():
            return self._load_pt_subject(pt_path)
        return self._load_legacy_subject()

    def _select_channels(self, eeg: torch.Tensor) -> torch.Tensor:
        if self.selected_channels_mode == "all":
            return eeg
        if self.selected_channels_mode != "posterior17":
            raise ValueError(f"Unsupported selected_channels_mode: {self.selected_channels_mode}")
        if self.ch_names is None:
            raise ValueError("Channel names are required for channel selection")
        ch_names_upper = [name.upper() for name in self.ch_names]
        selected_indices = [idx for idx, name in enumerate(ch_names_upper) if name in self.posterior_17_channels]
        if not selected_indices:
            raise RuntimeError("No posterior channels were found")
        self.ch_names = [self.ch_names[idx] for idx in selected_indices]
        return eeg[:, selected_indices, :]

    def _extract_time_window(self, eeg: torch.Tensor) -> torch.Tensor:
        if self.times is None:
            raise ValueError("Times are required for time-window extraction")
        start, end = self.time_window
        indices = (self.times >= start) & (self.times <= end)
        return eeg[..., indices]

    def _apply_eeg_aug(self, eeg: torch.Tensor) -> torch.Tensor:
        if not self.train or self.eeg_aug == "none":
            return eeg
        if self.eeg_aug != "smooth":
            raise ValueError(f"Unsupported eeg_aug: {self.eeg_aug}")
        padded = torch.nn.functional.pad(eeg, (1, 1), mode="replicate")
        return (padded[..., :-2] + padded[..., 1:-1] + padded[..., 2:]) / 3.0

    def _load_feature_dicts(self):
        suffix = "" if self.train else "_test"
        image_path = self.feature_path / f"image_original_features_clip_dict{suffix}.pt"
        text_path = self.feature_path / f"text_finegrain_features_clip_dict{suffix}.pt"
        depth_path = self.feature_path / f"image_depth_features_clip_dict{suffix}.pt"
        aug_path = self.feature_path / f"image_aug_features_clip_dict{suffix}.pt"

        image_features = self._torch_load(image_path)
        text_features = self._torch_load(text_path)
        depth_features = self._torch_load(depth_path)
        if aug_path.exists():
            aug_features = self._torch_load(aug_path)
        elif self.missing_aug_fallback == "original":
            aug_features = image_features
        else:
            raise FileNotFoundError(f"Missing augmented image features: {aug_path}")
        return image_features, text_features, depth_features, aug_features

    @staticmethod
    def _lookup_feature(feature_dict, image_name: str, feature_name: str) -> torch.Tensor:
        if image_name in feature_dict:
            value = feature_dict[image_name]
            return value.squeeze(0) if value.dim() > 1 else value
        base, _ = os.path.splitext(image_name)
        for extension in (".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"):
            candidate = base + extension
            if candidate in feature_dict:
                value = feature_dict[candidate]
                return value.squeeze(0) if value.dim() > 1 else value
        raise KeyError(f"Missing {feature_name} feature for {image_name}")

    def _resolve_image_feature(self, image_features: torch.Tensor, aug_image_features: torch.Tensor) -> torch.Tensor:
        if self.image_feature_source == "original":
            return image_features
        if self.image_feature_source == "aug":
            return aug_image_features
        if self.image_feature_source == "mix":
            mixed = 0.5 * (image_features + aug_image_features)
            return mixed / mixed.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        raise ValueError(f"Unsupported image_feature_source: {self.image_feature_source}")

    def __getitem__(self, index: int):
        eeg = self.data[index]
        img_index = self.sample_img_indices[index]
        image_path = self.image_items[int(img_index.item())] if len(self.image_items) > int(img_index.item()) else self.image_items[index]
        image_name = os.path.basename(image_path)
        image_features = self._lookup_feature(self.image_feature_dict, image_name, "image")
        text_features = self._lookup_feature(self.text_feature_dict, image_name, "text")
        depth_features = self._lookup_feature(self.depth_feature_dict, image_name, "depth")
        try:
            aug_image_features = self._lookup_feature(self.aug_image_feature_dict, image_name, "aug_image")
        except KeyError:
            if self.missing_aug_fallback != "original":
                raise
            aug_image_features = image_features
        image_features = self._resolve_image_feature(torch.as_tensor(image_features), torch.as_tensor(aug_image_features))
        return (
            torch.as_tensor(eeg).float(),
            torch.as_tensor(image_features).float(),
            torch.as_tensor(text_features).float(),
            torch.as_tensor(depth_features).float(),
            torch.as_tensor(img_index).long(),
        )

    def __len__(self) -> int:
        return int(self.data.shape[0])


class SplitHalfViewsDataset(Dataset):
    def __init__(self, base_dataset: EEGFeatureDataset, no_leakage: bool = True, low_trial_bias: bool = False) -> None:
        self.base_dataset = base_dataset
        self.no_leakage = no_leakage
        self.low_trial_bias = low_trial_bias
        self.group_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
        subjects = base_dataset.subject_labels.detach().cpu().tolist()
        image_indices = base_dataset.sample_img_indices.detach().cpu().tolist()
        for sample_index, key in enumerate(zip(subjects, image_indices)):
            self.group_indices[(int(key[0]), int(key[1]))].append(sample_index)
        self.valid_groups = [
            key
            for key, indices in self.group_indices.items()
            if (len(indices) >= 2 if self.no_leakage else len(indices) >= 1)
        ]
        if not self.valid_groups:
            raise RuntimeError("No valid groups for split-half training")

    def __len__(self) -> int:
        return len(self.valid_groups)

    def _sample_count(self, count: int) -> int:
        if not self.low_trial_bias:
            return count
        weights = np.array([1.0 / (value + 1) for value in range(count)], dtype=np.float64)
        weights = weights / weights.sum()
        return int(np.random.choice(np.arange(1, count + 1), p=weights))

    def _mean_eeg(self, indices: np.ndarray) -> torch.Tensor:
        k = self._sample_count(len(indices))
        if k < len(indices):
            indices = np.random.permutation(indices)[:k]
        eeg_views = [self.base_dataset[int(sample_index)][0] for sample_index in indices]
        return torch.stack(eeg_views, dim=0).mean(dim=0)

    def __getitem__(self, index: int):
        group_key = self.valid_groups[index]
        all_indices = np.array(self.group_indices[group_key], dtype=np.int64)
        permuted = np.random.permutation(all_indices)
        if self.no_leakage:
            half = len(permuted) // 2
            if half == 0:
                raise RuntimeError(f"Too few trials for group {group_key}")
            indices_a = permuted[:half]
            indices_b = permuted[half:]
        else:
            indices_a = permuted
            indices_b = permuted
        eeg_a = self._mean_eeg(indices_a)
        eeg_b = self._mean_eeg(indices_b)
        _, image_features, text_features, depth_features, img_index = self.base_dataset[int(permuted[0])]
        return (eeg_a, eeg_b), image_features, text_features, depth_features, img_index


class SplitMixDataModule(L.LightningDataModule):
    def __init__(
        self,
        eeg_data_path: str,
        image_root: str,
        feature_path: str,
        subject: str,
        test_subject: str | None = None,
        batch_size: int = 512,
        batch_size_eval: int = 200,
        num_workers: int = 0,
        pin_memory: bool = False,
        time_window: tuple[float, float] = (0.0, 1.0),
        use_split_half_train: bool = True,
        no_leakage: bool = True,
        low_trial_bias: bool = False,
        selected_channels_mode: str = "all",
        eeg_aug: str = "none",
        image_feature_source: str = "original",
        missing_aug_fallback: str = "original",
    ) -> None:
        super().__init__()
        self.eeg_data_path = eeg_data_path
        self.image_root = image_root
        self.feature_path = feature_path
        self.subject = subject
        self.test_subject = test_subject or subject
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.time_window = tuple(time_window)
        self.use_split_half_train = use_split_half_train
        self.no_leakage = no_leakage
        self.low_trial_bias = low_trial_bias
        self.selected_channels_mode = selected_channels_mode
        self.eeg_aug = eeg_aug
        self.image_feature_source = image_feature_source
        self.missing_aug_fallback = missing_aug_fallback
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is None:
            train_dataset = EEGFeatureDataset(
                eeg_data_path=self.eeg_data_path,
                image_root=self.image_root,
                feature_path=self.feature_path,
                subject=self.subject,
                train=True,
                time_window=self.time_window,
                selected_channels_mode=self.selected_channels_mode,
                eeg_aug=self.eeg_aug,
                image_feature_source=self.image_feature_source,
                missing_aug_fallback=self.missing_aug_fallback,
            )
            if self.use_split_half_train:
                train_dataset = SplitHalfViewsDataset(
                    base_dataset=train_dataset,
                    no_leakage=self.no_leakage,
                    low_trial_bias=self.low_trial_bias,
                )
            self.train_dataset = train_dataset

        if self.val_dataset is None or self.test_dataset is None:
            eval_dataset = EEGFeatureDataset(
                eeg_data_path=self.eeg_data_path,
                image_root=self.image_root,
                feature_path=self.feature_path,
                subject=self.test_subject,
                train=False,
                time_window=self.time_window,
                selected_channels_mode=self.selected_channels_mode,
                eeg_aug="none",
                image_feature_source=self.image_feature_source,
                missing_aug_fallback=self.missing_aug_fallback,
            )
            self.val_dataset = eval_dataset
            self.test_dataset = eval_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
