"""Convert local NVIDIA PAI AV clips to the AutoVLA SFT format."""

import hashlib
import io
import json
import os
import pickle
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


MICROSECONDS_PER_SECOND = 1_000_000
CAMERA_MAP = {
    "camera_front_wide_120fov": "front_camera",
    "camera_cross_left_120fov": "front_left_camera",
    "camera_cross_right_120fov": "front_right_camera",
}
REQUIRED_FEATURES = (*CAMERA_MAP.keys(), "egomotion")


class LocalFeatures:
    """Small, pickle-friendly equivalent of physical_ai_av.dataset.Features."""

    def __init__(self, features_df: pd.DataFrame):
        self.features_df = features_df

    def get_chunk_feature_filename(self, chunk_id: int, feature: str) -> str:
        return self.features_df.at[feature, "chunk_path"].format(chunk_id=chunk_id)

    def get_clip_files_in_zip(self, clip_id: str, feature: str) -> Dict[str, str]:
        templates = self.features_df.at[feature, "clip_files_in_zip"]
        if not isinstance(templates, dict):
            raise ValueError(f"Feature {feature!r} is not stored as a zip file")
        return {key: value.format(clip_id=clip_id) for key, value in templates.items()}


class SimplePAIInterface:
    """Read already-downloaded PAI data without Hugging Face network access."""

    def __init__(self, local_dir: str):
        self.local_dir = Path(os.path.expandvars(local_dir)).expanduser()
        clip_index_path = self.local_dir / "clip_index.parquet"
        features_path = self.local_dir / "features.csv"
        if not clip_index_path.is_file() or not features_path.is_file():
            raise FileNotFoundError(
                f"PAI root must contain clip_index.parquet and features.csv: {self.local_dir}"
            )

        self.clip_index = pd.read_parquet(clip_index_path)
        features_df = pd.read_csv(features_path, index_col="feature")
        missing_features = set(REQUIRED_FEATURES).difference(features_df.index)
        if missing_features:
            raise ValueError(f"PAI features.csv is missing: {sorted(missing_features)}")

        def parse_zip_layout(value):
            if isinstance(value, str):
                return json.loads(value)
            return value

        features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
            parse_zip_layout, na_action="ignore"
        )
        self.features = LocalFeatures(features_df)
        self.feature_presence = self._load_feature_presence()
        self._availability_cache: Dict[tuple, List[str]] = {}
        print(
            f"SimplePAIInterface: {len(self.clip_index)} indexed clips, "
            f"root={self.local_dir}"
        )

    def _load_feature_presence(self) -> Optional[pd.DataFrame]:
        current = self.local_dir / "metadata/feature_presence.parquet"
        legacy = self.local_dir / "metadata/sensor_presence.parquet"
        if current.is_file():
            return pd.read_parquet(current)
        if legacy.is_file():
            return pd.read_parquet(legacy).select_dtypes(include=bool)
        warnings.warn(
            "No feature_presence metadata found; availability will be checked from "
            "local chunk files only.",
            stacklevel=2,
        )
        return None

    def get_clip_chunk(self, clip_id: str) -> int:
        return int(self.clip_index.at[clip_id, "chunk"])

    def chunk_path(self, chunk_id: int, feature: str) -> Path:
        return self.local_dir / self.features.get_chunk_feature_filename(
            chunk_id, feature
        )

    def available_clip_ids(self, required_features: Iterable[str]) -> List[str]:
        required_features = tuple(required_features)
        if required_features in self._availability_cache:
            return list(self._availability_cache[required_features])
        valid = pd.Series(True, index=self.clip_index.index, dtype=bool)

        if self.feature_presence is not None:
            presence = self.feature_presence.reindex(self.clip_index.index)
            for feature in required_features:
                if feature in presence.columns:
                    valid &= presence[feature].fillna(False).astype(bool)

        chunks = self.clip_index["chunk"]
        for feature in required_features:
            existing_chunks = {
                int(chunk_id)
                for chunk_id in chunks.unique()
                if self.chunk_path(int(chunk_id), feature).is_file()
            }
            valid &= chunks.isin(existing_chunks)

        clip_ids = list(self.clip_index.index[valid])
        print(
            f"Local availability filter: {len(clip_ids)}/{len(self.clip_index)} clips "
            f"contain {', '.join(required_features)}"
        )
        self._availability_cache[required_features] = clip_ids
        return list(clip_ids)

    def get_clip_feature(self, clip_id: str, feature: str) -> Any:
        """Load one feature using the same packed-file semantics as physical_ai_av."""
        from physical_ai_av import calibration, egomotion, video

        chunk_filename = self.features.get_chunk_feature_filename(
            self.get_clip_chunk(clip_id), feature
        )
        local_path = self.local_dir / chunk_filename
        if not local_path.is_file():
            raise FileNotFoundError(local_path)

        with local_path.open("rb") as file_handle:
            if chunk_filename.endswith(".parquet"):
                feature_df = pd.read_parquet(file_handle).loc[clip_id]
                if feature == "sensor_extrinsics":
                    return calibration.SensorExtrinsics.from_extrinsics_df(feature_df)
                if feature == "camera_intrinsics":
                    return calibration.CameraIntrinsics.from_intrinsics_df(feature_df)
                if feature == "vehicle_dimensions":
                    return calibration.VehicleDimensions.from_dimensions_df(feature_df)
                return feature_df

            if not chunk_filename.endswith(".zip"):
                raise ValueError(f"Unexpected PAI file type: {chunk_filename}")

            clip_files = self.features.get_clip_files_in_zip(clip_id, feature)
            with zipfile.ZipFile(file_handle, "r") as zip_file:
                if feature == "egomotion":
                    ego_df = pd.read_parquet(
                        io.BytesIO(zip_file.read(clip_files["egomotion"]))
                    )
                    return egomotion.EgomotionState.from_egomotion_df(
                        ego_df
                    ).create_interpolator(ego_df["timestamp"].to_numpy(copy=True))
                if feature.startswith("camera"):
                    timestamps = pd.read_parquet(
                        io.BytesIO(zip_file.read(clip_files["frame_timestamps"]))
                    )["timestamp"].to_numpy(copy=True)
                    return video.SeekVideoReader(
                        video_data=io.BytesIO(zip_file.read(clip_files["video"])),
                        timestamps=timestamps,
                    )
                return {
                    key: pd.read_parquet(io.BytesIO(zip_file.read(filename)))
                    if filename.endswith(".parquet")
                    else io.BytesIO(zip_file.read(filename))
                    for key, filename in clip_files.items()
                }


def split_clip_ids(
    clip_ids: Sequence[str],
    split: str,
    validation_fraction: float,
    seed: int,
) -> List[str]:
    """Create a deterministic, disjoint clip-level train/validation split."""
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    selected = []
    for clip_id in clip_ids:
        digest = hashlib.sha256(f"{seed}:{clip_id}".encode("utf-8")).digest()
        score = int.from_bytes(digest[:8], "big") / 2**64
        is_validation = score < validation_fraction
        if (split == "val") == is_validation:
            selected.append(clip_id)

    rng = np.random.default_rng(seed + (1 if split == "val" else 0))
    rng.shuffle(selected)
    return selected


def _polygon_contour(
    position: torch.Tensor,
    heading: torch.Tensor,
    width: float,
    length: float,
) -> torch.Tensor:
    """Return [left-front, right-front, right-back, left-back]."""
    x, y = position[0], position[1]
    half_cos = 0.5 * torch.cos(heading)
    half_sin = 0.5 * torch.sin(heading)
    length_cos, length_sin = length * half_cos, length * half_sin
    width_cos, width_sin = width * half_cos, width * half_sin
    return torch.stack(
        (
            torch.stack((x + length_cos - width_sin, y + length_sin + width_cos)),
            torch.stack((x + length_cos + width_sin, y + length_sin - width_cos)),
            torch.stack((x - length_cos + width_sin, y - length_sin - width_cos)),
            torch.stack((x - length_cos - width_sin, y - length_sin + width_cos)),
        )
    )


class PAICodebookMatcher:
    """Exact deterministic form of AutoVLA TokenProcessor._match_agent_token."""

    def __init__(
        self,
        codebook_path: Optional[str] = None,
        *,
        code_book: Optional[torch.Tensor] = None,
        vehicle_width: float = 2.0,
        vehicle_length: float = 4.8,
    ):
        if code_book is None:
            if codebook_path is None:
                raise ValueError("codebook_path or code_book is required")
            resolved_codebook_path = Path(
                os.path.expandvars(os.path.expanduser(codebook_path))
            )
            with resolved_codebook_path.open("rb") as f:
                data = pickle.load(f)
            code_book = torch.as_tensor(data["token_all"]["veh"])

        self.code_book = code_book.to(dtype=torch.float32, device="cpu")
        if self.code_book.ndim != 4 or tuple(self.code_book.shape[-2:]) != (4, 2):
            raise ValueError(
                "Vehicle codebook must have shape [n_tokens, n_substeps, 4, 2], "
                f"got {tuple(self.code_book.shape)}"
            )
        self.token_end_contours = self.code_book[:, -1].contiguous()
        self.n_bins = int(self.code_book.shape[0])
        self.vehicle_width = float(vehicle_width)
        self.vehicle_length = float(vehicle_length)

    @staticmethod
    def _to_global(
        local_contours: torch.Tensor, position: torch.Tensor, heading: torch.Tensor
    ) -> torch.Tensor:
        cos_heading, sin_heading = torch.cos(heading), torch.sin(heading)
        x = (
            cos_heading * local_contours[..., 0]
            - sin_heading * local_contours[..., 1]
            + position[0]
        )
        y = (
            sin_heading * local_contours[..., 0]
            + cos_heading * local_contours[..., 1]
            + position[1]
        )
        return torch.stack((x, y), dim=-1)

    def match(self, gt_xy: np.ndarray, gt_heading: np.ndarray) -> np.ndarray:
        gt_xy_tensor = torch.as_tensor(gt_xy, dtype=torch.float32)
        gt_heading_tensor = torch.as_tensor(gt_heading, dtype=torch.float32)
        if gt_xy_tensor.ndim != 2 or gt_xy_tensor.shape[1] != 2:
            raise ValueError(
                f"gt_xy must have shape [steps, 2], got {gt_xy_tensor.shape}"
            )
        if gt_heading_tensor.shape != (gt_xy_tensor.shape[0],):
            raise ValueError("gt_heading length must match gt_xy")
        if (
            not torch.isfinite(gt_xy_tensor).all()
            or not torch.isfinite(gt_heading_tensor).all()
        ):
            raise ValueError("Ground-truth trajectory contains non-finite values")

        indices = []
        previous_position = torch.zeros(2, dtype=torch.float32)
        previous_heading = torch.tensor(0.0, dtype=torch.float32)
        for target_position, target_heading in zip(gt_xy_tensor, gt_heading_tensor):
            target_contour = _polygon_contour(
                target_position,
                target_heading,
                self.vehicle_width,
                self.vehicle_length,
            )
            token_world = self._to_global(
                self.token_end_contours, previous_position, previous_heading
            )
            distance = torch.norm(
                token_world - target_contour.unsqueeze(0), dim=-1
            ).sum(-1)
            token_index = int(torch.argmin(distance))
            selected_contour = token_world[token_index]

            indices.append(token_index)
            previous_position = selected_contour.mean(dim=0)
            forward_edge = selected_contour[0] - selected_contour[3]
            previous_heading = torch.atan2(forward_edge[1], forward_edge[0])

        return np.asarray(indices, dtype=np.int64)


class PAISFTDataset(Dataset):
    def __init__(
        self,
        dataset_interface: SimplePAIInterface,
        data_config: dict,
        model_config: dict,
        processor,
        split: str,
        max_samples: Optional[int] = None,
    ):
        if model_config.get("use_cot", False):
            raise NotImplementedError(
                "PAI SFT currently supports trajectory-only targets"
            )

        self.ds = dataset_interface
        self.processor = processor
        self.split = split
        self.anchor_time_s = float(data_config.get("anchor_time_s", 2.0))
        self.max_retries = int(data_config.get("max_retries", 20))
        self.frame_interval_s = float(data_config.get("frame_interval_s", 0.5))
        self.num_context_frames = int(data_config.get("num_context_frames", 4))
        trajectory_cfg = model_config["trajectory"]
        self.num_poses = int(trajectory_cfg["num_poses"])
        self.pose_interval_s = float(trajectory_cfg["interval_length"])
        video_cfg = model_config["video"]
        self.min_pixels = int(video_cfg["min_pixels"])
        self.max_pixels = int(video_cfg["max_pixels"])
        if self.anchor_time_s < 0:
            raise ValueError("anchor_time_s must be non-negative")
        if self.frame_interval_s <= 0 or self.pose_interval_s <= 0:
            raise ValueError("Frame and trajectory intervals must be positive")
        if self.num_context_frames <= 0 or self.num_poses <= 0:
            raise ValueError("Frame and pose counts must be positive")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be positive")
        if self.min_pixels <= 0 or self.max_pixels < self.min_pixels:
            raise ValueError("Invalid video pixel bounds")
        configured_horizon = float(trajectory_cfg.get("time_horizon", 0.0))
        actual_horizon = self.num_poses * self.pose_interval_s
        if configured_horizon and not np.isclose(configured_horizon, actual_horizon):
            raise ValueError(
                f"Trajectory horizon mismatch: configured={configured_horizon}, "
                f"num_poses*interval={actual_horizon}"
            )

        self.matcher = PAICodebookMatcher(
            model_config["codebook_cache_path"],
            vehicle_width=float(model_config.get("codebook_vehicle_width", 2.0)),
            vehicle_length=float(model_config.get("codebook_vehicle_length", 4.8)),
        )
        available_ids = self.ds.available_clip_ids(REQUIRED_FEATURES)
        self.clip_ids = split_clip_ids(
            available_ids,
            split=split,
            validation_fraction=float(data_config.get("validation_fraction", 0.01)),
            seed=int(data_config.get("split_seed", 42)),
        )
        if max_samples is not None:
            if int(max_samples) <= 0:
                raise ValueError("max_samples must be positive or null")
            self.clip_ids = self.clip_ids[: int(max_samples)]
        if not self.clip_ids:
            raise RuntimeError(f"No locally available clips remain for split={split}")
        print(f"PAI SFT {split} split: {len(self.clip_ids)} clips")

    def __len__(self):
        return len(self.clip_ids)

    def _context_timestamps_us(self) -> np.ndarray:
        first_time_s = self.anchor_time_s - self.frame_interval_s * (
            self.num_context_frames - 1
        )
        if first_time_s < 0:
            raise ValueError(
                "anchor_time_s is too early for the requested context frames"
            )
        timestamps_s = (
            first_time_s + np.arange(self.num_context_frames) * self.frame_interval_s
        )
        return np.rint(timestamps_s * MICROSECONDS_PER_SECOND).astype(np.int64)

    def _extract_camera_frames(self, clip_id: str) -> Dict[str, List[Image.Image]]:
        requested_timestamps = self._context_timestamps_us()
        camera_frames: Dict[str, List[Image.Image]] = {}
        for pai_camera, role in CAMERA_MAP.items():
            reader = self.ds.get_clip_feature(clip_id, pai_camera)
            try:
                if (
                    reader.timestamps is None
                    or len(reader.timestamps) < self.num_context_frames
                ):
                    raise ValueError(f"Not enough frames for {pai_camera}")
                if (
                    requested_timestamps[0] < reader.timestamps[0]
                    or requested_timestamps[-1] > reader.timestamps[-1]
                ):
                    raise ValueError(
                        f"Requested camera range {requested_timestamps[[0, -1]].tolist()} "
                        f"is outside {reader.timestamps[[0, -1]].tolist()}"
                    )
                images, _ = reader.decode_images_from_timestamps(requested_timestamps)
                if len(images) != self.num_context_frames:
                    raise ValueError(
                        f"Decoded {len(images)} frames, expected {self.num_context_frames}"
                    )
                camera_frames[role] = [
                    Image.fromarray(frame).convert("RGB") for frame in images
                ]
            finally:
                reader.close()
        return camera_frames

    @staticmethod
    def _yaw(state) -> float:
        euler = np.asarray(state.pose.rotation.as_euler("xyz"))
        return float(euler[-1])

    def _extract_ego_motion(self, clip_id: str) -> Dict[str, Any]:
        interpolator = self.ds.get_clip_feature(clip_id, "egomotion")
        anchor_us = int(round(self.anchor_time_s * MICROSECONDS_PER_SECOND))
        future_us = anchor_us + np.rint(
            np.arange(1, self.num_poses + 1)
            * self.pose_interval_s
            * MICROSECONDS_PER_SECOND
        ).astype(np.int64)

        if hasattr(interpolator, "time_range"):
            first_us, last_us = interpolator.time_range
            if anchor_us < first_us or future_us[-1] > last_us:
                raise ValueError(
                    f"Egomotion range {first_us}..{last_us} does not cover "
                    f"{anchor_us}..{future_us[-1]}"
                )

        current_state = interpolator(anchor_us)
        future_states = [interpolator(int(timestamp)) for timestamp in future_us]
        current_position = np.asarray(current_state.pose.translation, dtype=np.float64)[
            :2
        ]
        current_heading = self._yaw(current_state)
        future_positions = np.asarray(
            [
                np.asarray(state.pose.translation, dtype=np.float64)[:2]
                for state in future_states
            ]
        )
        future_headings = np.asarray([self._yaw(state) for state in future_states])

        delta = future_positions - current_position
        cos_heading, sin_heading = np.cos(current_heading), np.sin(current_heading)
        relative_positions = np.empty_like(delta)
        relative_positions[:, 0] = cos_heading * delta[:, 0] + sin_heading * delta[:, 1]
        relative_positions[:, 1] = (
            -sin_heading * delta[:, 0] + cos_heading * delta[:, 1]
        )
        relative_headings = np.arctan2(
            np.sin(future_headings - current_heading),
            np.cos(future_headings - current_heading),
        )

        velocity = np.asarray(current_state.velocity, dtype=np.float64)
        acceleration = np.asarray(current_state.acceleration, dtype=np.float64)
        result = {
            "gt_xy": relative_positions.astype(np.float32),
            "gt_heading": relative_headings.astype(np.float32),
            "velocity": float(np.linalg.norm(velocity[:2])),
            "acceleration": float(np.linalg.norm(acceleration[:2])),
        }
        if not all(
            np.isfinite(value).all()
            if isinstance(value, np.ndarray)
            else np.isfinite(value)
            for value in result.values()
        ):
            raise ValueError("Egomotion interpolation produced non-finite values")
        return result

    def _build_sample(self, clip_id: str) -> Dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        cameras = self._extract_camera_frames(clip_id)
        ego = self._extract_ego_motion(clip_id)
        action_indices = self.matcher.match(ego["gt_xy"], ego["gt_heading"])
        action_text = "".join(f"<action_{index}>" for index in action_indices)
        sample_rate_hz = 1.0 / self.frame_interval_s
        video_description = (
            f"comprising {self.num_context_frames} sequential frames sampled at "
            f"{sample_rate_hz:g} Hz."
        )

        system_content = [
            {
                "type": "text",
                "text": (
                    "You are an Advanced Driver Assistance and Full Self-Driving System. "
                    "You will be provided with video observations from the ego vehicle's "
                    "surrounding cameras, along with the vehicle's current dynamic states. "
                    "Your task is to predict the most appropriate driving action for the "
                    "next five seconds."
                ),
            }
        ]
        user_content = [
            {
                "type": "text",
                "text": (
                    "The autonomous vehicle is equipped with three cameras mounted at the "
                    "front, left, and right, enabling a comprehensive perception of the "
                    "surrounding environment."
                ),
            },
            {
                "type": "text",
                "text": f"The first video presents the front view of the vehicle, {video_description}",
            },
            {
                "type": "video",
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "sample_fps": sample_rate_hz,
                "video": cameras["front_camera"],
            },
            {
                "type": "text",
                "text": f"The second video presents the front-left view of the vehicle, {video_description}",
            },
            {
                "type": "video",
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "sample_fps": sample_rate_hz,
                "video": cameras["front_left_camera"],
            },
            {
                "type": "text",
                "text": f"The third video presents the front-right view of the vehicle, {video_description}",
            },
            {
                "type": "video",
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "sample_fps": sample_rate_hz,
                "video": cameras["front_right_camera"],
            },
            {
                "type": "text",
                "text": (
                    f"The current velocity of the vehicle is {ego['velocity']:.3f} m/s, "
                    f"and the current acceleration is {ego['acceleration']:.3f} m/s². "
                    "No route or navigation command is available for this clip. Based on "
                    "the observations and current dynamics, plan a safe action trajectory "
                    "for the autonomous vehicle over the next five seconds."
                ),
            },
        ]
        assistant_content = [
            {
                "type": "text",
                "text": (
                    f"<answer>\nThe final output action is: {action_text}\n</answer>"
                ),
            }
        ]
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            add_vision_id=True,
        )
        return {
            "text": text,
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "gt_trajectory": torch.as_tensor(ego["gt_xy"], dtype=torch.float32),
            "gt_action": torch.as_tensor(action_indices, dtype=torch.int64),
            "has_cot": False,
            "clip_id": clip_id,
        }

    def __getitem__(self, index: int):
        errors = []
        attempts = min(self.max_retries, len(self.clip_ids))
        for offset in range(attempts):
            clip_id = self.clip_ids[(index + offset) % len(self.clip_ids)]
            try:
                sample = self._build_sample(clip_id)
                if offset:
                    warnings.warn(
                        f"Skipped {offset} unreadable clip(s); using {clip_id}",
                        stacklevel=2,
                    )
                return sample
            except (
                FileNotFoundError,
                KeyError,
                OSError,
                ValueError,
                RuntimeError,
            ) as exc:
                errors.append(f"{clip_id}: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            f"Unable to read a valid {self.split} sample after {attempts} attempts. "
            + " | ".join(errors[-3:])
        )


class PAIDataCollator:
    def __init__(
        self,
        processor,
        ignore_index: int,
        action_start_id: int,
        n_action_tokens: int,
    ):
        self.processor = processor
        self.ignore_index = int(ignore_index)
        self.action_start_id = int(action_start_id)
        self.n_action_tokens = int(n_action_tokens)
        self.assistant_marker = processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        if not self.assistant_marker:
            raise RuntimeError("Could not tokenize Qwen assistant marker")

    @staticmethod
    def _find_subsequence(
        sequence: torch.Tensor, pattern: Sequence[int]
    ) -> Optional[int]:
        pattern_tensor = torch.as_tensor(pattern, dtype=sequence.dtype)
        for start in range(len(sequence) - len(pattern) + 1):
            if torch.equal(sequence[start : start + len(pattern)], pattern_tensor):
                return start
        return None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [feature["text"] for feature in features]
        video_inputs = []
        image_inputs = []
        for feature in features:
            if feature.get("video_inputs"):
                video_inputs.extend(feature["video_inputs"])
            if feature.get("image_inputs"):
                image_inputs.extend(feature["image_inputs"])
        if not video_inputs:
            raise RuntimeError("PAI batch contains no video inputs")

        batch = self.processor(
            text=texts,
            images=image_inputs or None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = self.ignore_index
        action_end_id = self.action_start_id + self.n_action_tokens
        for row, feature in enumerate(features):
            marker_start = self._find_subsequence(labels[row], self.assistant_marker)
            if marker_start is None:
                raise RuntimeError(
                    "Assistant marker not found in tokenized SFT example"
                )
            response_start = marker_start + len(self.assistant_marker)
            labels[row, :response_start] = self.ignore_index

            actual_action_ids = labels[row][
                (labels[row] >= self.action_start_id) & (labels[row] < action_end_id)
            ]
            expected_action_ids = feature["gt_action"] + self.action_start_id
            if not torch.equal(actual_action_ids.cpu(), expected_action_ids.cpu()):
                raise RuntimeError(
                    "Action targets were not tokenized atomically. "
                    f"Expected {expected_action_ids.tolist()}, got {actual_action_ids.tolist()}"
                )

        batch["labels"] = labels
        batch["gt_trajectory"] = torch.stack(
            [feature["gt_trajectory"] for feature in features]
        )
        batch["gt_action"] = torch.stack([feature["gt_action"] for feature in features])
        batch["has_cot"] = torch.tensor(
            [feature.get("has_cot", False) for feature in features], dtype=torch.bool
        )
        return batch
