"""PAI dataset → AutoVLA SFT format converter.

Loads clips from the NVIDIA PhysicalAI-Autonomous-Vehicles dataset using
the `physical_ai_av` package, extracts camera frames + ego motion, converts
the GT trajectory to AutoVLA action tokens, and builds the same prompt format
as AutoVLA's original SFTDataset.
"""

import contextlib
import io
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Add AutoVLA repo to path (set AUTOVLA_REPO_PATH env var)
_AUTOVLA_REPO = os.environ.get("AUTOVLA_REPO_PATH", "/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA")
sys.path.insert(0, _AUTOVLA_REPO)

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# Camera mapping: PAI camera names → AutoVLA camera roles
CAMERA_MAP = {
    "camera_front_wide_120fov": "front_camera",
    "camera_cross_left_120fov": "front_left_camera",
    "camera_cross_right_120fov": "front_right_camera",
}

NUM_CONTEXT_FRAMES = 4
FRAME_INTERVAL_S = 0.5  # 2 Hz
NUM_POSES = 10
POSE_INTERVAL_S = 0.5
TIME_HORIZON_S = 5.0
ACTION_START_ID = 151665


class LocalPAIDataset:
    """Wrapper around PhysicalAIAVDatasetInterface that reads from local_dir first.

    Overrides download_file and open_file to check local_dir before the HF cache.
    This bypasses the need for HF authentication when data is already downloaded.
    """

    def __init__(self, local_dir, revision="main"):
        from physical_ai_av import PhysicalAIAVDatasetInterface
        from physical_ai_av.utils import hf_interface

        # Create the real interface but with our patched methods
        self._real = PhysicalAIAVDatasetInterface.__new__(PhysicalAIAVDatasetInterface)
        # Patch download_file and open_file on the instance
        self._real.download_file = self._download_file.__get__(self._real)
        self._real.open_file = self._open_file.__get__(self._real)
        # Now call __init__ which uses our patched download_file
        self._real.__init__(local_dir=local_dir, revision=revision)

    @property
    def clip_index(self):
        return self._real.clip_index

    @property
    def features(self):
        return self._real.features

    def get_clip_chunk(self, clip_id):
        return self._real.get_clip_chunk(clip_id)

    def get_clip_feature(self, clip_id, feature, maybe_stream=False):
        return self._real.get_clip_feature(clip_id, feature, maybe_stream=maybe_stream)

    def _download_file(self, filename, **kwargs):
        if self.local_dir:
            local_path = Path(self.local_dir) / filename
            if local_path.exists():
                return str(local_path)
        # Fall back to original method
        from physical_ai_av.utils.hf_interface import HfRepoInterface
        return HfRepoInterface.download_file(self, filename, **kwargs)

    @contextlib.contextmanager
    def _open_file(self, filename, mode="rb", maybe_stream=False) -> Iterator[io.BytesIO]:
        if self.local_dir:
            local_path = Path(self.local_dir) / filename
            if local_path.exists():
                with open(local_path, mode) as f:
                    yield f
                    return
        # Fall back to original method
        from physical_ai_av.utils.hf_interface import HfRepoInterface
        with HfRepoInterface.open_file(self, filename, mode, maybe_stream) as f:
            yield f


class PAICodebookMatcher:
    """Matches GT trajectory waypoints to action token indices using the codebook."""

    def __init__(self, codebook_path: str):
        with open(codebook_path, "rb") as f:
            data = pickle.load(f)
        self.code_book = torch.tensor(data["token_all"]["veh"], dtype=torch.float32)
        self.n_bins = self.code_book.shape[0]

    def match(self, gt_xy: np.ndarray, gt_heading: np.ndarray) -> np.ndarray:
        n_steps = gt_xy.shape[0]
        indices = np.zeros(n_steps, dtype=np.int64)

        prev_pos = torch.tensor([[0.0, 0.0]])
        prev_head = torch.tensor([0.0])

        for i in range(n_steps):
            target_pos = torch.tensor(gt_xy[i:i+1], dtype=torch.float32)

            cb = self.code_book
            cb_flat = cb.reshape(-1, 4, 2)

            cos_h = torch.cos(prev_head)
            sin_h = torch.sin(prev_head)
            cb_global_x = cos_h * cb_flat[..., 0] - sin_h * cb_flat[..., 1] + prev_pos[0, 0]
            cb_global_y = sin_h * cb_flat[..., 0] + cos_h * cb_flat[..., 1] + prev_pos[0, 1]
            cb_global = torch.stack([cb_global_x, cb_global_y], dim=-1)

            cb_endpoints = cb_global[:, -1, :].reshape(self.n_bins, 6, 2)
            cb_mean = cb_endpoints.mean(dim=1)

            dist = torch.norm(cb_mean - target_pos, dim=-1)
            best_idx = torch.argmin(dist).item()
            indices[i] = best_idx

            prev_pos = cb_mean[best_idx].unsqueeze(0)
            best_candidate = cb_flat[best_idx * 6]
            dxy = best_candidate[-1] - best_candidate[0]
            prev_head = torch.atan2(dxy[1], dxy[0]).unsqueeze(0)

        return indices


class PAISFTDataset(Dataset):
    """SFT dataset that loads PAI clips and converts to AutoVLA format."""

    def __init__(
        self,
        pai_data_dir: str,
        model_config: dict,
        processor: AutoProcessor,
        using_cot: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.pai_data_dir = Path(pai_data_dir)
        self.processor = processor
        self.using_cot = using_cot
        self.model_config = model_config

        codebook_path = model_config.get("codebook_cache_path", "codebook_cache/agent_vocab.pkl")
        self.matcher = PAICodebookMatcher(codebook_path)

        self.ds = LocalPAIDataset(local_dir=str(self.pai_data_dir), revision="main")

        self.clip_ids = list(self.ds.clip_index["clip_id"].values)
        if max_samples is not None:
            self.clip_ids = self.clip_ids[:max_samples]

        print(f"PAI SFT dataset: {len(self.clip_ids)} clips from {self.pai_data_dir}")

    def __len__(self):
        return len(self.clip_ids)

    def _extract_camera_frames(self, clip_id: str) -> Optional[Dict[str, List[str]]]:
        """Extract 4 frames at 2Hz from each of the 3 cameras."""
        from PIL import Image
        from physical_ai_av import video as pai_video

        camera_frames = {}
        for pai_cam, autovla_cam in CAMERA_MAP.items():
            try:
                reader = self.ds.get_clip_feature(clip_id, pai_cam)
                if reader is None:
                    return None

                # Get timestamps of the first few frames
                ts = reader.timestamps
                if len(ts) < NUM_CONTEXT_FRAMES:
                    return None

                # Sample 4 frames at 2Hz from the start
                frame_ts = np.array([ts[0] + i * FRAME_INTERVAL_S * 1e6 for i in range(NUM_CONTEXT_FRAMES)])
                frame_ts = np.minimum(frame_ts, ts[-1])

                images, _ = reader.decode_images_from_timestamps(frame_ts)
                if len(images) < NUM_CONTEXT_FRAMES:
                    return None

                frame_paths = []
                for j, frame in enumerate(images[:NUM_CONTEXT_FRAMES]):
                    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    Image.fromarray(frame).save(tmp.name, quality=90)
                    frame_paths.append(tmp.name)

                camera_frames[autovla_cam] = frame_paths
            except Exception as e:
                print(f"  Camera {pai_cam} failed for clip {clip_id}: {e}")
                return None

        return camera_frames

    def _extract_ego_motion(self, clip_id: str) -> Optional[Dict]:
        """Extract ego motion and compute GT trajectory."""
        try:
            ego_interp = self.ds.get_clip_feature(clip_id, "egomotion")
            if ego_interp is None:
                return None

            # Get ego states at 10 future waypoints (0.5s intervals over 5s)
            # Start from the last context frame timestamp
            timestamps = [i * POSE_INTERVAL_S for i in range(NUM_POSES + 1)]
            ego_states = []
            for ts in timestamps[1:]:  # skip t=0 (current position)
                try:
                    state = ego_interp(ts)
                    ego_states.append(state)
                except Exception:
                    return None

            if len(ego_states) < NUM_POSES:
                return None

            # Extract position and heading from ego states
            positions = np.array([[s.pose.translation[0], s.pose.translation[1]] for s in ego_states])
            headings = np.array([s.pose.rotation.as_euler('z') for s in ego_states])
            velocity = np.sqrt(ego_states[0].velocity[0]**2 + ego_states[0].velocity[1]**2)
            acceleration = np.sqrt(ego_states[0].acceleration[0]**2 + ego_states[0].acceleration[1]**2)

            # Transform to ego frame (relative to first waypoint)
            x0, y0, h0 = positions[0, 0], positions[0, 1], headings[0]
            cos_h, sin_h = np.cos(-h0), np.sin(-h0)
            rel_pos = np.zeros_like(positions)
            for i in range(len(positions)):
                dx, dy = positions[i, 0] - x0, positions[i, 1] - y0
                rel_pos[i, 0] = cos_h * dx - sin_h * dy
                rel_pos[i, 1] = sin_h * dx + cos_h * dy
            rel_heading = headings - h0

            return {
                "gt_xy": rel_pos,
                "gt_heading": rel_heading,
                "velocity": float(velocity),
                "acceleration": float(acceleration),
            }
        except Exception as e:
            print(f"  Ego motion failed for clip {clip_id}: {e}")
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clip_id = self.clip_ids[idx]

        camera_frames = self._extract_camera_frames(clip_id)
        if camera_frames is None:
            return self.__getitem__((idx + 1) % len(self))

        ego_data = self._extract_ego_motion(clip_id)
        if ego_data is None:
            return self.__getitem__((idx + 1) % len(self))

        gt_action_idx = self.matcher.match(ego_data["gt_xy"], ego_data["gt_heading"])

        action_text = "".join([f"<action_{gt_action_idx[i]}>" for i in range(len(gt_action_idx))])

        front = camera_frames["front_camera"]
        left = camera_frames["front_left_camera"]
        right = camera_frames["front_right_camera"]

        system_content = [{
            "type": "text",
            "text": (
                "You are an Advanced Driver Assistance and Full Self-Driving System. "
                "You will be provided with video observations from the ego vehicle's surrounding cameras, along with the vehicle's current dynamic states. "
                "Your task is to predict the most appropriate driving action for the next five seconds."
            )
        }]

        user_content = [
            {"type": "text", "text": "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."},
            {"type": "text", "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
            {"type": "video", "min_pixels": 28*28*128, "max_pixels": 28*28*128, "video": [f"file://{f}" for f in front]},
            {"type": "text", "text": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
            {"type": "video", "min_pixels": 28*28*128, "max_pixels": 28*28*128, "video": [f"file://{f}" for f in left]},
            {"type": "text", "text": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
            {"type": "video", "min_pixels": 28*28*128, "max_pixels": 28*28*128, "video": [f"file://{f}" for f in right]},
            {"type": "text", "text": f"The current velocity of the vehicle is {ego_data['velocity']:.3f} m/s, and the current acceleration is {ego_data['acceleration']:.3f} m/s². The driving instruction is: straight. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."},
        ]

        assistant_content = [{
            "type": "text",
            "text": f"<answer>\nThe final output action is: {action_text}\n</answer>"
        }]

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )

        return {
            "text": text,
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "gt_trajectory": torch.tensor(ego_data["gt_xy"], dtype=torch.float32),
            "gt_action": torch.tensor(gt_action_idx, dtype=torch.int64),
            "has_cot": False,
            "clip_id": clip_id,
        }


class PAIDataCollator:
    def __init__(self, processor: AutoProcessor, ignore_index: int = -100):
        self.processor = processor
        self.ignore_index = ignore_index
        self.assistant_id = [151644, 77091]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        video_inputs = []
        image_inputs = []
        has_cot = []
        for f in features:
            video_inputs.extend(f.get("video_inputs", []))
            image_inputs.append(f.get("image_inputs"))
            has_cot.append(f.get("has_cot", False))

        batch = self.processor(
            text=texts,
            images=image_inputs if image_inputs[0] is not None else None,
            videos=video_inputs if video_inputs[0] is not None else None,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        assistant_id = torch.tensor(self.assistant_id)

        for i in range(labels.shape[0]):
            for j in range(len(labels[i]) - len(assistant_id) + 1):
                if torch.equal(labels[i][j:j + len(assistant_id)], assistant_id):
                    labels[i, :j] = self.ignore_index
                    break

        batch["labels"] = labels
        batch["gt_trajectory"] = torch.stack([f["gt_trajectory"] for f in features])
        batch["gt_action"] = torch.stack([f["gt_action"] for f in features])
        batch["has_cot"] = torch.tensor(has_cot)

        return batch
