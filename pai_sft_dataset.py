"""PAI dataset → AutoVLA SFT format converter.

Loads clips from the NVIDIA PhysicalAI-Autonomous-Vehicles dataset using
the `physical_ai_av` package, extracts camera frames + ego motion, converts
the GT trajectory to AutoVLA action tokens, and builds the same prompt format
as AutoVLA's original SFTDataset.
"""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

# Sampling: 4 frames at 2Hz = 2 seconds of history
NUM_CONTEXT_FRAMES = 4
FRAME_INTERVAL_S = 0.5  # 2 Hz

# Trajectory: 10 waypoints at 0.5s intervals = 5 seconds future
NUM_POSES = 10
POSE_INTERVAL_S = 0.5
TIME_HORIZON_S = 5.0

ACTION_START_ID = 151665


class PAICodebookMatcher:
    """Matches GT trajectory waypoints to action token indices using the codebook.

    The codebook has shape (n_bins, 6, 4, 2): n_bins discrete action tokens,
    each with 6 candidate sub-trajectories of 4 timesteps (x, y).

    For each GT waypoint, we find the action token whose best candidate
    sub-trajectory (transformed to global frame) ends closest to the GT position.
    """

    def __init__(self, codebook_path: str):
        with open(codebook_path, "rb") as f:
            data = pickle.load(f)
        self.code_book = torch.tensor(data["token_all"]["veh"], dtype=torch.float32)
        # (n_bins, 6, 4, 2)
        self.n_bins = self.code_book.shape[0]

    def match(self, gt_xy: np.ndarray, gt_heading: np.ndarray) -> np.ndarray:
        """Match GT trajectory to action token indices.

        Args:
            gt_xy: (N, 2) GT positions in ego frame (relative to start)
            gt_heading: (N,) GT headings in ego frame

        Returns:
            (N,) array of action token indices
        """
        n_steps = gt_xy.shape[0]
        indices = np.zeros(n_steps, dtype=np.int64)

        prev_pos = torch.tensor([[0.0, 0.0]])
        prev_head = torch.tensor([0.0])

        for i in range(n_steps):
            # Target: GT position at step i
            target_pos = torch.tensor(gt_xy[i:i+1], dtype=torch.float32)  # (1, 2)

            # Transform all codebook candidates to global frame
            # code_book: (n_bins, 6, 4, 2) → flatten to (n_bins*6, 4, 2)
            cb = self.code_book  # (n_bins, 6, 4, 2)
            cb_flat = cb.reshape(-1, 4, 2)  # (n_bins*6, 4, 2)

            # Rotate by prev_head and translate by prev_pos
            cos_h = torch.cos(prev_head)
            sin_h = torch.sin(prev_head)
            # (n_bins*6, 4, 2) rotated
            cb_global_x = cos_h * cb_flat[..., 0] - sin_h * cb_flat[..., 1] + prev_pos[0, 0]
            cb_global_y = sin_h * cb_flat[..., 0] + cos_h * cb_flat[..., 1] + prev_pos[0, 1]
            cb_global = torch.stack([cb_global_x, cb_global_y], dim=-1)  # (n_bins*6, 4, 2)

            # For each action token, take the mean of the 6 candidates' endpoints
            cb_endpoints = cb_global[:, -1, :]  # (n_bins*6, 2)
            cb_endpoints = cb_endpoints.reshape(self.n_bins, 6, 2)  # (n_bins, 6, 2)
            cb_mean = cb_endpoints.mean(dim=1)  # (n_bins, 2)

            # Find closest action token to target
            dist = torch.norm(cb_mean - target_pos, dim=-1)  # (n_bins,)
            best_idx = torch.argmin(dist).item()
            indices[i] = best_idx

            # Update prev_pos and prev_head for next step
            best_endpoint = cb_mean[best_idx]  # (2,)
            prev_pos = best_endpoint.unsqueeze(0)  # (1, 2)

            # Update heading from codebook: direction from point[0] to point[-1]
            best_candidate = cb_flat[best_idx * 6]  # (4, 2) — first candidate
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
        clip_index_path: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.pai_data_dir = Path(pai_data_dir)
        self.processor = processor
        self.using_cot = using_cot
        self.model_config = model_config

        # Load codebook matcher
        codebook_path = model_config.get("codebook_cache_path", "codebook_cache/agent_vocab.pkl")
        self.matcher = PAICodebookMatcher(codebook_path)

        # Load PAI clip index
        from physical_ai_av import PhysicalAIAVDatasetInterface
        self.ds = PhysicalAIAVDatasetInterface(local_dir=str(self.pai_data_dir))

        # Get clip IDs
        self.clip_ids = list(self.ds.clip_index["clip_id"].values)
        if max_samples is not None:
            self.clip_ids = self.clip_ids[:max_samples]

        print(f"PAI SFT dataset: {len(self.clip_ids)} clips from {self.pai_data_dir}")

    def __len__(self):
        return len(self.clip_ids)

    def _extract_camera_frames(self, clip_id: str) -> Optional[Dict[str, List[str]]]:
        """Extract 4 frames at 2Hz from each of the 3 cameras."""
        from physical_ai_av import video as pai_video
        import tempfile

        camera_frames = {}
        for pai_cam, autovla_cam in CAMERA_MAP.items():
            try:
                # Load video for this camera
                video_data = self.ds.get_clip_video(clip_id, pai_cam)
                if video_data is None:
                    return None

                # Extract 4 frames at 2Hz
                frames = pai_video.extract_frames(
                    video_data,
                    timestamps=[0.0, 0.5, 1.0, 1.5],  # 4 frames at 2Hz
                    output_format="numpy",
                )
                if len(frames) < NUM_CONTEXT_FRAMES:
                    return None

                # Save frames to temp files
                frame_paths = []
                for j, frame in enumerate(frames[:NUM_CONTEXT_FRAMES]):
                    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    from PIL import Image
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
            ego_df = self.ds.get_clip_labels(clip_id, "egomotion")
            if ego_df is None or len(ego_df) == 0:
                return None

            from physical_ai_av.egomotion import EgomotionState
            from physical_ai_av.utils import interpolation

            # Get ego motion states at the frame timestamps
            # We need 10 future waypoints at 0.5s intervals starting from t=1.5s (end of context)
            timestamps = [1.5 + i * POSE_INTERVAL_S for i in range(NUM_POSES)]

            # Interpolate ego motion at these timestamps
            ego_states = []
            for ts in timestamps:
                state = self._interpolate_ego(ego_df, ts)
                if state is not None:
                    ego_states.append(state)

            if len(ego_states) < NUM_POSES:
                return None

            # Extract position (x, y) and heading
            positions = np.array([[s["x"], s["y"]] for s in ego_states])
            headings = np.array([s["heading"] for s in ego_states])
            velocity = ego_states[0]["speed"]
            acceleration = ego_states[0]["accel"]

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
                "gt_xy": rel_pos,  # (10, 2) in ego frame
                "gt_heading": rel_heading,  # (10,) in ego frame
                "velocity": velocity,
                "acceleration": acceleration,
            }
        except Exception as e:
            print(f"  Ego motion failed for clip {clip_id}: {e}")
            return None

    def _interpolate_ego(self, ego_df, ts: float) -> Optional[dict]:
        """Interpolate ego motion at timestamp ts (seconds from clip start)."""
        from physical_ai_av.egomotion import EgomotionState
        from physical_ai_av.utils import interpolation

        # Convert ts to the ego_df's time column (usually microseconds)
        ts_us = ts * 1e6

        # Find bracketing rows
        times = ego_df["timestamp"].values
        if ts_us < times[0] or ts_us > times[-1]:
            return None

        idx = np.searchsorted(times, ts_us)
        if idx == 0:
            row = ego_df.iloc[0]
        elif idx >= len(times):
            row = ego_df.iloc[-1]
        else:
            # Linear interpolation between idx-1 and idx
            t0, t1 = times[idx-1], times[idx]
            alpha = (ts_us - t0) / (t1 - t0) if t1 > t0 else 0.0
            row0 = ego_df.iloc[idx-1]
            row1 = ego_df.iloc[idx]

            x = row0["x"] + alpha * (row1["x"] - row0["x"])
            y = row0["y"] + alpha * (row1["y"] - row0["y"])
            heading = np.arctan2(
                row0["y"] + alpha * (row1["y"] - row0["y"]) - row0["y"],
                row0["x"] + alpha * (row1["x"] - row0["x"]) - row0["x"],
            ) if idx > 0 else 0.0
            vx = row0["vx"] + alpha * (row1["vx"] - row0["vx"])
            vy = row0["vy"] + alpha * (row1["vy"] - row0["vy"])
            ax = row0["ax"] + alpha * (row1["ax"] - row0["ax"])
            ay = row0["ay"] + alpha * (row1["ay"] - row0["ay"])

            speed = np.sqrt(vx**2 + vy**2)
            accel = np.sqrt(ax**2 + ay**2)

            return {"x": x, "y": y, "heading": heading, "speed": speed, "accel": accel}

        # Single row (exact match)
        speed = np.sqrt(row["vx"]**2 + row["vy"]**2)
        accel = np.sqrt(row["ax"]**2 + row["ay"]**2)
        heading = np.arctan2(row["vy"], row["vx"]) if abs(row["vx"]) + abs(row["vy"]) > 0.01 else 0.0

        return {"x": row["x"], "y": row["y"], "heading": heading, "speed": speed, "accel": accel}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clip_id = self.clip_ids[idx]

        # 1. Extract camera frames
        camera_frames = self._extract_camera_frames(clip_id)
        if camera_frames is None:
            # Return a dummy sample (will be filtered by collator)
            return self.__getitem__((idx + 1) % len(self))

        # 2. Extract ego motion and GT trajectory
        ego_data = self._extract_ego_motion(clip_id)
        if ego_data is None:
            return self.__getitem__((idx + 1) % len(self))

        # 3. Convert GT trajectory → action token indices
        gt_action_idx = self.matcher.match(ego_data["gt_xy"], ego_data["gt_heading"])

        # 4. Convert indices to action token text
        action_text = ""
        for i in range(len(gt_action_idx)):
            action_text += f"<action_{gt_action_idx[i]}>"

        # 5. Build prompt (same format as AutoVLA SFT)
        front = camera_frames["front_camera"]
        left = camera_frames["front_left_camera"]
        right = camera_frames["front_right_camera"]

        velocity = ego_data["velocity"]
        acceleration = ego_data["acceleration"]

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
            {"type": "text", "text": f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current acceleration is {acceleration:.3f} m/s². The driving instruction is: straight. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."},
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

        # 6. Process with Qwen processor
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
    """Collate PAI SFT samples into batched tensors."""

    def __init__(self, processor: AutoProcessor, ignore_index: int = -100):
        self.processor = processor
        self.ignore_index = ignore_index
        self.assistant_id = [151644, 77091]  # Qwen2.5-VL assistant token

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
