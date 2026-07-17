"""PAI dataset → AutoVLA SFT format converter.

Reads PAI clips from a local directory (no HF authentication needed).
"""

import io
import json
import os
import pickle
import sys
import tempfile
import types
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_AUTOVLA_REPO = os.environ.get("AUTOVLA_REPO_PATH", "/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA")
sys.path.insert(0, _AUTOVLA_REPO)

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

CAMERA_MAP = {
    "camera_front_wide_120fov": "front_camera",
    "camera_cross_left_120fov": "front_left_camera",
    "camera_cross_right_120fov": "front_right_camera",
}
NUM_CONTEXT_FRAMES = 4
FRAME_INTERVAL_S = 0.5
NUM_POSES = 10
POSE_INTERVAL_S = 0.5


class SimplePAIInterface:
    """Reads PAI data directly from local directory, no HF cache/API needed."""

    def __init__(self, local_dir: str):
        self.local_dir = Path(local_dir)
        self.clip_index = pd.read_parquet(self.local_dir / "clip_index.parquet")
        features_df = pd.read_csv(self.local_dir / "features.csv", index_col="feature")
        features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
            json.loads, na_action="ignore"
        )
        self.features = self._build_features(features_df)
        # Dummy feature_presence: all True
        self.feature_presence = pd.DataFrame(
            {col: True for col in features_df.index},
            index=self.clip_index.index,
        )
        self.chunk_feature_presence = (
            pd.concat([self.clip_index[["chunk"]], self.feature_presence], axis=1)
            .groupby("chunk").any()
        )
        print(f"SimplePAIInterface: {len(self.clip_index)} clips, {len(features_df)} features")

    def _build_features(self, features_df):
        ns = types.SimpleNamespace()
        ns.ALL = set()
        for directory, group in features_df.groupby("directory"):
            setattr(ns, directory.upper(), types.SimpleNamespace(
                **{f.upper().replace(".", "_"): f for f in group.index},
                ALL=set(group.index),
            ))
            ns.ALL.update(set(group.index))
        ns.get_chunk_feature_filename = lambda chunk_id, feature: features_df.at[feature, "chunk_path"].format(chunk_id=chunk_id)
        ns.get_clip_files_in_zip = lambda clip_id, feature: {
            k: v.format(clip_id=clip_id) for k, v in features_df.at[feature, "clip_files_in_zip"].items()
        }
        return ns

    def get_clip_chunk(self, clip_id: str) -> int:
        return self.clip_index.at[clip_id, "chunk"]

    def get_clip_feature(self, clip_id: str, feature: str, maybe_stream: bool = False) -> Any:
        """Load a clip feature from local zip/parquet files."""
        from physical_ai_av import calibration, egomotion, video

        chunk_filename = self.features.get_chunk_feature_filename(
            self.get_clip_chunk(clip_id), feature
        )
        local_path = self.local_dir / chunk_filename
        if not local_path.exists():
            print(f"  File not found: {local_path}")
            return None

        with open(local_path, "rb") as f:
            if chunk_filename.endswith(".parquet"):
                feature_df = pd.read_parquet(f).loc[clip_id]
                if feature == "sensor_extrinsics":
                    return calibration.SensorExtrinsics.from_extrinsics_df(feature_df)
                elif feature == "camera_intrinsics":
                    return calibration.CameraIntrinsics.from_intrinsics_df(feature_df)
                elif feature == "vehicle_dimensions":
                    return calibration.VehicleDimensions.from_dimensions_df(feature_df)
                else:
                    return feature_df
            elif chunk_filename.endswith(".zip"):
                clip_files = self.features.get_clip_files_in_zip(clip_id, feature)
                with zipfile.ZipFile(f, "r") as zf:
                    if feature == "egomotion":
                        ego_df = pd.read_parquet(io.BytesIO(zf.read(clip_files["egomotion"])))
                        return egomotion.EgomotionState.from_egomotion_df(
                            ego_df
                        ).create_interpolator(ego_df["timestamp"].to_numpy(copy=True))
                    elif feature.startswith("camera"):
                        return video.SeekVideoReader(
                            video_data=io.BytesIO(zf.read(clip_files["video"])),
                            timestamps=pd.read_parquet(
                                io.BytesIO(zf.read(clip_files["frame_timestamps"]))
                            )["timestamp"].to_numpy(copy=True),
                        )
                    else:
                        return {
                            k: pd.read_parquet(io.BytesIO(zf.read(v)))
                            if v.endswith(".parquet")
                            else io.BytesIO(zf.read(v))
                            for k, v in clip_files.items()
                        }
        return None


class PAICodebookMatcher:
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
            cb_flat = self.code_book.reshape(-1, 4, 2)
            cos_h, sin_h = torch.cos(prev_head), torch.sin(prev_head)
            gx = cos_h * cb_flat[..., 0] - sin_h * cb_flat[..., 1] + prev_pos[0, 0]
            gy = sin_h * cb_flat[..., 0] + cos_h * cb_flat[..., 1] + prev_pos[0, 1]
            cb_global = torch.stack([gx, gy], dim=-1)
            cb_mean = cb_global[:, -1, :].reshape(self.n_bins, 6, 2).mean(dim=1)
            dist = torch.norm(cb_mean - target_pos, dim=-1)
            best_idx = torch.argmin(dist).item()
            indices[i] = best_idx
            prev_pos = cb_mean[best_idx].unsqueeze(0)
            dxy = cb_flat[best_idx * 6][-1] - cb_flat[best_idx * 6][0]
            prev_head = torch.atan2(dxy[1], dxy[0]).unsqueeze(0)
        return indices


class PAISFTDataset(Dataset):
    def __init__(self, pai_data_dir, model_config, processor, using_cot=False, max_samples=None):
        self.pai_data_dir = Path(pai_data_dir)
        self.processor = processor
        self.using_cot = using_cot
        codebook_path = model_config.get("codebook_cache_path", "codebook_cache/agent_vocab.pkl")
        self.matcher = PAICodebookMatcher(codebook_path)
        self.ds = SimplePAIInterface(str(self.pai_data_dir))
        self.clip_ids = list(self.ds.clip_index.index.values)
        if max_samples is not None:
            self.clip_ids = self.clip_ids[:max_samples]
        print(f"PAI SFT dataset: {len(self.clip_ids)} clips")

    def __len__(self):
        return len(self.clip_ids)

    def _extract_camera_frames(self, clip_id):
        from PIL import Image
        camera_frames = {}
        for pai_cam, role in CAMERA_MAP.items():
            try:
                reader = self.ds.get_clip_feature(clip_id, pai_cam)
                if reader is None:
                    return None
                ts = reader.timestamps
                if len(ts) < NUM_CONTEXT_FRAMES:
                    return None
                frame_ts = np.array([ts[0] + i * FRAME_INTERVAL_S * 1e6 for i in range(NUM_CONTEXT_FRAMES)])
                frame_ts = np.minimum(frame_ts, ts[-1])
                images, _ = reader.decode_images_from_timestamps(frame_ts)
                if len(images) < NUM_CONTEXT_FRAMES:
                    return None
                paths = []
                for frame in images[:NUM_CONTEXT_FRAMES]:
                    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    Image.fromarray(frame).save(tmp.name, quality=90)
                    paths.append(tmp.name)
                camera_frames[role] = paths
            except Exception as e:
                print(f"  Camera {pai_cam} failed for {clip_id}: {e}")
                return None
        return camera_frames

    def _extract_ego_motion(self, clip_id):
        try:
            interp = self.ds.get_clip_feature(clip_id, "egomotion")
            if interp is None:
                return None
            timestamps = [(i+1) * POSE_INTERVAL_S for i in range(NUM_POSES)]
            states = []
            for ts in timestamps:
                try:
                    s = interp(ts)
                    states.append(s)
                except Exception:
                    return None
            if len(states) < NUM_POSES:
                return None
            positions = np.array([[s.pose.translation[0], s.pose.translation[1]] for s in states])
            headings = np.array([s.pose.rotation.as_euler('xyz')[2] for s in states])
            vel = np.sqrt(states[0].velocity[0]**2 + states[0].velocity[1]**2)
            acc = np.sqrt(states[0].acceleration[0]**2 + states[0].acceleration[1]**2)
            x0, y0, h0 = positions[0, 0], positions[0, 1], headings[0]
            cos_h, sin_h = np.cos(-h0), np.sin(-h0)
            rel_pos = np.zeros_like(positions)
            for i in range(len(positions)):
                dx, dy = positions[i, 0] - x0, positions[i, 1] - y0
                rel_pos[i, 0] = cos_h * dx - sin_h * dy
                rel_pos[i, 1] = sin_h * dx + cos_h * dy
            return {"gt_xy": rel_pos, "gt_heading": headings - h0, "velocity": float(vel), "acceleration": float(acc)}
        except Exception as e:
            print(f"  Ego motion failed for {clip_id}: {e}")
            return None

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        cam = self._extract_camera_frames(clip_id)
        if cam is None:
            return self.__getitem__((idx + 1) % len(self))
        ego = self._extract_ego_motion(clip_id)
        if ego is None:
            return self.__getitem__((idx + 1) % len(self))
        gt_idx = self.matcher.match(ego["gt_xy"], ego["gt_heading"])
        action_text = "".join([f"<action_{gt_idx[i]}>" for i in range(len(gt_idx))])
        front, left, right = cam["front_camera"], cam["front_left_camera"], cam["front_right_camera"]
        v, a = ego["velocity"], ego["acceleration"]

        system_content = [{"type": "text", "text": "You are an Advanced Driver Assistance and Full Self-Driving System. You will be provided with video observations from the ego vehicle's surrounding cameras, along with the vehicle's current dynamic states. Your task is to predict the most appropriate driving action for the next five seconds."}]
        user_content = [
            {"type": "text", "text": "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."},
            {"type": "text", "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
            {"type": "video", "min_pixels": 28*28*128, "max_pixels": 28*28*128, "video": [f"file://{f}" for f in front]},
            {"type": "text", "text": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
            {"type": "video", "min_pixels": 28*28*128, "max_pixels": 28*28*128, "video": [f"file://{f}" for f in left]},
            {"type": "text", "text": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
            {"type": "video", "min_pixels": 28*28*128, "max_pixels": 28*28*128, "video": [f"file://{f}" for f in right]},
            {"type": "text", "text": f"The current velocity of the vehicle is {v:.3f} m/s, and the current acceleration is {a:.3f} m/s². The driving instruction is: straight. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."},
        ]
        assistant_content = [{"type": "text", "text": f"<answer>\nThe final output action is: {action_text}\n</answer>"}]
        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]
        image_inputs, video_inputs = process_vision_info(messages)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        return {"text": text, "image_inputs": image_inputs, "video_inputs": video_inputs, "gt_trajectory": torch.tensor(ego["gt_xy"], dtype=torch.float32), "gt_action": torch.tensor(gt_idx, dtype=torch.int64), "has_cot": False, "clip_id": clip_id}


class PAIDataCollator:
    def __init__(self, processor, ignore_index=-100):
        self.processor = processor
        self.ignore_index = ignore_index
        self.assistant_id = [151644, 77091]

    def __call__(self, features):
        texts = [f["text"] for f in features]
        video_inputs, image_inputs, has_cot = [], [], []
        for f in features:
            video_inputs.extend(f.get("video_inputs", []))
            image_inputs.append(f.get("image_inputs"))
            has_cot.append(f.get("has_cot", False))
        batch = self.processor(text=texts, images=image_inputs if image_inputs[0] is not None else None, videos=video_inputs if video_inputs[0] is not None else None, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        aid = torch.tensor(self.assistant_id)
        for i in range(labels.shape[0]):
            for j in range(len(labels[i]) - len(aid) + 1):
                if torch.equal(labels[i][j:j + len(aid)], aid):
                    labels[i, :j] = self.ignore_index
                    break
        batch["labels"] = labels
        batch["gt_trajectory"] = torch.stack([f["gt_trajectory"] for f in features])
        batch["gt_action"] = torch.stack([f["gt_action"] for f in features])
        batch["has_cot"] = torch.tensor(has_cot)
        return batch
