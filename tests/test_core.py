import ast
import unittest
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from pai_sft_dataset import (
    PAIDataCollator,
    PAICodebookMatcher,
    PAISFTDataset,
    _polygon_contour,
    split_clip_ids,
)


def load_checkpoint_key_functions():
    source = (
        Path(__file__).parents[1].joinpath("run_sft.py").read_text(encoding="utf-8")
    )
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"normalize_checkpoint_key", "checkpoint_key_candidates"}
    ]
    namespace = {"Tuple": Tuple}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "run_sft.py", "exec"),
        namespace,
    )
    return namespace["normalize_checkpoint_key"], namespace["checkpoint_key_candidates"]


normalize_checkpoint_key, checkpoint_key_candidates = load_checkpoint_key_functions()


class SplitTests(unittest.TestCase):
    def test_train_and_validation_are_disjoint_and_deterministic(self):
        clip_ids = [f"clip-{index:04d}" for index in range(1000)]
        train = split_clip_ids(clip_ids, "train", validation_fraction=0.1, seed=42)
        val = split_clip_ids(clip_ids, "val", validation_fraction=0.1, seed=42)

        self.assertFalse(set(train).intersection(val))
        self.assertEqual(set(train).union(val), set(clip_ids))
        self.assertEqual(
            train,
            split_clip_ids(clip_ids, "train", validation_fraction=0.1, seed=42),
        )


class CodebookMatcherTests(unittest.TestCase):
    def test_matches_repeated_forward_token(self):
        width, length = 2.0, 4.8
        forward = _polygon_contour(
            torch.tensor([1.0, 0.0]), torch.tensor(0.0), width, length
        )
        left = _polygon_contour(
            torch.tensor([0.0, 1.0]), torch.tensor(np.pi / 2), width, length
        )
        code_book = torch.zeros((2, 6, 4, 2), dtype=torch.float32)
        code_book[0, -1] = forward
        code_book[1, -1] = left
        matcher = PAICodebookMatcher(code_book=code_book)

        indices = matcher.match(
            np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            np.asarray([0.0, 0.0], dtype=np.float32),
        )

        np.testing.assert_array_equal(indices, np.asarray([0, 0]))


class EgomotionTimingTests(unittest.TestCase):
    class FakeRotation:
        def __init__(self, yaw):
            self.yaw = yaw

        def as_euler(self, _sequence):
            return np.asarray([0.0, 0.0, self.yaw])

    class FakePose:
        def __init__(self, x, yaw):
            self.translation = np.asarray([x, 0.0, 0.0])
            self.rotation = EgomotionTimingTests.FakeRotation(yaw)

    class FakeState:
        def __init__(self, seconds):
            self.pose = EgomotionTimingTests.FakePose(2.0 * seconds, 0.0)
            self.velocity = np.asarray([2.0, 0.0, 0.0])
            self.acceleration = np.asarray([0.0, 0.0, 0.0])

    class FakeInterpolator:
        time_range = (-1_000_000, 20_000_000)

        def __init__(self):
            self.calls = []

        def __call__(self, timestamp):
            self.calls.append(timestamp)
            return EgomotionTimingTests.FakeState(timestamp / 1_000_000)

    class FakeInterface:
        def __init__(self, interpolator):
            self.interpolator = interpolator

        def get_clip_feature(self, _clip_id, feature):
            if feature != "egomotion":
                raise AssertionError(feature)
            return self.interpolator

    def test_future_is_in_microseconds_and_relative_to_anchor(self):
        interpolator = self.FakeInterpolator()
        dataset = PAISFTDataset.__new__(PAISFTDataset)
        dataset.ds = self.FakeInterface(interpolator)
        dataset.anchor_time_s = 2.0
        dataset.num_poses = 10
        dataset.pose_interval_s = 0.5

        ego = dataset._extract_ego_motion("clip")

        self.assertEqual(interpolator.calls[0], 2_000_000)
        self.assertEqual(interpolator.calls[-1], 7_000_000)
        self.assertAlmostEqual(float(ego["gt_xy"][0, 0]), 1.0)
        self.assertAlmostEqual(float(ego["gt_xy"][-1, 0]), 10.0)
        self.assertAlmostEqual(ego["velocity"], 2.0)


class CheckpointKeyTests(unittest.TestCase):
    def test_normalizes_published_autovla_prefixes(self):
        self.assertEqual(
            normalize_checkpoint_key("autovla.vlm.model.embed_tokens.weight"),
            "vlm.model.embed_tokens.weight",
        )
        self.assertEqual(
            normalize_checkpoint_key(
                "_forward_module.module.drivevla.vlm.visual.patch_embed.proj.weight"
            ),
            "vlm.visual.patch_embed.proj.weight",
        )
        self.assertEqual(
            normalize_checkpoint_key("model.embed_tokens.weight"),
            "vlm.model.embed_tokens.weight",
        )
        self.assertIn(
            "vlm.model.visual.patch_embed.proj.weight",
            checkpoint_key_candidates("model.visual.patch_embed.proj.weight"),
        )


class CollatorTests(unittest.TestCase):
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text != "<|im_start|>assistant\n" or add_special_tokens:
                raise AssertionError((text, add_special_tokens))
            return [10, 11, 12]

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = CollatorTests.FakeTokenizer()

        def __call__(self, **kwargs):
            if len(kwargs["videos"]) != 1:
                raise AssertionError("Expected one video")
            return {
                "input_ids": torch.tensor([[1, 10, 11, 12, 100, 101, 2, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]]),
            }

    def test_masks_prompt_and_padding_but_keeps_assistant_action_tokens(self):
        collator = PAIDataCollator(
            processor=self.FakeProcessor(),
            ignore_index=-100,
            action_start_id=100,
            n_action_tokens=2,
        )
        batch = collator(
            [
                {
                    "text": "unused",
                    "video_inputs": [object()],
                    "image_inputs": None,
                    "gt_trajectory": torch.zeros((2, 2)),
                    "gt_action": torch.tensor([0, 1]),
                    "has_cot": False,
                }
            ]
        )

        self.assertEqual(
            batch["labels"].tolist(),
            [[-100, -100, -100, -100, 100, 101, 2, -100]],
        )


if __name__ == "__main__":
    unittest.main()
