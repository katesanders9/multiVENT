from __future__ import absolute_import, division, print_function, unicode_literals

import io
import json
import logging
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

LOG = logging.getLogger(__name__)


class MSRVTT_DataLoader(Dataset):
    """MSRVTT dataset loader."""

    def __init__(
        self,
        csv_path,
        features_path,
        tokenizer,
        max_words=30,
        feature_framerate=1.0,
        max_frames=100,
        image_resolution=224,
        frame_order=0,
        slice_framepos=0,
        transforms=None,
    ):
        print("MSRVTT_DataLoader")
        print("max_words", max_words)
        print("feature_framerate", feature_framerate)
        print("max_frames", max_frames)
        print("frame_order", slice_framepos)
        print("slice_framepos", slice_framepos)

        self.data = pd.read_csv(csv_path)
        print("test size", len(self.data))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.image_resolution = image_resolution
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Compose(
                [
                    Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(image_resolution),
                    lambda image: image.convert("RGB"),
                    ToTensor(),
                    Normalize(
                        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                    ),
                ]
            )

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer(sentence)

            words = words.squeeze()

        return words

    def _get_rawvideo_dec(self, choice_video_ids, s=None, e=None):
        # speed up video decode via decord.
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)

        max_video_length = [0] * len(choice_video_ids)

        # T x 3 x H x W
        video = np.zeros(
            (
                len(choice_video_ids),
                self.max_frames,
                1,
                3,
                self.image_resolution,
                self.image_resolution,
            ),
            dtype=np.float32,
        )

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0.0 else 0.0
            end_time = end_time if end_time >= 0.0 else 0.0
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1

        for i, video_id in enumerate(choice_video_ids):
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))

            vreader = VideoReader(video_path, ctx=cpu(0))

            fps = vreader.get_avg_fps()

            f_start = 0 if start_time is None else int(start_time * fps)
            f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
            num_frames = f_end - f_start + 1
            if num_frames > 0:
                sample_fps = int(self.feature_framerate)
                t_stride = int(round(float(fps) / sample_fps))

                """Get uniform sample of frames."""
                vlen = len(vreader)
                acc_samples = min(self.max_frames, vlen)
                intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
                ranges = []
                for idx, interv in enumerate(intervals[:-1]):
                    ranges.append((interv, intervals[idx + 1] - 1))
                sample_pos = [(x[0] + x[1]) // 2 for x in ranges]

                patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
                patch_images = torch.stack([self.transforms(img) for img in patch_images])

                patch_images = patch_images.unsqueeze(1)

                slice_len = patch_images.shape[0]
                max_video_length[i] = (
                    max_video_length[i] if max_video_length[i] > slice_len else slice_len
                )
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = patch_images
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id = self.data["video_id"].values[idx]
        sentence = self.data["sentence"].values[idx]

        text = self._get_text(video_id, sentence)
        video, video_mask = self._get_rawvideo_dec([video_id])
        return text, video


class MSRVTT_TrainDataLoader(Dataset):
    """MSRVTT train dataset loader."""

    def __init__(
        self,
        csv_path,
        json_path,
        features_path,
        tokenizer,
        max_words=30,
        feature_framerate=1.0,
        max_frames=100,
        unfold_sentences=False,
        image_resolution=224,
        frame_order=0,
        slice_framepos=0,
        transforms=None,
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, "r"))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.image_resolution = image_resolution
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        if self.unfold_sentences:
            train_video_ids = list(self.csv["video_id"].values)
            print("...ids", len(train_video_ids))
            self.sentences_dict = {}
            for itm in self.data["sentences"]:
                if itm["video_id"] in train_video_ids:
                    if len(self.sentences_dict) == 1:
                        print("...", len(self.sentences_dict), itm["video_id"], itm["caption"])
                    self.sentences_dict[len(self.sentences_dict)] = (
                        itm["video_id"],
                        itm["caption"],
                    )

            self.sample_len = len(self.sentences_dict)
            print("-->total", self.sample_len)
        else:
            num_sentences = 0
            self.sentences = defaultdict(list)
            s_video_id_set = set()
            for itm in self.data["sentences"]:
                self.sentences[itm["video_id"]].append(itm["caption"])
                num_sentences += 1
                s_video_id_set.add(itm["video_id"])

            # Use to find the clips in the same video
            self.parent_ids = {}
            self.children_video_ids = defaultdict(list)
            for itm in self.data["videos"]:
                vid = itm["video_id"]
                url_posfix = itm["url"].split("?v=")[-1]
                self.parent_ids[vid] = url_posfix
                self.children_video_ids[url_posfix].append(vid)
            self.sample_len = len(self.csv)

        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Compose(
                [
                    Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(image_resolution),
                    lambda image: image.convert("RGB"),
                    ToTensor(),
                    Normalize(
                        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                    ),
                ]
            )

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer(caption)
            else:
                words = self._get_single_text(video_id)

            words = words.squeeze()

        return words

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer(caption)
        return words

    def _get_rawvideo_dec(self, choice_video_ids, s=None, e=None):
        # speed up video decode via decord.

        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)

        max_video_length = [0] * len(choice_video_ids)

        # T x 3 x H x W
        video = np.zeros(
            (
                len(choice_video_ids),
                self.max_frames,
                1,
                3,
                self.image_resolution,
                self.image_resolution,
            ),
            dtype=np.float32,
        )

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0.0 else 0.0
            end_time = end_time if end_time >= 0.0 else 0.0
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1

        for i, video_id in enumerate(choice_video_ids):
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))

            vreader = VideoReader(video_path, ctx=cpu(0))

            fps = vreader.get_avg_fps()
            f_start = 0 if start_time is None else int(start_time * fps)
            f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
            num_frames = f_end - f_start + 1
            if num_frames > 0:
                # T x 3 x H x W

                sample_fps = int(self.feature_framerate)
                t_stride = int(round(float(fps) / sample_fps))

                """Get uniform sample of frames."""
                vlen = len(vreader)
                acc_samples = min(self.max_frames, vlen)
                intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
                ranges = []
                for idx, interv in enumerate(intervals[:-1]):
                    ranges.append((interv, intervals[idx + 1] - 1))
                sample_pos = [(x[0] + x[1]) // 2 for x in ranges]

                patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
                patch_images = torch.stack([self.transforms(img) for img in patch_images])

                patch_images = patch_images.unsqueeze(1)

                slice_len = patch_images.shape[0]

                max_video_length[i] = (
                    max_video_length[i] if max_video_length[i] > slice_len else slice_len
                )
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = patch_images
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv["video_id"].values[idx], None

        text = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo_dec([video_id])
        return text, video
