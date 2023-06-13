from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import io
import json
import logging
import math
import os
import random
import re
from collections import defaultdict

import decord
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


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


class MultiVENTDataset(Dataset):
    """MultiVENT dataset loader."""

    def __init__(
        self,
        metadata_path,
        category_path,
        event_path,
        language_path,
        video_path,
        tokenizer,
        max_frames=100,
        image_resolution=224,
    ):
        self.video_path = video_path
        self.metadata_path = metadata_path
        self.data = pd.read_csv(metadata_path)
        self.max_frames = max_frames
        self.image_resolution = image_resolution
        self.tokenizer = tokenizer

        self.category_dict = {}
        with open(category_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # toss headers
            for id, category in reader:
                self.category_dict[category] = int(id)
        LOG.info("category %s", self.category_dict)

        self.event_dict = {}
        with open(event_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # toss headers
            for id, event in reader:
                self.event_dict[event] = int(id)
        LOG.info("event %s", self.event_dict)

        self.language_dict = {}
        with open(language_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # toss headers
            for id, language in reader:
                self.language_dict[language] = int(id)
        LOG.info("language %s", self.language_dict)

        self.transform = Compose(
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

        LOG.info("MultiVENT dataset")
        LOG.info("metadata_path %s", metadata_path)
        LOG.info("video_path %s", self.video_path)
        LOG.info("videos %s", len(self.data))

    def __len__(self):
        return len(self.data)

    def uniform_sample_frames(self, num_frames, vlen):
        """Get uniform sample of frames."""
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
        return frame_idxs

    def __getitem__(self, idx):
        video_rec = self.data.iloc[idx]

        video_key = video_rec["video_path"]

        lang = self.language_dict[video_rec["language"]]
        category = self.category_dict[video_rec["category"]]
        event = self.event_dict[video_rec["event"]]

        curr_video = os.path.join(self.video_path, remove_prefix(video_key, "/"))

        video_name = os.path.splitext(os.path.basename(curr_video))[0]

        video_reader = decord.VideoReader(curr_video, num_threads=1)
        vlen = len(video_reader)
        fps_vid = math.ceil(video_reader.get_avg_fps())

        video = np.zeros(
            (
                1,
                self.max_frames,
                1,
                3,
                self.image_resolution,
                self.image_resolution,
            ),
            dtype=np.float32,
        )

        frame_idxs = self.uniform_sample_frames(self.max_frames, vlen)
        if len(frame_idxs) < self.max_frames:
            LOG.info("...less then max frames %s %s", video_key, len(frame_idxs))

        patch_images = [Image.fromarray(f) for f in video_reader.get_batch(frame_idxs).asnumpy()]
        patch_images = torch.stack([self.transform(img) for img in patch_images])
        patch_images = patch_images.unsqueeze(1)
        slice_len = patch_images.shape[0]
        video[0][:slice_len, ...] = patch_images

        desc = video_rec["description"]
        text = self.tokenizer(desc)
        text = text.squeeze()

        return text, video, lang, category, event, video_name
