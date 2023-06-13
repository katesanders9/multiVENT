import logging
import os
import sys
import timeit

import hydra
import numpy as np
import open_clip
import torch
from omegaconf import DictConfig, OmegaConf
from open_clip import CLIP, CLIPTextCfg, CLIPVisionCfg, CustomTextCLIP
from open_clip.tokenizer import HFTokenizer
from scipy.special import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

from transformers.models.clip.tokenization_clip_fast import CLIPTokenizerFast

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
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
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
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



@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    """Execute the command with the given arguments."""
    print(OmegaConf.to_yaml(cfg))

    start_time = timeit.default_timer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_vision_cfg = CLIPVisionCfg(**cfg.clip_model.vision_cfg)
    clip_text_cfg = CLIPTextCfg(**cfg.clip_model.text_cfg)

    model = CustomTextCLIP(
        embed_dim=cfg.clip_model.embed_dim,
        vision_cfg=clip_vision_cfg,
        text_cfg=clip_text_cfg,
        output_dict=True,
    )

    tokenizer = HFTokenizer(cfg.tokenizer_path)

    LOG.info("...loading encoder checkpoint: path %s", cfg.checkpoint_path)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=torch.device("cpu"))

    state_dict = checkpoint["state_dict"]
    for key in list(state_dict.keys()):
        if "module.text" in key:
            state_dict[key.replace("module.text.", "text.")] = state_dict.pop(key)
        if "module.visual" in key:
            state_dict[key.replace("module.visual.", "visual.")] = state_dict.pop(key)
        if "module.logit_scale" in key:
            state_dict[key.replace("module.logit_scale", "logit_scale")] = state_dict.pop(key)

    model.load_state_dict(checkpoint["state_dict"], strict=True)

    model.eval()
    model.to(device)

    eval_dataset = MSRVTT_DataLoader(
        csv_path=cfg.test_csv,
        features_path=cfg.features_path,
        max_words=cfg.max_words,
        feature_framerate=cfg.feature_framerate,
        tokenizer=tokenizer,
        max_frames=cfg.max_frames,
        frame_order=cfg.eval_frame_order,
        slice_framepos=cfg.slice_framepos,
    )

    LOG.info("eval %s", len(eval_dataset))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.per_device_train_batch_size,
        num_workers=cfg.dataloader_num_workers,
        shuffle=False,
        drop_last=False,
        persistent_workers=True,
    )

    batch_sequence_output_list, batch_visual_output_list = [], []

    with torch.no_grad():
        pbar = tqdm(iter(eval_loader), leave=False, total=len(eval_loader))
        for inputs in pbar:
            input_ids, video = inputs

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

            input_ids = input_ids.to(device)
            video = video.to(device)

            outputs = model(video, input_ids)

            image_embeds = outputs["image_features"]
            _, dim = image_embeds.shape
            image_embeds = image_embeds.view(b, bs, dim)
            image_embeds_mean = image_embeds.mean(dim=1)

            visual_output = image_embeds_mean.cpu().detach().numpy()
            sequence_output = outputs["text_features"].cpu().detach().numpy()

            batch_sequence_output_list.append(sequence_output)
            batch_visual_output_list.append(visual_output)

        vis_list = [batch for batch in batch_visual_output_list]
        seq_list = [batch for batch in batch_sequence_output_list]

        features_file = os.path.join(cfg.output, cfg.feature_file_name + ".npz")
        LOG.info("loading features from %s", features_file)
        np.savez(
            features_file,
            metadata=np.vstack(seq_list),
            video=np.vstack(vis_list),
        )

    total_time = timeit.default_timer() - start_time
    LOG.info("Ran the script in %.3f seconds", total_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
