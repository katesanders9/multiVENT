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

from video_retrieval.data.dataloader_msrvtt_retrieval_laion import MSRVTT_DataLoader
from video_retrieval.data.multivent_retrieval_csv import MultiVENTDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
LOG = logging.getLogger(__name__)


def compute_metrics(x):
    """Computer R @ evaluation measure."""
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics["R1"] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics["R5"] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics["R10"] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics["MR"] = np.median(ind) + 1
    metrics["MedianR"] = metrics["MR"]
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics


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

    eval_dataset = MultiVENTDataset(
        metadata_path=cfg.multivent_metadata_path,
        category_path=cfg.multivent_category_path,
        event_path=cfg.multivent_event_path,
        language_path=cfg.multivent_language_path,
        video_path=cfg.video_retrieval_path,
        tokenizer=tokenizer,
        max_frames=cfg.max_frames,
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
            input_ids, video, lang, category, event, video_name = inputs

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
