""" Score retrieval. """
import logging
import os
import sys
import timeit

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.special import softmax

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
LOG = logging.getLogger(__name__)


def compute_metrics(x):
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
    metrics["Count"] = len(x)
    return metrics


def get_stats(data, title="", indicies=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if indicies is not None:
        batch_sequence_output_list = [
            torch.from_numpy(data["metadata"][indicies]).float().to(device)
        ]
        batch_visual_output_list = [torch.from_numpy(data["video"][indicies]).float().to(device)]
    else:
        batch_sequence_output_list = [torch.from_numpy(data["metadata"]).float().to(device)]
        batch_visual_output_list = [torch.from_numpy(data["video"]).float().to(device)]

    sim_matrix = []
    for idx1, b1 in enumerate(batch_sequence_output_list):
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_visual_output_list):
            visual_output = batch_visual_output_list[idx2]

            # normalized features
            visual_output = visual_output / visual_output.norm(p=2, dim=-1, keepdim=True)
            sequence_output = sequence_output / sequence_output.norm(p=2, dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = torch.tensor([100.0]).to(device)  # model.logit_scale.exp()
            b1b2_logits = torch.matmul(sequence_output, visual_output.t()) * logit_scale

            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)

    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    sim_matrix_dsl = sim_matrix * softmax(sim_matrix, axis=0)
    np.save('/exp/ksanders/video/sim_matrix.npy', sim_matrix_dsl)

    dsl_tv_metrics = compute_metrics(sim_matrix_dsl)

    LOG.info("------------------------------------------------------------")
    LOG.info("Text-to-Video:")
    LOG.info(
        "\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f}  - Median R: {:.1f} - Mean R: {:.1f}".format(
            dsl_tv_metrics["R1"],
            dsl_tv_metrics["R5"],
            dsl_tv_metrics["R10"],
            dsl_tv_metrics["MR"],
            dsl_tv_metrics["MeanR"],
        )
    )


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    """Execute the command with the given arguments."""

    start_time = timeit.default_timer()

    features_file = os.path.join(cfg.output, cfg.feature_file_name + ".npz")
    LOG.info("loading features from %s", features_file)
    data = np.load(features_file)

    LOG.info("-----------------------------------------------------")
    get_stats(data, title="All", indicies=None)

    total_time = timeit.default_timer() - start_time
    LOG.info("Ran the script in %.3f seconds", total_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
