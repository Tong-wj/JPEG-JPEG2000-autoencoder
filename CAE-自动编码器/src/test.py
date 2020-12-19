import os
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import logger
from data_loader import ImageFolder96p
from utils import dump_cfg, get_args, get_config, save_imgs

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models"))

# from cae_32x32x32_noise import CAE
# from cae_16x8x8_zero_pad_bin import CAE
# from cae_32x32x32_zero_pad_bin import CAE
# from cae_48x32x32_zero_pad_bin import CAE
# from cae_32x32x32_zero_pad import CAE
# from cae_96x16x16_mirror_pad_round import CAE
from cae_96x16x16_test import CAE
# from models import CAE

def main():
    # classes
    classes = ["plane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    # models

    args = get_args()
    config = get_config(args)
    for label in classes:
        config.dataset_path="E:/PycharmProjects/data/cae_96_96_num_1000/" + label
        test(config,label)


def prologue(cfg: Namespace, *varargs) -> None:
    # sanity checks
    assert cfg.chkpt not in [None, ""]
    assert cfg.device == "cpu" or (cfg.device == "cuda" and torch.cuda.is_available())

    # dirs
    base_dir = f"../experiments/{cfg.exp_name}"

    os.makedirs(f"{base_dir}/out", exist_ok=True)

    dump_cfg(f"{base_dir}/test_config.txt", vars(cfg))


def epilogue(cfg: Namespace, *varargs) -> None:
    pass


def test(cfg: Namespace, label) -> None:
    logger.info("=== Testing ===")

    # initial setup
    prologue(cfg)

    model = CAE()
    model.load_state_dict(torch.load(cfg.chkpt))
    model.eval()
    if cfg.device == "cuda":
        model.cuda()

    logger.info("Loaded model")

    dataset = ImageFolder96p(cfg.dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=cfg.shuffle)

    logger.info("Loaded data")

    loss_criterion = nn.MSELoss()

    for batch_idx, data in enumerate(dataloader, start=1):
        img, patches, _ = data
        if cfg.device == 'cuda':
            patches = patches.cuda()

        if batch_idx % cfg.batch_every == 0:
            pass

        out = torch.zeros(1, 1, 3, 128, 128)
        # out = torch.zeros(6, 10, 3, 128, 128)
        # enc = torch.zeros(6, 10, 16, 8, 8)
        avg_loss = 0

        for i in range(1):
            for j in range(1):
                x = Variable(patches[:, :, i, j, :, :]).cuda()
                y = model(x)
                out[i, j] = y.data

                loss = loss_criterion(y, x)
                avg_loss += (1 / 1) * loss.item()

        logger.debug('[%5d/%5d] avg_loss: %f' % (batch_idx, len(dataloader), avg_loss))

        # save output
        out = np.transpose(out, (0, 3, 1, 4, 2))
        out = np.reshape(out, (128, 128, 3))
        out = np.transpose(out, (2, 0, 1))

        save_imgs(imgs=out.unsqueeze(0), to_size=(3, 128, 128), name=f"../experiments/{cfg.exp_name}/test_out/STL10_num_1000/{label}/{batch_idx}.png")

        # y = torch.cat((img[0], out), dim=2).unsqueeze(0)
        # save_imgs(imgs=y, to_size=(3, 128, 2 * 128),
        #           name=f"../experiments/{cfg.exp_name}/test_out/STL10_num_1000/{label}/{batch_idx}.png")

    # final setup
    epilogue(cfg)


if __name__ == '__main__':
    main()
