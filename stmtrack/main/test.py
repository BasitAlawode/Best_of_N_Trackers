# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(10)])

import argparse
import os.path as osp

from loguru import logger

import torch
# torch.set_num_threads(4)

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.utils import complete_path_wt_root_in_cfg


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='',
                        type=str,
                        help='experiment configuration')

    return parser


def build_stmtrack_tester(task_cfg):
    # build model
    model = model_builder.build("track", task_cfg.model)
    # build pipeline
    pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)
    # build tester
    testers = tester_builder("track", task_cfg.tester, "tester", pipeline)
    return testers


if __name__ == '__main__':
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    # resolve config
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()

    torch.multiprocessing.set_start_method('spawn', force=True)

    if task == 'track':
        testers = build_stmtrack_tester(task_cfg)
    else:
        raise ValueError('Undefined task: {}'.format(task))
    for tester in testers:
        tester.test()
