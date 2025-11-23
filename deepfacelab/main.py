from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
from pathlib import Path

import tensorflow

from deepfacelab.core import osex
from deepfacelab.core.leras.nn import nn
from deepfacelab.mainscripts.trainer import main as trainer
from deepfacelab.models import saehd

from deepfacelab import config

if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 12):
    raise Exception("This program requires at least Python 3.6")

tensorflow.compat.v1.disable_v2_behavior()
logger = logging.getLogger(__name__)

MODELS = {"SAEHD": saehd.SAEHDModel}


def main():
    logging.basicConfig(level=logging.DEBUG)

    # Fix for linux
    multiprocessing.set_start_method("spawn")

    nn.initialize_main_env()

    class fixPathAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

    exit_code = 0

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cpu-num",
        type=int,
        default=min(multiprocessing.cpu_count(), 8)
    )
    subparsers = parser.add_subparsers()

    def process_train(arguments):
        osex.set_process_lowest_prio()

        trainer(
            model_class=MODELS[arguments.model_name],
            saved_models_path=Path(arguments.model_dir),
            training_data_src_path=Path(arguments.training_data_src_dir),
            training_data_dst_path=Path(arguments.training_data_dst_dir),
            pretraining_data_path=(
                Path(arguments.pretraining_data_dir)
                if arguments.pretraining_data_dir is not None
                else None
            ),
            pretrained_model_path=(
                Path(arguments.pretrained_model_dir)
                if arguments.pretrained_model_dir is not None
                else None
            ),
            no_preview=arguments.no_preview,
            force_model_name=arguments.force_model_name,
            force_gpu_idxs=(
                [int(x) for x in arguments.force_gpu_idxs.split(",")]
                if arguments.force_gpu_idxs is not None
                else None
            ),
            cpu_only=arguments.cpu_only,
            silent_start=arguments.silent_start,
            execute_programs=[[int(x[0]), x[1]] for x in arguments.execute_program],
            debug=arguments.debug,
        )

    p = subparsers.add_parser("train", help="Trainer")

    p.add_argument(
        "--training-data-src-dir",
        required=True,
        action=fixPathAction,
        dest="training_data_src_dir",
        help="Dir of extracted SRC faceset.",
    )
    p.add_argument(
        "--training-data-dst-dir",
        required=True,
        action=fixPathAction,
        dest="training_data_dst_dir",
        help="Dir of extracted DST faceset.",
    )
    p.add_argument(
        "--pretraining-data-dir",
        action=fixPathAction,
        dest="pretraining_data_dir",
        default=None,
        help="Optional dir of extracted faceset that will be used in pretraining mode.",
    )
    p.add_argument(
        "--pretrained-model-dir",
        action=fixPathAction,
        dest="pretrained_model_dir",
        default=None,
        help="Optional dir of pretrain model files. (Currently only for Quick96).",
    )
    p.add_argument(
        "--model-dir",
        required=True,
        action=fixPathAction,
        dest="model_dir",
        help="Saved models dir.",
    )
    p.add_argument(
        "--model",
        required=True,
        dest="model_name",
        choices=list(MODELS.keys()),
        help="Model class name.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        dest="debug",
        default=False,
        help="Debug samples.",
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        dest="no_preview",
        default=False,
        help="Disable preview window.",
    )
    p.add_argument(
        "--force-model-name",
        dest="force_model_name",
        default=None,
        help="Forcing to choose model name from model/ folder.",
    )
    p.add_argument(
        "--cpu-only",
        action="store_true",
        dest="cpu_only",
        default=False,
        help="Train on CPU.",
    )
    p.add_argument(
        "--force-gpu-idxs",
        dest="force_gpu_idxs",
        default=None,
        help="Force to choose GPU indexes separated by comma.",
    )
    p.add_argument(
        "--silent-start",
        action="store_true",
        dest="silent_start",
        default=False,
        help="Silent start. Automatically chooses Best GPU and last used model.",
    )

    p.add_argument(
        "--execute-program",
        dest="execute_program",
        default=[],
        action="append",
        nargs="+",
    )
    p.set_defaults(func=process_train)

    arguments = parser.parse_args()
    config.CONFIG = config.Config(
        cpu_number=arguments.cpu_num
    )
    arguments.func(arguments)

    if exit_code == 0:
        logger.info("Done.")

    exit(exit_code)


if __name__ == "__main__":
    main()

"""
import code
code.interact(local=dict(globals(), **locals()))
"""
