# -*- coding: utf-8 -*-
'''
@Time    : 2021/6/11 15:58
@Author  : Wang Qiang
@FileName: train_utils.py
'''
import os
import numpy as np
import torch
import torch.nn.modules.loss
import random
import argparse

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_single_graph(data, graph_index):
    data_single = {}
    for key in data:
        if 'adj' in key or 'features' in key:
            data_single[key] = data[key][graph_index]
        else:
            data_single[key] = data[key]
    return data_single

def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    return " ".join(
        ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])


def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
            [
                d
                for d in os.listdir(models_dir)
                if os.path.isdir(os.path.join(models_dir, d))
            ]
        ).astype(np.int)
        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max() + 1)
        else:
            dir_id = "1"
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir


def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        if isinstance(config_dict[param], tuple):
            default, description = config_dict[param]
        else:
            default = config_dict[param]
            description = 'None'
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                        f"--{param}",
                        action="append",
                        type=type(default[0]),
                        default=default,
                        help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser

def save_embeddings(args, embeddings, indicators=None):
    import datetime, pickle
    dt = datetime.datetime.now()
    date = f"{dt.year}_{dt.month}_{dt.day}"
    models_dir = os.path.join(args.save_dir, args.model, r"{}".format(args.dataset), date, r"ProcessID{}".format(args.processID))
    save_dir = get_dir_name(models_dir)
    save_dict = {"args": args, "embeddings": embeddings, "indicators": indicators}
    with open(os.path.join(save_dir, 'embeddings.pkl'), "wb") as f:
        pickle.dump(save_dict, f)

def save_indicators(args, data):
    import datetime, pickle
    dt = datetime.datetime.now()
    date = f"{dt.year}_{dt.month}_{dt.day}"
    models_dir = os.path.join(args.result_dir, r"{}".format(args.dataset), args.model, r"dim{}".format(args.dim), date)
    save_dir = get_dir_name(models_dir)
    save_dict = {"args": args, "data": data}
    save_file = os.path.join(save_dir, 'data.pkl')
    with open(save_file, "wb") as f:
        pickle.dump(save_dict, f)
    return save_file