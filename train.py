import os
import sys
import argparse
from configs.import_utils import parse_config
from core.trainer import Trainer
from torch.backends import cudnn

cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default=os.path.join("configs", "base_config.py"), help="Path to config file")
    config = parser.parse_args()
    return config

def main():
    train_args = parse_args()
    opts = parse_config(train_args.model_config_path)
    trainer = Trainer(opts)
    trainer.train()

if __name__ == "__main__":
    main()