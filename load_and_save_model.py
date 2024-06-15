import os
import sys
import argparse

import torch
from diffusers import StableDiffusionPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", help="The bin file path to load the learned embeddings from, typically has the pattern checkpoint-N/pytorch_model.bin")
    parser.add_argument("--save_path", help="The path to save the pipeline.")
    parser.add_argument("--num_tokens", type=int, help="The number of tokens optimized during training")
    config = parser.parse_args()

    model_save_path = config.save_path
    os.makedirs(model_save_path, exist_ok=True)

    ckpt_path = config.ckpt_path

    ckpt = torch.load(ckpt_path)

    tokens = [f"<t{idx}>" for idx in range(1, config.num_tokens + 1)]

    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

    num_added_tokens = pipe.tokenizer.add_tokens(tokens)

    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

    pipe.text_encoder.load_state_dict(ckpt)

    pipe.save_pretrained(model_save_path)



