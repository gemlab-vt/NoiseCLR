import os
import sys
import itertools
import random
import math
from tqdm import tqdm
import json

import torch
import torch.nn.functional
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import numpy as np

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from packaging import version

from core.dataset import DiscoveryDataset
from core.loss import get_negative_sum, get_positive_sum

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

logger = get_logger(__name__)

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def save_weights(weights, args):
    logger.info("Saving embeddings")
    learned_weights_dict = {"weights": weights.detach().cpu()}
    if args.test:
        weight_path = os.path.join(args.output_dir, "test_weights.bin")
    else:
        weight_path = os.path.join(args.output_dir, "weights.bin")
    torch.save(learned_weights_dict, weight_path)

def save_progress(text_encoder, placeholder_token_id, accelerator, args, placeholder_tokens):
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = dict()
    for token_idx in range(len(placeholder_token_id)):
        token = placeholder_tokens[token_idx]
        embedding = learned_embeds[token_idx].detach().cpu()
        learned_embeds_dict[token] = embedding
    embed_path = os.path.join(args.output_dir, "learned_embeds.bin")
    torch.save(learned_embeds_dict, embed_path)

class Trainer:
    def __init__(self, opts):
        self.opts = opts
        logging_dir = os.path.join(self.opts.output_dir, self.opts.logging_dir)

        self.temperature = self.opts.temperature

        # Create the accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps = self.opts.gradient_accumulation_steps,
            mixed_precision=self.opts.mixed_precision,
            log_with="tensorboard",
            project_dir=logging_dir
        )

        # Set optional seed
        if self.opts.seed is not None:
            set_seed(self.opts.seed)
        
        # Create output directory
        if self.accelerator.is_main_process:
            if self.opts.output_dir is not None:
                os.makedirs(self.opts.output_dir, exist_ok=True)

        # Resume from checkpoint
        if self.opts.resume_from_checkpoint:
            self.opts.pretrained_model = self.opts.resume_dir
            print(f"Resuming from {self.opts.pretrained_model}")

        # Load the tokenizer
        if self.opts.tokenizer_name:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.opts.tokenizer_name)
        elif self.opts.pretrained_model:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.opts.pretrained_model, subfolder="tokenizer")

        # Populate the placeholder tokens and add them to the tokenizer - Directions
        self.placeholder_tokens = [f"<t{token_idx + 1}>" for token_idx in range(self.opts.placeholder_token_count)]
        num_added_tokens = self.tokenizer.add_tokens(self.placeholder_tokens)

        if num_added_tokens != 0 and num_added_tokens != len(self.placeholder_tokens):
            raise ValueError(f"Some of the tokens added are already present in the tokenizer. Please check the list of added tokens again: {self.placeholder_tokens}")

        # Convert the initializer tokens and placeholdet tokens to ids
        if self.opts.initializer_tokens != "":
            self.initializer_tokens = [x.strip() for x in self.opts.initializer_tokens.split(",")]
        else:
            self.initializer_tokens = []

        if len(self.initializer_tokens) == 0:
            if self.opts.resume_from_checkpoint:
                logger.info("* Resume the embeddings of placeholder tokens *")
                print("* Resume the embeddings of placeholder tokens *")
            else:
                logger.info("* Initialize the newly added placeholder token with the random embeddings *")
                print("* Initialize the newly added placeholder token with the random embeddings *")
            self.token_ids = self.tokenizer.convert_tokens_to_ids(self.placeholder_tokens)
        else:
            logger.info("* Initialize the newly added placeholder token with the embeddings of the initializer token *")
            print("* Initialize the newly added placeholder token with the embeddings of the initializer token *")
            self.token_ids = self.tokenizer.encode(self.initializer_tokens, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(self.token_ids) > len(self.initializer_tokens):
                raise ValueError("The initializer token must be a single token.")

        self.initializer_token_ids = self.token_ids
        self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(self.placeholder_tokens)

        # Load models
        self.text_encoder = CLIPTextModel.from_pretrained(self.opts.pretrained_model, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(self.opts.pretrained_model, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.opts.pretrained_model, subfolder="unet")

        # Resize token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Initialize newly added placeholder tokens with embeddings of initializer token
        self.token_embeds = self.text_encoder.get_input_embeddings().weight.data
        self.token_embeds[self.placeholder_token_ids] = self.token_embeds[self.initializer_token_ids]
        if self.opts.normalize_word:
            self.token_embeds[self.placeholder_token_ids] = F.normalize(self.token_embeds[self.placeholder_token_ids], dim=1, p=2)
        
        # Freezing vae and unet
        freeze_params(self.vae.parameters())
        freeze_params(self.unet.parameters())
        
        # Freeze all parameters except token embeddings in text encoder
        params_to_freeze = itertools.chain(
            self.text_encoder.text_model.encoder.parameters(),
            self.text_encoder.text_model.final_layer_norm.parameters(),
            self.text_encoder.text_model.embeddings.position_embedding.parameters()
        )
        freeze_params(params_to_freeze)

        if self.opts.gradient_checkpointing:
            self.unet.train()
            self.text_encoder.gradient_checkpointing_enable()
            self.unet.enable_gradient_checkpointing()

        if self.opts.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available.")

        if self.opts.scale_lr:
            self.opts.learning_rate = (
                self.opts.learning_rate * self.opts.gradient_accumulation_steps * self.accelerator.num_processes
            )

        # Initialize the dataset
        self.train_dataset = DiscoveryDataset(
            data_root_dir=self.opts.train_data_dir,
            tokenizer=self.tokenizer,
            size=self.opts.resolution,
            repeats=self.opts.repeats,
            center_crop=self.opts.center_crop,
            placeholder_tokens=self.placeholder_tokens
        )

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(
            self.text_encoder.get_input_embeddings().parameters(),
            lr=self.opts.learning_rate,
            betas=(self.opts.adam_beta1, self.opts.adam_beta2),
            weight_decay=self.opts.adam_weight_decay,
            eps=self.opts.adam_epsilon
        )

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.opts.train_batch_size, shuffle=True)
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.opts.pretrained_model, subfolder="scheduler")

        self.lr_scheduler = get_scheduler(
            self.opts.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.opts.lr_warmup_steps * self.opts.gradient_accumulation_steps,
            num_training_steps=self.opts.num_train_iters * self.opts.gradient_accumulation_steps
        )

        self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        
        # Move vae and unet to the device
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.opts.gradient_accumulation_steps)
        self.max_train_steps = self.opts.num_train_iters
        self.num_train_epochs = math.ceil(self.max_train_steps / self.num_update_steps_per_epoch)

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("checkpoints", config=vars(self.opts))
        
        print("Setup Complete!")

    def train(self):
        total_batch_size = self.opts.train_batch_size * self.accelerator.num_processes * self.opts.gradient_accumulation_steps
        print(f"total_batch_size: {total_batch_size}")

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.opts.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.opts.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")

        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous state
        if self.opts.resume_from_checkpoint:
            if self.opts.resume_from_checkpoint != "latest":
                path = os.path.basename(self.opts.resume_from_checkpoint)
            else:
                # Get most recent checkpoint
                dirs = os.listdir(self.opts.resume_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.opts.resume_from_checkpoint}' does not exist. Starting a new training run."
                )

                self.opts.resume_from_checkpoint = None
                self.max_train_steps = self.opts.num_train_iters

            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.opts.resume_dir, path))
                global_step = int(path.split("-")[-1])
                resume_global_step = global_step * self.opts.gradient_accumulation_steps
                first_epoch = global_step // self.num_update_steps_per_epoch
                resume_step = resume_global_step % (self.num_update_steps_per_epoch * self.opts.gradient_accumulation_steps)
                self.max_train_steps = global_step + self.opts.num_train_iters
            
            self.opts.num_train_epochs = math.ceil(self.max_train_steps / self.num_update_steps_per_epoch)

        # Keep original embeddings as reference
        orig_embeds_params = self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight.data.clone()

        progress_bar = tqdm(range(self.opts.num_train_iters))
        progress_bar.set_description("Steps")

        print("Starting Training")
        print(f"First Epoch: {first_epoch}")
        print(f"Global Step: {global_step}")
        print(f"Num Train Epochs: {self.num_train_epochs}")
        print(f"Num iters: {self.max_train_steps}")
        print(f"Update steps per epoch: {self.num_update_steps_per_epoch}")
        print(f"Num batches in dataloader: {len(self.train_dataloader)}")

        for epoch in range(first_epoch, self.num_train_epochs):
            self.text_encoder.train()
            # Break if reached the final iteration at the end of the epoch
            if self.accelerator.is_main_process and global_step >= self.opts.num_train_iters:
                # Save from last state
                pipeline = StableDiffusionPipeline.from_pretrained(
                                self.opts.pretrained_model,
                                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                                vae=self.vae,
                                unet=self.unet,
                                tokenizer=self.tokenizer
                            )
                pipeline.save_pretrained(self.opts.output_dir)
                save_progress(self.text_encoder, self.initializer_token_ids, self.accelerator, self.opts, self.placeholder_tokens)
                save_path = os.path.join(self.opts.output_dir, f"checkpoint-{global_step}")
                self.accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
                break

            for step, batch in enumerate(self.train_dataloader):
                # Break if reached the final iteration inside the epoch
                if self.accelerator.is_main_process and global_step >= self.opts.num_train_iters:
                    pipeline = StableDiffusionPipeline.from_pretrained(
                                self.opts.pretrained_model,
                                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                                vae=self.vae,
                                unet=self.unet,
                                tokenizer=self.tokenizer
                            )
                    pipeline.save_pretrained(self.opts.output_dir)
                    save_progress(self.text_encoder, self.initializer_token_ids, self.accelerator, self.opts, self.placeholder_tokens)
                    save_path = os.path.join(self.opts.output_dir, f"checkpoint-{global_step}")
                    self.accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    break

                if self.opts.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % self.opts.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with self.accelerator.accumulate(self.text_encoder):
                    pixel_values = batch["img_pixel_values"]
                    # Randomly sample a token for the iteration as the selected token
                    token_idx = random.choice(range(len(self.placeholder_tokens)))

                    latents = self.vae.encode(pixel_values).latent_dist.sample().detach()
                    latents = latents * 0.18215

                    noise = torch.randn(latents.shape).to(latents.device)
                    timestep = torch.randint(0, self.noise_scheduler.num_train_timesteps, (1,), device=latents.device).long()

                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timestep)

                    # Predict unconditional noise
                    uncond_text_ids = self.tokenizer(
                        "",
                        padding="max_length",
                        truncation=True,
                        max_length=self.tokenizer.model_max_length,
                        return_tensors="pt"
                    ).input_ids.to(latents.device)

                    uncond_hidden_states = self.text_encoder(uncond_text_ids)[0].repeat(latents.shape[0], 1, 1)
                    uncond_noise_pred = self.unet(noisy_latents, timestep, uncond_hidden_states).sample

                    pos_sum, pos_count = 0, 0
                    neg_sum, neg_count = 0, 0

                    # Compute embeddings for selected sample
                    # Randomly select the target direction
                    selected_token = self.placeholder_tokens[token_idx]
                    # Filter out the set of directions
                    tokens_to_compare = [token for token in self.placeholder_tokens if token != selected_token]
                    # Randomly sample the set of directions to compare for computation
                    tokens_to_compare = random.sample(tokens_to_compare, self.opts.tokens_per_iter)

                    selected_token_ids = self.tokenizer(
                        selected_token,
                        padding="max_length",
                        truncation=True,
                        max_length=self.tokenizer.model_max_length,
                        return_tensors="pt"
                    ).input_ids.to(latents.device)

                    compared_token_ids = self.tokenizer(
                        tokens_to_compare,
                        padding="max_length",
                        truncation=True,
                        max_length=self.tokenizer.model_max_length,
                        return_tensors="pt"
                    ).input_ids.to(latents.device)

                    # Conditional embeddings for selected direction
                    interest_hidden_states = self.text_encoder(selected_token_ids)[0].repeat(latents.shape[0], 1, 1)
                    interest_noise_pred = self.unet(noisy_latents, timestep, interest_hidden_states).sample

                    # Feature difference for the selected direction
                    interest_noise_pred = interest_noise_pred - uncond_noise_pred

                    pos_sum, pos_count = get_positive_sum(interest_noise_pred, self.temperature)

                    # Predict noise for learned concepts other than selected
                    for idx in range(len(tokens_to_compare)):
                        if latents.shape[0] == 1: # Batch size == 1, cannot compute contrastive objective
                            continue

                        # Conditional embeddings for compared directions
                        token_hidden_states = self.text_encoder(compared_token_ids[idx][None, :])[0].repeat(latents.shape[0], 1, 1)
                        token_noise_pred = self.unet(noisy_latents, timestep, token_hidden_states).sample

                        # Target feature difference for compared directions
                        token_noise_pred = token_noise_pred - uncond_noise_pred

                        token_neg_sum, token_neg_count = get_negative_sum(interest_noise_pred, token_noise_pred, self.temperature)
                        neg_sum += token_neg_sum
                        neg_count += token_neg_count
                    
                    pos_sum = pos_sum / pos_count
                    neg_sum = neg_sum / neg_count
                    loss = -torch.log(pos_sum / neg_sum)

                    # Update gradients
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # Ensuring that weights other than newly added tokens are not updated
                    index_no_updates = torch.ones(len(self.tokenizer), dtype=torch.bool)
                    index_no_updates[self.placeholder_token_ids] = False
                    if self.accelerator.num_processes > 1:
                        grads = self.text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads = self.text_encoder.get_input_embeddings().weight.grad

                    grads.data[index_no_updates, :] = grads.data[index_no_updates, :].fill_(0)

                    with torch.no_grad():
                        self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]

                    # Checking if the accelerator performed an optimization step behind the scenes
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)

                        global_step += 1
                        if global_step % self.opts.save_steps == 0:
                            save_progress(self.text_encoder, self.initializer_token_ids, self.accelerator, self.opts, self.placeholder_tokens)

                        if global_step % self.opts.checkpointing_steps == 0:
                            if self.accelerator.is_main_process:
                                save_path = os.path.join(self.opts.output_dir, f"checkpoint-{global_step}")
                                self.accelerator.save_state(save_path)
                                logger.info(f"Saved state to {save_path}")
                        
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": self.lr_scheduler.get_last_lr()[0]
                    }

                    progress_bar.set_postfix(**logs)
                    self.accelerator.log(logs, step=global_step)

                    if global_step > self.max_train_steps or global_step >= self.opts.num_train_iters:
                        break

                    if self.accelerator.sync_gradients and global_step % self.opts.validation_step == 0:
                        folder = os.path.join(self.opts.output_dir, f"generated_samples_{global_step}")
                        os.makedirs(folder, exist_ok=True)
                        logger.info("Running validation...")

                        pipeline = DiffusionPipeline.from_pretrained(
                            self.opts.pretrained_model,
                            text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                            tokenizer=self.tokenizer,
                            unet=self.unet,
                            vae=self.vae,
                            torch_dtype=self.weight_dtype,
                            safety_checker=None
                        )
                        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
                        pipeline = pipeline.to(self.accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        generator = (
                            None if self.opts.seed is None else torch.Generator(device=self.accelerator.device).manual_seed(self.opts.seed)
                        )
                        images = []
                        prompts = []

                        for placeholder in self.placeholder_tokens:
                            prompts.append(f"{placeholder}")

                        for prompt in prompts:
                            image_list = pipeline(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator)
                            image = image_list.images[0]
                            image.save(os.path.join(folder, f"{prompt}.png"))
                            images.append(image)

                        del pipeline
                        torch.cuda.empty_cache()

                        if self.accelerator.is_main_process and global_step % self.opts.checkpointing_steps == 0:
                            pipeline = StableDiffusionPipeline.from_pretrained(
                                self.opts.pretrained_model,
                                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                                vae=self.vae,
                                unet=self.unet,
                                tokenizer=self.tokenizer
                            )
                            pipeline.save_pretrained(self.opts.output_dir)
                            save_progress(self.text_encoder, self.initializer_token_ids, self.accelerator, self.opts, self.placeholder_tokens)
                            save_path = os.path.join(self.opts.output_dir, f"checkpoint-{global_step}")
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                            del pipeline
                            torch.cuda.empty_cache()
                
                self.accelerator.wait_for_everyone()
        
        self.accelerator.end_training()
