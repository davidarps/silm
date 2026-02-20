#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
import time
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType, find_executable_batch_size, ProfileKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, GradientAccumulationPlugin
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import multiprocessing

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Instantiate UDGNConfig
from silm.udgn import UDGN, UDGNConfig
from silm.structformer import *
from silm.structformer_in_parser import *
from silm.gpst.config import *
from silm.gpst.generative_r2d2_fast import *
from silm.gpst.r2d2_insideoutside import *
from silm.gpst.data_collator import *
from silm.pos_enc import RobertaEmbeddingsWithSinusoidal

UDGNConfig.register_for_auto_class()
UDGN.register_for_auto_class()
AutoConfig.register("udgn", UDGNConfig)
AutoModelForMaskedLM.register(UDGNConfig, UDGN)
StructFormer_In_ParserConfig.register_for_auto_class()
StructFormer_In_ParserModel.register_for_auto_class()
AutoConfig.register("structformer_in_parser", StructFormer_In_ParserConfig)
AutoModelForMaskedLM.register(StructFormer_In_ParserConfig, StructFormer_In_ParserModel)
GPSTConfig.register_for_auto_class()
AutoConfig.register("gpst", GPSTConfig)
GPST.register_for_auto_class()
AutoModelForCausalLM.register(GPSTConfig, GPST)
StructFormerConfig.register_for_auto_class()
StructFormerModel.register_for_auto_class()
AutoConfig.register("structformer", StructFormerConfig)
AutoModelForMaskedLM.register(StructFormerConfig, StructFormerModel)


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.46.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Masked Language Modeling task",
        fromfile_prefix_chars='@'
        )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--objective_function",
        default="mlm",
        choices=["mlm","alm"],
        help="Training autoregressive vs. masked LMs",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=None,
        help="""Batch size (per device) for the evaluation dataloader. 
        If None, take the same as the largest batch_size that fits the GPU memory, as determined 
        by the find_executable_batch_size function. This largest batch size is also taken if 
        it is smaller than the handed value for per_device_eval_batch_size.""",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        ),
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--num_processes", 
        type=int,
        default=1,
        help="The number of available processes. DataLoaders use min(1, num_processes-1) workers."
        )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--train_for_time",
        type=int,
        default=None,
        help=("If the training script should run for a set number of minutes (including preprocessing but excluding a final save).",
              "Checking the clock after each backward pass"
              )
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Additional file where logs can be written"
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        help="Enabling mixed precision training"
    )
    parser.add_argument(
        "--gpst_io_loss", 
        type=str, 
        default="struct_loss")
    parser.add_argument(
        "--gpst_gen_loss",
        type=str,
        default="non_struct_loss_fullscale")
    parser.add_argument(
        "--track_param_and_gradient_properties", 
        action="store_true",
        help="Whether to track gradient and parameter norms, std.s, means etc."
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mlm_no_trainer", args)

    # Starting the clock when training for a set amount of time
    if args.train_for_time:
        start_time = time.time()
    time_is_up = False

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    if args.mixed_precision:
        accelerator_log_kwargs["mixed_precision"] = args.mixed_precision
    gradient_accumuluation_steps = args.gradient_accumulation_steps
    gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=gradient_accumuluation_steps)

    # profiling 
    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(output)
        output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
        print(output)
        #p.export_chrome_trace("tmp_traces/trace_" + str(p.step_num) + ".json")

    profile_kwargs = ProfileKwargs(
        activities=["cpu", "cuda"],
        schedule_option={"wait": 200, "warmup": 1, "active": 1, "skip_first": 0},
        on_trace_ready=trace_handler
    )
    accelerator_log_kwargs["kwargs_handlers"]=[profile_kwargs]
    accelerator = Accelerator(gradient_accumulation_plugin=gradient_accumulation_plugin, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.log_file:
        logfilehandler = logging.FileHandler(args.log_file)
        logger.logger.addHandler(logfilehandler)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name, args.dataset_config_name, trust_remote_code=args.trust_remote_code
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                trust_remote_code=args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                trust_remote_code=args.trust_remote_code,
            )
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.


    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    if args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace() and not "│ │ │" in line
            ]
            model_inputs = tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )
            too_long = [i for i,inp in enumerate(model_inputs["input_ids"]) if inp[-3] != tokenizer.pad_token_id]
            while len(too_long) != 0:
                l = [s[:len(s)//2] for s in model_inputs["input_ids"]]
                decoded = [tokenizer.decode(s, skip_special_tokens=True) for s in l]
                model_inputs = tokenizer(
                    decoded,
                    padding=padding,
                    truncation=True,
                    max_length=max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )
                too_long = [i for i,inp in enumerate(model_inputs["input_ids"]) if inp[-3] != 1]
            
            #for i in too_long:
            #    model_inputs["input_ids"][i][-3] = tokenizer.eos_token_id
            #    model_inputs["input_ids"][i][-2] = tokenizer.pad_token_id
            #    model_inputs["input_ids"][i][-1] = tokenizer.pad_token_id
            #    model_inputs["attention_mask"][i][-2] = 0
            #    model_inputs["attention_mask"][i][-1] = 0
            return model_inputs

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.num_processes,
                #remove_columns=cols_to_remove,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.num_processes,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with accelerator.main_process_first():
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.num_processes,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Conditional for small test subsets
    if len(train_dataset) > 3:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")


    # Dropping unimportant columns
    cols_to_remove = [text_column_name, "tree", "token_type_ids"]
    cols_to_remove = [c for c in cols_to_remove if c in train_dataset.column_names]
    train_dataset_filtered      = train_dataset.remove_columns(cols_to_remove)
    eval_dataset_filtered = eval_dataset.remove_columns(cols_to_remove)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value

    @find_executable_batch_size(starting_batch_size=args.per_device_train_batch_size)
    def training_loop(batch_size):
        logger.info(f"Starting training loop with batch size {batch_size}")
        nonlocal time_is_up, accelerator, gradient_accumulation_plugin, gradient_accumuluation_steps
        accelerator.free_memory()
        
        # if we diminished the batch size, we increase number of accum. steps
        if batch_size < args.per_device_train_batch_size:
            gradient_accumuluation_steps *= 2
        accelerator.gradient_accumulation_steps = gradient_accumuluation_steps
        print("Changed number of accumulation steps to ", accelerator.gradient_accumulation_steps)

        # Data collator
        if isinstance(config, GPSTConfig):
            ctx_size = 96 if "baby" in args.config_name else 192
            data_collator = GPSTDataCollator(tokenizer=tokenizer, ctx_size=ctx_size)
        else:
            # This one will take care of randomly masking the tokens.
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)
        
        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_dataset_filtered, shuffle=True, collate_fn=data_collator, batch_size=batch_size, num_workers=max(1, args.num_processes-1), prefetch_factor=2
        )
        eval_batch_size = min(args.per_device_eval_batch_size, batch_size) if args.per_device_eval_batch_size else batch_size
        eval_dataloader = DataLoader(eval_dataset_filtered, collate_fn=data_collator, batch_size=eval_batch_size, num_workers=max(1, args.num_processes-1), prefetch_factor=2)

        # Model
        if args.model_name_or_path:
            if args.objective_function=="mlm":
                model = AutoModelForMaskedLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    trust_remote_code=args.trust_remote_code,
                )
            else: 
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    trust_remote_code=args.trust_remote_code,
                )
        else:
            logger.info("Training new model from scratch")
            logger.info("Using objective function: " + args.objective_function)
            if args.objective_function=="mlm":
                model = AutoModelForMaskedLM.from_config(config, trust_remote_code=args.trust_remote_code)
            else: 
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

        if config.config_type=="RoBERTaConfig" and config.position_embedding_type == "sinusoidal":
            print("Build sinusoidal embeddings")
            model.roberta.embeddings = RobertaEmbeddingsWithSinusoidal(config)


        logger.info(f"Training a {model._get_name()} with {model.num_parameters()} parameters")
        logger.info(f"Training a {model._get_name()} with the following config: ")
        logger.info(config)

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes,
        )

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

        # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
        #if accelerator.distributed_type == DistributedType.TPU:
        #    model.tie_weights()

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # Figure out how many steps we should save the Accelerator states
        checkpointing_steps = args.checkpointing_steps
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)

        if args.with_tracking:
            accelerator.init_trackers("mlm_no_trainer", experiment_config)

        # Train!
        total_batch_size = batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                checkpoint_path = args.resume_from_checkpoint
                path = os.path.basename(args.resume_from_checkpoint)
                if "step_" in checkpoint_path.split("/")[-3]:
                    training_difference = int(checkpoint_path.split("/")[-3].split("_")[1])
                else:
                    training_difference = os.path.splitext(path)[0]
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)
                training_difference = os.path.splitext(path)[0]

            accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            # Extract `epoch_{i}` or `step_{i}`


            if "epoch" in str(training_difference):
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                if isinstance(training_difference, str):
                    training_difference = int(training_difference.replace("step_", ""))
                resume_step = training_difference * accelerator.gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                completed_steps = resume_step // accelerator.gradient_accumulation_steps
                resume_step -= starting_epoch * len(train_dataloader)

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)
       
        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            steps_since_zero_grad=0
            torch.cuda.empty_cache()
            if args.with_tracking:
                total_loss = 0
                loss=0.0
            if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
                active_dataloader = train_dataloader
            with accelerator.profile() as prof:
                for step, batch in enumerate(active_dataloader):
                    with accelerator.accumulate(model):
                        try: 
                            #torch.autograd.set_detect_anomaly(True)
                            #batch_cpu = {k:v.cpu() for k,v in batch.items() if isinstance(v, torch.Tensor)}
                            outputs = model(**batch) # logits should be batch_size x max_sent_len x voc_size
                            if isinstance(config, GPSTConfig):
                                io_loss = getattr(outputs, args.gpst_io_loss, None)
                                if io_loss is not None:
                                    WeightedSumFunc.a_ij_require_grad = True
                                    accelerator.backward(io_loss, retain_graph=True)
                                WeightedSumFunc.a_ij_require_grad = False
                                accelerator.backward(getattr(outputs, args.gpst_gen_loss))
                            else:
                                loss = outputs.loss 
                                # We keep track of the loss at each epoch
                                if args.with_tracking:
                                    total_loss += loss.detach().float()
                                accelerator.backward(loss)

                            steps_since_zero_grad+=1
                            if args.track_param_and_gradient_properties and steps_since_zero_grad==accelerator.gradient_accumulation_steps:
                                log_params_and_grads = dict()
                                for n, p in model.named_parameters():
                                    log_params_and_grads[f"{n}/para_mean"] = p.mean().item()
                                    log_params_and_grads[f"{n}/para_std"] = p.std().item()
                                    log_params_and_grads[f"{n}/para_max"] = p.max().item()
                                    log_params_and_grads[f"{n}/para_min"] = p.min().item()
                                    log_params_and_grads[f"{n}/grad_mean"] = p.grad.mean().item()
                                    log_params_and_grads[f"{n}/grad_std"] = p.grad.std().item()
                                    log_params_and_grads[f"{n}/grad_max"] = p.grad.max().item()
                                    log_params_and_grads[f"{n}/grad_min"] = p.grad.min().item()
                                    log_params_and_grads[f"{n}/grad_norm"] = torch.linalg.norm(p).item()
                                accelerator.log(log_params_and_grads, step=completed_steps)
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()
                            prof.step()
                        except torch.OutOfMemoryError as oom:
                            logger.warning(f"Raising OOM, batch size {batch_size}")
                            raise oom
                        except RuntimeError as e:
                            caught=False
                            #batch = batch_cpu
                            if not isinstance(config, GPSTConfig):
                                for i in range(batch["input_ids"].shape[0]):
                                    for j in range(batch["attention_mask"][i].sum()):
                                        if batch["input_ids"][i,j]==tokenizer.pad_token_id:
                                            logger.warning("Problems with a batch where the pad token appears inside the text. Skipping this batch")
                                            logger.warning("Sentence index in batch: ", i, "Token in sentence: ", j)
                                            logger.warning("input_ids: ", batch["input_ids"][i])
                                            logger.warning("Sentence: ", tokenizer.decode(batch["input_ids"][i]))
                                            caught=True
                            if not caught: 
                                raise e
                        
                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        completed_steps += 1    
                        if args.with_tracking:                            
                            if isinstance(config, GPSTConfig):
                                log_dict = {
                                    "lr_step": lr_scheduler.get_last_lr()[0] # assuming same lr for all groups (method might return list)
                                }
                                for att, v in vars(outputs).items():
                                    if "loss" in att:
                                        log_dict[f"train_{att}"] = v
                                accelerator.log(log_dict, step=completed_steps)
                            else:
                                accelerator.log({
                                    "train_loss_step": loss,
                                    "lr_step": lr_scheduler.get_last_lr()[0] # assuming same lr for all groups (method might return list)
                                },
                                step=completed_steps)
                        steps_since_zero_grad=0
                    if isinstance(checkpointing_steps, int):
                        if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                            output_dir = f"step_{completed_steps}"
                            if args.output_dir is not None:
                                output_dir_acc = os.path.join(args.output_dir, output_dir, "accelerator/")
                                output_dir_model = os.path.join(args.output_dir, output_dir, "model/")
                            accelerator.save_state(output_dir_acc, safe_serialization=False)
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(
                                output_dir_model, is_main_process=accelerator.is_main_process, save_function=accelerator.save, safe_serialization=False
                            )
                    if args.train_for_time:
                        now = time.time()
                        elapsed_time = now - start_time
                        if (elapsed_time / 60) > args.train_for_time:
                            time_is_up = True
                            logger.info(f"Now: {now} the time of {args.train_for_time} is up!")
                    if time_is_up or completed_steps >= args.max_train_steps:
                        break

            model.eval()
            torch.cuda.empty_cache()
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    try:
                        outputs = model(**batch)
                        if isinstance(config, GPSTConfig):
                            loss = outputs.gpt_loss
                        else:
                            loss = outputs.loss
                        if not loss.isnan():
                            losses.append(accelerator.gather_for_metrics(loss.repeat(eval_batch_size)))
                    except torch.OutOfMemoryError as oom:
                        logger.warning(f"Raising OOM, batch size {batch_size}")
                        raise oom
                    except RuntimeError as e:
                        caught=False
                        for i in range(batch["input_ids"].shape[0]):
                            for j in range(batch["attention_mask"][i].sum()):
                                if batch["input_ids"][i,j]==tokenizer.pad_token_id:
                                    print("Problems with a batch where the pad token appears inside the text. Skipping this batch")
                                    print("Sentence index in batch: ", i, "Token in sentence: ", j)
                                    print("input_ids: ", batch["input_ids"][i])
                                    print("Sentence: ", tokenizer.decode(batch["input_ids"][i]))
                                    caught=True
                        if not caught: 
                            raise e

            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

            if args.with_tracking:
                accelerator.log(
                    {
                        "perplexity": perplexity,
                        "eval_loss": eval_loss,
                        "train_loss": total_loss,
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

            if args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, safe_serialization=False
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    api.upload_folder(
                        commit_message=f"Training in progress epoch {epoch}",
                        folder_path=args.output_dir,
                        repo_id=repo_id,
                        repo_type="model",
                        token=args.hub_token,
                    )

            if args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir, safe_serialization=False)

            if time_is_up:
                logger.info("Finished evaluation after the time was up. Saving and shutting down now")
                break

        if args.with_tracking:
            accelerator.end_training()

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, safe_serialization=False
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                if args.push_to_hub:
                    api.upload_folder(
                        commit_message="End of training",
                        folder_path=args.output_dir,
                        repo_id=repo_id,
                        repo_type="model",
                        token=args.hub_token,
                    )
                with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                    json.dump({"perplexity": perplexity}, f)
    training_loop()


if __name__ == "__main__":
    main()