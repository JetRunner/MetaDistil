# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import gc
import random
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from copy import deepcopy as cp
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup
                          )

from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)
from distillation_meta import MetaPatientDistillation

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def clone_weights(first_module, second_module):
    for first_param, second_param in zip(first_module.parameters(), second_module.parameters()):
        first_param.data = torch.clone(second_param.data)


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_optimizer_and_scheduler(params, t_total, args, teacher=False):
    params = list(params)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if teacher:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.teacher_learning_rate, eps=args.adam_epsilon)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)
    return optimizer, scheduler


def train(args, train_dataset, held_dataset, t_model, s_model, order, d_criterion, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.num_held_batches == 0:
        held_sampler = RandomSampler(held_dataset) if args.local_rank == -1 else DistributedSampler(held_dataset)
    else:
        held_sampler = RandomSampler(held_dataset,
                                     replacement=True,
                                     num_samples=args.num_held_batches * args.train_batch_size * args.gradient_accumulation_steps
                                     )
    held_dataloader = DataLoader(held_dataset, sampler=held_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    t_optimizer, t_scheduler = get_optimizer_and_scheduler(t_model.named_parameters(), t_total, args, teacher=True)
    s_optimizer, s_scheduler = get_optimizer_and_scheduler(s_model.named_parameters(), t_total, args)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    global_round = 0
    log_step_counter = 0

    held_tr_loss = 0.0

    assume_total_avg_loss = 0.0
    assume_train_avg_loss = 0.0
    assume_soft_avg_loss = 0.0
    assume_distill_avg_loss = 0.0

    real_total_avg_loss = 0.0
    real_train_avg_loss = 0.0
    real_soft_avg_loss = 0.0
    real_distill_avg_loss = 0.0

    s_model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for epoch in train_iterator:

        total_steps_one_epoch = len(train_dataloader)

        if epoch == 0:
            batches_buffer = []
        else:
            # Sanity check
            assert batches_buffer == []

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for d_step, d_batch in enumerate(epoch_iterator):

            batches_buffer.append((d_step, d_batch))

            if (d_step + 1) % (args.num_meta_batches * args.gradient_accumulation_steps) != 0 and (d_step + 1) != total_steps_one_epoch:
                continue

            #########################################
            #           Step 1: Assume S'           #
            #########################################

            # Time machine!
            fast_weights = OrderedDict((name, param) for (name, param) in s_model.named_parameters())
            s_model_backup_state_dict, s_optimizer_backup_state_dict = cp(s_model.state_dict()), cp(
                s_optimizer.state_dict())

            s_model.train()
            t_model.eval()

            for step, batch in batches_buffer:

                batch = tuple(t.to(args.device) for t in batch)
                input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]

                assume_train_loss, assume_soft_loss, assume_pkd_loss = d_criterion(
                    t_model=t_model,
                    s_model=s_model if step == 0 else fast_weights,
                    order=order,
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    args=args,
                    teacher_grad=True)

                assume_loss = args.alpha * assume_soft_loss + (
                    1 - args.alpha) * assume_train_loss + args.beta * assume_pkd_loss

                if args.n_gpu > 1:
                    assume_loss = assume_loss.mean()  # mean() to average on multi-gpu parallel training
                    assume_train_loss = assume_train_loss.mean()
                    assume_soft_loss = assume_soft_loss.mean()
                    assume_pkd_loss = assume_pkd_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    assume_loss = assume_loss / args.gradient_accumulation_steps
                    assume_train_loss = assume_train_loss / args.gradient_accumulation_steps
                    assume_soft_loss = assume_soft_loss / args.gradient_accumulation_steps
                    assume_pkd_loss = assume_pkd_loss / args.gradient_accumulation_steps

                # There is no optimizer.step() so we can always update the fast_weights without waiting
                grads = torch.autograd.grad(assume_loss, s_model.parameters() if step == 0 else fast_weights.values(),
                                            create_graph=True, retain_graph=True)

                fast_weights = OrderedDict(
                    (name, param - args.assume_s_step_size * grad) for ((name, param), grad) in
                    zip(fast_weights.items(), grads))

                assume_total_avg_loss += assume_loss.item()
                assume_train_avg_loss += assume_train_loss.item()
                assume_soft_avg_loss += assume_soft_loss.item()
                assume_distill_avg_loss += assume_pkd_loss.item()

            #########################################
            #  Step 2: Train T with S' on HELD set  #
            #########################################

            s_prime_loss = None
            held_batch_num = 0

            t_model.train()

            for step, batch in enumerate(held_dataloader):
                batch = tuple(t.to(args.device) for t in batch)
                input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]

                s_prime_step_loss = d_criterion.s_prime_forward(
                    s_prime=fast_weights,
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    args=args)

                if args.n_gpu > 1:
                    s_prime_step_loss = s_prime_step_loss.mean()  # mean() to average on multi-gpu parallel training

                if s_prime_loss is None:
                    s_prime_loss = s_prime_step_loss
                else:
                    s_prime_loss += s_prime_step_loss

                held_batch_num += 1

            s_prime_loss /= held_batch_num

            t_grads = torch.autograd.grad(s_prime_loss, t_model.parameters())

            for p, gr in zip(t_model.parameters(), t_grads):
                p.grad = gr

            torch.nn.utils.clip_grad_norm_(t_model.parameters(), args.max_grad_norm)

            held_tr_loss += s_prime_loss.item()

            t_optimizer.step()
            t_scheduler.step()

            # Manual zero_grad
            for p in t_model.parameters():
                p.grad = None

            for p in s_model.parameters():
                p.grad = None

            del t_grads
            del grads
            del fast_weights

            #########################################
            #        Step 3: Actually update S      #
            #########################################

            # We use the Time Machine!
            s_model.load_state_dict(s_model_backup_state_dict)
            s_optimizer.load_state_dict(s_optimizer_backup_state_dict)

            del s_model_backup_state_dict, s_optimizer_backup_state_dict

            s_model.train()
            t_model.eval()

            for step, batch in batches_buffer:

                batch = tuple(t.to(args.device) for t in batch)
                input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]

                real_train_loss, real_soft_loss, real_pkd_loss = d_criterion(
                    t_model=t_model,
                    s_model=s_model,
                    order=order,
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    args=args,
                    teacher_grad=False)

                real_loss = args.alpha * real_soft_loss + (
                    1 - args.alpha) * real_train_loss + args.beta * real_pkd_loss

                if args.n_gpu > 1:
                    real_loss = real_loss.mean()  # mean() to average on multi-gpu parallel training
                    real_train_loss = real_train_loss.mean()
                    real_soft_loss = real_soft_loss.mean()
                    real_pkd_loss = real_pkd_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    real_loss = real_loss / args.gradient_accumulation_steps
                    real_train_loss = real_train_loss / args.gradient_accumulation_steps
                    real_soft_loss = real_soft_loss / args.gradient_accumulation_steps
                    real_pkd_loss = real_pkd_loss / args.gradient_accumulation_steps

                real_loss.backward()
                torch.nn.utils.clip_grad_norm_(s_model.parameters(), args.max_grad_norm)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    s_optimizer.step()
                    s_scheduler.step()
                    s_optimizer.zero_grad()

                real_total_avg_loss += real_loss.item()
                real_train_avg_loss += real_train_loss.item()
                real_soft_avg_loss += real_soft_loss.item()
                real_distill_avg_loss += real_pkd_loss.item()

            global_step += len(batches_buffer)
            log_step_counter += len(batches_buffer)
            global_round += 1
            batches_buffer = []

            if args.local_rank in [-1, 0] and args.logging_rounds > 0 and global_round % args.logging_rounds == 0:
                # Log metrics

                logger.info(
                    f"assume_loss: {assume_total_avg_loss / log_step_counter}, held_loss: {held_tr_loss / log_step_counter}, real_loss: {real_total_avg_loss / log_step_counter}")

                if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, s_model, tokenizer)
                    for key, value in results.items():
                        tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

                tb_writer.add_scalar('s_lr', s_scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('t_lr', t_scheduler.get_lr()[0], global_step)

                tb_writer.add_scalar('assume_total_loss', assume_total_avg_loss / log_step_counter, global_step)
                tb_writer.add_scalar('assume_train_loss', assume_train_avg_loss / log_step_counter, global_step)
                tb_writer.add_scalar('assume_soft_loss', assume_soft_avg_loss / log_step_counter, global_step)
                tb_writer.add_scalar('assume_pkd_loss', assume_distill_avg_loss / log_step_counter, global_step)

                tb_writer.add_scalar('held_tr_loss', held_tr_loss / log_step_counter, global_step)

                tb_writer.add_scalar('real_total_loss', real_total_avg_loss / log_step_counter, global_step)
                tb_writer.add_scalar('real_train_loss', real_train_avg_loss / log_step_counter, global_step)
                tb_writer.add_scalar('real_soft_loss', real_soft_avg_loss / log_step_counter, global_step)
                tb_writer.add_scalar('real_pkd_loss', real_distill_avg_loss / log_step_counter, global_step)

                log_step_counter = 0

                held_tr_loss = 0.0

                assume_total_avg_loss = 0.0
                assume_train_avg_loss = 0.0
                assume_soft_avg_loss = 0.0
                assume_distill_avg_loss = 0.0

                real_total_avg_loss = 0.0
                real_train_avg_loss = 0.0
                real_soft_avg_loss = 0.0
                real_distill_avg_loss = 0.0

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                base_output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                t_output_dir = os.path.join(base_output_dir, 't')
                s_output_dir = os.path.join(base_output_dir, 's')
                for output_dir in [t_output_dir, s_output_dir]:
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                s_model_to_save = s_model.module if hasattr(s_model, 'module') else s_model
                t_model_to_save = t_model.module if hasattr(t_model, 'module') else t_model

                s_model_to_save.save_pretrained(s_output_dir)
                t_model_to_save.save_pretrained(t_output_dir)

                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

            if args.memory_saving:
                gc.collect()
                torch.cuda.empty_cache()

        # Save model checkpoint for each epoch
        base_output_dir = os.path.join(args.output_dir, 'checkpoint-epoch-{}'.format(epoch + 1))
        t_output_dir = os.path.join(base_output_dir, 't')
        s_output_dir = os.path.join(base_output_dir, 's')
        for output_dir in [t_output_dir, s_output_dir]:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        s_model_to_save = s_model.module if hasattr(s_model, 'module') else s_model
        t_model_to_save = t_model.module if hasattr(t_model, 'module') else t_model

        s_model_to_save.save_pretrained(s_output_dir)
        t_model_to_save.save_pretrained(t_output_dir)

        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

    return global_step, real_total_avg_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3]}
                input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        logging.info("eval_loss: %s", str(eval_loss))
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, held=False):
    processor = processors[task]()
    output_mode = output_modes[task]
    data_dir = args.data_dir

    logger.info("Creating features from dataset file at %s", data_dir)
    label_list = processor.get_labels()
    if evaluate:
        if held:
            examples = processor.get_held_examples(data_dir)
        else:
            examples = processor.get_dev_examples(data_dir)
    else:
        examples = processor.get_train_examples(data_dir)
    features = convert_examples_to_features(examples, label_list, args.max_seq_length,
                                            tokenizer, output_mode,
                                            cls_token_at_end=False,
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            cls_token_segment_id=1,
                                            pad_on_left=False,
                                            pad_token_segment_id=0)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--teacher_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--student_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--log_dir", default='logs', type=str, help="The log data dir.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default='tmp/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--num_hidden_layers', default=6, type=int)
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="Train loss ratio.")
    parser.add_argument("--beta", default=100.0, type=float,
                        help="Distillation loss ratio.")
    parser.add_argument("--temperature", default=5.0, type=float,
                        help="Distillation temperature for soft target.")
    parser.add_argument("--select", default="skip", type=str)

    ## Other parameters
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--assume_s_step_size", default=5e-5, type=float,
                        help="LR for updating the student.")
    parser.add_argument("--teacher_learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_meta_batches", default=1, type=int, help="How many batches for one round meta learning")
    parser.add_argument("--num_held_batches", default=0, type=int,
                        help="How many batches randomly sampled for teacher updating. 0 means to use all data.")

    parser.add_argument('--logging_rounds', type=int, default=10,
                        help="Log every X meta learning rounds.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--memory_saving", action='store_true',
                        help="Try to avoid CUDA OOM error.")
    parser.add_argument("--logits_mse", action='store_true',
                        help="Using logits mse instead of softmax mse")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--head_mask_path', type=str, default=None)
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
        args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(filename="./log"+os.getenv("CUDA_VISIBLE_DEVICES"),format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.teacher_model, do_lower_case=args.do_lower_case)

    t_config = BertConfig.from_pretrained(args.teacher_model)
    t_config.num_labels = num_labels
    t_config.finetuning_task = args.task_name
    t_config.output_hidden_states = True
    t_model = model_class.from_pretrained(args.teacher_model, config=t_config)

    s_config = BertConfig.from_pretrained(args.student_model)
    s_config.num_hidden_layers = args.num_hidden_layers
    s_config.num_labels = num_labels
    s_config.finetuning_task = args.task_name
    s_config.output_hidden_states = True
    s_model = model_class.from_pretrained(args.student_model, config=s_config)
    if args.head_mask_path:
        s_model.bert.encoder.head_mask = np.load(args.head_mask_path)

    d_criterion = MetaPatientDistillation(t_config, s_config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # Distributed and parallel training
    t_model.to(args.device)
    s_model.to(args.device)
    d_criterion.to(args.device)

    if args.local_rank != -1:
        t_model = torch.nn.parallel.DistributedDataParallel(t_model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)
        s_model = torch.nn.parallel.DistributedDataParallel(s_model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)

    elif args.n_gpu > 1:
        t_model = torch.nn.DataParallel(t_model)
        s_model = torch.nn.DataParallel(s_model)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        # To Chunshu: Here I just use the dev set for simplicity. Need to tweak the code if you use a split held set.
        held_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True, held=True)

        if args.select == 'last':
            order = list(range(t_config.num_hidden_layers - 1))
            order = torch.LongTensor(order[-(s_config.num_hidden_layers - 1):])

        elif args.select == 'skip':
            order = list(range(t_config.num_hidden_layers - 1))
            every_num = t_config.num_hidden_layers // s_config.num_hidden_layers
            order = torch.LongTensor(order[(every_num - 1)::every_num])
        else:
            print('layer selection must be in [entropy, attn, dist, every]')
        order, _ = order[:(s_config.num_hidden_layers - 1)].sort()

        global_step, tr_loss = train(args, train_dataset, held_dataset, t_model, s_model, order, d_criterion, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = s_model.module if hasattr(s_model,
                                                  'module') else s_model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        s_model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        s_model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            epoch = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            s_model = model_class.from_pretrained(checkpoint)
            s_model.to(args.device)
            result = evaluate(args, s_model, tokenizer, prefix=epoch)
            result = dict((k + '_{}'.format(epoch), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
