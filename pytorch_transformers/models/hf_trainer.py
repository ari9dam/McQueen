from __future__ import absolute_import

import argparse
import logging
import os
import random
import sys
import time
import json
import numpy as np
import torch

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.modeling_bert import BertModel
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
from hf_bert_mcq_parallel_reader import BertMCQParallelReader
from hf_bert_mcq_parallel import BertMCQParallel
from util import cleanup_global_logging,prepare_global_logging
from hf_bert_mcq_concat import BertMCQConcat
from hf_bert_mcq_concat_reader import BertMCQConcatReader
from hf_bert_mcq_weighted_sum import BertMCQWeightedSum
from hf_bert_mcq_simple_sum import BertMCQSimpleSum
from hf_bert_mcq_mac import BertMCQMAC

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
MODELS = {
    "mcq_parallel": (BertModel, BertMCQParallelReader)
}


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def _get_loss_accuracy(loss:float,accuracy:float):
    return "accuracy: %.4f"%accuracy + ", loss: %.4f"%loss

def _show_runtime(seconds:int):
    return f"{(seconds//3600)} hours : {(seconds//60)} minutes"

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--training_data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The training data path")
    parser.add_argument("--validation_data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The validation data path")

    parser.add_argument("--mcq_model", default=None, type=str, required=True,
                        help="choose one from the list: bert-mcq-parallel-max, "
                             "bert-mcq_parallel-weighted-sum, bert-mcq-concat, mac-bert")

    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--max_grad_norm", default=None, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout", default=0.0, type=float,
                        help="dropout")
    parser.add_argument("--eval_freq", default=0, type=int,
                        help="Evaluation steps frequency. Default is at the end of each epoch. "
                             "You can also increase the frequency")
    parser.add_argument('--tie_weights_weighted_sum',
                        action='store_true',
                        help="Whether to tie the weights for the weighted sum model")
    parser.add_argument('--max_number_premises',
                        type=int,
                        default=None,
                        help="Number of premise sentences to use at max")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    # true batch size
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir,"mcq_inputs.json"),'w') as f:
        json.dump(vars(args),f,indent=2)

    stdout_handler = prepare_global_logging(args.output_dir, False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    data_reader = None
    if args.mcq_model == 'bert-mcq-parallel-max':
        model = BertMCQParallel.from_pretrained(args.bert_model,
                                                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed_{}'.format(args.local_rank)))
        data_reader = BertMCQParallelReader()
    elif args.mcq_model == 'bert-mcq-concat':
        model = BertMCQConcat.from_pretrained(args.bert_model,
                                                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed_{}'.format(args.local_rank)))
        data_reader = BertMCQConcatReader()
    elif args.mcq_model == 'bert-mcq-weighted-sum':
        model = BertMCQWeightedSum.from_pretrained(args.bert_model,
                                                   tie_weights = args.tie_weights_weighted_sum,
                                                   cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                          'distributed_{}'.format(args.local_rank)))
        data_reader = BertMCQParallelReader()
    elif args.mcq_model == 'bert-mcq-simple-sum':
        model = BertMCQSimpleSum.from_pretrained(args.bert_model,

                                                   cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                          'distributed_{}'.format(args.local_rank)))
        data_reader = BertMCQParallelReader()
    elif args.mcq_model == 'bert-mcq-mac':
        model = BertMCQMAC.from_pretrained(args.bert_model,
                                                   cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                          'distributed_{}'.format(args.local_rank)))
        data_reader = BertMCQParallelReader()
    else:
        logger.error(f"Invalid MCQ model name {args.mcq_model}")
        exit(0)

    if args.do_train:
        # Prepare data loader
        # get data loader for train/dev
        train_data = data_reader.read(args.training_data_path, tokenizer, args.max_seq_length,
                                      args.max_number_premises)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        eval_data = data_reader.read(args.validation_data_path, tokenizer, args.max_seq_length,
                                      args.max_number_premises)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # num_train_optimization_steps, dividing by effective batch size
        t_total = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_train_epochs

        num_train_optimization_steps = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare optimizer
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        model.to(device)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        number_of_batches_per_epoch = len(train_dataloader)
        if args.eval_freq > 0:
            steps_to_eval = args.eval_freq

        else:
            steps_to_eval = number_of_batches_per_epoch

        logger.info("***** Training *****")
        logger.info("  num examples = %d", len(train_data))
        logger.info("  batch size = %d", args.train_batch_size)
        logger.info("  num steps = %d", num_train_optimization_steps)
        logger.info("  number of Gpus= %d", n_gpu)
        logger.info("***** Evaluation *****")
        logger.info("  num examples = %d", len(eval_data))
        logger.info("  batch size = %d", args.eval_batch_size)

        best_acc = 0.0
        best_epoch = 1

        for epoch_index in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch_start_time = time.time()
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            tq = tqdm(train_dataloader, desc="Iteration")
            acc = 0
            for step, batch in enumerate(tq):
                batch = tuple(t.to(device) for t in batch)
                input_ids, segment_ids, input_mask, label_ids = batch
                outputs = model(input_ids, segment_ids, input_mask, label_ids)
                loss = outputs[0]
                logits = outputs[1]
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                tmp_accuracy = accuracy(logits, label_ids)
                acc += tmp_accuracy

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    if args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()  # Update learning rate schedule
                    optimizer.step()
                    model.zero_grad()
                    global_step += 1

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                tq.set_description(_get_loss_accuracy(tr_loss / nb_tr_steps,acc / nb_tr_examples))

                # TODO: always eval on last batch
                # For now select the batch_size appropriately
                if (((step + 1) % steps_to_eval == 0) or (step+1)==number_of_batches_per_epoch )\
                        and args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                    model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    etq = tqdm(eval_dataloader, desc="Validating")
                    for input_ids, segment_ids, input_mask, label_ids in etq:
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)

                        with torch.no_grad():
                            outputs = model(input_ids, segment_ids, input_mask, label_ids)
                            tmp_eval_loss = outputs[0]
                            logits = outputs[1]

                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        tmp_eval_accuracy = accuracy(logits, label_ids)

                        eval_loss += tmp_eval_loss.mean().item()
                        eval_accuracy += tmp_eval_accuracy

                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                        etq.set_description(
                            _get_loss_accuracy(eval_loss / nb_eval_steps, eval_accuracy / nb_eval_examples))

                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = eval_accuracy / nb_eval_examples
                    logger.info(f"epoch, step | {epoch_index}, {step}")
                    logger.info("            |   Training |  Validation")
                    logger.info("accuracy    |   %.4f"%(acc / nb_tr_examples)+
                                "  |   %.4f"%eval_accuracy)
                    logger.info("loss        |   %.4f" % (tr_loss / nb_tr_steps) +
                                "  |   %.4f" % eval_loss)
                    best_acc = max(best_acc, eval_accuracy)

                    if eval_accuracy == best_acc:
                        best_epoch = (epoch_index, step)
                        logger.info(
                            "best validation performance so far %.4f: " % best_acc + ", best epoch: " + str(best_epoch)
                            + ". saving current model to " + args.output_dir)

                        # Save a trained model, configuration and tokenizer
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self

                        # If we save using the predefined names, we can load using `from_pretrained`
                        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir)
                model.train()
            epoch_end_time = time.time()
            logger.info(f"time it took to finish the epoch {epoch_index} of {args.num_train_epochs} is "
                        + _show_runtime(epoch_end_time - epoch_start_time))

        # Does this even make sense to output?
        result = {'eval_accuracy': best_acc,
                  'global_step': global_step,
                  'best_epoch':best_epoch}
        cleanup_global_logging(stdout_handler)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
