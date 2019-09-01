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
    return np.sum(outputs == labels),outputs

def _get_loss_accuracy(loss:float,accuracy:float):
    return "accuracy: %.4f"%accuracy + ", loss: %.4f"%loss

def _show_runtime(seconds:int):
    return f"{(seconds//3600)} hours : {(seconds//60)} minutes"

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The training data path")
    parser.add_argument("--output_data_path",
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
    parser.add_argument("--model_dir",
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
    parser.add_argument("--error_only",
                        action='store_true',
                        help="Whether to filter errors. Labels are needed")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--max_number_premises',
                        type=int,
                        default=None,
                        help="Number of premise sentences to use at max")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()


    if not os.path.exists(args.model_dir) and not os.listdir(args.model_dir):
        raise ValueError("Model directory ({}) doesnot exists.".format(args.model_dir))

    stdout_handler = prepare_global_logging(args.model_dir, False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    data_reader = None
        
    if args.mcq_model == 'bert-mcq-parallel-max':
        model = BertMCQParallel.from_pretrained(args.model_dir,
                                                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed_{}'.format(args.local_rank)))
        data_reader = BertMCQParallelReader()
    elif args.mcq_model == 'bert-mcq-concat':
        model = BertMCQConcat.from_pretrained(args.model_dir,
                                                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed_{}'.format(args.local_rank)))
        data_reader = BertMCQConcatReader()
    elif args.mcq_model == 'bert-mcq-weighted-sum':
        model = BertMCQWeightedSum.from_pretrained(args.model_dir,
                                                   tie_weights = args.tie_weights_weighted_sum,
                                                   cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                          'distributed_{}'.format(args.local_rank)))
        data_reader = BertMCQParallelReader()
    elif args.mcq_model == 'bert-mcq-simple-sum':
        model = BertMCQSimpleSum.from_pretrained(args.model_dir,

                                                   cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                          'distributed_{}'.format(args.local_rank)))
        data_reader = BertMCQParallelReader()
    elif args.mcq_model == 'bert-mcq-mac':
        model = BertMCQMAC.from_pretrained(args.model_dir,
                                                   cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                          'distributed_{}'.format(args.local_rank)))
        data_reader = BertMCQParallelReader()
    else:
        logger.error(f"Invalid MCQ model name {args.mcq_model}")
        exit(0)

        
    # Load Data To Score:
    eval_data = data_reader.read(args.input_data_path, tokenizer, args.max_seq_length,
                                      args.max_number_premises)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    model.to(device)


    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)


    logger.info("***** Evaluation *****")
    logger.info("  num examples = %d", len(eval_data))
    logger.info("  batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    etq = tqdm(eval_dataloader, desc="Scoring")
    prediction_list = []
    gold_labels = []
    scores = []
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
            tmp_eval_accuracy,predictions = accuracy(logits, label_ids)
            
            scores.extend(logits)
            gold_labels.extend(label_ids)
            prediction_list.extend(predictions)        

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

            etq.set_description(_get_loss_accuracy(eval_loss / nb_eval_steps, eval_accuracy / nb_eval_examples))

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples

        cleanup_global_logging(stdout_handler)
        output_score_file = os.path.join(args.output_data_path,"score_file.txt")
        output_only_preds = os.path.join(args.output_data_path,"predictions.txt")
        output_with_labels = os.path.join(args.output_data_path,"pred_labels.txt")
        with open(output_score_file, "w") as scorefile:
            for score in scores:
                scorefile.write(str(score)+"\n")
        with open(output_only_preds,"w") as onlypreds, open(output_with_labels,"w") as predlabels:
            for pred,label in zip(prediction_list,gold_labels):
                onlypreds.write(str(pred)+"\n")
                predlabels.write(str(pred)+"\t"+str(label)+"\t"+str(pred==label)+"\n")
            

if __name__ == "__main__":
    main()
