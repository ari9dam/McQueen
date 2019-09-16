from __future__ import absolute_import

import os
import sys

import pickle
import numpy as np
import collections
import json
import operator
import torch
from random import shuffle
import gc
import jsonlines
import argparse
import logging
import random
import time

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.modeling_bert import BertModel
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer

sys.path.append("/home/pbanerj6/github/McQueen/pytorch_transformers/models/")
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

gc.enable()

Input = collections.namedtuple("Input","idx passage a b c d label")

verbose=False

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum()).tolist()

def read_social_data(fname):
    data = {}
    with jsonlines.open(fname) as reader:
        for row in tqdm(reader,desc="Reading Data:"):
            choices = row["choices"]
            label = row["gold_label"]
            data[row["id"]]=Input(idx=row["id"],passage=row["premises"][0][0],a=choices[0],b=choices[1],c=choices[2],d="none of the above",label=int(label))
    return data


def read_physical_data(fname):
    data = {}
    with jsonlines.open(fname) as reader:
        for row in tqdm(reader,desc="Reading Data:"):
            choices = row["choices"]
            label = row["gold_label"]
            data[row["id"]]=Input(idx=row["id"],passage=row["premises"][0],a=choices[0],b=choices[1],c="all of the above",d="none of the above",label=int(label))
    return data
    
    
def create_score_file(fname,data,scores):
    with jsonlines.open(fname+"_scores.jsonl", mode='w') as writer:
        for row in data.keys():
            if verbose:
                print(data[row],scores[row])
            writer.write({"id":row,"score":scores[row][data[row].label]})
    
def init_weak_learner():
    sys.path.append("/home/pbanerj6/github/OBQA/bert")
    from models.bert_qa import BertQA
    model_dir = "/scratch/pbanerj6/obqa"
    model_dir += "/bertqa-withir64s-all-35-256/"
    binfile = "pytorch_model.bin."+str(3)
    model =  BertQA( output_dir=model_dir,topk=5,
                bert_model="bert-large-cased",do_lower_case=False,
                eval_batch_size=64,max_seq_length=128,num_labels=4,action="predict",model=binfile)
    return model

def score_weak_learner_v2(fname,data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    model_dir = "/scratch/pbanerj6/social/social_mcq_concat"
    model = BertMCQConcat.from_pretrained(model_dir,cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),'distributed_{}'.format(-1)))
    model.to(device)
    model = torch.nn.DataParallel(model) 

    data_reader = BertMCQConcatReader()
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking", do_lower_case=True)
    eval_data = data_reader.read_json(json=data,tokenizer=tokenizer, max_seq_len=128,max_number_premises=10)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=128)
    etq = tqdm(eval_dataloader, desc="Scoring")
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
            for logit,label in zip(logits,label_ids):
                scores.append(softmax(logit)[label])
                    
    with jsonlines.open(fname+"_v2_scores.jsonl", mode='w') as writer:
        for row,score in zip(data,scores):
            if verbose:
                print(row["id"],score)
            writer.write({"id":row["id"],"score":score})
            
            
def score_weak_learner_physical_v2(fname,data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    model_dir = "/scratch/kkpal/serdir_bertlgww_concat_kb_1e5/"
    model = BertMCQConcat.from_pretrained(model_dir,cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),'distributed_{}'.format(-1)))
    model.to(device)
    model = torch.nn.DataParallel(model) 

    data_reader = BertMCQConcatReader()
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking", do_lower_case=True)
    eval_data = data_reader.read_json(json=data,tokenizer=tokenizer, max_seq_len=128,max_number_premises=10)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=128)
    etq = tqdm(eval_dataloader, desc="Scoring")
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
            for logit,label in zip(logits,label_ids):
                scores.append(softmax(logit)[label])
                    
    with jsonlines.open(fname+"_v2_scores.jsonl", mode='w') as writer:
        for row,score in zip(data,scores):
            if verbose:
                print(row["id"],score)
            writer.write({"id":row["id"],"score":score})
                

    
if __name__ == "__main__":
    
    model = init_weak_learner()
    
    typet = sys.argv[1]
    
    if typet == "social":
        fname = sys.argv[2]
        data = read_social_data(fname)
        scores = model.predict_weak({"test":data},"test-1")
        scores = scores[1]
        create_score_file(fname,data,scores)
    elif typet == "social_v2":
        fname = sys.argv[2]
        data = read_social_data(fname)
        d2 = []
        for idx,val in data.items():
            row={}
            row["id"]=idx
            row["premise"]=val.passage
            row["choices"]=[val.a,val.b,val.c]
            row["gold_label"]=val.label
            d2.append(row)
        score_weak_learner_v2(fname,d2)
            
    elif typet == "physical":
        fname = sys.argv[2]
        data = read_physical_data(fname)
        scores = model.predict_weak({"test":data},"test-1")
        scores = scores[1]
        create_score_file(fname,data,scores)
        
    elif typet == "physical_v2":
        fname = sys.argv[2]
        data = read_physical_data(fname)
        d2 = []
        for idx,val in data.items():
            row={}
            row["id"]=idx
            row["premise"]=val.passage
            row["choices"]=[val.a,val.b,val.c]
            row["gold_label"]=val.label
            d2.append(row)
        score_weak_learner_physical_v2(fname,d2)
        