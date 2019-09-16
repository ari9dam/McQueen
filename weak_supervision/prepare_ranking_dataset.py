from tqdm import tqdm
import argparse
import pickle
import numpy as np
import collections
import json
import operator
import torch
from random import shuffle
import gc
import jsonlines
import os
import sys
import operator
import random


random.seed(42)
np.random.seed(42)


def create_social_ranking(fname,scorefname,outname):
    with jsonlines.open(fname) as datafile, jsonlines.open(scorefname) as scorefile, jsonlines.open(outname,mode="w") as writer:
        for row,scores in tqdm(zip(datafile,scorefile),desc="Writing File"):
            idx = row["id"]
            fact = row["fact"]
            passage = row['passage']
            score = scores['score']
            choices = row['choices']
            choice_str = ' | '.join(choices)
            premise = passage + " . " + choice_str
            writer.write({"idx":idx,"premise":premise,"hypo":fact,"label":score})
            
def create_social_ranking_scaled(fname,scorefname,outname,scale=True,topk=None):
    with jsonlines.open(fname) as datafile, jsonlines.open(scorefname) as scorefile, jsonlines.open(outname,mode="w") as writer:
        row_scores = {}
        for row,scores in tqdm(zip(datafile,scorefile),desc="Merging File"):
            idx = row["id"]
            fact = row["fact"]
            passage = row['passage']
            score = scores['score']
            choices = row['choices']
            choice_str = ' | '.join(choices)
            premise = passage + " . " + choice_str
            
            ix = idx.split(":")[0]
            if ix not in row_scores:
                row_scores[ix]={}
                row_scores[ix]['row'] = {"id":ix,"premise":premise}
                row_scores[ix]['facts']= []
            row_scores[ix]['facts'].append([fact,score])
            
        for ix,data in tqdm(row_scores.items(),desc="Scaling Scores"):
            facts = data["facts"]
            if topk is None:
                topk=len(facts)
            sorted_facts = list(sorted(facts, key=operator.itemgetter(1),reverse=True))[0:topk]
            if scale:
                max_val = sorted_facts[0][1]
                min_val = sorted_facts[-1][1]
                scaled_facts = []
                for tup in sorted_facts:
                    tup[1] = (tup[1]-min_val)/(max_val-min_val)
                    scaled_facts.append(tup)
                sorted_facts=scaled_facts
            data["facts"]=sorted_facts
            row_scores[ix]=data
            
        for ix,data in tqdm(row_scores.items(),desc="Writing Scores"):
            premise = data['row']["premise"]
            for fix,tup in enumerate(data["facts"]):
                idx = ix+":"+str(fix)
                fact = tup[0]
                score = tup[1]
                writer.write({"idx":idx,"premise":premise,"hypo":fact,"label":score})
                
                
def convert_to_label(facts,label=1):
    return [ (tup[0],label) for tup in facts ]
        
def get_facts_labels(facts):
    fs= [ tup[0] for tup in facts]
    labels = [ tup[1] for tup in facts]
    return fs,labels
                
def create_social_mcml_scaled(fname,scorefname,outname,scale=True,topk=None):
    with jsonlines.open(fname) as datafile, jsonlines.open(scorefname) as scorefile, jsonlines.open(outname,mode="w") as writer:
        row_scores = {}
        for row,scores in tqdm(zip(datafile,scorefile),desc="Merging File"):
            idx = row["id"]
            fact = row["fact"]
            passage = row['passage']
            score = scores['score']
            choices = row['choices']
            choice_str = ' | '.join(choices)
            premise = passage + " . " + choice_str
            
            ix = idx.split(":")[0]
            if ix not in row_scores:
                row_scores[ix]={}
                row_scores[ix]['row'] = {"id":ix,"premise":premise}
                row_scores[ix]['facts']= []
            row_scores[ix]['facts'].append([fact,score])
            
        for ix,data in tqdm(row_scores.items(),desc="Scaling Scores"):
            facts = data["facts"]
            if topk is None:
                topk=len(facts)
            sorted_facts = list(sorted(facts, key=operator.itemgetter(1),reverse=True))[0:topk]
            if scale:
                max_val = sorted_facts[0][1]
                min_val = sorted_facts[-1][1]
                scaled_facts = []
                for tup in sorted_facts:
                    tup[1] = (tup[1]-min_val)/(max_val-min_val)
                    scaled_facts.append(tup)
                sorted_facts=scaled_facts
            data["facts"]=sorted_facts
            row_scores[ix]=data
            
        for ix,data in tqdm(row_scores.items(),desc="Writing Scores"):
            premise = data['row']["premise"]
            
            top5 = convert_to_label(data["facts"][0:5],label=1)
            rest15 = data["facts"][5:]
            shuffle(rest15)
            rest15=rest15[0:15]
            wrong5 = convert_to_label(rest15[0:5],label=0)
            wrong10 = convert_to_label(rest15[5:10],label=0)
            wrong15 = convert_to_label(rest15[10:],label=0)  
            s1 = top5+wrong5
            s2 = top5+wrong10
            s3 = top5+wrong15
            
            for nix,ss in enumerate([s1,s2,s3]):
                shuffle(ss)
                fs,labels = get_facts_labels(ss)
                writer.write({"idx":ix+":"+str(nix),"premise":premise,"choices":fs,"labels":labels})
            
    
    
if __name__ == "__main__":
    typet = sys.argv[1]
    fname = sys.argv[2]
    scorefname = sys.argv[3]
    outname = sys.argv[4]
    topk = None
    if len(sys.argv)>5:
        topk = int(sys.argv[5])
    
    if typet == "social":
        create_social_ranking(fname,scorefname,outname)
    elif typet == "social_scaled":
        create_social_ranking_scaled(fname,scorefname,outname,scale=True,topk=topk)
    elif typet == "social_unscaled":
        create_social_ranking_scaled(fname,scorefname,outname,scale=False,topk=topk)
    elif typet == "social_mcml":
        create_social_mcml_scaled(fname,scorefname,outname,scale=False)
