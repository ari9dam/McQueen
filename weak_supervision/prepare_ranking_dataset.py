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
                
                
def convert_to_label(facts,label=1,scaled=False):
    if not scaled:
        return [ (tup[0],label) for tup in facts ]
    else:
        return [ (tup[0],tup[1]) for tup in facts ]
        
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
            
        min_len = 100
        max_len = 0
        skipped = 0
        all_skipped = 0
        mp_sk = {}
        for ix,data in tqdm(row_scores.items(),desc="Writing Scores"):
            premise = data['row']["premise"]
            
#             top9 = convert_to_label(data["facts"][0:9],label=1)
#             rest9s = data["facts"][9:]
#             shuffle(rest9s)
#             rest9s=rest9s[0:27]
#             wrong9 = convert_to_label(rest9s[0:9],label=0)
#             wrong18 = convert_to_label(rest9s[9:18],label=0)
#             wrong27 = convert_to_label(rest9s[18:],label=0)  
#             s1 = top9+wrong9
#             s2 = top9+wrong18
#             s3 = top9+wrong27
            
            top15 = convert_to_label(data["facts"][0:15],label=1,scaled=scale)
        
            if len(data["facts"])<=30:
                randfn = random.choices
            else:
                randfn = random.sample
                
            wrong15_1 = convert_to_label(randfn(data["facts"][15:],k=15),label=0,scaled=scale)
            wrong15_2 = convert_to_label(randfn(data["facts"][15:],k=15),label=0,scaled=scale)
            wrong15_3 = convert_to_label(randfn(data["facts"][15:],k=15),label=0,scaled=scale)
            
            s1 = top15+wrong15_1
            s2 = top15+wrong15_2
            s3 = top15+wrong15_3

            min_len = min([min_len,len(s1),len(s2),len(s3)])
            max_len = max([max_len,len(s1),len(s2),len(s3)])
            sk_cur = 0
            for nix,ss in enumerate([s1,s2,s3]):
                if len(ss) < 30:
                    skipped+=1
                    sk_cur+=1
                    continue
                shuffle(ss)
                fs,labels = get_facts_labels(ss)
                if scale:
                    min_val,max_val = min(labels),max(labels)
                    den = max_val-min_val
                    labels= [ (label-min_val)/den for label in labels]
                    
                writer.write({"idx":ix+":"+str(nix),"premise":premise,"choices":fs,"labels":labels})
            mp_sk[sk_cur] = mp_sk.get(sk_cur,0)+1
            if sk_cur==3:
                all_skipped+=1
        print(f"Lengths is: {min_len,max_len}")
        print(f"Skipped : {skipped}")
        print(f"All Skipped : {all_skipped,mp_sk}")
    
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
        create_social_mcml_scaled(fname,scorefname,outname,scale=True)
