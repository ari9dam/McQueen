import pandas as pd
import csv
import json
import numpy as np
from tqdm import tqdm
import ast
import operator
import pickle
import spacy
import sys
import jsonlines


import spacy
nlp = spacy.load('en_core_web_lg',disable=["ner","parser","tagger"])

docmap = {}


def get_doc(docs):
    if docs in docmap:
        return docmap[docs]
    docmap[docs] = nlp(docs)
    return docmap[docs]

def save_maps(fname,mmap):
    with open(fname, 'wb+') as handle:
        pickle.dump(mmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def create_merged_facts_map(df):
    merged_map = {}
    for index, row in tqdm(df.iterrows(),desc="Creating Map"):
        qidx = row['qid'].split(":")[0]
        if qidx not in merged_map:
            merged_map[qidx]={}
            merged_map[qidx]['answerlist']=[]
            merged_map[qidx]['facts']={}
        merged_map[qidx]['passage'] = row['passage']
        merged_map[qidx]['answerlist'].append(row['answer'])
        if row['label'] == 1:
            merged_map[qidx]['label'] = row['qid'].split(":")[1]
        irfacts = ast.literal_eval(row['irfacts'])
        for tup in irfacts:
            fact = tup[0]
            score = float(tup[1])
            if fact in merged_map[qidx]['facts']:
                current_score = merged_map[qidx]['facts'][fact]
                score = max(score,current_score)
            merged_map[qidx]['facts'][fact] = score
            get_doc(fact)
            
    sorted_merged_map = {}
    for qid in tqdm(merged_map.keys(),desc="Sorting:"):
        sorted_merged_map[qid]=merged_map[qid]
        sorted_merged_map[qid]['facts'] = list(sorted(merged_map[qid]['facts'].items(), key=operator.itemgetter(1),reverse=True))
    return sorted_merged_map



def create_unmerged_facts_map(df):
    unmap = {}
    for index, row in tqdm(df.iterrows(),desc="Creating Map"):
        qidx =  row['qid'].split(":")[0]
        opt  =  row['qid'].split(":")[1]
        if qidx not in unmap:
            unmap[qidx]={}
            unmap[qidx]['answerlist']=[]
            unmap[qidx]['facts']={}
        if opt not in unmap[qidx]['facts']:
            unmap[qidx]['facts'][opt]={}
        unmap[qidx]['passage'] = row['passage']
        unmap[qidx]['answerlist'].append(row['answer'])
#         if row['label'] == 1:
#             unmap[qidx]['label'] = row['qid'].split(":")[1]
        irfacts = ast.literal_eval(row['irfacts'])
        for tup in irfacts:
            fact = tup[0]
            score = float(tup[1])
            unmap[qidx]['facts'][opt][fact]=score
            get_doc(fact)
            
    sorted_merged_map = {}
    for qid in tqdm(unmap.keys(),desc="Sorting:"):
        sorted_merged_map[qid]=unmap[qid]
        sorted_merged_map[qid]['facts']['0'] = list(sorted(unmap[qid]['facts']['0'].items(), key=operator.itemgetter(1),reverse=True))
        sorted_merged_map[qid]['facts']['1'] = list(sorted(unmap[qid]['facts']['1'].items(), key=operator.itemgetter(1),reverse=True))
        #sorted_merged_map[qid]['facts']['2'] = list(sorted(unmap[qid]['facts']['2'].items(), key=operator.itemgetter(1),reverse=True))

    return sorted_merged_map


def rerank_using_spacy(row,topk=20,choice=None):
    passage = row['passage']
    choices = row['answerlist']
    query = passage
    query_doc0 = get_doc(query+" . " + choices[0])
    query_doc1 = get_doc(query+" . " + choices[1])
#     query_doc2 = get_doc(query+" . " + choices[2])
    
    if not choice:
        facts = row['facts']
    else:
        facts = row['facts'][choice]
        query_doc = get_doc(query+" . " + choices[int(choice)])
    reranked_facts = {}
    for fact_tup in facts:
        fact_doc = get_doc(fact_tup[0])
        if not choice:
            new_score= max(fact_doc.similarity(query_doc0),
                           fact_doc.similarity(query_doc1))*fact_tup[1]
        else:
            new_score = fact_doc.similarity(query_doc)
        reranked_facts[fact_tup[0]]=new_score
    facts = list(sorted(reranked_facts.items(),key=operator.itemgetter(1),reverse=True))
    non_redundant_facts = []
    non_redundant_facts.append(facts[0])
    count=1
    lastfactdoc = get_doc(facts[0][0])
    while count<topk:
        temp = {}
        for fact_tup in facts:
            fact_doc = get_doc(fact_tup[0])
            temp[fact_tup[0]] = (1-fact_doc.similarity(lastfactdoc))*fact_tup[1]
        facts = list(sorted(temp.items(),key=operator.itemgetter(1),reverse=True))
        chosen_fact_tup = facts[0]
        non_redundant_facts.append(chosen_fact_tup)
        lastfactdoc = get_doc(chosen_fact_tup[0])
        facts.pop(0)
        count+=1
    return non_redundant_facts

def create_reranked_map(merged_map,topk=20):
    reranked_map = {}
    for qidx,row in tqdm(merged_map.items(),desc="Reranking:"):
        row['facts'] = rerank_using_spacy(row)
        reranked_map[qidx]=row
    return reranked_map

def create_reranked_umap(unmap,topk=20):
    reranked_map = {}
    for qidx,row in tqdm(unmap.items(),desc="Reranking:"):
        row['facts']['0'] = rerank_using_spacy(row,choice='0')
        row['facts']['1'] = rerank_using_spacy(row,choice='1')
#         row['facts']['2'] = rerank_using_spacy(row,choice='2')
        reranked_map[qidx]=row
    return reranked_map



def create_kb_data_no_question(merged_map,fname,typet):
#     with jsonlines.open("../../data/"+fname+".jsonl", mode='w') as writer:
    with jsonlines.open(""+fname+".jsonl", mode='w') as writer:
        for qidx,row in tqdm(merged_map.items(),desc="Writing PH:"):
            facts = []
            facts.append( [tup[0] for tup in row['facts']['0'][0:10]])
            facts.append( [tup[0] for tup in row['facts']['1'][0:10]])
#             choices = row['answerlist']
            passage = row['passage']
            choices = [passage+" "+eachans for eachans in row['answerlist']]
#             label = row['label']
#             writer.write({"id":qidx,"question":[],"premises":facts,"choices":choices,"gold_label":label})
            writer.write({"id":qidx,"question":[],"premises":facts,"choices":choices})
            
            
if __name__ == "__main__":
    
    dev_fn = sys.argv[1] # inp filename 
    outfilename = sys.argv[2] #out filename -- physical_dev_kb_noq_v2
    dev_df = pd.read_csv(dev_fn,delimiter="\t",names=['qid','passage','answer','irkeys','irfacts'])
    dev_merged = create_unmerged_facts_map(dev_df)
    dev_merged_reranked = create_reranked_umap(dev_merged)
    create_kb_data_no_question(dev_merged_reranked,outfilename,"")