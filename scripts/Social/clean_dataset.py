import pandas as pd
from tqdm import tqdm
import ast
import operator
import pickle
import spacy
import sys
import jsonlines

types = ["WNTS","RCTN","DESC","MOTI","NDS","EFCT","DEFT"]    


def read_file(fname):
    train_list =[]
    with jsonlines.open(fname) as reader:
        for obj in tqdm(reader,desc="Reading:"):
            train_list.append(obj)
    return train_list        

def clean_list(flist):
    for row in flist:
        npremises = []
        for premise in row["premises"]:
            npremise =[]
            for fact in premise:
                for t in types:
                    fact = fact.replace(t,"").strip()
                npremise.append(fact)
            npremises.append(npremise)
        row["premises"]=npremises
    return flist

def write_file(flist,fname):
    with jsonlines.open(fname, mode='w') as writer:
        for row in tqdm(flist,desc="Writing PH:"):
            writer.write(row)
        
dev = sys.argv[1]

write_file(clean_list(read_file(dev)),dev)



