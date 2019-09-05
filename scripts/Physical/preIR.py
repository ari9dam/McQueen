import pandas as pd
import os
import json
import numpy as np
import csv
import sys
from tqdm import tqdm


from stop_words import get_stop_words
from nltk.corpus import stopwords
import string

def remove_stopwords(sent):
    sent = sent.lower().split(" ")
    stop_words = list(get_stop_words('en'))         #About 900 stopwords
    nltk_words = list(stopwords.words('english')) #About 150 stopwords
    stop_words.extend(nltk_words)

    output = list(set([w for w in sent if not w in stop_words]))
    s = " ".join(output)
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s


def createIRFile(inpFile, outFile):
    with open(inpFile,'r') as csvin, open(outFile, 'w') as tsvout:
        csvin = csv.reader(csvin)
        next(csvin)
        for row in csvin:
#             if row[-1]=='1':
#                 a1=0
#                 a2=1
#             else:
#                 a1=1
#                 a2=0

            s1query = row[1]+" "+row[2]
            s2query = row[1]+" "+row[3]
            s1query = remove_stopwords(s1query)
            s2query = remove_stopwords(s2query)
#             sent1 = row[0]+":0"+"\t"+row[0]+"\t"+row[2]+"\t"+str(a1)+"\t"+s1query+"\n"
#             sent2 = row[0]+":1"+"\t"+row[0]+"\t"+row[3]+"\t"+str(a2)+"\t"+s2query+"\n"
            sent1 = row[0]+":0"+"\t"+row[1]+"\t"+row[2]+"\t"+s1query+"\n"
            sent2 = row[0]+":1"+"\t"+row[1]+"\t"+row[3]+"\t"+s2query+"\n"
            tsvout.write(sent1+sent2)
            
            
if __name__ == "__main__":
    
#     inpFile = "../data/train.csv"
#     outFile = "ir_search_train_fullchoice_wikisingle.tsv"

#     devFileX = "data/dev.jsonl"
#     devFileY = "data/dev-labels.lst"
    devFileX = sys.argv[1]
    #devFileY = sys.argv[2]

    #dataTrain = pd.read_json(trainFileX, lines=True)
    dataDev = pd.read_json(devFileX, lines=True)
    
    #dataTrainY = pd.read_json(trainFileY, lines=True)
#     dataDevY = pd.read_json(devFileY, lines=True)
    
#     dataTrain['ans'] = dataTrainY
#     dataDev['ans'] = dataDevY

    #Preprocessing
    dataDev['goal'] = dataDev.goal.str.replace("\r"," ").str.strip()
    dataDev['sol1'] = dataDev.sol1.str.replace("\r"," ").str.strip()
    dataDev['sol2'] = dataDev.sol2.str.replace("\r"," ").str.strip()

    
    dataDev.to_csv("dev.csv",index=False)

    inpFile = "dev.csv"
    outFile = sys.argv[2]
    createIRFile(inpFile, outFile)