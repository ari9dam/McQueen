import jsonlines
from tqdm import tqdm
import json
import string
import nltk
nltk.download("stopwords")
nltk.download("punkt")


from nltk.corpus import stopwords
import sys

stpwords =  set(stopwords.words('english'))

import spacy
nlp = spacy.load('en_core_web_lg',disable=["ner",])

def get_verbs_adj(sent):
    doc = get_doc(sent)
    verbs_adj = []
    for token in doc:
        if token.pos_ in ["VERB","ADJ","NOUN"]:
            verbs_adj.append(token.text)
    return verbs_adj

def get_keywords2(inp):
    return ' '.join(set(get_verbs_adj(inp)))
        
def get_nli_label(ansIndex,label):
    return 1 if ansIndex == label else 0
        
def strip_punct(s):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return s

def get_keywords(inp):
    inp = strip_punct(inp)
    inp = inp.lower()
    uniq_words = set(inp.split(" "))
    return ' '.join(set(uniq_words - stpwords))
        
        
def create_tsv_dataset_for_ir(qlist,labels,fname):
    with open(fname,"w+") as ofd:
        for index,tup in tqdm(enumerate(zip(qlist,labels))):
            qsn = tup[0]
            label = int(tup[1])-1
            passage = (qsn['context']+" . " +qsn["question"]).replace("\t"," ").replace('\n'," ")
            ansA = qsn["answerA"].replace("\t"," ").replace('\n'," ")
            ansB = qsn["answerB"].replace("\t"," ").replace('\n'," ")
            ansC = qsn["answerC"].replace("\t"," ").replace('\n'," ")
            
            ofd.write("%s\t%s\t%s\t%d\t%s\n"%(str(index)+":0",passage,ansA,get_nli_label(0,label),passage+" " + ansA))
            ofd.write("%s\t%s\t%s\t%d\t%s\n"%(str(index)+":1",passage,ansB,get_nli_label(1,label),passage+" " + ansB))
            ofd.write("%s\t%s\t%s\t%d\t%s\n"%(str(index)+":2",passage,ansC,get_nli_label(2,label),passage+" " + ansC))


input_path = sys.argv[1]

val_dict = []
with jsonlines.open(input_path) as reader:
    for obj in reader:
        val_dict.append(obj)
        
        
val_labels = []
with open(input_path) as tlabels:
    for line in tlabels.readlines():
        val_labels.append('0')
            
create_tsv_dataset_for_ir(val_dict,val_labels,"dev_ir.tsv")