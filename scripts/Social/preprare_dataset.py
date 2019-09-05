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

input_path = sys.argv[1]


val_dict = []
with jsonlines.open(input_path) as reader:
    for obj in reader:
        val_dict.append(obj)
        
        
val_labels = []
with open(input_path) as tlabels:
    for line in tlabels.readlines():
        val_labels.append('0')
        
        
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
            
            ofd.write("%s\t%s\t%s\t%d\t%s\n"%(str(index)+":0",passage,ansA,get_nli_label(0,label),get_keywords(passage+" " + ansA)))
            ofd.write("%s\t%s\t%s\t%d\t%s\n"%(str(index)+":1",passage,ansB,get_nli_label(1,label),get_keywords(passage+" " + ansB)))
            ofd.write("%s\t%s\t%s\t%d\t%s\n"%(str(index)+":2",passage,ansC,get_nli_label(2,label),get_keywords(passage+" " + ansC)))
            
create_tsv_dataset_for_ir(val_dict,val_labels,"dev_ir.tsv")