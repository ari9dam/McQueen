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

def lemmatized_keywords(sentence):
    doc = get_doc(sentence)
    words = []
    stoplist = ['?','would','other','result','why','what','which','how','when','whom','did','does','do','alex','casey', 'riley', 'jessie', 'jackie', 'avery', 'jaime', 'peyton', 'kerry', 'jody', 'kendall','peyton', 'skyler', 'frankie', 'pat', 'quinn']
    for token in doc:
        if token.text.lower() in stoplist or token.text.lower() in stpwords:
            continue
        words.append(token.lemma_)
    lemma = ' '.join(words)
    return lemma

def get_type_of_atomic(sent):
    if ("others want" in sent) or ("wants" in sent) :
        return 'WNTS'
    if "feel" in sent:
        return "RCTN"
    if "is seen as" in sent:
        return "DESC"
    if "wanted" in sent:
        return "MOTI"
    if "needed" in sent:
        return "NDS"
    if "effect" in sent:
        return "EFCT"

def get_type_of_quesn(qsn,row):
    if ("want" in qsn) or ('do next' in qsn):
        return 'WNTS'
    if ("feel" in qsn) or ('emotion' in qsn) or ('look to' in qsn) or ("wonder" in qsn):
        return "RCTN"
    if ("describe" in qsn) or ('think of' in qsn) or ('kind of' in qsn) or ('type of' in qsn):
        return "DESC"
    if ("why" in qsn) or ("what reason" in qsn) or ("because" in qsn):
        return "MOTI"
    if ("need" in qsn) or ("what does" in qsn) or ("what did" in qsn) :
        return "NDS"
    if ("will" in qsn) or ("happen" in qsn):
        return "EFCT"
    if ("what did" in qsn and "do" in qsn) or ("what is" in qsn and "do" in qsn) or ("do" in qsn and "afterwards" in qsn) or ("what might" in qsn and "do" in qsn) or ("what would" in qsn and "do" in qsn) :
        return "EFCT"
    if ("how would" in qsn) or ("react" in qsn):
        return "RCTN"
    if ("do" in qsn):
        return "EFCT"
    
#     print("ERROR! NOT FOUND:" + qsn,row)
    return "DEFT"
        
        
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

def create_tsv_dataset_for_ir_lemma_with_type(qlist,labels,fname):
    with open(fname,"w+") as ofd:
        for index,tup in tqdm(enumerate(zip(qlist,labels))):
            qsn = tup[0]
            label = int(tup[1])-1
            passage = (qsn['context']+" . " +qsn["question"]).replace("\t"," ").replace('\n'," ")
            ansA = qsn["answerA"].replace("\t"," ").replace('\n'," ")
            ansB = qsn["answerB"].replace("\t"," ").replace('\n'," ")
            ansC = qsn["answerC"].replace("\t"," ").replace('\n'," ")
            typet = get_type_of_quesn(qsn["question"].lower(),qsn)
            if typet is None:
                print(qsn,label)
            ofd.write("%s\t%s\t%s\t%d\t%s\n"%(str(index)+":0",passage,ansA,get_nli_label(0,label),typet + " " + lemmatized_keywords(passage+" " + ansA)))
            ofd.write("%s\t%s\t%s\t%d\t%s\n"%(str(index)+":1",passage,ansB,get_nli_label(1,label),typet + " " + lemmatized_keywords(passage+" " + ansB)))
            ofd.write("%s\t%s\t%s\t%d\t%s\n"%(str(index)+":2",passage,ansC,get_nli_label(2,label),typet + " " + lemmatized_keywords(passage+" " + ansC)))


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
create_tsv_dataset_for_ir_lemma_with_type(val_dict,val_labels,"dev_ir_lemma.tsv")