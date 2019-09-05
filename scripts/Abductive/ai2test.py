import jsonlines
from tqdm import tqdm
import json
import sys
#parse val data

input_path = sys.argv[1]

val_dict = []
with jsonlines.open(input_path) as reader:
    for obj in reader:
        val_dict.append(obj)
        
val_labels = []
with open(input_path) as tlabels:
    for line in tlabels.readlines():
        val_labels.append('1')

#verify data
mapto = {'1':"hyp1",'2':"hyp2"}
def view_data(X,y,i):
    print(json.dumps(X[i], indent=4, sort_keys=False))
    print("Answer:",X[i][mapto[y[i]]])
#view_data(train_dict,train_labels,1201)

#prprocessing to generate keywords
def prepn(inp):
    from nltk.corpus import stopwords 
    from nltk.tokenize import word_tokenize 

    stop_words = set(stopwords.words('english')) 

    word_tokens = word_tokenize(inp) 

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 

    filtered_sentence = [] 

    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    return TreebankWordDetokenizer().detokenize(filtered_sentence)
    #print(word_tokens) 
    #print(filtered_sentence)

#prepn('This is a sample sentence, showing off the stop words filtration.')
    
#create tsv file needed for IR
def create_tsv_dataset(qlist,labels,fname):
    with open(fname,"w+") as ofd:
        for index,tup in tqdm(enumerate(zip(qlist,labels))):
            if index == 0:
                print(tup)
            qsn = tup[0]
            if index == 0:
                print(qsn)
            label = tup[1]
            if index == 1:
                print(label)
                print(int(label)-1)
            obs1 = (qsn['obs1']).replace("\t"," ").replace('\n'," ")
            obs2 = (qsn['obs2']).replace("\t"," ").replace('\n'," ")
            #passage2 = (qsn['obs1']+qsn["hyp2"]+qsn["obs2"]).replace("\t"," ").replace('\n'," ")
            if index == 1:
                print(obs1)
                print(obs2)
            hyp1 = qsn["hyp1"].replace("\t"," ").replace('\n'," ")
            if index == 1:
                print(hyp1)
            hyp2 = qsn["hyp2"].replace("\t"," ").replace('\n'," ")
            if index == 1:
                print(hyp2)
            
            #ansC = qsn["answerC"].replace("\t"," ").replace('\n'," ")
            indexn1=str(index) + ':'+ '0'
            indexn2=str(index) + ':'+ '1'
            #print(indexn1)
            obs1n=prepn(obs1)
            obs2n=prepn(obs2)
            hyp1n=prepn(hyp1)
            hyp2n=prepn(hyp2)
            col1=obs1n+hyp1n+obs2n
            col2=obs1n+hyp2n+obs2n
        
#             col1=obs1+obs2
#             col2=obs1+obs2
            if int(label)==1:
                ofd.write("%s\t%s\t%s\t%s\t%d\t%s\n"%(indexn1,obs1,hyp1,obs2,1,col1))
                ofd.write("%s\t%s\t%s\t%s\t%d\t%s\n"%(indexn2,obs1,hyp2,obs2,0,col2))
            elif int(label)==2:
                ofd.write("%s\t%s\t%s\t%s\t%d\t%s\n"%(indexn1,obs1,hyp1,obs2,0,col1))
                ofd.write("%s\t%s\t%s\t%s\t%d\t%s\n"%(indexn2,obs1,hyp2,obs2,1,col2))
                
#generate tsv file
create_tsv_dataset(val_dict,val_labels,"devfinal.tsv")

#if getting out of range error, run following code (useful for train data where size is big)
# import sys
# import csv

# csv.field_size_limit(sys.maxsize)
                
# python ir_from_aristo.py devfinal.tsv
#it will generate devfinal.tsv.out file

#rereanked jsonl file
# python merge_ir.py mnli_simple
# this will generate devf.jsonl and devr.jsonl, both are same, use anyone.

#Best model path: /scratch/srmishr1/Sunpower/2015_swt_5e6_001_11
#--bert_model bert-large-uncased-whole-word-masking --mcq_model bert-mcq-weighted-sum --tie_weights_weighted_sum
#dev data is in /scratch/srmishr1/fullstorydata/devkb.jsonl, verify and check if the accuracy you are getting is 0.7513. If yes, go ahead and test.


