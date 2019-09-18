import pandas as pd
from tqdm import tqdm
import ast
import operator
import pickle
import spacy
import sys
import jsonlines



print("Loading Spacy")
nlp = spacy.load('en_core_web_lg',disable=["ner","parser","tagger"])

import nltk
from nltk.corpus import stopwords
stpwords =  set(stopwords.words('english'))

def load_doc_map(path_to_docmap):
    pickle_in = open(path_to_docmap,"rb")
    doc_map = pickle.load(pickle_in)
    pickle_in.close()
    return doc_map

print("Loading DocMap")
# docmap = load_doc_map("/home/pbanerj6/github/socialiqa/notebooks/redocmap.pickled")
docmap={}

def get_doc(docs):
    if docs in docmap:
        return docmap[docs]
    docmap[docs] = nlp(docs)
    return docmap[docs]

def save_maps(fname,mmap):
    with open(fname, 'wb+') as handle:
        pickle.dump(mmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
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
        
def get_type_of_quesn(qsn):
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
    return "DEFT"

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
            
    sorted_merged_map = {}
    for qid in tqdm(merged_map.keys(),desc="Sorting:"):
        sorted_merged_map[qid]=merged_map[qid]
        sorted_merged_map[qid]['facts'] = list(sorted(merged_map[qid]['facts'].items(), key=operator.itemgetter(1),reverse=True))
    return sorted_merged_map

def create_unmerged_facts_map(df,filter=False):
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
        question = row['passage'].split(' . ')[1]
        question_type = get_type_of_quesn(question.lower())
        
        unmap[qidx]['answerlist'].append(row['answer'])
        if row['label'] == 1:
            unmap[qidx]['label'] = row['qid'].split(":")[1]
        irfacts = ast.literal_eval(row['irfacts'])
        flist = []
        for tup in irfacts:
            fact = tup[0]
            if filter and question_type not in fact and question_type != "DEFT":
                continue
                
            score = float(tup[1])
            unmap[qidx]['facts'][opt][fact]=score        
            
    sorted_merged_map = {}
    for qid in tqdm(unmap.keys(),desc="Sorting:"):
        sorted_merged_map[qid]=unmap[qid]
        sorted_merged_map[qid]['facts']['0'] = list(sorted(unmap[qid]['facts']['0'].items(), key=operator.itemgetter(1),reverse=True))
        sorted_merged_map[qid]['facts']['1'] = list(sorted(unmap[qid]['facts']['1'].items(), key=operator.itemgetter(1),reverse=True))
        sorted_merged_map[qid]['facts']['2'] = list(sorted(unmap[qid]['facts']['2'].items(), key=operator.itemgetter(1),reverse=True))

    return sorted_merged_map

def create_swag_data(merged_map,fname,typet):
    with open("../data/"+typet+"/"+fname, mode='w') as writer:
        for qidx,row in tqdm(merged_map.items(),desc="Writing Swag:"):
            facts = [ tup[0] for tup in row['facts'][0:10]]
            choices = row['answerlist']
            passage = row['passage']
            facts = " . ".join(facts)
            choices = row['answerlist']
            label = row['label']
            writer.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(qidx,facts,passage,choices[0],choices[1],choices[2],label))
            
def create_s(train_fn,dev_fn,trainout,devout,typet):
    train_df = pd.read_csv(train_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    dev_df = pd.read_csv(dev_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    train_merged = create_merged_facts_map(train_df)
    dev_merged = create_merged_facts_map(dev_df)
    create_swag_data(train_merged,trainout,typet)
    create_swag_data(dev_merged,devout,typet)
    
def rerank_using_spacy(row,topk=20,choice=None):
    passage = row['passage']
    choices = row['answerlist']
    query = passage
    query_doc0 = get_doc(query+" . " + choices[0])
    query_doc1 = get_doc(query+" . " + choices[1])
    query_doc2 = get_doc(query+" . " + choices[2])
    
    if not choice:
        facts = row['facts']
    else:
        facts = row['facts'][choice]
        query_doc = get_doc(query+" . " + choices[int(choice)])
    
    if len(facts)==0:
        return facts
    
    reranked_facts = {}
    for fact_tup in facts:
        fact_doc = get_doc(fact_tup[0])
        if not choice:
            new_score= max(fact_doc.similarity(query_doc0),fact_doc.similarity(query_doc1),fact_doc.similarity(query_doc2))*fact_tup[1]
        else:
            new_score = fact_doc.similarity(query_doc)
        reranked_facts[fact_tup[0]]=new_score
    facts = list(sorted(reranked_facts.items(),key=operator.itemgetter(1),reverse=True))
    non_redundant_facts = []
    non_redundant_facts.append(facts[0])
    count=1
    lastfactdoc = get_doc(facts[0][0])
    while count<topk and len(facts)>0:
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
        row['facts']['2'] = rerank_using_spacy(row,choice='2')
        reranked_map[qidx]=row
    return reranked_map

def create_file_for_ir_scoring(unmap,fname,max_facts=-1):
    with open(fname,"w") as ofd:
        for qidx,row in tqdm(unmap.items(),desc="Writing to IR:"):
            passage = row['passage']
            choices = row['answerlist']
            for cix,choice in enumerate(choices):
                label = 1 if int(row['label'])==cix else 0
                facts = row['facts'][str(cix)]
                if max_facts != -1:
                    facts = row['facts'][str(cix)][0:max_facts]
                for idx,fact in enumerate(facts):
                    query = lemmatized_keywords(passage +" "+choice)
                    ofd.write("%s\t%s\t%s\t%s\t%s\t%d\n"%(str(qidx)+":"+str(cix)+":"+str(idx),passage,choice,fact[0],query,label))

def create_ir_files(train_fn,dev_fn,trainout,devout,typet):
    train_df = pd.read_csv(train_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    dev_df = pd.read_csv(dev_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    train_merged = create_unmerged_facts_map(train_df)
    dev_merged = create_unmerged_facts_map(dev_df)
    create_file_for_ir_scoring(train_merged,trainout,max_facts=50)
    create_file_for_ir_scoring(dev_merged,devout)

def create_multinli(train_fn,dev_fn,trainout,devout,typet,no_train=False):
    if not no_train:
        train_df = pd.read_csv(train_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
        train_merged = create_unmerged_facts_map(train_df)
        train_merged = create_reranked_umap(train_merged)
        create_multinli_data(train_merged,trainout,typet)
    dev_df = pd.read_csv(dev_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    dev_merged = create_unmerged_facts_map(dev_df)
    dev_merged = create_reranked_umap(dev_merged)
    create_multinli_data(dev_merged,devout,typet)

def create_multinli_cont(train_fn,dev_fn,trainout,devout,typet,no_train=False):
    if not no_train:
        train_df = pd.read_csv(train_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
        train_merged = create_unmerged_facts_map(train_df)
        train_merged = create_reranked_umap(train_merged)
        create_multinli_with_prem_first(train_merged,trainout,typet)
        create_multinli_with_prem_first_score(train_merged,trainout+"_score",typet)
    dev_df = pd.read_csv(dev_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    dev_merged = create_unmerged_facts_map(dev_df)
    dev_merged = create_reranked_umap(dev_merged)
    create_multinli_with_prem_first(dev_merged,devout,typet)
    create_multinli_with_prem_first_score(dev_merged,devout+"_score",typet)


def create_multinli_cont_score(train_fn,dev_fn,trainout,devout,typet,no_train=False):
    if not no_train:
        train_df = pd.read_csv(train_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
        train_merged = create_unmerged_facts_map(train_df)
        train_merged = create_reranked_umap(train_merged)
        create_multinli_with_prem_first_score(train_merged,trainout,typet)
    dev_df = pd.read_csv(dev_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    dev_merged = create_unmerged_facts_map(dev_df)
    dev_merged = create_reranked_umap(dev_merged)
    create_multinli_with_prem_first_score(dev_merged,devout,typet)

def create_multinli_knowledge(train_fn,dev_fn,trainout,devout,typet,no_train=False):
    if not no_train:
        train_df = pd.read_csv(train_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
        train_merged = create_unmerged_facts_map(train_df)
        train_merged = create_reranked_umap(train_merged)
        create_multinli_data_knowledge(train_merged,trainout,typet)
    dev_df = pd.read_csv(dev_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    dev_merged = create_unmerged_facts_map(dev_df)
    dev_merged = create_reranked_umap(dev_merged)
    create_multinli_data_knowledge(dev_merged,devout,typet)
    
def create_multinli_filtered(train_fn,dev_fn,trainout,devout,typet):
    train_df = pd.read_csv(train_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    dev_df = pd.read_csv(dev_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    train_merged = create_unmerged_facts_map(train_df,filter=True)
    dev_merged = create_unmerged_facts_map(dev_df,filter=True)
    train_merged = create_reranked_umap(train_merged)
    dev_merged = create_reranked_umap(dev_merged)
    create_multinli_data(dev_merged,devout,typet)
    create_multinli_data(train_merged,trainout,typet)
    
def create_multinli_filt_uniq(train_fn,dev_fn,trainout,devout,typet):
    train_df = pd.read_csv(train_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    dev_df = pd.read_csv(dev_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    train_merged = create_unmerged_facts_map(train_df,filter=True)
    dev_merged = create_unmerged_facts_map(dev_df,filter=True)
    train_merged = create_reranked_umap(train_merged)
    dev_merged = create_reranked_umap(dev_merged)
    create_multinli_data_unique(dev_merged,devout,typet)
    create_multinli_data_unique(train_merged,trainout,typet)
    
def create_multinli_unique(train_fn,dev_fn,trainout,devout,typet):
    train_df = pd.read_csv(train_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    dev_df = pd.read_csv(dev_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    train_merged = create_unmerged_facts_map(train_df)
    dev_merged = create_unmerged_facts_map(dev_df)
    train_merged = create_reranked_umap(train_merged)
    dev_merged = create_reranked_umap(dev_merged)
    create_multinli_data_unique(dev_merged,devout,typet)
    create_multinli_data_unique(train_merged,trainout,typet)
    
def create_reranked_s(train_fn,dev_fn,trainout,devout,typet):
    train_df = pd.read_csv(train_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    dev_df = pd.read_csv(dev_fn,delimiter="\t",names=['qid','passage','answer','label','irkeys','irfacts'])
    train_merged = create_merged_facts_map(train_df)
    dev_merged = create_merged_facts_map(dev_df)
    train_merged = create_reranked_map(train_merged)
    dev_merged = create_reranked_map(dev_merged)
    create_swag_data(train_merged,trainout,typet)
    create_swag_data(dev_merged,devout,typet)
    
def get_probable_names(doc):
    probable_names = []
    for token in doc:
        if token.pos_ == 'NOUN' and token.dep_ in ['nsubj','nmod']:
            probable_names.append(token.text)
    return probable_names

def create_multinli_data(merged_map,fname,typet):
    with jsonlines.open(fname+".jsonl", mode='w') as writer:
        for qidx,row in tqdm(merged_map.items(),desc="Writing PH:"):
            facts = []
            facts.append( [tup[0] for tup in row['facts']['0'][0:10]])
            facts.append( [tup[0] for tup in row['facts']['1'][0:10]])
            facts.append( [tup[0] for tup in row['facts']['2'][0:10]])
            passage = row['passage']
            choices = [passage + " . " + row['answerlist'][0],passage + " . " + row['answerlist'][1],passage + " . " + row['answerlist'][2]]
            writer.write({"id":qidx,"premises":facts,"choices":choices,"gold_label":0})

def create_multinli_with_prem_first(merged_map,fname,typet):
    with jsonlines.open(fname+".jsonl", mode='w') as writer:
        for qidx,row in tqdm(merged_map.items(),desc="Writing PH:"):

            passage = row['passage']

            facts = [[passage],[passage],[passage]]
            facts.extend( [tup[0] + " . "+passage for tup in row['facts']['0'][0:10]])
            facts.extend( [tup[0] + " . "+passage for tup in row['facts']['1'][0:10]])
            facts.extend( [tup[0] + " . "+passage for tup in row['facts']['2'][0:10]])

            choices = row['answerlist']
            writer.write({"id":qidx,"premises":facts,"choices":choices,"gold_label":row['label']})

def append_context(tup,passage):
    return [tup[0] + " . "+passage,tup[1]]

def create_multinli_with_prem_first_score(merged_map,fname,typet):
    with jsonlines.open(fname+".jsonl", mode='w') as writer:
        for qidx,row in tqdm(merged_map.items(),desc="Writing PH:"):

            passage = row['passage']
            context = passage.split(" . ")[0]
            question = passage.split(" . ")[1]

            facts = [[[passage,1]],[[passage,1]],[[passage,1]]]

            facts.extend([append_context(tup,passage) for tup in row['facts']['0'][0:10]])
            facts.extend([append_context(tup,passage) for tup in row['facts']['1'][0:10]])
            facts.extend([append_context(tup,passage) for tup in row['facts']['2'][0:10]])

            choices = row['answerlist']
            writer.write({"id":qidx,"premises":facts,"choices":choices,"gold_label":row['label']})
  

def create_multinli_data_knowledge(merged_map,fname,typet):
    with jsonlines.open(fname+".jsonl", mode='w') as writer:
        for qidx,row in tqdm(merged_map.items(),desc="Writing PH:"):
            passage = row['passage']
            facts = []
            facts.extend( [tup[0] for tup in row['facts']['0'][0:20]])
            facts.extend( [tup[0] for tup in row['facts']['1'][0:20]])
            facts.extend( [tup[0] for tup in row['facts']['2'][0:20]])
            choices = [row['answerlist'][0],row['answerlist'][1],row['answerlist'][2]]
            for fix,fact in enumerate(set(facts)):
                nqidx = qidx+":"+str(fix)
                f1 = fact + " " + passage
                writer.write({"id":nqidx,"fact":fact,"passage":passage,"premises":[[f1],[f1],[f1]],"choices":choices,"gold_label":row['label']})        
            
def create_multinli_data_unique(merged_map,fname,typet):
    with jsonlines.open("../data/"+typet+"/"+fname+".jsonl", mode='w') as writer:
        for qidx,row in tqdm(merged_map.items(),desc="Writing PH:"):
            a= [tup[0] for tup in row['facts']['0']]
            b= [tup[0] for tup in row['facts']['1']]
            c= [tup[0] for tup in row['facts']['2']]
            
            a1 = [x for x in a if ((x not in b) and (x not in c))]
            b1 = [x for x in b if ((x not in a) and (x not in c))]
            c1 = [x for x in c if ((x not in a) and (x not in b))]
            
            set_of_uniqs = set(a1).union(set(b1)).union(set(c1))
            
            present_in_all = {}
            for ch in ['0','1','2']:
                for tup in row['facts'][ch]:
                    if tup[0] not in set_of_uniqs:
                        sc = present_in_all.get(tup[0],0)
                        sc = max(sc,tup[1])
                        present_in_all[tup[0]]=sc
            present_in_all = list(sorted(present_in_all.items(),key=operator.itemgetter(1),reverse=True))
            
            a1 = a1[0:10]
            b1 = b1[0:10]
            c1 = c1[0:10]
            
            a1.extend([tup[0] for tup in present_in_all[0:(10-len(a1))]])
            b1.extend([tup[0] for tup in present_in_all[0:(10-len(b1))]])
            b1.extend([tup[0] for tup in present_in_all[0:(10-len(c1))]])
            
            facts = [a1[0:10],b1[0:10],c1[0:10]]
            passage = row['passage']
            
            choices = [passage + " . " + row['answerlist'][0],passage + " . " + row['answerlist'][1],passage + " . " + row['answerlist'][2]]
            label = row['label']
            writer.write({"id":qidx,"premises":facts,"choices":choices,"gold_label":label})
    
    
if __name__ == "__main__":

    # print("Initialize Doc Cache")
    # with open("atomic_knowledge_sentences.txt","r") as ifd:
    #     flist = []
    #     for line in ifd:
    #         flist.append(line.strip())
    #     print("Generating Docs")
    #     for doc,fact in tqdm(zip(nlp.pipe(flist,batch_size=10000),flist),desc="Caching"):
    #         docmap[fact]=doc
    
    typet = sys.argv[1]

    if typet == 'swag_simple':
        create_reranked_s("../data/simple_ir/train-ir.tsv.out","../data/simple_ir/dev-ir.tsv.out","train_swag_rr.tsv","dev_swag_rr.tsv","simple_ir")
    elif typet == 'swag_vadj':
        create_reranked_s("../data/verb_adj_ir/train-ir2.tsv.out","../data/verb_adj_ir/dev-ir2.tsv.out","train_swag_rr.tsv","dev_swag_rr.tsv","verb_adj_ir")
    elif typet == 'swag_nvadj':
        create_reranked_s("../data/n_verb_adj_ir/train-ir3.tsv.out","../data/n_verb_adj_ir/dev-ir3.tsv.out","train_swag_rr.tsv","dev_swag_rr.tsv","n_verb_adj_ir")
    elif typet == 'mnli_simple':
        create_multinli("../data/simple_ir/train-ir.tsv.out","../data/simple_ir/dev-ir.tsv.out","train","dev","simple_ir")
    elif typet == 'mnli_vadj':
        create_multinli("../data/verb_adj_ir/train-ir2.tsv.out","../data/verb_adj_ir/dev-ir2.tsv.out","train","dev","verb_adj_ir")
    elif typet == 'mnli_nvadj':
        create_multinli("../data/n_verb_adj_ir/train-ir3.tsv.out","../data/n_verb_adj_ir/dev-ir3.tsv.out","train","dev","n_verb_adj_ir")
    elif typet == 'lemma':
        create_multinli("../data/train-lemma.tsv.out","../data/dev-lemma.tsv.out","train_lemma","dev_lemma","")
    elif typet == 'unique':
        create_multinli_unique("../data/train-lemma.tsv.out","../data/dev-lemma.tsv.out","train_l_u2","dev_l_u2","")
    elif typet == 'typed':
        create_multinli("","dev_ir_lemma.tsv.out","","dev_typed","",no_train=True)
    elif typet == 'filtered':
        create_multinli_filtered("../data/train-lemma-type.tsv.out","../data/dev-lemma-type.tsv.out","train_filtered","dev_filtered","")
    elif typet == 'filt_uniq':
        create_multinli_filt_uniq("../data/train-lemma-type.tsv.out","../data/dev-lemma-type.tsv.out","train_filt_uniq","dev_filt_uniq","")
    elif typet == 'ir_files':
        create_ir_files("../data/train-lemma.tsv.out","../data/dev-lemma.tsv.out","train_ir2.tsv","dev_ir.tsv","")
    elif typet == "eval":
        create_multinli("","dev_ir.tsv.out","","dev","",no_train=True)
    elif typet == "lemma_knowledge":
        create_multinli_knowledge("train_ir.tsv.out","dev_ir.tsv.out","train","dev","")
    elif typet == "with_perm":
        create_multinli_cont("train_ir.tsv.out","dev_ir.tsv.out","train_perm","dev_perm","")

