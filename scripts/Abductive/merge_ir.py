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

def load_doc_map(path_to_docmap):
    pickle_in = open(path_to_docmap,"rb")
    doc_map = pickle.load(pickle_in)
    pickle_in.close()
    return doc_map

print("Loading DocMap")
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
        #merged_map[qidx]['passage'] = row['passage']
        merged_map[qidx]['answerlist'].append(row['obs1']+row['answer']+row['obs2'])
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

def create_unmerged_facts_map(df):
    unmap = {}
    for index, row in tqdm(df.iterrows(),desc="Creating Map"):
#         print(row)
#         print(row['qid'])
#         input()
        qidx =  row['qid'].split(":")[0]
        opt  =  row['qid'].split(":")[1]
        if qidx not in unmap:
            unmap[qidx]={}
            unmap[qidx]['answerlist']=[]
            unmap[qidx]['facts']={}
        if opt not in unmap[qidx]['facts']:
            unmap[qidx]['facts'][opt]={}
        #unmap[qidx]['passage'] = row['passage']
        unmap[qidx]['answerlist'].append(row['obs1']+row['answer']+row['obs2'])
        if row['label'] == 1:
            unmap[qidx]['label'] = row['qid'].split(":")[1]
        irfacts = ast.literal_eval(row['irfacts'])
        for tup in irfacts:
            fact = tup[0]
            score = float(tup[1])
            unmap[qidx]['facts'][opt][fact]=score
            
    sorted_merged_map = {}
    for qid in tqdm(unmap.keys(),desc="Sorting:"):
        #print(unmap[qid]['facts']['2'])
        sorted_merged_map[qid]=unmap[qid]
        sorted_merged_map[qid]['facts']['0'] = list(sorted(unmap[qid]['facts']['0'].items(), key=operator.itemgetter(1),reverse=True))
        sorted_merged_map[qid]['facts']['1'] = list(sorted(unmap[qid]['facts']['1'].items(), key=operator.itemgetter(1),reverse=True))
        #sorted_merged_map[qid]['facts']['2'] = list(sorted(unmap[qid]['facts']['2'].items(), key=operator.itemgetter(1),reverse=True))

    return sorted_merged_map

# def create_swag_data(merged_map,fname,typet):
#     with open("../data/"+typet+"/"+fname, mode='w') as writer:
#         for qidx,row in tqdm(merged_map.items(),desc="Writing Swag:"):
#             facts = [ tup[0] for tup in row['facts'][0:10]]
#             choices = row['answerlist']
#             #passage = row['passage']
#             facts = " . ".join(facts)
#             choices = row['answerlist']
#             label = row['label']
#             writer.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(qidx,facts,choices[0],choices[1],choices[2],label))
            
# def create_s(train_fn,dev_fn,trainout,devout,typet):
#     train_df = pd.read_csv(train_fn,delimiter="\t",names=['qid','answer','label','irkeys','irfacts'])
#     dev_df = pd.read_csv(dev_fn,delimiter="\t",names=['qid','answer','label','irkeys','irfacts'])
#     train_merged = create_merged_facts_map(train_df)
#     dev_merged = create_merged_facts_map(dev_df)
#     create_swag_data(train_merged,trainout,typet)
#     create_swag_data(dev_merged,devout,typet)
#topk=20.
    
def rerank_using_spacy(row,topk=2,choice=None):
    #passage = row['passage']
    choices = row['answerlist']
    #query = passage
    #query_doc0 = get_doc(query+" . " + choices[0])
    #query_doc1 = get_doc(query+" . " + choices[1])
    query_doc0 = get_doc(choices[0])
    query_doc1 = get_doc(choices[1])
    #query_doc2 = get_doc(query+" . " + choices[2])
    
    if not choice:
        facts = row['facts']
    else:
        facts = row['facts'][choice]
        query_doc = get_doc(choices[int(choice)])
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

def create_reranked_map(merged_map,topk=2):
    reranked_map = {}
    for qidx,row in tqdm(merged_map.items(),desc="Reranking:"):
        row['facts'] = rerank_using_spacy(row)
        reranked_map[qidx]=row
    return reranked_map

def create_reranked_umap(unmap,topk=2):
    reranked_map = {}
    for qidx,row in tqdm(unmap.items(),desc="Reranking:"):
        row['facts']['0'] = rerank_using_spacy(row,choice='0')
        row['facts']['1'] = rerank_using_spacy(row,choice='1')
        #row['facts']['2'] = rerank_using_spacy(row,choice='2')
        reranked_map[qidx]=row
    return reranked_map

def create_multinli(train_fn,dev_fn,trainout,devout,typet):
    train_df = pd.read_csv(train_fn,delimiter="\t",names=['qid','obs1','answer','obs2','label','irkeys','irfacts'])
    dev_df = pd.read_csv(dev_fn,delimiter="\t",names=['qid','obs1','answer','obs2','label','irkeys','irfacts'])
    #dev_df['irfacts']=dev_df['irfacts'].str[40:]
    train_merged = create_unmerged_facts_map(train_df)
    dev_merged = create_unmerged_facts_map(dev_df)
    train_merged = create_reranked_umap(train_merged)
    dev_merged = create_reranked_umap(dev_merged)
    create_multinli_data(dev_merged,devout,typet)
    create_multinli_data(train_merged,trainout,typet)
    
# def create_reranked_s(train_fn,dev_fn,trainout,devout,typet):
#     train_df = pd.read_csv(train_fn,delimiter="\t",names=['qid','answer','label','irkeys','irfacts'])
#     dev_df = pd.read_csv(dev_fn,delimiter="\t",names=['qid','answer','label','irkeys','irfacts'])
#     train_merged = create_merged_facts_map(train_df)
#     dev_merged = create_merged_facts_map(dev_df)
#     train_merged = create_reranked_map(train_merged)
#     dev_merged = create_reranked_map(dev_merged)
#     create_swag_data(train_merged,trainout,typet)
#     create_swag_data(dev_merged,devout,typet)
    
# def get_probable_names(doc):
#     probable_names = []
#     for token in doc:
#         if token.pos_ == 'NOUN' and token.dep_ in ['nsubj','nmod']:
#             probable_names.append(token.text)
#     return probable_names

def create_multinli_data(merged_map,fname,typet):
    unique_name_list = {'alex'}
    max_names_in = 0
    with jsonlines.open(fname+".jsonl", mode='w') as writer:
        for qidx,row in tqdm(merged_map.items(),desc="Writing PH:"):
            facts = []
            facts.append( [tup[0] for tup in row['facts']['0'][0:10]])
            facts.append( [tup[0] for tup in row['facts']['1'][0:10]])
            #facts.append( [tup[0] for tup in row['facts']['2'][0:10]])
            choices = row['answerlist']
            #passage = row['passage']
            label = row['label']
            #prob_names = get_probable_names( get_doc(row['passage']))
            #unique_name_list = unique_name_list.union(set(prob_names))
            #max_names_in = max(max_names_in,len(set(prob_names)))
            writer.write({"id":qidx,"premises":facts,"choices":choices,"gold_label":label})
    #print(unique_name_list,max_names_in)
    
    
if __name__ == "__main__":

    typet = sys.argv[1]
    if typet == 'swag_simple':
        create_reranked_s("../data/simple_ir/train-ir.tsv.out","../data/simple_ir/dev-ir.tsv.out","train_swag_rr.tsv","dev_swag_rr.tsv","simple_ir")
    elif typet == 'swag_vadj':
        create_reranked_s("../data/verb_adj_ir/train-ir2.tsv.out","../data/verb_adj_ir/dev-ir2.tsv.out","train_swag_rr.tsv","dev_swag_rr.tsv","verb_adj_ir")
    elif typet == 'swag_nvadj':
        create_reranked_s("../data/n_verb_adj_ir/train-ir3.tsv.out","../data/n_verb_adj_ir/dev-ir3.tsv.out","train_swag_rr.tsv","dev_swag_rr.tsv","n_verb_adj_ir")
    elif typet == 'mnli_simple':
        create_multinli("devfinal.tsv.out","devfinal.tsv.out","devf","devr","simple_ir")
    elif typet == 'mnli_vadj':
        create_multinli("../data/verb_adj_ir/train-ir2.tsv.out","../data/verb_adj_ir/dev-ir2.tsv.out","train","dev","verb_adj_ir")
    elif typet == 'mnli_nvadj':
        create_multinli("../data/n_verb_adj_ir/train-ir3.tsv.out","../data/n_verb_adj_ir/dev-ir3.tsv.out","train","dev","n_verb_adj_ir")