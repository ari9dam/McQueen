import itertools
from typing import List
import json
import logging
import torch
from tqdm import tqdm
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from torch.utils.data import TensorDataset
import os
import pickle

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class RoBertaMCQConcatReader:
    
    def __init__(self,debug=False):
        self.truncated=0
        self.tokenized_map = {}
        self.debug=debug
    
    @staticmethod
    def _truncate_tokens(tokens_a, tokens_b, max_length):
        """
        Truncate a from the start and b from the end until total is less than max_length.
        At each step, truncate the longest one
        """
        while len(tokens_a) + len(tokens_b) > max_length:
            if len(tokens_a) > 0:
                tokens_a.pop()
            else:
                tokens_b.pop()
        return tokens_a, tokens_b
    
    
    def load_cache(self,path_to_docmap):
        pickle_in = open(path_to_docmap,"rb")
        cached_obj = pickle.load(pickle_in)
        pickle_in.close()
        return cached_obj

    def save_cache(self,obj,fname):
        with open(fname, 'wb+') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    def bert_features_from_qa(self, tokenizer, max_pieces: int, question: str, answer: str, context: str = None):
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        
        question_tokens = self.tokenized_map.get(question,tokenizer.tokenize(question))
        self.tokenized_map[question]=question_tokens
        
        if context is not None:
            context_tokens = self.tokenized_map.get(context,tokenizer.tokenize(context))
            self.tokenized_map[context] = context_tokens
            #Append sep tokens
            question_tokens = context_tokens + [sep_token] + question_tokens
        
        choice_tokens = self.tokenized_map.get(answer,tokenizer.tokenize(answer))
        self.tokenized_map[answer]=choice_tokens
        
        question_tokens, choice_tokens = self._truncate_tokens(question_tokens, choice_tokens, max_pieces - 4)

        tokens = [cls_token] + question_tokens + [sep_token]+ [sep_token] + choice_tokens + [sep_token]
        segment_ids = list(itertools.repeat(0, len(question_tokens) + 3)) + \
                      list(itertools.repeat(1, len(choice_tokens) + 1))
        
        if self.debug:
            logger.info("Generated tokens %s\n %s", str(tokens), str(segment_ids))
        
        return tokens, segment_ids

    def text_to_instance(self,  # type: ignore
                         tokenizer,  # type: ignore
                         max_seq_length: int,
                         premises: List[List[str]],
                         choices: List[str],
                         question: str = None,
                         max_number_premises=None):
        tokens = []
        token_type_ids = []
        if isinstance(premises,str):
            premises = [premises]
        if isinstance(premises[0], str):
            premises = [premises] * len(choices)
        for premise, hypothesis in zip(premises, choices):

            per_choice_tokens = []
            per_choice_token_ids = []
            # two major keys
            # ph: [cls]all_premise[sep]hypothesis[sep]
            # two different segment_ids
            # join all premise sentences
            if not max_number_premises :
                max_number_premises = len(premise)
            concatenated_premise = " ".join(premise[0:max_number_premises])

            if question is None:
                ph_tokens, ph_token_type_ids = self.bert_features_from_qa(tokenizer, max_seq_length,
                                                                          question=concatenated_premise,
                                                                          answer=hypothesis)
            else:
                ph_tokens, ph_token_type_ids = self.bert_features_from_qa(tokenizer, max_seq_length,
                                                                          question=question,
                                                                          context=concatenated_premise,
                                                                          answer=hypothesis)

            # tokenize
            input_ids = tokenizer.convert_tokens_to_ids(ph_tokens)
            if self.debug:
                print(f"Premise: {premise}, Hypothesis :{hypothesis}")
                print(f"Concatenated Premise: {concatenated_premise}")
                print(f"Tokens : {ph_tokens}, TokenIds: {ph_token_type_ids}")
                print(f"InputIds : {input_ids}")
            
            
            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            ph_token_type_ids += padding
            tokens.append(input_ids)
            token_type_ids.append(ph_token_type_ids)

        return (tokens, token_type_ids)

    def read(self, file_path: str, tokenizer, max_seq_len: int, max_number_premises:int=None):
        all_tokens = []
        all_segment_ids = []
        all_labels = []
        
        file_name = file_path.split("/")[-1]
        dir_name = os.path.dirname(file_path)
        cache_file_path =  os.path.join(dir_name, 'cached_concat_roberta_{}_{}_{}'.format(file_name,max_seq_len,max_number_premises))
        if os.path.exists(cache_file_path):
            logger.info("Loading features from cached file %s", cache_file_path)
            features = self.load_cache(cache_file_path)
            all_tokens = features['tokens']
            all_segment_ids = features['segmentids']
            all_masks = features['masks']
            all_labels = features.get('labels',None)
            if all_labels is not None:
                return TensorDataset(all_tokens, all_segment_ids, all_masks, all_labels)
            else:
                return TensorDataset(all_tokens, all_segment_ids, all_masks)
        

        with open(file_path, 'r') as te_file:
            logger.info("Reading MCQ instances for 'bert mcq parallel' from jsonl dataset at: %s", file_path)
            for line in tqdm(te_file,desc="preparing dataset:"):
                if line.strip() == '':
                    continue
                example = json.loads(line)
                label = None
                if "gold_label" in example:
                    label = int(example["gold_label"])

                premises = example["premises"]
                choices = example["choices"]
                question = example["question"] if "question" in example else None
                
                if question is not None:
                    updated_choices =[]
                    for choice in choices:
                        if question not in choice:
                            updated_choices.append(question + choice)
                        else:
                            updated_choices.append(choice)
                    question = None
                    choices = updated_choices
                
                
                pp_tokens, pp_segment_ids = self.text_to_instance(tokenizer, max_seq_len, premises, choices, question,
                                                                  max_number_premises)

                assert len(pp_tokens) == len(pp_segment_ids)
                all_tokens.append(pp_tokens)
                all_segment_ids.append(pp_segment_ids)
                all_labels.append(label)


        all_tokens = torch.tensor(all_tokens, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        if None in all_labels:
            all_labels = None
        else:
            all_labels = torch.tensor(all_labels, dtype=torch.long)
        all_masks = (all_tokens != 0).long().clone().detach()
        
        
        features={}
        features['tokens']=all_tokens 
        features['segmentids']=all_segment_ids 
        features['masks']=all_masks
        features['labels']=all_labels
        self.save_cache(features,cache_file_path)
        
        if all_labels is not None:
            return TensorDataset(all_tokens, all_segment_ids, all_masks, all_labels)
        else:
            return TensorDataset(all_tokens, all_segment_ids, all_masks)
        
        
    def read_json(self,json,tokenizer, max_seq_len: int, max_number_premises:int=None):
        all_tokens = []
        all_segment_ids = []
        all_labels = []
        for row in tqdm(json,desc="Reading Json"):
            premises = row["premise"]
            choices = row["choices"]
            label = row["gold_label"]
            pp_tokens, pp_segment_ids = self.text_to_instance(tokenizer, max_seq_len, premises, choices, None,
                                                                  max_number_premises)

            assert len(pp_tokens) == len(pp_segment_ids)
        
            all_tokens.append(pp_tokens)
            all_segment_ids.append(pp_segment_ids)
            all_labels.append(label)
        all_tokens = torch.tensor(all_tokens, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        if None in all_labels:
            all_labels = None
        else:
            all_labels = torch.tensor(all_labels, dtype=torch.long)
        
        all_masks = (all_tokens != 0).long().clone().detach()
        return TensorDataset(all_tokens, all_segment_ids, all_masks, all_labels)
        
    def read_single_instance(self,premises :str,choices:list, label:int, tokenizer, max_seq_len: int, max_number_premises:int=None):
        all_tokens = []
        all_segment_ids = []
        all_labels = []
        pp_tokens, pp_segment_ids = self.text_to_instance(tokenizer, max_seq_len, premises, choices, None,
                                                                  max_number_premises)

        assert len(pp_tokens) == len(pp_segment_ids)
        
        all_tokens.append(pp_tokens)
        all_segment_ids.append(pp_segment_ids)
        all_labels.append(label)
        all_tokens = torch.tensor(all_tokens, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        if None in all_labels:
            all_labels = None
        else:
            all_labels = torch.tensor(all_labels, dtype=torch.long)
        
        all_masks = (all_tokens != 0).long().clone().detach()
        return TensorDataset(all_tokens, all_segment_ids, all_masks, all_labels)


def main():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    reader = RoBertaMCQConcatReader(debug=True)
    out = reader.read("dummy_data.jsonl", tokenizer, 20)
    print(len(out))
    tokens, segs, masks, labels = out[0]
    print(tokens.size())
    print(tokens)
    print(segs)
    print(masks)
    print(labels.size()) # shoud be 0


if __name__ == "__main__":
    main()
