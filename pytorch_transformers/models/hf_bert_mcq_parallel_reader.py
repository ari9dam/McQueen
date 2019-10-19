import itertools
from typing import List
import json
import logging
import torch
from tqdm import tqdm
import pickle
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.utils.data import TensorDataset
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class BertMCQParallelReader:
    
    def __init__(self):
        self.truncated=0
        self.tokenized_map = {}
        
        
    def load_cache(self,path_to_docmap):
        pickle_in = open(path_to_docmap,"rb")
        cached_obj = pickle.load(pickle_in)
        pickle_in.close()
        return cached_obj

    def save_cache(self,obj,fname):
        with open(fname, 'wb+') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _truncate_tokens(self,tokens_a, tokens_b, max_length):
        """
        Truncate a from the end and b from the end until total is less than max_length.
        At each step, truncate the longest one
        """
        truncated=False
        while len(tokens_a) + len(tokens_b) > max_length:
            if len(tokens_a) > 0:
                tokens_a.pop()
                truncated=True
            else:
                tokens_b.pop()
                truncated=True
        if truncated:
            self.truncated+=1
        return tokens_a, tokens_b

    def bert_features_from_qa(self, tokenizer, max_pieces: int, question: str, answer: str, context: str = None):
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        
        question_tokens = self.tokenized_map.get(question,tokenizer.tokenize(question))
        self.tokenized_map[question]=question_tokens
        
        if context is not None:
            context_tokens = self.tokenized_map.get(context,tokenizer.tokenize(context))
            self.tokenized_map[context] = context_tokens
            #Append sep tokens
            question_tokens = context_tokens + [sep_token] + question_tokens
        
        choice_tokens = self.tokenized_map.get(answer,tokenizer.tokenize(answer))
        self.tokenized_map[answer]=choice_tokens
        
        question_tokens, choice_tokens = self._truncate_tokens(question_tokens, choice_tokens, max_pieces - 3)

        tokens = [cls_token] + question_tokens + [sep_token] + choice_tokens + [sep_token]
        segment_ids = list(itertools.repeat(0, len(question_tokens) + 2)) + \
                      list(itertools.repeat(1, len(choice_tokens) + 1))
        return tokens, segment_ids

    def text_to_instance(self,  # type: ignore
                         tokenizer,  # type: ignore
                         max_seq_length: int,
                         premises: List[List[str]],
                         choices: List[str],
                         question: str = None,
                         max_number_premises:int = None):

        tokens = []
        token_type_ids = []

        for premise, hypothesis in zip(premises, choices):
            if isinstance(premises[0], str):
                premises = [premises] * len(choices)
            per_choice_tokens = []
            per_choice_token_ids = []
            # two major keys
            # ph: [cls]all_premise[sep]hypothesis[sep]
            # two different segment_ids
            # join all premise sentences
            if not max_number_premises:
                max_number_premises = len(premise)
            for sentence in premise[0:max_number_premises]:
                if question is None:
                    ph_tokens, ph_token_type_ids = self.bert_features_from_qa(tokenizer, max_seq_length,
                                                                              question=sentence, answer=hypothesis)
                else:
                    ph_tokens, ph_token_type_ids = self.bert_features_from_qa(tokenizer, max_seq_length,
                                                                              question=question, context=sentence,
                                                                              answer=hypothesis)
                # tokenize
                input_ids = tokenizer.convert_tokens_to_ids(ph_tokens)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                ph_token_type_ids += padding
                per_choice_tokens.append(input_ids)
                per_choice_token_ids.append(ph_token_type_ids)

            if max_number_premises == 0 and question is not None:
                ph_tokens, ph_token_type_ids = self.bert_features_from_qa(tokenizer, max_seq_length,
                                                                          question=question, answer=hypothesis)
                input_ids = tokenizer.convert_tokens_to_ids(ph_tokens)
                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding

                ph_token_type_ids += padding
                per_choice_tokens.append(input_ids)
                per_choice_token_ids.append(ph_token_type_ids)

            tokens.append(per_choice_tokens)
            token_type_ids.append(per_choice_token_ids)

        return (tokens, token_type_ids)

    def read(self, file_path: str, tokenizer, max_seq_len: int, max_number_premises:int=None):
        file_name = file_path.split("/")[-1]
        dir_name = os.path.dirname(file_path)
        cache_file_path =  os.path.join(dir_name, 'cached_bert_{}_{}_{}'.format(file_name,max_seq_len,max_number_premises))
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
        
        all_tokens = []
        all_segment_ids = []
        all_labels = []
#         max_number_premises = 0
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
                
                pp_tokens, pp_segment_ids = self.text_to_instance(tokenizer, max_seq_len, premises, choices,
                                                                  question, max_number_premises)
                for per_choice_pp in pp_tokens:
                    max_number_premises = max(max_number_premises, len(per_choice_pp))
                assert len(pp_tokens) == len(pp_segment_ids)
                all_tokens.append(pp_tokens)
                all_segment_ids.append(pp_segment_ids)
                all_labels.append(label)


            # pad and make a tensor
            padding = [0] * max_seq_len
            for pp_tokens, pp_segment_ids in zip(all_tokens, all_segment_ids):
                for per_choice_pp, per_choice_sg_id in zip(pp_tokens, pp_segment_ids):
                    for i in range(0, max_number_premises - len(per_choice_pp)):
                        per_choice_pp.append(padding)
                        per_choice_sg_id.append(padding)
                        
        logger.info("Truncated Pairs:%d",self.truncated)

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
        
    def read_single_instance(self,premise :str,choices:list, label:int, tokenizer, max_seq_len: int, max_number_premises:int=None):
        all_tokens = []
        all_segment_ids = []
        all_labels = []
        pp_tokens, pp_segment_ids = self.text_to_instance(tokenizer, max_seq_len, premises, choices,
                                                                  question, max_number_premises)
        for per_choice_pp in pp_tokens:
            max_number_premises = max(max_number_premises, len(per_choice_pp))
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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    reader = BertMCQParallelReader()
    
    out = reader.read("dummy_data.jsonl", tokenizer, 70, None)
    print(len(out))
    tokens, segs, masks, labels = out[0]
#     print(tokens.size())
#     print(segs)
#     print(masks)
#     print(labels.size()) # shoud be 0


if __name__ == "__main__":
    main()
