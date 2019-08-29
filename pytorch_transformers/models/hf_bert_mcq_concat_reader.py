import itertools
from typing import List
import json
import logging
import torch
from tqdm import tqdm
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class BertMCQConcatReader:
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

    def bert_features_from_qa(self, tokenizer, max_pieces: int, question: str, answer: str, context: str = None):
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        question_tokens = tokenizer.tokenize(question)
        if context is not None:
            context_tokens = tokenizer.tokenize(context)
            question_tokens = context_tokens + [sep_token] + question_tokens
        choice_tokens = tokenizer.tokenize(answer)
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
                         max_number_premises=None):
        debug = False
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
            if debug:
                print(f"Premise: {premise}, Hypothesis :{hypothesis}")
                print(f"Concatenated Premise: {concatenated_premise}")
                print(f"Tokens : {ph_tokens}, TokenIds: {ph_token_type_ids}")
            # tokenize
            input_ids = tokenizer.convert_tokens_to_ids(ph_tokens)
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
        if all_labels is not None:
            return TensorDataset(all_tokens, all_segment_ids, all_masks, all_labels)
        else:
            return TensorDataset(all_tokens, all_segment_ids, all_masks)


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    reader = BertMCQConcatReader()
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
