from overrides import overrides
from typing import Iterator, List, Dict,Union
import itertools
from allennlp.data.dataset_readers import DatasetReader
import numpy as np
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ListField, SequenceLabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.common.file_utils import cached_path
import json
import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

@DatasetReader.register("bert_mcq_parallel")
class BertMCQParallelReader(DatasetReader):

    def __init__(self, tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 max_pieces: int = 512) -> None:
        super().__init__(lazy=True)
        self._token_indexers = token_indexers
        self._tokenizer = tokenizer
        self._max_pieces = max_pieces

    @staticmethod
    def _truncate_tokens(tokens_a, tokens_b, max_length):
        """
        Truncate a from the start and b from the end until total is less than max_length.
        At each step, truncate the longest one
        """
        while len(tokens_a) + len(tokens_b) > max_length:
            if len(tokens_a)>0:
                tokens_a.pop(0)
            else:
                tokens_b.pop()
        return tokens_a, tokens_b

    def bert_features_from_qa(self, question: str, answer: str, context: str = None):
        cls_token = Token("[CLS]")
        sep_token = Token("[SEP]")
        question_tokens = self._tokenizer.tokenize(question)
        if context is not None:
            context_tokens = self._tokenizer.tokenize(context)
            question_tokens = context_tokens + [sep_token] + question_tokens
        choice_tokens = self._tokenizer.tokenize(answer)
        question_tokens, choice_tokens = self._truncate_tokens(question_tokens, choice_tokens, self._max_pieces - 3)

        tokens = [cls_token] + question_tokens + [sep_token] + choice_tokens + [sep_token]
        segment_ids = list(itertools.repeat(0, len(question_tokens) + 2)) + \
                      list(itertools.repeat(1, len(choice_tokens) + 1))
        return tokens, segment_ids

    @overrides
    def text_to_instance(self,  # type: ignore
                         premises: Union[List[str],List[List[str]]],
                         choices: List[str],
                         label: int = None,
                         question: str = None) -> Instance:
        number_of_choices = len(choices)
        if isinstance(premises[0],str):
            premises = [premises]*number_of_choices

        # create an empty dictionary to store the input
        fields: Dict[str, Field] = {}
        tokens = []
        token_type_ids = []

        for premise, hypothesis in zip(premises, choices):

            per_choice_tokens = [ ]
            per_choice_token_ids = []
            # two major keys
            # ph: [cls]all_premise[sep]hypothesis[sep]
            # two different segment_ids
            # join all premise sentences
            for sentence in premise:
                if question is None:
                    ph_tokens, ph_token_type_ids = self.bert_features_from_qa(question=sentence,answer=hypothesis)
                else:
                    ph_tokens, ph_token_type_ids = self.bert_features_from_qa(question=question, context=sentence,
                                                                              answer=hypothesis)
                # create a simple textfield for hypothesis
                tokens_field = TextField(ph_tokens, self._token_indexers)
                per_choice_tokens.append(tokens_field)
                per_choice_token_ids.append(SequenceLabelField(ph_token_type_ids, tokens_field))

            if len(premise)==0 and question is not None:
                ph_tokens, ph_token_type_ids = self.bert_features_from_qa(question=question, answer=hypothesis)
                # create a simple textfield for hypothesis
                tokens_field = TextField(ph_tokens, self._token_indexers)
                per_choice_tokens.append(tokens_field)
                per_choice_token_ids.append(SequenceLabelField(ph_token_type_ids, tokens_field))

            tokens.append(ListField(per_choice_tokens))
            token_type_ids.append(ListField(per_choice_token_ids))

        if label is not None:
            fields['label'] = LabelField(label,skip_indexing=True)

        fields['tokens'] = ListField(tokens)
        fields['token_type_ids'] = ListField(token_type_ids)

        return Instance(fields)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as te_file:
            logger.info("Reading MCQ instances for 'bert mcq parallel' from jsonl dataset at: %s", file_path)
            for line in te_file:
                if line.strip()=='':
                    continue
                example = json.loads(line)
                label = None
                if "gold_label" in example:
                    label = example["gold_label"]
                if label == '-':
                    # These were cases where the annotators disagreed; we'll just skip them.  It's
                    # like 800 out of 500k examples in the snli training data.
                    continue

                premises = example["premises"]
                choices = example["choices"]
                question = example["question"] if "question" in example else None

                yield self.text_to_instance(premises, choices, label,question)
