from overrides import overrides
from typing import Iterator, List, Dict,Union
import itertools
from allennlp.data.dataset_readers import DatasetReader

from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ListField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.common.file_utils import cached_path
import json
import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer


@DatasetReader.register("bert_mcq_reader")
class BERTMCQDatasetReader(DatasetReader):

    def __init__(self, tokenizer: Tokenizer=None,
                 token_indexers: Dict[str, TokenIndexer]=None) -> None:
        super().__init__(lazy=True)
        self._token_indexers = token_indexers
        self._tokenizer = tokenizer


    def bert_features_from_qa(self, question: str, answer: str, context: str = None):
        cls_token = Token("[CLS]")
        sep_token = Token("[SEP]")
        question_tokens = self._tokenizer.tokenize(question)
        if context is not None:
            context_tokens = self._tokenizer.tokenize(context)
            question_tokens = context_tokens + [sep_token] + question_tokens
        choice_tokens = self._tokenizer.tokenize(answer)

        tokens = [cls_token] + question_tokens + [sep_token] + choice_tokens + [sep_token]
        segment_ids = list(itertools.repeat(0, len(question_tokens) + 2)) + \
                      list(itertools.repeat(1, len(choice_tokens) + 1))
        return tokens, segment_ids

    @overrides
    def text_to_instance(self,  # type: ignore
                         premises: Union[List[str],List[List[str]]],
                         choices: List[str],
                         label: int = None) -> Instance:
        number_of_choices = len(choices)
        if isinstance(premises[0],str):
            premises = [premises]*number_of_choices

        # create an empty dictionary to store the input
        fields: Dict[str, Field] = {}
        tokens = []
        token_type_ids = []

        for premise, hypothesis in zip(premises,choices):

            # two major keys
            # ph: [cls]all_premise[sep]hypothesis[sep]
            # two different segment_ids


            # join all premise sentences
            all_premise = " ".join(premise)
            ph_tokens, ph_token_type_ids = self.bert_features_from_qa(question=all_premise,answer=hypothesis)

            #create a simple textfield for hypothesis
            tokens_field = TextField(ph_tokens, self._token_indexers)
            tokens.append(tokens_field)
            token_type_ids.append(SequenceLabelField(ph_token_type_ids, tokens_field))

        if label is not None:
            fields['label'] = LabelField(label,skip_indexing=True)

        fields['tokens']  = ListField(tokens)
        fields['token_type_ids'] = ListField(token_type_ids)

        return Instance(fields)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as te_file:
            logger.info("Reading multi-sentence textual entailment instances from jsonl dataset at: %s", file_path)
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

                yield self.text_to_instance(premises, choices, label)


def main():
    print("testing data reader")
    reader = BERTMCQDatasetReader()
    instance = reader.text_to_instance(["Phillamon sit on the snow-covered bench.", "Mary Loves Mia."], ["hey","hi"])

    print(instance)
    print({k: v.__class__.__name__ for k, v in instance.fields.items()})
    instances = reader.read("AbductiveNLI/big_mcq_abductive_train.jsonl")
    all_instance_fields_and_types: List[Dict[str, str]] = [{k: v.__class__.__name__
                                                            for k, v in x.fields.items()}
                                                           for x in instances]
    print(all_instance_fields_and_types)
    # Check all the field names and Field types are the same for every instance.
    if not all([all_instance_fields_and_types[0] == x for x in all_instance_fields_and_types]):
        print("You cannot construct a Batch with non-homogeneous Instances.")

#main()


