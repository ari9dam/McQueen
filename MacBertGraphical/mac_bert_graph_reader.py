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


# from allennlp.data.token_indexers.wordpiece_indexer import WordpieceIndexer
# from allennlp.data.tokenizers import Token, Tokenizer
# from pytorch_pretrained_bert import BertTokenizer
#
#
# class BertMultiNLITokenizer(Tokenizer):
#     def __init__(self, pretrained_model: str):
#         self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
#         print("In bert-multimli tokenizer: loaded")
#
#     @overrides
#     def tokenize(self, text: str) -> List[Token]:
#         return [Token(token) for token in self.tokenizer.tokenize(text)]
#
#
#
# class BertMultiNLITokenIndexer(WordpieceIndexer):
#     def __init__(self,
#                  pretrained_model: str,
#                  max_pieces: int = 512) -> None:
#         bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model)
#         print("In BertMultiNLITokenIndexer: loaded")
#         super().__init__(vocab=bert_tokenizer.vocab,
#                          wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
#                          max_pieces=max_pieces,
#                          namespace="bert",
#                          separator_token="[SEP]")


@DatasetReader.register("mac_bert_graph_mcq")
class MACReader(DatasetReader):

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
                         coverage: List[List[List[float]]],
                         label: int = None) -> Instance:
        number_of_choices = len(choices)
        if isinstance(premises[0],str):
            premises = [premises]*number_of_choices

        # create an empty dictionary to store the input
        fields: Dict[str, Field] = {}
        all_nodes = []
        all_nodes_token_ids = []
        all_links = []
        all_link_token_ids = []

        if len(coverage) != len(choices):

            logger.error("the dimension of coverage and choices did not match")
            exit(0)

        max_len = 0
        max_premises = 0
        for arr,p in zip(coverage,premises):

            if len(arr) != len(p):
                logger.error("the dimension of coverage and premises did not match")
                exit(0)
            max_premises = max([max_premises,len(p)])
            max_len = max([max_len,max([len(a) for a in arr])])

        # padding
        np_coverage = np.zeros([len(coverage),max_premises,max_len])
        for c_idx in range(len(coverage)):
            for p_idx in range(len(coverage[c_idx])):
                np_coverage[c_idx,p_idx,0:len(coverage[c_idx][p_idx])] = coverage[c_idx][p_idx]

        fields['coverage'] = ArrayField(np_coverage)

        for premise, hypothesis in zip(premises, choices):

            # two major keys
            # ph: [cls]all_premise[sep]hypothesis[sep]
            # two different segment_ids
            # join all premise sentences
            nodes = []
            node_ids = []
            for sen in premise:
                node_tokens, node_token_type_ids = self.bert_features_from_qa(question=sen,
                                                                          answer=hypothesis)
                node_tokens_field = TextField(node_tokens, self._token_indexers)
                nodes.append(node_tokens_field)
                node_ids.append(SequenceLabelField(node_token_type_ids, node_tokens_field))

            all_nodes.append(ListField(nodes))
            all_nodes_token_ids.append(ListField(node_ids))
            links_segment_2d = []
            links_2d = []

            for i in range(0, len(premise)):
                tokenized_links_field = []
                type_ids_of_links = []
                for j in range(0, len(premise)):
                    if i == j:
                        continue
                    else:
                        pp_tokens, pp_token_type_ids = self.bert_features_from_qa(question=premise[i],
                                                                            answer=hypothesis, context=premise[j])
                        pp_tokens_field = TextField(pp_tokens, self._token_indexers)
                        tokenized_links_field.append(pp_tokens_field)
                        type_ids_of_links.append(SequenceLabelField(pp_token_type_ids, pp_tokens_field))
                links_2d.append(ListField(tokenized_links_field))
                links_segment_2d.append(ListField(type_ids_of_links))

            if len(premise) >= 2:
                all_links.append(ListField(links_2d))
                all_link_token_ids.append(ListField(links_segment_2d))
            else:
                # add an empty list field
                empty_tokens_field = [TextField([], self._token_indexers)]
                empty_type_ids_of_links = [SequenceLabelField([], empty_tokens_field[0])]
                all_links.append(ListField(ListField(empty_tokens_field)))
                all_link_token_ids.append(ListField(ListField(empty_type_ids_of_links)))

        if label is not None:
            fields['label'] = LabelField(label,skip_indexing=True)

        fields['node_tokens'] = ListField(all_nodes)
        fields['node_token_type_ids'] = ListField(all_nodes_token_ids)
        fields['links_tokens'] = ListField(all_links)
        fields['links_token_type_ids'] = ListField(all_link_token_ids)

        return Instance(fields)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as te_file:
            logger.info("Reading MCQ instances from jsonl dataset at: %s", file_path)
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
                coverage = example["coverage"]

                yield self.text_to_instance(premises, choices, coverage, label)


def main():
    print("testing data reader")
    reader = MACReader()
    instance = reader.text_to_instance(["Phillamon sit on the snow-covered bench.", "Mary Loves Mia."], ["hey","hi"])

    print(instance)
    print({k: v.__class__.__name__ for k, v in instance.fields.items()})
    instances = reader.read("AppendeOverlapMatrix/coverage_mcq_abductive_train.jsonl")
    all_instance_fields_and_types: List[Dict[str, str]] = [{k: v.__class__.__name__
                                                            for k, v in x.fields.items()}
                                                           for x in instances]
    print(all_instance_fields_and_types)
    # Check all the field names and Field types are the same for every instance.
    if not all([all_instance_fields_and_types[0] == x for x in all_instance_fields_and_types]):
        print("You cannot construct a Batch with non-homogeneous Instances.")

#main()


