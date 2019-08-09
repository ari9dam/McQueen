from typing import Iterator, List, Dict, Union
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import WordTokenizer
from allennlp.common.file_utils import cached_path
import json
import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("mac_seq_dataset_reader")
class MACSeqDatasetReader(DatasetReader):

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = WordTokenizer()

    def text_to_instance(self,  # type: ignore
                         premises: Union[List[str], List[List[str]]],
                         choices: List[str],
                         label: int = None) -> Instance:

        number_of_choices = len(choices)
        if isinstance(premises[0], str):
            premises = [premises] * number_of_choices

        # create an empty dictionary to store the input
        fields: Dict[str, Field] = {}
        all_premises = []
        all_choices = []
        for premise, hypothesis in zip(premises, choices):

            # hypothesis is a sentence, tokentize it to get List[Token]
            tokenized_hypothesis = self._tokenizer.tokenize(hypothesis)


            # create a ListField for premise since  it is a list of sentences
            tokenized_premises_field = []
            for premise_sentence in premise:
                tokenized_premises_field.append(TextField(self._tokenizer.tokenize(premise_sentence),
                                                          self._token_indexers))

            all_premises.append(ListField(tokenized_premises_field))

            #create a simple textfield for hypothesis
            all_choices.append(TextField(tokenized_hypothesis, self._token_indexers))

        if label is not None:
            fields['label'] = LabelField(label,skip_indexing=True)

        fields['premises'] = ListField(all_premises)
        fields['choices'] = ListField(all_choices)


        return Instance(fields)


    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as te_file:
            logger.info("Reading MAC seq instances from jsonl dataset at: %s", file_path)
            for line in te_file:
                example = json.loads(line)
                label = None
                if "gold_label" in example:
                    label = example["gold_label"]
                if label == '-':
                    # These were cases where the annotators disagreed; we'll just skip them.  It's
                    # like 800 out of 500k examples in the snli training data.
                    continue

                premise = example["premises"]
                hypothesis = example["choices"]
                if len(premise)==0:
                    print("empty")
                    print(line)
                    continue

                yield self.text_to_instance(premise, hypothesis, label)

def main():
    print("testing data reader")
    reader = MACSeqDatasetReader()
    instance = reader.text_to_instance(["After doing so, ask the children what characteristics the non rolling objects have that the rolling objects don't have.","Let children experiment by rolling various objects down their ramps.","Lecture Demos: Roll different objects down a ramp again.","We then revisit the expression for the acceleration of an object rolling down a ramp.","The objects that do not roll should be put aside.","For example, children get to roll various objects down a ramp, to discover which object rolls best.","What objects were the most fun to watch as they rolled down the ramp?","If I want my students to understand that a rolling ball can be a force upon another object which would cause that object to move, I can position a block at the bottom of my ramp and ask, \"I wonder what will happen to the block when I roll my ball down the ramp?\"","Roll different objects down a ramp.","Experiment with balls and other objects to see which will roll down the ramps."], "Mary hates Mia.")
    print(instance)
    print(instance.__getitem__("metadata").metadata)


#main()