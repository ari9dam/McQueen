from overrides import overrides
import json
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import numpy as np
from typing import List
from allennlp.common.util import JsonDict, sanitize

@Predictor.register('mcqueen_only_answer_predictor')
class McQueenOnlyAnswerPredictor(Predictor):

    def predict(self, source: str) -> JsonDict:
        print("called predict()")
        return self.predict_json(json.loads(source))

    @overrides
    def _json_to_instance(self, example: JsonDict) -> Instance:
        print("called _json_to_instance()")
        premises = example["premises"]
        choices = example["choices"]
        coverage = example["coverage"]
        question = example["question"] if "question" in example else None

        if coverage is None and question is None:
            return self._dataset_reader.text_to_instance(premises,choices)
        if coverage is None and question is not None:
            return self._dataset_reader.text_to_instance(premises, choices, question = question)
        if coverage is not None and question is None:
            return self._dataset_reader.text_to_instance(premises, choices, coverage = coverage)
        return self._dataset_reader.text_to_instance(premises, choices, coverage = coverage, question=question)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        print("called predict_json")
        instance = self._my_json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance)

        answer_index = np.argmax(outputs['probs']) + 1
        return {"answer_index":str(answer_index)}

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        print("called batch predict_json")
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)