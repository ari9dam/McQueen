from typing import Dict, Optional, List, Any

import torch
import numpy as np
from typing import Dict, Union
from allennlp.common.checks import check_dimensions_match
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from pytorch_pretrained_bert.modeling import BertModel
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum, get_final_encoder_states, replace_masked_values
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules import FeedForward, InputVariationalDropout

@Model.register("bert_mcq_parallel")
class BertMCQParallel(Model):
    '''
       get the pooled output from ph batches of size b,size
       flatten the links tokens and pool the output
       reshape to b,n,n-1,size
       reduce to b,n,size (max pool)
       reduce further to b,size (avg, max pool)
       combine two add linear layer to compute the score
    '''

    def ff(self, input_dim, hidden_dim, output_dim):
        return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(hidden_dim, output_dim))

    def __init__(self,
                 vocab: Vocabulary,
                 bert_model: Union[str, BertModel],
                 dropout: float = 0.0,
                 trainable: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)

        if isinstance(bert_model, str):
            self.bert_model = PretrainedBertModel.load(bert_model)
        else:
            self.bert_model = bert_model

        self.bert_model.requires_grad = trainable
        in_features = self.bert_model.config.hidden_size

        self._dropout = torch.nn.Dropout(p=dropout)

        self._classification_layer = torch.nn.Linear(in_features, 1)

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self._classification_layer)

    def forward(self, # type: ignore
                tokens: Dict[str, torch.LongTensor],
                token_type_ids: torch.LongTensor,
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
                ) -> Dict[str, torch.Tensor]:

        debug = False
        # batch_size, num_of_choices, max_premise_perchoice, L
        input_ids = tokens['tokens']
        # batch_size, L
        input_mask = (input_ids != 0).long()

        # shape: batch_size*num_choices*max_premise_perchoice, max_len
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = input_mask.view(-1, input_mask.size(-1))

        # shape: batch_size*num_choices*max_premise_perchoice, hidden_dim
        _, pooled_ph = self.bert_model(input_ids=flat_input_ids,
                                    token_type_ids=flat_token_type_ids,
                                    attention_mask=flat_attention_mask)

        if debug:
            print(f"input_ids.size() = {input_ids.size()}")
            print(f"token_type_ids.size() = {token_type_ids.size()}")
            print(f"pooled_ph.size() = {pooled_ph.size()}")

        # batch*choice, max_premise_per_choice, hidden_dim
        pooled_ph = pooled_ph.view(-1,input_ids.size(2),pooled_ph.size(-1))

        max_pooled_ph,_ = torch.max(pooled_ph,dim=1,keepdim=False)

        if debug:
            print(f"max_pooled_ph.size() = {max_pooled_ph.size()}")

            max_pooled_ph = self._dropout(max_pooled_ph)

        # apply classification layer
        logits = self._classification_layer(max_pooled_ph)

        # shape: batch_size,num_choices
        reshaped_logits = logits.view(-1, input_ids.size(1))
        if debug:
            print(f"reshaped_logits = {reshaped_logits}")

        probs = torch.nn.functional.softmax(reshaped_logits, dim=-1)

        output_dict = {"logits": reshaped_logits, "probs": probs}

        if label is not None:
            loss = self._loss(reshaped_logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(reshaped_logits, label)

        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self._accuracy.get_metric(reset),
                }