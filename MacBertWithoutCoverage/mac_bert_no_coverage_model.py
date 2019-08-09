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



@Model.register("multi_bert_mcq_model")
class BERTMultiSentenceNLI(Model):
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
                 two_bert: bool = False,
                 projection_dim: int = 300,
                 label_namespace: str = "labels",
                 trainable: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)

        if isinstance(bert_model, str):
            self.bert_model = PretrainedBertModel.load(bert_model,not two_bert)
            self.link_model = PretrainedBertModel.load(bert_model)
        else:
            self.bert_model = bert_model
            self.link_model = bert_model

        self.bert_model.requires_grad = trainable
        self._projection_dim = projection_dim
        in_features = self.bert_model.config.hidden_size + 2*self._projection_dim

        self._dropout = torch.nn.Dropout(p=dropout)

        self._classification_layer = torch.nn.Linear(in_features, 1)
        self._projection_layer = torch.nn.Sequential(torch.nn.Linear(self.bert_model.config.hidden_size,
                                                                     self._projection_dim),
                                   torch.nn.ReLU())

        #self._classification_layer = self.ff(in_features, self.bert_model.config.hidden_size, 1)

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self._classification_layer)

    def forward(self, # type: ignore
                tokens: Dict[str, torch.LongTensor],
                token_type_ids: torch.LongTensor,
                links_tokens: Dict[str, torch.LongTensor],
                links_token_type_ids:torch.LongTensor,
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
                ) -> Dict[str, torch.Tensor]:
        debug = False
        # batch_size, L
        input_ids = tokens['tokens']
        # batch_size, L
        input_mask = (input_ids != 0).long()

        # shape: batch_size*num_choices, max_len
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = input_mask.view(-1, input_mask.size(-1))

        # shape: batch_size*num_choices, hidden_dim
        _, pooled_ph = self.bert_model(input_ids=flat_input_ids,
                                    token_type_ids=flat_token_type_ids,
                                    attention_mask=flat_attention_mask)

        # batch_size,num_of_choices, N_P, N_P-1,L1
        links = links_tokens['tokens']
        # batch_size,num_of_choices, N_P, N_P-1,L1
        link_type_ids = links_token_type_ids
        batch_size, num_of_choices, max_premise,c, link_len = links.size()
        links = links.view(-1, link_len)
        link_type_ids = link_type_ids.view(-1, link_len)
        link_mask = (links != 0).long()
        # batch_size*N_P* N_P-1*choices,  CLS embedding
        _, pooled_links = self.link_model(input_ids=links,
                                       token_type_ids=link_type_ids,
                                       attention_mask=link_mask)

        projected_links = self._projection_layer(pooled_links)
        projected_links = projected_links.view(batch_size*num_of_choices, max_premise,max_premise-1,-1)

        reduced_link_mask,_ = torch.max(link_mask.view(batch_size*num_of_choices,max_premise,
                                                        max_premise-1,-1),dim=-1,keepdim=False)

        # batch_size, N_p, CLS embedding
        link_maxpooled, _ = replace_masked_values(
            projected_links, reduced_link_mask.unsqueeze(-1), -1e7
        ).max(dim=-2)
        print(link_maxpooled.size())
        # batch_size, N_P
        average_link_mask,_ = torch.max(reduced_link_mask,dim=-1,keepdim=False)
        average_link_mask[average_link_mask==0] = 0.001
        average_link_mask = average_link_mask.double().float().cuda()
        print(average_link_mask.size())
        link_max_summary,_ = replace_masked_values(
            link_maxpooled, average_link_mask.unsqueeze(-1), -1e7
        ).max(dim=-2)
        link_avg_summary = torch.sum(link_maxpooled * average_link_mask.unsqueeze(-1), -2) / torch.sum(
            average_link_mask, dim=1, keepdim=True)

        # compute enhanced key
        if debug:
            print(pooled_ph.size())
            print(link_max_summary.size())
            print(link_avg_summary.size())
        pooled = torch.cat(
            (pooled_ph, link_max_summary, link_avg_summary),
            dim=-1
        )

        pooled = self._dropout(pooled)

        # apply classification layer
        logits = self._classification_layer(pooled)

        # shape: batch_size,num_choices
        reshaped_logits = logits.view(-1, num_of_choices)
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