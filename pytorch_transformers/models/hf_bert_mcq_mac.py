# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import

import logging
import torch
from torch import nn


from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BertMCQMAC(BertPreTrainedModel):
    def __init__(self, config):
        super(BertMCQMAC, self).__init__(config)
        config.output_attentions = True
        self.bert = BertModel(config)
        self._dropout = nn.Dropout(config.hidden_dropout_prob)
        self._classification_layer = nn.Linear(config.hidden_size, 1)
        self._key_components_detection_layer = nn.Linear(3*config.hidden_size, 1)
        self._attention_layer = 13
        self._attention_head = 4
        self.apply(self.init_weights)

    def forward(self,  # type: ignore
                input_ids,
                token_type_ids,
                input_mask,
                labels = None):

        debug = False

        # shape: batch_size*num_choices*max_premise_perchoice, max_len
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = input_mask.view(-1, input_mask.size(-1))

        # shape: batch_size*num_choices*max_premise_perchoice, ...,...
        top_layer, pooled_ph, all_attentions = self.bert(input_ids=flat_input_ids,
                                       token_type_ids=flat_token_type_ids,
                                       attention_mask=flat_attention_mask)

        # apply weighting layer
        ind_weights = self._classification_layer(pooled_ph)
        ind_weights = ind_weights.view(-1, input_ids.size(2)) # batch_size*num_choices, max_premise_perchoice
        ind_weights = torch.nn.functional.softmax(ind_weights, dim=-1)

        # reshape pooled_ph
        pooled_ph = pooled_ph.view(-1, input_ids.size(2), pooled_ph.size(1))

        # compute link strengths
        # step 1: detect the words that are present in hyp
        # shape: batch_size*num_choices*max_premise_perchoice,max_seq_len,max_seq_len
        cross_segment_attentions = all_attentions[self._attention_layer].transpose(0,1)[self._attention_head]
        cross_segment_attentions = cross_segment_attentions * flat_token_type_ids.double().float().unsqueeze(1)
        # batch_size*num_choices*max_premise_perchoice,max_seq_len
        cross_segment_attentions,_ = cross_segment_attentions.max(dim=-1)
        cross_segment_attentions = 1.0 - cross_segment_attentions
        # step 2: detect link words
        # step 3: create link description
        # compute link strength
        if debug:
            print(f"key_word_weights.size() = {cross_segment_attentions.size()}")
        # keep only weights for context or segement 0
        segment_0_mask = flat_attention_mask * (1.0-flat_attention_mask)
        key_word_weights = cross_segment_attentions * segment_0_mask.float()
        if debug:
            print(f"key_word_weights.size() = {key_word_weights.size()}")
        # shape: batch*num_choice*max_premise, max_seq_len, hidden_dim
        top_layer = top_layer * key_word_weights.unsqueeze(-1)
        if debug:
            print(f"top_layer.size() = {top_layer.size()}")

        # compute key,
        # many possibilities exists
        # e.g. sum the vecors or run a lstm and get the final states
        # batch_size*num_choices*max_premise_perchoice, key_size
        keys = torch.sum(top_layer,1)
        if debug:
            print(f"keys.size() = {keys.size()}")
        # reshape: batch_size*num_choices, max_premise_per_choice, key_size
        keys = keys.view(-1, input_ids.size(-2), keys.size(-1))
        if debug:
            print(f"keys.size() = {keys.size()}")
        # shape: batch_size*num_choices, max_premise_per_choice, max_premise_per_choice
        link_strength = torch.einsum('bpd,bcd -> bpc', [keys, keys])
        # fill diagonal with zero and normalize link strength
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        link_strength = link_strength * ((1 - torch.eye(link_strength.size(-1), device=device)).unsqueeze(0))
        link_strength = link_strength.view(link_strength.size(0),-1)
        link_strength = torch.nn.functional.softmax(link_strength, dim=-1).view(link_strength.size(0),
                                                                                input_ids.size(-2),input_ids.size(-2))
        # shape: batch_size*num_choices, max_premise_per_choice
        link_strength_max,_ = torch.max(link_strength,-1)
        # weights = ind_weights * (1 + link_strength_max)
        weights = ind_weights -  (1 - ind_weights)* link_strength_max
        if debug:
            print(f"input_ids.size() = {input_ids.size()}")
            print(f"token_type_ids.size() = {weights.size()}")
            print(f"link_strength_max.size() = {link_strength_max.size()}")
            print(f"link_strength.size() = {link_strength.size()}")

        # multiply each element by the corresponding scores
        weighted_ph = pooled_ph * weights.unsqueeze(-1)

        #reshape: batch*num_choices, hidden_dim
        weighted_ph = torch.sum(weighted_ph,1)

        #apply classification layer
        logits = self._classification_layer(weighted_ph)

        if debug:
            print(f"logits.size() = {logits.size()}")

        # shape: batch_size,num_choices
        reshaped_logits = logits.view(-1, input_ids.size(1))
        if debug:
            print(f"reshaped_logits = {reshaped_logits}")

        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss, reshaped_logits, prob)



