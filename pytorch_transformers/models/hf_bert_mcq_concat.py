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


class BertMCQConcat(BertPreTrainedModel):
    def __init__(self, config):
        super(BertMCQConcat, self).__init__(config)
        self.bert = BertModel(config)
        self._dropout = nn.Dropout(config.hidden_dropout_prob)
        self._classification_layer = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_weights)

    def forward(self,  # type: ignore
                input_ids,      # batch_size, number_of_choices, max_seq_len
                token_type_ids, # batch_size, number_of_choices, max_seq_len
                input_mask,     # batch_size, number_of_choices, max_seq_len
                labels = None):

        debug = False

        # shape: batch_size*num_choices, max_len
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = input_mask.view(-1, input_mask.size(-1))

        if debug:
            print(f"flat_input_ids = {flat_input_ids}")
            print(f"flat_token_type_ids = {flat_token_type_ids}")
            print(f"flat_attention_mask = {flat_attention_mask}")

        # shape: batch_size*num_choices, hidden_dim
        _, pooled = self.bert(input_ids=flat_input_ids,
                                    token_type_ids=flat_token_type_ids,
                                    attention_mask=flat_attention_mask)
        if debug:
            print(f"pooled = {pooled}")
            print(f"labels = {labels}")

        pooled = self._dropout(pooled)

        # apply classification layer
        # shape: batch_size*num_choices, 1
        logits = self._classification_layer(pooled)

        if debug:
            print(f"logits = {logits}")

        # shape: batch_size,num_choices
        reshaped_logits = logits.view(-1, input_ids.size(1))

        if debug:
            print(f"reshaped_logits = {reshaped_logits}")

        probs = torch.nn.functional.softmax(reshaped_logits, dim=-1)

        outputs = (reshaped_logits, probs)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss, reshaped_logits, prob)



