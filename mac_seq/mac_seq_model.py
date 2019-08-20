from typing import Dict, Optional, List, Any

import torch
import overrides

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules.matrix_attention.dot_product_matrix_attention import DotProductMatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum, get_final_encoder_states, replace_masked_values
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules import FeedForward, InputVariationalDropout

@Model.register("mac_seq_model")
class MultiSentenceNLI(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 projection_feedforward: FeedForward,
                 key_projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 link_key_encoder: Seq2SeqEncoder,
                 key_compare_feedforward: FeedForward,
                 output_feedforward: FeedForward,
                 output_logit: FeedForward,
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super().__init__(vocab, regularizer)
        self.label_map = vocab.get_token_to_index_vocabulary('labels')
        l_map = [None] * len(self.label_map)
        for lb, lb_idx in self.label_map.items():
            l_map[lb_idx] = lb
        self.label_map = l_map

        self._text_field_embedder = text_field_embedder
        self._word_embedding_dimension = text_field_embedder.get_output_dim()
        self._sentence_encoder = encoder
        self._encoded_word_dimension = self._sentence_encoder.get_output_dim()

        self._matrix_attention = DotProductMatrixAttention()
        self._projection_feedforward = projection_feedforward
        self._key_projection_feedforward = key_projection_feedforward


        self._inference_encoder = inference_encoder
        self._link_key_encoder = link_key_encoder
        self._embedded_key_dimension = self._link_key_encoder.get_output_dim()
        self._key_compare_feedforward = key_compare_feedforward

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

        self._output_feedforward = output_feedforward
        self._output_logit = output_logit

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        check_dimensions_match(encoder.get_output_dim() * 4, projection_feedforward.get_input_dim(),
                               "encoder output dim", "projection feedforward input")
        check_dimensions_match(encoder.get_output_dim() * 4, key_projection_feedforward.get_input_dim(),
                               "encoder output dim", "projection feedforward input")
        check_dimensions_match(projection_feedforward.get_output_dim(), inference_encoder.get_input_dim(),
                               "proj feedforward output dim", "inference lstm input dim")
        check_dimensions_match(key_projection_feedforward.get_output_dim(), link_key_encoder.get_input_dim(),
                               "key proj feedforward output dim", "link key lstm input dim")
        check_dimensions_match(key_projection_feedforward.get_output_dim(), link_key_encoder.get_input_dim(),
                               "key proj feedforward output dim", "inference lstm input dim")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                premises: Dict[str, torch.LongTensor],
                choices: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
                ) -> Dict[str, torch.Tensor]:

        debug = False

        batch_size, num_choices, max_number_of_premise_sens, max_premise_len = premises['tokens'].size()
        batch_size, num_choices, max_hyp_len = choices['tokens'].size()

        # shape: (bacth_size, num_choices, max_number_of_premise_sens, max_premise_len, word_embedding)
        embedded_premise = self._text_field_embedder(premises, num_wrapping_dims=2)
        # shape: (bacth_size, num_choices, max_hypothesis_len, word_embedding)
        embedded_hypothesis = self._text_field_embedder(choices, num_wrapping_dims=1)

        if debug:
            print(f" premise['tokens'].size() = {premises['tokens'].size()}")
            print(f" hypothesis['tokens'].size() = {choices['tokens'].size()}")
            print(f" embedded_premise.size() = {embedded_premise.size()}")

        # shape: (bacth_size*choice*max_number_of_premise_phrases, max_premise_phrase_len, word_embedding)
        flatten_embedded_premise = embedded_premise.view(-1, max_premise_len,
                                                            self._word_embedding_dimension)
        flatten_embedded_hypothesis = embedded_hypothesis.view(-1,max_hyp_len, self._word_embedding_dimension)

        #shape: (bacth_size*choices, max_hypothesis_len)
        hypothesis_mask = get_text_field_mask(choices, num_wrapping_dims=1).float()

        flatten_hypothesis_mask = hypothesis_mask.view(-1, max_hyp_len)

        # shape: (bacth_size, choices, max_number_of_premise_sens, max_premise_len)
        premise_mask = get_text_field_mask(premises, num_wrapping_dims=2).float()
        premise_mask = premise_mask.view(-1,max_number_of_premise_sens, max_premise_len)
        # shape: (bacth_size*choice*max_number_of_premise_sens, max_premise_len)
        flatten_premise_mask = premise_mask.view(-1, max_premise_len)
        # shape: (bacth_size, max_number_of_premise_sens*max_premise_len)
        concatenated_premise_mask = premise_mask.view(-1, max_number_of_premise_sens*max_premise_len)

        # apply dropout for LSTM
        if self.rnn_input_dropout:
            flatten_embedded_premise = self.rnn_input_dropout(flatten_embedded_premise)
            flatten_embedded_hypothesis = self.rnn_input_dropout(flatten_embedded_hypothesis)

        # encode premise and hypothesis
        if debug:
            print(f"flatten_embedded_premise.size(): {flatten_embedded_premise.size()}")
            print(f"flatten_premise_mask.size(): {flatten_premise_mask.size()}")
            print(f"flatten_embedded_hypothesis.size(): {flatten_embedded_hypothesis.size()}")
            print(f"flatten_hypothesis_mask.size(): {flatten_hypothesis_mask.size()}")

        flatten_encoded_premise = self._sentence_encoder(flatten_embedded_premise, flatten_premise_mask)
        flatten_encoded_hypothesis = self._sentence_encoder(flatten_embedded_hypothesis, flatten_hypothesis_mask)

        ## create several tensors to help the remaining operations
        encoded_premise = flatten_encoded_premise.view(-1, max_number_of_premise_sens,
                                                          max_premise_len, self._encoded_word_dimension)
        concatenated_encoded_premise = encoded_premise.view(-1, max_number_of_premise_sens*max_premise_len,
                                                               self._encoded_word_dimension)

        if debug:
            print(f"encoded_premise.size(): {encoded_premise.size()}")
            print(f"flatten_encoded_hypothesis.size(): {flatten_encoded_hypothesis.size()}")

        # Shape: (batch_size*numchoices, max_number_of_premise_sens, premise_length, hypothesis_length)
        similarity_matrix_premimse = torch.einsum('bnmd,bpd -> bnmp', [encoded_premise, flatten_encoded_hypothesis])

        # Shape: (batch_size*numchoices, max_number_of_premise_sens, premise_length, hypothesis_length)
        p2h_attention = masked_softmax(similarity_matrix_premimse, flatten_hypothesis_mask)

        if debug:
            print(f"p2h_attention.size(): {p2h_attention.size()}")
            print(f"similarity_matrix_premimse.size(): {similarity_matrix_premimse.size()}")

        # weighted_sum
        # Shape: (batch_size*numchoices, max_number_of_premise_sens, premise_length, encoded_word_embedding)
        attended_hypothesis = torch.sum(torch.einsum('bpsl,bld->bpsld', [p2h_attention, flatten_encoded_hypothesis]),-2)
        attended_hypothesis = attended_hypothesis.view(-1,max_premise_len,self._encoded_word_dimension)

        if debug:
            print(f"attended_hypothesis.size(): {p2h_attention.size()}")

        # Shape: (batch_size*choice, hypothesis_length, premise_length*max_number_of_premise_sens)
        similarity_matrix_hypothesis = self._matrix_attention(flatten_encoded_hypothesis, concatenated_encoded_premise)
        # Shape: (batch_size, hypothesis_length, premise_length*max_number_of_premise_sens)
        h2p_attention = masked_softmax(similarity_matrix_hypothesis, concatenated_premise_mask)
        # Shape: (batch_size, hypothesis_length, encoded_word_embedding)
        attended_premise = weighted_sum(concatenated_encoded_premise, h2p_attention)

        if debug:
            print(f"similarity_matrix_hypothesis.size(): {similarity_matrix_hypothesis.size()}")
            print(f"h2p_attention.size(): {h2p_attention.size()}")

        # the "enhancement" layer
        premise_enhanced = torch.cat(
            (flatten_encoded_premise, attended_hypothesis,
             flatten_encoded_premise - attended_hypothesis,
             flatten_encoded_premise * attended_hypothesis),
            dim=-1
        )
        hypothesis_enhanced = torch.cat(
            (flatten_encoded_hypothesis, attended_premise,
             flatten_encoded_hypothesis - attended_premise,
             flatten_encoded_hypothesis * attended_premise),
            dim=-1
        )


        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.
        projected_premise_keys = self._key_projection_feedforward(premise_enhanced)
        projected_enhanced_premise = self._projection_feedforward(premise_enhanced)
        projected_enhanced_hypothesis = self._projection_feedforward(hypothesis_enhanced)

        if debug:
            print(f"projected_premise_keys.size() = {projected_premise_keys.size()}")
            print(f"projected_enhanced_premise.size() = {projected_enhanced_premise.size()}")
            print(f"projected_enhanced_hypothesis.size() = {projected_enhanced_hypothesis.size()}")

        # Run the inference layer
        if self.rnn_input_dropout:
            projected_premise_keys = self.rnn_input_dropout(projected_premise_keys)
            projected_enhanced_premise = self.rnn_input_dropout(projected_enhanced_premise)
            projected_enhanced_hypothesis = self.rnn_input_dropout(projected_enhanced_hypothesis)
        v_ai = self._inference_encoder(projected_enhanced_premise, flatten_premise_mask)
        v_bi = self._inference_encoder(projected_enhanced_hypothesis, flatten_hypothesis_mask)
        encoded_keys = self._link_key_encoder(projected_premise_keys, flatten_premise_mask)

        if debug:
            print(f"v_ai.size() = {v_ai.size()}")
            print(f"v_bi.size() = {v_bi.size()}")
            print(f"encoded_keys.size() = {encoded_keys.size()}")

        ##########################################################################################
        # link detection layer. The task of this layer is to see how well things are connected
        # link_ij describes how well key_i and key_j are similar
        # link_i describes how well key_i is linked
        # link describes how well normally keys are linked
        ##########################################################################################
        flatten_premise_mask_sum = torch.sum(
            flatten_premise_mask, 1, keepdim=True
        )
        flatten_premise_mask_sum[flatten_premise_mask_sum==0] = 0.0001
        v_a_avg = torch.sum(v_ai * flatten_premise_mask.unsqueeze(-1), dim=1) / flatten_premise_mask_sum
        v_a_avg_reshaped = v_a_avg.view(-1,max_number_of_premise_sens,v_a_avg.size(-1))
        # prepare the mask to get the final score
        reduced_premise_mask,_ = torch.max(premise_mask,dim=-1,keepdim=False)

        one_flatten_premise_mask = flatten_premise_mask.clone()
        one_flatten_premise_mask[:,0] = 1
        # get the final states ( link key )
        embedded_premise_keys = get_final_encoder_states(encoded_keys,one_flatten_premise_mask,bidirectional=False)
        if debug:
            print(f"embedded_premise_keys.size() = {embedded_premise_keys.size()}")
            print(f" self._embedded_key_dimension = { self._embedded_key_dimension }")
        embedded_premise_keys = embedded_premise_keys.view(-1,max_number_of_premise_sens, self._embedded_key_dimension)
        if debug:
            print(f"embedded_premise_keys.size() = {embedded_premise_keys.size()}")
            print(f"reduced_premise_mask.size() = {reduced_premise_mask.size()}")

        #print("The dimension of keys is"+ str(embedded_premise_keys.size()))
        # compute key similarity matrix
        key_similarity_matrix = self._matrix_attention(embedded_premise_keys, embedded_premise_keys)
        # compute mask
        key_mask = torch.bmm(reduced_premise_mask.unsqueeze(2), reduced_premise_mask.unsqueeze(1))
        cuda_device = self._get_prediction_device()
        diagonal = torch.eye(max_number_of_premise_sens).cuda(cuda_device).unsqueeze(0)
        key_mask = key_mask - diagonal
        key_mask = torch.nn.functional.relu(key_mask)

        if debug:
            print(f"key_similarity_matrix.size() = {key_similarity_matrix.size()}")
            print(f"key_mask = { key_mask.size() }")

        # compute masked softmax
        # Shape: (batch_size*choice, max_number_of_premise_sens, max_number_of_premise_sens)
        key2key_attention = masked_softmax(key_similarity_matrix, key_mask)
        # compute attended key
        # Shape: (batch_size*choice,max_number_of_premise_sens, _embedded_key_dimension)
        attended_keys = weighted_sum(embedded_premise_keys, key2key_attention)
        if debug:
            print(f"key2key_attention = {key2key_attention.size()}")
            print(f"key_mask.size() = {key_mask.size()}")
            print(f"key_similarity_matrix.size() = {key_similarity_matrix.size()}")


        # compute enhanced key
        links_enhanced = torch.cat(
            (v_a_avg_reshaped, embedded_premise_keys, attended_keys),
            dim=-1
        )

        # compute projection
        projected_enhanced_links = self._key_compare_feedforward(links_enhanced)
        # The pooling layer -- max and avg pooling.
        # (batch_size, model_dim)
        link_max, _ = replace_masked_values(
            projected_enhanced_links, reduced_premise_mask.unsqueeze(-1), -1e7
        ).max(dim=1)

        v_b_max, _ = replace_masked_values(
            v_bi, flatten_hypothesis_mask.unsqueeze(-1), -1e7
        ).max(dim=1)
        reduced_premise_mask = torch.sum(
            reduced_premise_mask, 1, keepdim=True
        )
        reduced_premise_mask[reduced_premise_mask==0] = 0.0001
        link_avg = torch.sum(projected_enhanced_links * reduced_premise_mask.unsqueeze(-1), dim=1) / reduced_premise_mask

        v_b_avg = torch.sum(v_bi * flatten_hypothesis_mask.unsqueeze(-1), dim=1) / torch.sum(
            flatten_hypothesis_mask, 1, keepdim=True
        )

        # Now concat
        # (batch_size, model_dim * 2 * 4)
        v_all = torch.cat(( link_max, link_avg, v_b_avg, v_b_max), dim=-1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        if self.dropout:
            v_all = self.dropout(v_all)

        if debug:
            print(f"links_enhanced = {links_enhanced.size()}")
            print(f"projected_enhanced_links.size() = {projected_enhanced_links.size()}")
            print(f"v_b_max.size() = {v_b_max.size()}")
            print(f"v_b_avg.size() = {v_b_avg.size()}")
            print(f"link_avg.size() = {link_avg.size()}")
            print(f"link_max.size() = {link_max.size()}")
            print(f"_output_feedforward.get_input_dim() = {self._output_feedforward.get_input_dim()}")


        output_hidden = self._output_feedforward(v_all)

        logits = self._output_logit(output_hidden)

        # shape: batch_size,num_choices
        reshaped_logits = logits.view(-1, num_choices)
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

