from typing import Dict, Union

from overrides import overrides
import torch
from pytorch_pretrained_bert.modeling import BertModel

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel
from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("bert_mcq")
class BertForClassification(Model):
    """
    An AllenNLP Model that runs pretrained BERT,
    takes the pooled output, and adds a Linear layer on top.
    If you want an easy way to use BERT for classification, this is it.
    Note that this is a somewhat non-AllenNLP-ish model architecture,
    in that it essentially requires you to use the "bert-pretrained"
    token indexer, rather than configuring whatever indexing scheme you like.


    Parameters
    ----------
    vocab : ``Vocabulary``
    bert_model : ``Union[str, BertModel]``
        The BERT model to be wrapped. If a string is provided, we will call
        ``BertModel.from_pretrained(bert_model)`` and use the result.
    num_labels : ``int``, optional (default: None)
        How many output classes to predict. If not provided, we'll use the
        vocab_size for the ``label_namespace``.
    index : ``str``, optional (default: "bert")
        The index of the token indexer that generates the BERT indices.
    label_namespace : ``str``, optional (default : "labels")
        Used to determine the number of classes if ``num_labels`` is not supplied.
    trainable : ``bool``, optional (default : True)
        If True, the weights of the pretrained BERT model will be updated during training.
        Otherwise, they will be frozen and only the final linear layer will be trained.
    initializer : ``InitializerApplicator``, optional
        If provided, will be used to initialize the final linear layer *only*.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 bert_model: Union[str, BertModel],
                 dropout: float = 0.0,
                 num_labels: int = None,
                 index: str = "bert",
                 label_namespace: str = "labels",
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
        self._index = index
        initializer(self._classification_layer)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                token_type_ids: torch.LongTensor,
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a bert-pretrained token indexer)
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        debug = False
        # shape: batch_size, num_choices, max_len
        input_ids = tokens["tokens"]

        input_mask = (input_ids != 0).long()

        # shape: batch_size*num_choices, max_len
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = input_mask.view(-1, input_mask.size(-1))

        if debug:
            print(f"flat_input_ids = {flat_input_ids}")
            print(f"flat_token_type_ids = {flat_token_type_ids}")
            print(f"flat_attention_mask = {flat_attention_mask}")


        # shape: batch_size*num_choices, hidden_dim
        _, pooled = self.bert_model(input_ids=flat_input_ids,
                                    token_type_ids=flat_token_type_ids,
                                    attention_mask=flat_attention_mask)
        if debug:
            print(f"pooled = {pooled}")

        pooled = self._dropout(pooled)

        if debug:
            print(f"pooled = {pooled}")

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

        output_dict = {"logits": reshaped_logits, "probs": probs}

        if label is not None:
            loss = self._loss(reshaped_logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(reshaped_logits, label)

        return output_dict



    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics