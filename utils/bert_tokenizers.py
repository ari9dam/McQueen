from allennlp.data.token_indexers.wordpiece_indexer import WordpieceIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from pytorch_pretrained_bert import BertTokenizer
from overrides import overrides
from typing import List
from allennlp.data.token_indexers.token_indexer import TokenIndexer

@Tokenizer.register("bert-multinli")
class BertMultiNLITokenizer(Tokenizer):
    def __init__(self, pretrained_model: str):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(token) for token in self.tokenizer.tokenize(text)]

@TokenIndexer.register("bert-multinli")
class BertMultiNLITokenIndexer(WordpieceIndexer):
    def __init__(self,
                  pretrained_model: str,
                 max_pieces: int = 512) -> None:
        bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        print("In BertMultiNLITokenIndexer: loaded")
        super().__init__(vocab=bert_tokenizer.vocab,
                         wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                         max_pieces=max_pieces,
                         namespace="bert",
                         separator_token="[SEP]")