# -*- coding: utf-8 -*-
"""Contains utilities for generating embeddings with use of pre-trained models."""

from typing import List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

class BERTEmbedder:
    """Generates embeddings using a model from the BERT family.
    
    The embedder supports running the processing in batches.
    """

    def __init__(self,
                 pretrained_model_name: str,
                 device: torch.device,
                 batch_size: int):
        """
        Args:
            pretrained_model_name: The name of the model on the HF hub.
            device: Device to run processing on. Note: The embeddings are moved to CPU before
                being returned.
            batch_size: Maximal size of a batch in batch processing.
        """

        self._embedder = AutoModel.from_pretrained(pretrained_model_name).to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self._device = device
        self._batch_size = batch_size

    def obtain_paired_bert_embeddings(self, sentences: List[str]) -> List[torch.Tensor]:
        """Obtains paired BERT embedding for a list of sentences.

        Paired embedding is the output hidden state corresponding to the [CLS] token
        for two sentences passed as input.
        """

        tokenized_sentences = [self._tokenizer.tokenize(sentence) for sentence in sentences]

        input_tokens = [['[CLS]'] + prev_tokens + ['[SEP]'] + next_tokens + ['[SEP]']
                        for prev_tokens, next_tokens
                        in zip(tokenized_sentences, tokenized_sentences[1:])]

        max_length = max(len(tokens) for tokens in input_tokens)

        input_ids = [self._tokenizer.convert_tokens_to_ids(tokens) for tokens in input_tokens]
        input_ids = [ids + [0] * (max_length - len(ids)) for ids in input_ids]

        token_type_ids = [[0] + [0] * (len(prev_tokens) + 1) + [1] * (len(next_tokens) + 1)
                          for prev_tokens, next_tokens
                          in zip(tokenized_sentences, tokenized_sentences[1:])]
        token_type_ids = [ids + [0] * (max_length - len(ids)) for ids in token_type_ids]

        attention_mask = [[1] * len(tokens) + [0] * (max_length - len(tokens))
                          for tokens in input_tokens]

        outputs = self._run_bert_in_batches(input_ids, attention_mask, token_type_ids)

        return [outputs[i, 0] for i in range(len(sentences) - 1)]

    def obtain_bert_embeddings_for_sentences(self, sentences: List[str]) -> List[torch.Tensor]:
        """Obtains BERT embeddings for each sentence in a list of sentences."""

        tokenized_sentences = [self._tokenizer.tokenize(sentence) for sentence in sentences]

        input_tokens = [['[CLS]'] + tokens + ['[SEP]'] for tokens in tokenized_sentences]
        max_length = max(len(tokens) for tokens in input_tokens)

        input_ids = [self._tokenizer.convert_tokens_to_ids(tokens) for tokens in input_tokens]
        input_ids = [ids + [0] * (max_length - len(ids)) for ids in input_ids]

        attention_mask = [[1] * len(tokens) + [0] * (max_length - len(tokens))
                          for tokens in input_tokens]

        outputs = self._run_bert_in_batches(input_ids, attention_mask)

        return [outputs[i][1:len(tokenized_sentence) + 1]
                for i, tokenized_sentence in enumerate(tokenized_sentences)]

    def _run_bert_in_batches(self,
                             input_ids: List[List[int]],
                             attention_mask: List[List[int]],
                             token_type_ids: Optional[List[List[int]]] = None) -> torch.Tensor:

        outputs: List[torch.Tensor] = []

        for i in range(0, len(input_ids), self._max_sentences_in_batch):
            batch_input_ids = input_ids[i:i + self._max_sentences_in_batch]
            batch_attention_mask = attention_mask[i:i + self._max_sentences_in_batch]

            if token_type_ids is not None:
                batch_token_type_ids = token_type_ids[i:i + self._max_sentences_in_batch]
            else:
                batch_token_type_ids = None

            with torch.no_grad():
                model_output = self._embedder(
                    input_ids=torch.tensor(batch_input_ids).to(self._device),
                    attention_mask=torch.tensor(batch_attention_mask).to(self._device),
                    token_type_ids=(torch.tensor(batch_token_type_ids).to(self._device)
                                    if batch_token_type_ids is not None else None)
                )
                batch_outputs = model_output.last_hidden_state.cpu()
                outputs.append(batch_outputs)

        return torch.cat(outputs, dim=0)
        

    def obtain_bert_embeddings(self, bert_tokens: List[str]) -> torch.Tensor:
        """Obtains BERT embeddings for the given BERT tokens."""

        input_tokens = ['[CLS]'] + bert_tokens + ['[SEP]']
        input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)

        with torch.no_grad():
            model_output = self._embedder(torch.tensor([input_ids]).to(self._device))
            outputs = model_output.last_hidden_state.cpu()

        return outputs[0][1:-1]