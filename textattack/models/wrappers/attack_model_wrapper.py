"""
AttackModelWrapper class
--------------------------

"""
import torch
import transformers

import textattack
from textattack.models.helpers import T5ForTextToText
from textattack.models.tokenizers import T5Tokenizer

from .pytorch_model_wrapper import PyTorchModelWrapper

torch.cuda.empty_cache()


class AttackModelWrapper(PyTorchModelWrapper):
    """A model wrapper queries a model with a list of text inputs.

    Classification-based models return a list of lists, where each sublist
    represents the model's scores for a given input.

    Text-to-text models return a list of strings, where each string is the
    output – like a translation or summarization – for a given input.
    """

    def __init__(self, model, tokenizer):
        # assert isinstance(
        #     model, (transformers.PreTrainedModel, T5ForTextToText)
        # ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        # assert isinstance(
        #     tokenizer,
        #     (
        #         transformers.PreTrainedTokenizer,
        #         transformers.PreTrainedTokenizerFast,
        #         T5Tokenizer,
        #     ),
        # ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)
        # print("text_input_list: ", text_input_list)
        with torch.no_grad():
            outputs = self.model(mlm=False, input_ids=inputs_dict['input_ids'], attention_mask=inputs_dict['attention_mask'], 
                        token_type_ids=inputs_dict['token_type_ids'])

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits
        

    def _tokenize(self, inputs):
        """Helper method for `tokenize`"""
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]


    def tokenize(self, inputs, strip_prefix=False):
        """Helper method that tokenizes input strings
        Args:
            inputs (list[str]): list of input strings
            strip_prefix (bool): If `True`, we strip auxiliary characters added to tokens as prefixes (e.g. "##" for BERT, "Ġ" for RoBERTa)
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        tokens = self._tokenize(inputs)
        if strip_prefix:
            # `aux_chars` are known auxiliary characters that are added to tokens
            strip_chars = ["##", "Ġ", "__"]
            # TODO: Find a better way to identify prefixes. These depend on the model, so cannot be resolved in ModelWrapper.

            def strip(s, chars):
                for c in chars:
                    s = s.replace(c, "")
                return s

            tokens = [[strip(t, strip_chars) for t in x] for x in tokens]

        return tokens
