from __future__ import annotations

import torch
from torch import nn


class QwenDecoder(nn.Module):
    """Frozen Qwen3 causal decoder that accepts multimodal input embeddings."""

    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B", torch_dtype=torch.float16):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=torch_dtype,
        #     trust_remote_code=True,
        # )


        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.config.use_cache = False

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        **generation_kwargs,
    ):
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

