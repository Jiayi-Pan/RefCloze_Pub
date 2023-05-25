from torch import nn
import torch

class ModelForMaskedLM(nn.Module):
    def __init__(self, raw_model):
        super(ModelForMaskedLM, self).__init__()
        self.model = raw_model
        self.model.backbone = None
        return raw_model.transformer.tokenizer
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        batch_size = input_ids.shape[0]

        query_spans = []

        # the obj query embeddings
        this_obj_query = self.model.query_embed.weight.repeat(batch_size, 1, 1)
        query_embed = this_obj_query.clone()
        query_spans.append({"type": "obj_query", "span": this_obj_query.shape[1]})

        encoded_text = self.text_encoder(**deepcopy(tokenized))
