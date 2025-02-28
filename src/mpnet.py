import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any

from transformers.modeling_outputs import (
    MaskedLMOutput,
    BaseModelOutput,
    SequenceClassifierOutput,
    MultipleChoiceModelOutput,
)

import bert_padding
from src.bert_layers.model import FlexBertPreTrainedModel
from src.bert_layers.configuration_bert import FlexBertConfig
from src.bert_layers.embeddings import get_embedding_layer
from src.bert_layers.normalization import get_norm_layer
from src.bert_layers.model import FlexBertPredictionHead
from src.bert_layers.initialization import ModuleType, init_weights

# Import the MPNet encoder
from src.bert_layers.mpnet_encoder import get_mpnet_encoder


class MPNetModel(FlexBertPreTrainedModel):
    """
    MPNet model based on FlexBERT architecture.
    
    This model implements the MPNet paper:
    "MPNet: Masked and Permuted Pre-training for Language Understanding"
    """
    
    def __init__(self, config: FlexBertConfig):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.embeddings = get_embedding_layer(config)
        
        # Encoder
        self.encoder = get_mpnet_encoder(config)
        
        # Final normalization layer if using pre-norm attention
        if config.final_norm:
            self.final_norm = get_norm_layer(config)
        else:
            self.final_norm = None
            
        # Initialize weights
        self.post_init()
        
    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings
        
    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        is_predicted_mask: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        """
        Forward pass for MPNet model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            position_ids: Position IDs for positional embeddings
            is_predicted_mask: Boolean mask indicating tokens in the predicted set vs context
            indices: Indices for unpadded sequence (for unpadded mode)
            cu_seqlens: Cumulative sequence lengths (for unpadded mode)
            max_seqlen: Maximum sequence length (for unpadded mode)
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            BaseModelOutput or tuple: Model outputs
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        embedding_output = self.embeddings(input_ids, position_ids)
        
        # Encoder forward pass
        if hasattr(self.encoder, "forward"):
            encoder_outputs = self.encoder(
                hidden_states=embedding_output,
                attention_mask=attention_mask,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                is_predicted_mask=is_predicted_mask,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            
            sequence_output = encoder_outputs["last_hidden_state"]
            all_hidden_states = encoder_outputs.get("hidden_states", None)
        else:
            # Fallback in case the encoder doesn't support the full interface
            sequence_output = self.encoder(
                embedding_output,
                attention_mask=attention_mask,
            )
            all_hidden_states = None
            
        # Apply final normalization if needed
        if self.final_norm is not None:
            sequence_output = self.final_norm(sequence_output)
            
        if not return_dict:
            return (sequence_output, all_hidden_states)
            
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=all_hidden_states,
            attentions=None,  # MPNet doesn't return attention probabilities by default
        )
            
    def _init_weights(self, module: Optional[nn.Module] = None, reset_params: Optional[bool] = None):
        assert (module is None) != (reset_params is None), "arg module xor reset_params must be specified"
        if module:
            self._init_module_weights(module)
        else:
            assert isinstance(reset_params, bool)
            self.embeddings._init_weights(reset_params=reset_params)
            self.encoder._init_weights(reset_params=reset_params)

            if reset_params and self.config.final_norm:
                self.final_norm.reset_parameters()

    def reset_parameters(self):
        self._init_weights(reset_params=True)


class MPNetForMaskedLM(FlexBertPreTrainedModel):
    """
    MPNet model with a masked language modeling head.
    
    This implements the masked and permuted pre-training task described in the MPNet paper.
    """
    
    def __init__(self, config: FlexBertConfig):
        super().__init__(config)
        self.config = config
        
        # Base MPNet model
        self.mpnet = MPNetModel(config)
        
        # MLM prediction head
        self.head = FlexBertPredictionHead(config)
        
        # Output projection
        if config.tie_word_embeddings:
            decoder_weights = self.mpnet.embeddings.tok_embeddings.weight
        else:
            decoder_weights = nn.Linear(config.hidden_size, config.vocab_size, bias=False).weight
            
        self.decoder = nn.Linear(decoder_weights.size(1), decoder_weights.size(0), bias=config.decoder_bias)
        self.decoder.weight = decoder_weights
        
        # Initialize weights
        self._init_weights(reset_params=False)
        
    def get_output_embeddings(self):
        return self.decoder
        
    def set_output_embeddings(self, new_embeddings):
        self.decoder = new_embeddings
        
    def _init_weights(self, module: Optional[nn.Module] = None, reset_params: Optional[bool] = None):
        assert (module is None) != (reset_params is None), "arg module xor reset_params must be specified"
        if module:
            self._init_module_weights(module)
        else:
            assert isinstance(reset_params, bool)
            self.mpnet._init_weights(reset_params=reset_params)
            self.head._init_weights(reset_params=reset_params)

            # Output weights
            if not self.config.tie_word_embeddings:
                init_weights(self.config, self.decoder, self.config.hidden_size, type_of_module=ModuleType.final_out)
                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        is_predicted_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        """
        Forward pass for masked language modeling with MPNet.
        
        Args:
            input_ids: Input token IDs with masking applied
            attention_mask: Attention mask for padding
            position_ids: Position IDs to track original token positions
            is_predicted_mask: Boolean mask indicating tokens to predict vs context tokens
            labels: Labels for masked LM (-100 for tokens not to predict)
            indices, cu_seqlens, max_seqlen: Unpadded sequence parameters
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            MaskedLMOutput or tuple: Model outputs
        """
        # Get model outputs
        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            is_predicted_mask=is_predicted_mask,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Apply prediction head
        prediction_scores = self.decoder(self.head(sequence_output))
        
        loss = None
        if labels is not None:
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            
            # Flatten the outputs and labels
            prediction_scores = prediction_scores.view(-1, self.config.vocab_size)
            labels = labels.view(-1)
            
            # Only compute loss on labels that are not -100
            masked_positions = labels != -100
            if masked_positions.sum() > 0:
                loss = loss_fct(prediction_scores[masked_positions], labels[masked_positions])
            else:
                loss = prediction_scores.new_tensor(0.0)
                
        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
            
        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=None,  # MPNet doesn't return attention probabilities by default
        )


# Additional models can be implemented as needed (e.g., for sequence classification)
class MPNetForSequenceClassification(FlexBertPreTrainedModel):
    """MPNet for sequence classification tasks."""
    
    def __init__(self, config: FlexBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.mpnet = MPNetModel(config)
        self.dropout = nn.Dropout(config.head_class_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self._init_weights(reset_params=False)
        
    def _init_weights(self, module: Optional[nn.Module] = None, reset_params: Optional[bool] = None):
        assert (module is None) != (reset_params is None), "arg module xor reset_params must be specified"
        if module:
            self._init_module_weights(module)
        else:
            assert isinstance(reset_params, bool)
            self.mpnet._init_weights(reset_params=reset_params)
            init_weights(self.config, self.classifier, self.config.hidden_size, type_of_module=ModuleType.final_out)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        is_predicted_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """Forward pass for sequence classification."""
        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            is_predicted_mask=is_predicted_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # Pool the output - using CLS token (first token)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # Compute loss
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
                
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )