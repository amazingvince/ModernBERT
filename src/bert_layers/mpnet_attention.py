import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union

from src.bert_layers.attention import FlexBertAttentionBase
from src.bert_layers.configuration_bert import FlexBertConfig
from src.bert_layers.initialization import ModuleType, init_weights

class MPNetAttention(FlexBertAttentionBase):
    """
    MPNet attention implementation for FlexBERT.
    
    This implements the key aspects of MPNet attention:
    1. Bidirectional attention for non-predicted tokens (masked positions)
    2. Autoregressive attention for predicted tokens with position compensation
    """
    
    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attn_qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attn_qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attn_qkv_bias)
        
        # Output projection
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_out_bias)
        
        # Dropouts
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(config.attn_out_dropout_prob) if config.attn_out_dropout_prob > 0.0 else nn.Identity()

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.q_proj,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.k_proj,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.v_proj,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.o_proj,
            layer_dim=self.config.hidden_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: Optional[torch.Tensor] = None,
        is_predicted_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        """
        Forward pass for MPNet attention.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask (torch.Tensor): Attention mask of shape [batch_size, 1, seq_len, seq_len]
            position_ids (torch.Tensor, optional): Position IDs for position-based attention
            is_predicted_mask (torch.Tensor, optional): Boolean tensor of shape [batch_size, seq_len] 
                                                       where True indicates tokens to be predicted (with autoregressive attention)
                                                       and False indicates tokens used for context (with bidirectional attention)
            past_key_value (Tuple[torch.Tensor], optional): Cached key and value projection states
            output_attentions (bool): Whether to return attention scores
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size]
            Optional[torch.Tensor]: Attention weights if output_attentions=True
        """
        batch_size, seq_length = hidden_states.shape[:2]

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # Cache key/values if using past_key_value
        if past_key_value is not None:
            # Reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if past_key_value is not None else None
        
        # Compute attention scores
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Create causal mask for predicted tokens, bidirectional for non-predicted ones
        if is_predicted_mask is not None:
            # Create bidirectional attention mask (all 0s)
            bidirectional_mask = torch.zeros(
                (batch_size, 1, seq_length, seq_length),
                device=hidden_states.device,
                dtype=attention_scores.dtype
            )
            
            # Create causal mask (upper triangle is -10000.0)
            causal_mask = torch.triu(
                torch.full(
                    (seq_length, seq_length),
                    -10000.0,
                    device=hidden_states.device,
                    dtype=attention_scores.dtype
                ),
                diagonal=1
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
            
            # Expand is_predicted_mask to proper shape for broadcasting
            is_predicted_expanded = is_predicted_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, seq_length, seq_length)
            
            # Apply causal mask to predicted tokens, bidirectional to non-predicted
            mpnet_mask = torch.where(is_predicted_expanded, causal_mask, bidirectional_mask)
            
            # Apply the MPNet-specific mask
            attention_scores = attention_scores + mpnet_mask
            
        # Apply general attention mask if provided
        if attention_mask is not None:
            # Convert mask to proper format if needed (we expect a 4D mask)
            if attention_mask.dim() == 2:
                # [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * -10000.0
            elif attention_mask.dim() == 3:
                # [batch_size, 1, seq_length] -> [batch_size, 1, 1, seq_length]
                attention_mask = attention_mask.unsqueeze(1)
            
            attention_scores = attention_scores + attention_mask
        
        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Attention output
        context_layer = torch.matmul(attention_probs, value_states)
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)
        
        # Output projection and dropout
        output = self.o_proj(context_layer)
        output = self.output_dropout(output)
        
        if output_attentions:
            return output, attention_probs, past_key_value
        return output


class MPNetPaddedAttention(MPNetAttention):
    """MPNet attention implementation for padded inputs in FlexBERT."""
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        is_predicted_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward for padded inputs - delegates to parent class but manages padding properly."""
        return super().forward(hidden_states, attention_mask, position_ids, is_predicted_mask)


class MPNetUnpadAttention(MPNetAttention):
    """MPNet attention implementation for unpadded inputs in FlexBERT."""
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: torch.Tensor,
        attn_mask: torch.Tensor,
        is_predicted_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for MPNet attention with unpadded sequences.
        
        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: (total_nnz,)
            attn_mask: (batch, max_seqlen)
            is_predicted_mask: (batch, max_seqlen) or None
        
        Returns:
            attention: (total_nnz, dim)
        """
        import bert_padding
        
        # First pad the inputs to be able to compute attention with our algorithm
        padded_states = bert_padding.pad_input(hidden_states, indices, cu_seqlens.shape[0] - 1, max_seqlen)
        
        # For is_predicted_mask, we need to extract the relevant parts if it exists
        padded_is_predicted = None
        if is_predicted_mask is not None:
            # Assuming is_predicted_mask matches the padded dimensions
            padded_is_predicted = is_predicted_mask
        
        # Compute attention on padded inputs
        attn_output = super().forward(padded_states, attn_mask, None, padded_is_predicted)
        
        # Unpad the outputs
        return bert_padding.unpad_input_only(attn_output, torch.squeeze(attn_mask) == 1)