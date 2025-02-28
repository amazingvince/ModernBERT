import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

class MPNetAttention(nn.Module):
    """MPNet attention implementation using PyTorch's flex_attention.
    
    This implements the key aspects of MPNet attention:
    1. Bidirectional attention for non-predicted tokens
    2. Autoregressive attention for predicted tokens with position compensation
    """
    
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Output projection
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.p_dropout = config.attention_probs_dropout_prob
        
        # Scale factor for attention scores
        self.scale = 1.0 / (self.attention_head_size ** 0.5)
        
        # Block size for block sparse attention (can be tuned for performance)
        self.block_size = 64
    
    def mpnet_attention_mask(self, batch, head, q_idx, k_idx):
        """
        Implements the MPNet attention pattern:
        - Non-predicted tokens can attend to all tokens (bidirectional)
        - Predicted tokens can only attend to previous tokens including non-predicted and 
          earlier predicted tokens (autoregressive)
        
        This approach implements position compensation by using token positions
        rather than sequence positions.
        """
        # Calculate predicted zone boundary
        # This is a simplified approach - in a real implementation you would need to
        # know which tokens are predicted vs non-predicted
        batch_info = getattr(self, '_batch_info', None)
        if batch_info is None:
            # Default fallback - assuming causal attention
            return q_idx >= k_idx
            
        # Retrieve batch-specific information
        boundary, token_positions = batch_info[batch]
        
        # Get the actual positions of the query and key tokens
        q_pos = token_positions[q_idx]
        k_pos = token_positions[k_idx]
        
        if q_idx < boundary:
            # Non-predicted token: bidirectional attention
            return torch.tensor(True, device=q_idx.device)
        else:
            # Predicted token: can attend to non-predicted tokens and 
            # previous predicted tokens (position-based autoregressive)
            # The comparison checks if:
            #   1. The key is a non-predicted token (k_idx < boundary), OR
            #   2. The key is a predicted token that appears before the query token
            return (k_idx < boundary) | (k_pos < q_pos)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        position_info=None,
        output_attentions=False,
    ):
        """
        Forward pass for MPNet attention using flex_attention.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
            head_mask: (num_heads,) for head pruning
            position_info: tuple with (boundary, token_positions) where:
                - boundary: index separating non-predicted tokens from predicted tokens
                - token_positions: original positions of tokens in the sequence
        
        Returns:
            attention output: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Store batch information for mask_mod
        if position_info is not None:
            self._batch_info = position_info
        
        # Project inputs to queries, keys, and values
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = query_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        key_layer = key_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        value_layer = value_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        
        # Transpose to get dimensions [batch_size, num_heads, seq_length, head_size]
        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)
        
        # Create block mask for MPNet attention pattern
        # For simplicity, we'll create one mask per batch item
        # In a real implementation, each sequence might have different boundaries
        block_mask = create_block_mask(
            self.mpnet_attention_mask,
            batch_size,
            self.num_attention_heads,
            seq_length,
            seq_length,
            device=hidden_states.device,
            BLOCK_SIZE=self.block_size
        )
        
        # Apply flex attention with our custom mask
        context_layer = flex_attention(
            query_layer,
            key_layer,
            value_layer,
            block_mask=block_mask,
            scale=self.scale,
            kernel_options={"dropout_p": self.p_dropout if self.training else 0.0}
        )
        
        # Transpose back to [batch_size, seq_length, num_heads, head_size]
        context_layer = context_layer.permute(0, 2, 1, 3)
        
        # Reshape to [batch_size, seq_length, hidden_size]
        context_layer = context_layer.reshape(batch_size, seq_length, self.all_head_size)
        
        # Apply output projection and dropout
        output = self.output(context_layer)
        output = self.dropout(output)
        
        # Clean up batch info after use
        if hasattr(self, '_batch_info'):
            delattr(self, '_batch_info')
        
        return output


class MPNetSelfAttention(nn.Module):
    """MPNet self-attention layer that integrates with FlexBERT."""
    
    def __init__(self, config, layer_id=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.attention = MPNetAttention(config)
        self.output_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        indices=None,
        attn_mask=None,
        position_ids=None,
    ):
        """
        Forward pass for MPNet self-attention within FlexBERT framework.
        
        Args:
            hidden_states: (total_nnz, dim) for unpadded input or (batch_size, seq_len, dim) for padded
            cu_seqlens: (batch + 1,) cumulative sequence lengths for unpadded input
            max_seqlen: maximum sequence length
            indices: (total_nnz,) indices for unpadded input
            attn_mask: (batch, max_seqlen) attention mask
            position_ids: (batch, max_seqlen) position IDs
        
        Returns:
            output: (total_nnz, dim) or (batch_size, seq_len, dim)
        """
        # Handle unpadded input (convert to padded format for flex_attention)
        is_unpadded = cu_seqlens is not None and indices is not None
        if is_unpadded:
            # Convert unpadded to padded format
            batch_size = cu_seqlens.shape[0] - 1
            padded_hidden_states = []
            position_info = []
            
            for b in range(batch_size):
                # Get sequence for this batch
                start_idx = cu_seqlens[b]
                end_idx = cu_seqlens[b + 1]
                seq_len = end_idx - start_idx
                
                # Pad to max_seqlen
                batch_hidden = torch.zeros(
                    (max_seqlen, hidden_states.size(-1)),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                batch_hidden[:seq_len] = hidden_states[start_idx:end_idx]
                padded_hidden_states.append(batch_hidden)
                
                # Create position info for this batch
                # In a real implementation, you would need to know the boundary between
                # non-predicted and predicted tokens
                # For simplicity, let's assume first 85% tokens are non-predicted
                boundary = int(seq_len * 0.85)  # 85% tokens are non-predicted
                # Use position_ids if provided, otherwise use sequence positions
                if position_ids is not None:
                    token_positions = position_ids[b, :seq_len]
                else:
                    token_positions = torch.arange(seq_len, device=hidden_states.device)
                position_info.append((boundary, token_positions))
            
            # Stack batches
            padded_hidden_states = torch.stack(padded_hidden_states)
            
            # Process with attention
            attention_output = self.attention(
                padded_hidden_states,
                attention_mask=attn_mask,
                position_info=position_info,
            )
            
            # Convert back to unpadded format
            output = torch.zeros_like(hidden_states)
            for b in range(batch_size):
                start_idx = cu_seqlens[b]
                end_idx = cu_seqlens[b + 1]
                seq_len = end_idx - start_idx
                output[start_idx:end_idx] = attention_output[b, :seq_len]
        else:
            # Handle padded input
            batch_size = hidden_states.size(0)
            seq_len = hidden_states.size(1)
            
            # Create position info for each batch item
            position_info = []
            for b in range(batch_size):
                # Simplification: assuming first 85% tokens are non-predicted
                valid_len = attn_mask[b].sum().item() if attn_mask is not None else seq_len
                boundary = int(valid_len * 0.85)  # 85% tokens are non-predicted
                # Use position_ids if provided, otherwise use sequence positions
                if position_ids is not None:
                    token_positions = position_ids[b, :valid_len]
                else:
                    token_positions = torch.arange(valid_len, device=hidden_states.device)
                position_info.append((boundary, token_positions))
            
            # Process with attention
            output = self.attention(
                hidden_states,
                attention_mask=attn_mask,
                position_info=position_info,
            )
        
        # Apply output transformation
        output = self.output_dense(output)
        output = self.output_dropout(output)
        output = self.output_layernorm(hidden_states + output)
        
        return output