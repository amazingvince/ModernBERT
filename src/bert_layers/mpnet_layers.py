import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

from src.bert_layers.configuration_bert import FlexBertConfig
from src.bert_layers.normalization import get_norm_layer
from src.bert_layers.mlp import get_mlp_layer, FlexBertMLPBase
from src.bert_layers.attention import FlexBertAttentionBase
from src.bert_layers.layers import FlexBertLayerBase
from src.bert_layers.initialization import ModuleType, init_weights

# Import the MPNet attention implementations
from src.bert_layers.mpnet_attention import MPNetAttention, MPNetPaddedAttention, MPNetUnpadAttention


class MPNetLayerBase(FlexBertLayerBase):
    """Base class for MPNet layers in FlexBERT architecture."""
    
    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.config = config
        self.layer_id = layer_id


class MPNetPaddedPreNormLayer(MPNetLayerBase):
    """MPNet layer with pre-normalization for padded inputs."""
    
    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.skip_first_prenorm and config.embed_norm and layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = get_norm_layer(config)
        self.attn = MPNetPaddedAttention(config, layer_id=layer_id)
        self.mlp_norm = get_norm_layer(config)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.attn_norm.reset_parameters()
            self.mlp_norm.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        is_predicted_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-normalization."""
        attn_output = hidden_states + self.attn(
            self.attn_norm(hidden_states), 
            attention_mask,
            position_ids,
            is_predicted_mask
        )
        return attn_output + self.mlp(self.mlp_norm(attn_output))


class MPNetUnpadPreNormLayer(MPNetLayerBase):
    """MPNet layer with pre-normalization for unpadded inputs."""
    
    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.skip_first_prenorm and config.embed_norm and layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = get_norm_layer(config)
        self.attn = MPNetUnpadAttention(config, layer_id=layer_id)
        self.mlp_norm = get_norm_layer(config)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.attn_norm.reset_parameters()
            self.mlp_norm.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_predicted_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-normalization for unpadded inputs."""
        normed_states = self.attn_norm(hidden_states)
        attn_output = hidden_states + self.attn(
            normed_states, 
            cu_seqlens, 
            max_seqlen, 
            indices, 
            attn_mask,
            is_predicted_mask
        )
        return attn_output + self.mlp(self.mlp_norm(attn_output))


class MPNetPaddedPostNormLayer(MPNetLayerBase):
    """MPNet layer with post-normalization for padded inputs."""
    
    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.attn = MPNetPaddedAttention(config, layer_id=layer_id)
        self.attn_norm = get_norm_layer(config)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)
        self.mlp_norm = get_norm_layer(config)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.attn_norm.reset_parameters()
            self.mlp_norm.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        is_predicted_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with post-normalization."""
        attn_output = self.attn_norm(
            hidden_states + self.attn(hidden_states, attention_mask, position_ids, is_predicted_mask)
        )
        return self.mlp_norm(attn_output + self.mlp(attn_output))


class MPNetUnpadPostNormLayer(MPNetLayerBase):
    """MPNet layer with post-normalization for unpadded inputs."""
    
    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.attn = MPNetUnpadAttention(config, layer_id=layer_id)
        self.attn_norm = get_norm_layer(config)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)
        self.mlp_norm = get_norm_layer(config)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.attn_norm.reset_parameters()
            self.mlp_norm.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_predicted_mask: Optional[torch.Tensor