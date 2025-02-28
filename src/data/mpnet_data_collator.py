import random
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import math
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


@dataclass
class MPNetDataCollator:
    """
    Data collator for MPNet pretraining.
    
    This collator implements the data preprocessing strategy described in the MPNet paper:
    1. Randomly permute the sequence
    2. Choose rightmost portion (determined by mlm_probability) of tokens to predict
    3. For non-predicted (context) tokens, apply bidirectional attention
    4. For predicted tokens, apply autoregressive attention with position compensation
    5. Apply masking to input tokens with an MLM-style strategy
    
    Args:
        tokenizer: Tokenizer to use for encoding/decoding
        mlm_probability: Probability of choosing a token for prediction
        mask_probability: Probability of masking a token in the predicted set (typically higher than mlm_probability)
    """
    
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    mask_probability: float = 0.80
    random_replacement_probability: float = 0.10
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError("The tokenizer does not have a mask token. Please use a different tokenizer.")
            
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of examples for MPNet pretraining.
        
        Args:
            examples: List of tokenized examples with 'input_ids' and 'attention_mask'
            
        Returns:
            Batch dictionary with:
            - input_ids: Input tokens with masking applied
            - attention_mask: Attention mask for padding
            - is_predicted_mask: Boolean mask indicating which tokens are in the predicted set
            - labels: Labels for loss computation (-100 for tokens not being predicted)
            - position_ids: Position IDs after permutation (used for position compensation)
        """
        batch = self._collate_batch(examples)
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size, seq_length = input_ids.size()
        
        # Create a permutation for each sequence
        permuted_indices, permuted_input_ids, permuted_attention_mask, position_ids = self._permute_sequences(
            input_ids, attention_mask
        )
        
        # Split into predicted and non-predicted tokens
        is_predicted_mask, predicted_indices = self._create_prediction_mask(permuted_attention_mask)
        
        # Apply masking strategy to predicted tokens
        masked_input_ids, labels = self._mask_predicted_tokens(
            permuted_input_ids, is_predicted_mask, predicted_indices
        )
        
        # Prepare final batch
        batch = {
            "input_ids": masked_input_ids,
            "attention_mask": permuted_attention_mask,
            "is_predicted_mask": is_predicted_mask,
            "labels": labels,
            "position_ids": position_ids,
        }
        
        return batch
    
    def _collate_batch(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate examples into a batch.
        """
        # Convert list of dictionaries to dictionary of lists
        batch = {
            key: [example[key] for example in examples] 
            for key in examples[0].keys()
        }
        
        # Special handling for input_ids and attention_mask to ensure proper padding
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        
        # If examples are already tensors
        if isinstance(input_ids[0], torch.Tensor):
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
        else:
            input_ids = torch.tensor([ids for ids in input_ids])
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        elif isinstance(attention_mask[0], torch.Tensor):
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            )
        else:
            attention_mask = torch.tensor([mask for mask in attention_mask])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def _permute_sequences(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly permute sequences while preserving padding.
        
        Returns:
            - permuted_indices: List of permutation indices per sequence
            - permuted_input_ids: Input IDs after permutation
            - permuted_attention_mask: Attention mask after permutation  
            - position_ids: Position IDs tracking original positions
        """
        batch_size, seq_length = input_ids.size()
        permuted_input_ids = input_ids.clone()
        permuted_attention_mask = attention_mask.clone()
        position_ids = torch.zeros_like(input_ids)
        permuted_indices = []
        
        for i in range(batch_size):
            # Find non-padding positions
            valid_indices = torch.nonzero(attention_mask[i], as_tuple=True)[0]
            num_valid = len(valid_indices)
            
            # Create random permutation for non-padding tokens
            perm = torch.randperm(num_valid, device=input_ids.device)
            permuted_indices.append(valid_indices[perm])
            
            # Apply permutation to input_ids
            permuted_input_ids[i, :num_valid] = input_ids[i, valid_indices[perm]]
            
            # Create position_ids that track original positions
            for j, idx in enumerate(perm):
                position_ids[i, j] = valid_indices[idx]
            
            # Keep padding as is
            if num_valid < seq_length:
                permuted_input_ids[i, num_valid:] = input_ids[i, num_valid:]
                position_ids[i, num_valid:] = torch.arange(
                    num_valid, seq_length, device=input_ids.device
                )
            
        return permuted_indices, permuted_input_ids, permuted_attention_mask, position_ids
    
    def _create_prediction_mask(
        self, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Create a mask to identify tokens to predict (rightmost portion of permuted sequence).
        
        Returns:
            - is_predicted_mask: Boolean tensor where True indicates tokens to predict
            - predicted_indices: List of indices per sequence of tokens to predict
        """
        batch_size, seq_length = attention_mask.size()
        is_predicted_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        predicted_indices = []
        
        for i in range(batch_size):
            # Get valid token positions
            valid_indices = torch.nonzero(attention_mask[i], as_tuple=True)[0]
            num_valid = len(valid_indices)
            
            # Determine how many tokens to predict based on mlm_probability
            num_to_predict = max(1, int(num_valid * self.mlm_probability))
            
            # Choose the rightmost portion as MPNet does
            prediction_indices = valid_indices[-num_to_predict:]
            predicted_indices.append(prediction_indices)
            
            # Set the mask for these positions
            is_predicted_mask[i, prediction_indices] = True
            
        return is_predicted_mask, predicted_indices
    
    def _mask_predicted_tokens(
        self, 
        input_ids: torch.Tensor, 
        is_predicted_mask: torch.Tensor,
        predicted_indices: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply masking strategy to predicted tokens: 80% mask, 10% random, 10% unchanged.
        
        Returns:
            - masked_input_ids: Input ids with masking applied
            - labels: Labels for loss computation (-100 for non-predicted tokens)
        """
        batch_size, seq_length = input_ids.size()
        masked_input_ids = input_ids.clone()
        
        # Initialize labels with -100 (ignored in loss)
        labels = torch.full_like(input_ids, -100)
        
        for i in range(batch_size):
            # Set labels for predicted tokens
            predicted_positions = predicted_indices[i]
            labels[i, predicted_positions] = input_ids[i, predicted_positions].clone()
            
            # Create random numbers for masking strategy
            rand = torch.rand(len(predicted_positions), device=input_ids.device)
            
            # Indices for each masking strategy
            mask_indices = predicted_positions[rand < self.mask_probability]
            random_indices = predicted_positions[(rand >= self.mask_probability) & 
                                             (rand < self.mask_probability + self.random_replacement_probability)]
            
            # Apply mask token
            masked_input_ids[i, mask_indices] = self.tokenizer.mask_token_id
            
            # Apply random tokens
            if len(random_indices) > 0:
                random_words = torch.randint(
                    len(self.tokenizer), 
                    (len(random_indices),), 
                    device=input_ids.device
                )
                masked_input_ids[i, random_indices] = random_words
        
        return masked_input_ids, labels