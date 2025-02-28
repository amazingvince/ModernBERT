import torch
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class MPNetDataCollator:
    """
    Data collator for MPNet pretraining.
    It permutes sequences and prepares inputs according to the MPNet paper:
    - Permutes the sequence
    - Chooses rightmost 15% tokens to predict
    - Adds mask tokens before predicted part
    - Prepares position information and attention masks
    """
    
    def __init__(
        self,
        tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=None,
        return_tensors="pt",
        whole_word_mask=True,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.whole_word_mask = whole_word_mask
        self.vocab_size = tokenizer.vocab_size
    
    def __call__(self, examples):
        # Prepare batch
        batch = self._prepare_batch(examples)
        
        # Permute sequences
        input_ids, attention_mask, position_ids, permutation_indices = self._permute_sequences(batch)
        
        # Split into non-predicted and predicted parts
        non_pred_ids, pred_ids, pred_positions, mask_indices = self._split_into_parts(
            input_ids, permutation_indices, attention_mask
        )
        
        # Create final inputs with mask tokens
        final_input_ids, final_attention_mask, final_position_ids, labels = self._create_final_inputs(
            non_pred_ids, pred_ids, pred_positions, mask_indices, input_ids, position_ids, attention_mask
        )
        
        return {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_mask,
            "position_ids": final_position_ids,
            "labels": labels,
            "cu_seqlens": self._get_cu_seqlens(final_attention_mask),
            "max_seqlen": final_input_ids.size(1),
            "indices": self._get_indices(final_attention_mask),
        }
    
    def _prepare_batch(self, examples):
        # Convert examples to tensors
        batch = {
            "input_ids": [example["input_ids"] for example in examples],
            "attention_mask": [example["attention_mask"] for example in examples]
        }
        return batch
    
    def _permute_sequences(self, batch):
        input_ids_list = batch["input_ids"]
        attention_mask_list = batch["attention_mask"]
        
        permuted_input_ids = []
        permuted_attention_mask = []
        permuted_position_ids = []
        permutation_indices_list = []
        
        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            # Get valid token indices (non-padding)
            valid_indices = [i for i, mask in enumerate(attention_mask) if mask == 1]
            seq_len = len(valid_indices)
            
            # Create permutation
            permutation = list(range(seq_len))
            random.shuffle(permutation)
            
            # Map to original indices
            permutation_indices = [valid_indices[i] for i in permutation]
            
            # Create permuted sequence
            permuted_ids = [input_ids[i] for i in permutation_indices] + [
                input_ids[i] for i in range(len(input_ids)) if i not in valid_indices
            ]
            
            # Create permuted attention mask
            permuted_mask = [1] * seq_len + [0] * (len(attention_mask) - seq_len)
            
            # Create position ids (original positions)
            position_ids = list(range(len(input_ids)))
            permuted_pos = [position_ids[i] for i in permutation_indices] + [
                position_ids[i] for i in range(len(position_ids)) if i not in valid_indices
            ]
            
            permuted_input_ids.append(permuted_ids)
            permuted_attention_mask.append(permuted_mask)
            permuted_position_ids.append(permuted_pos)
            permutation_indices_list.append(permutation_indices)
        
        return (
            torch.tensor(permuted_input_ids),
            torch.tensor(permuted_attention_mask),
            torch.tensor(permuted_position_ids),
            permutation_indices_list
        )
    
    def _split_into_parts(self, input_ids, permutation_indices, attention_mask):
        batch_size, seq_len = input_ids.size()
        
        non_pred_ids = []
        pred_ids = []
        pred_positions = []
        mask_indices = []
        
        for i in range(batch_size):
            valid_len = attention_mask[i].sum().item()
            c = int(valid_len * (1 - self.mlm_probability))  # Number of non-predicted tokens
            
            # Split sequence
            non_pred = input_ids[i, :c].tolist()
            pred = input_ids[i, c:valid_len].tolist()
            
            # Get positions of predicted tokens in original sequence
            pred_pos = permutation_indices[i][c:valid_len]
            
            # Create mask indices (positions where mask tokens will be placed)
            mask_idx = list(range(c, valid_len))
            
            non_pred_ids.append(non_pred + [self.tokenizer.pad_token_id] * (c - len(non_pred)))
            pred_ids.append(pred + [self.tokenizer.pad_token_id] * (valid_len - c - len(pred)))
            pred_positions.append(pred_pos + [0] * (valid_len - c - len(pred_pos)))
            mask_indices.append(mask_idx + [0] * (valid_len - c - len(mask_idx)))
        
        return (
            torch.tensor(non_pred_ids),
            torch.tensor(pred_ids),
            torch.tensor(pred_positions),
            torch.tensor(mask_indices)
        )
    
    def _create_final_inputs(self, non_pred_ids, pred_ids, pred_positions, mask_indices, 
                            original_ids, position_ids, attention_mask):
        batch_size = non_pred_ids.size(0)
        non_pred_len = non_pred_ids.size(1)
        pred_len = pred_ids.size(1)
        
        # Create final input ids with mask tokens
        final_input_ids = []
        final_attention_mask = []
        final_position_ids = []
        labels = []
        
        for i in range(batch_size):
            # Get non-predicted part
            non_pred = non_pred_ids[i, :non_pred_len].tolist()
            
            # Add mask tokens
            masks = [self.tokenizer.mask_token_id] * pred_len
            
            # Add predicted tokens (for content stream)
            pred = pred_ids[i, :pred_len].tolist()
            
            # Combine all parts
            final_ids = non_pred + masks + pred
            
            # Create attention mask
            final_mask = [1] * (len(non_pred) + len(masks) + len(pred))
            
            # Create position ids - match original positions
            non_pred_pos = position_ids[i, :non_pred_len].tolist()
            pred_pos = [position_ids[i, pos].item() if pos < position_ids.size(1) else 0 
                       for pos in pred_positions[i, :pred_len].tolist()]
            
            # Final position ids
            final_pos = non_pred_pos + pred_pos + pred_pos
            
            # Create labels (-100 for non-predicted tokens)
            label = [-100] * (len(non_pred) + len(masks)) + pred
            
            # Pad if needed
            max_len = max(len(final_ids), len(final_mask), len(final_pos), len(label))
            final_ids = final_ids + [self.tokenizer.pad_token_id] * (max_len - len(final_ids))
            final_mask = final_mask + [0] * (max_len - len(final_mask))
            final_pos = final_pos + [0] * (max_len - len(final_pos))
            label = label + [-100] * (max_len - len(label))
            
            final_input_ids.append(final_ids)
            final_attention_mask.append(final_mask)
            final_position_ids.append(final_pos)
            labels.append(label)
        
        return (
            torch.tensor(final_input_ids),
            torch.tensor(final_attention_mask),
            torch.tensor(final_position_ids),
            torch.tensor(labels)
        )
    
    def _get_cu_seqlens(self, attention_mask):
        # Calculate cumulative sequence lengths for unpadded attention
        cu_seqlens = torch.zeros(attention_mask.size(0) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(attention_mask.sum(dim=1), dim=0)
        return cu_seqlens
    
    def _get_indices(self, attention_mask):
        # Get indices for unpadded tokens
        indices = torch.zeros_like(attention_mask, dtype=torch.int32)
        for i in range(attention_mask.size(0)):
            indices[i, :attention_mask[i].sum()] = torch.arange(attention_mask[i].sum())
        return indices.view(-1)[attention_mask.view(-1) == 1]