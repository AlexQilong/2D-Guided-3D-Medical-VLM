from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import numpy as np
import random

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Phi3Config, Phi3Model, Phi3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..lamed_arch import LamedMetaModel, LamedMetaForCausalLM
import torch.nn.functional as F

batch_count = 0

def compute_cross_entropy_guidance_loss(generated_logits, guidance_tokens, guidance_attention_mask):
    """
    Computes cross-entropy loss between model output logits and 2D report token IDs (guidance_tokens).
    Ignores padding using the attention mask.
    """
    # Shift logits and labels for next-token prediction (language modeling style)
    shift_logits = generated_logits[:, :-1, :].contiguous()  # [B, seq_len - 1, vocab]
    shift_labels = guidance_tokens[:, 1:].contiguous()       # [B, seq_len - 1]
    shift_mask = guidance_attention_mask[:, 1:].contiguous() # [B, seq_len - 1]

    # Flatten
    loss_logits = shift_logits.view(-1, shift_logits.size(-1))      # [B * (seq_len - 1), vocab]
    loss_labels = shift_labels.view(-1)                             # [B * (seq_len - 1)]
    loss_mask = shift_mask.view(-1).float()                         # [B * (seq_len - 1)]

    # Compute token-level loss
    loss_per_token = F.cross_entropy(loss_logits, loss_labels, reduction='none', label_smoothing=0.1)

    # Mask and average
    masked_loss = (loss_per_token * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    return masked_loss


def compute_kl_loss(generated_logits, guidance_tokens, guidance_attention_mask):
    """
    KL divergence between model's predicted logits and 2D guidance tokens (one-hot).
    """
    # One-hot encode the target tokens: [B, T, V]
    guidance_onehot = F.one_hot(guidance_tokens, num_classes=generated_logits.size(-1)).float()

    # Apply attention mask
    mask = guidance_attention_mask.unsqueeze(-1)  # [B, T, 1]
    guidance_onehot = guidance_onehot * mask

    # Convert to probability distribution
    guidance_probs = guidance_onehot / (guidance_onehot.sum(dim=-1, keepdim=True) + 1e-8)

    # Get log probs from model
    log_probs = F.log_softmax(generated_logits, dim=-1)  # [B, T, V]

    # KL divergence per token: [B, T]
    token_kl = F.kl_div(log_probs, guidance_probs, reduction='none').sum(dim=-1)

    # Mask and average
    kl_loss = (token_kl * guidance_attention_mask).sum() / (guidance_attention_mask.sum() + 1e-8)

    return kl_loss


def repetition_penalty_loss(logits):
    probs = F.softmax(logits, dim=-1)
    top_ids = torch.argmax(probs, dim=-1)  # [B, T]
    repeat_flags = (top_ids[:, 1:] == top_ids[:, :-1]).float()
    return repeat_flags.sum() / (repeat_flags.numel() + 1e-8)


def compute_guidance_loss(generated_logits, guidance_tokens, guidance_attention_mask, attention_mask, debug=False, tokenizer=None):
    """
    Simplified guidance loss that focuses on the main summarization task.
    If debug=True, prints input/output and their token lengths.
    """
    # Only compute loss on the guidance portion (2D reports)
    # This helps the model learn to understand the 2D reports without conflicting objectives
    
    # Get the guidance portion of the sequence
    guidance_length = guidance_tokens.size(1)
    guidance_logits = generated_logits[:, :guidance_length, :]

    if debug:
        real_input_length = attention_mask[0].sum().item()
        print("[DEBUG] Real input length:", real_input_length)
        print("[DEBUG] guidance_tokens shape:", guidance_tokens.shape)
        print("[DEBUG] guidance_logits shape:", guidance_logits.shape)
        print("[DEBUG] guidance_attention_mask shape:", guidance_attention_mask.shape)
        print("[DEBUG] guidance token length:", guidance_tokens.shape[1])
        print("[DEBUG] logits token length:", guidance_logits.shape[1])
        if tokenizer is not None:
            # Print first example decoded
            try:
                print("[DEBUG] Decoded input tokens:", tokenizer.decode(guidance_tokens[0].tolist(), skip_special_tokens=True))
            except Exception as e:
                print("[DEBUG] Could not decode tokens:", e)
        print("[DEBUG] First 5 logits (softmax):", torch.softmax(guidance_logits[0, :5], dim=-1).detach().cpu().numpy())

    # Standard cross-entropy loss
    ce_loss = F.cross_entropy(
        guidance_logits.view(-1, guidance_logits.size(-1)),
        guidance_tokens.view(-1),
        ignore_index=-100,
        label_smoothing=0.1
    )
    # loss = ce_loss
    
    # KL loss
    kl_loss = compute_kl_loss(guidance_logits, guidance_tokens, guidance_attention_mask)
    # Repetition penalty
    rep_loss = repetition_penalty_loss(guidance_logits)
    # Weighted sum (weights can be adjusted as needed)
    loss = ce_loss + kl_loss + 0.1 * rep_loss

    return loss


def get_3D_slices(image):
    """
    Extracts axial, sagittal, and coronal slices from a 3D image tensor.
    
    Args:
        image (torch.Tensor): 5D tensor of shape (B, C, D, H, W).
    
    Returns:
        axial_slice, sagittal_slice, coronal_slice (torch.Tensor): 2D slices of shape (B, C, H, W).
    """
    B, C, D, H, W = image.shape
    
    # # Compute middle indices
    # axial_index = D // 2
    # sagittal_index = W // 2
    # coronal_index = H // 2

    # Select random indices
    axial_index = random.randint(0, D - 1)      # Select random depth slice
    sagittal_index = random.randint(0, W - 1)   # Select random width slice
    coronal_index = random.randint(0, H - 1)    # Select random height slice

    # Extract slices
    axial_slice = image[:, :, axial_index, :, :]     # (B, C, H, W)
    sagittal_slice = image[:, :, :, :, sagittal_index]  # (B, C, D, H)
    coronal_slice = image[:, :, :, coronal_index, :]  # (B, C, D, W)
    
    return axial_slice, sagittal_slice, coronal_slice


class LamedPhi3Config(Phi3Config):
    model_type = "lamed_phi3"


class LamedPhi3Model(LamedMetaModel, Phi3Model):
    config_class = LamedPhi3Config
    def __init__(self, config: Phi3Config):
        super(LamedPhi3Model, self).__init__(config)


class LamedPhi3ForCausalLMSum(LamedMetaForCausalLM, Phi3ForCausalLM):
    config_class = LamedPhi3Config

    def __init__(self, config):
        super(LamedPhi3ForCausalLMSum, self).__init__(config)
        self.model = LamedPhi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            images: Optional[torch.FloatTensor] = None,
            input_ids: torch.LongTensor = None,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            segs: Optional[torch.FloatTensor] = None,
            guidance_tokens: Optional[torch.Tensor] = None,  # Multiple 2D report tokens
            guidance_attention_mask: Optional[torch.Tensor] = None,
            summarization_tokens: Optional[torch.Tensor] = None,
            summarization_attention_mask: Optional[torch.Tensor] = None,

            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # print(self.tokenizer.convert_ids_to_tokens(input_ids[0][:20]))

        # TEXT-ONLY MODE: Skip image processing to save context length (dont use it cause image matters!!!)
        # if images is None:
        #     # print("TEXT-ONLY MODE")
        #     # Direct text processing without image tokens
        #     outputs = super().forward(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_values=past_key_values,
        #         inputs_embeds=inputs_embeds,
        #         labels=labels,
        #         use_cache=use_cache,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict
        #     )
            
        #     logits = outputs.logits.float()
            
        #     # Simple loss: focus on the main task (3D summarization)
        #     # The model should learn to generate 3D summaries from 2D guidance
        #     ce_3d_loss = outputs.loss
            
        #     # Optional: Add a small guidance loss to help understand 2D reports
        #     # But keep it minimal to avoid conflicts
        #     if guidance_tokens is not None:
        #         ce_2d_loss = compute_guidance_loss(logits, guidance_tokens, guidance_attention_mask, debug=True, tokenizer=self.tokenizer if hasattr(self, 'tokenizer') else None)
        #         # Simple weighted combination
        #         outputs.loss = ce_3d_loss + 0.1 * ce_2d_loss
        #     else:
        #         outputs.loss = ce_3d_loss
                
        #     return outputs

        # ORIGINAL IMAGE MODE (for backward compatibility)
        input_ids_pre = input_ids

        if inputs_embeds is None:
            # print("IMAGE MODE")
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )

            outputs =  super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            logits = outputs.logits.float()

            # Simple loss: focus on the main task (3D summarization)
            # The model should learn to generate 3D summaries from 2D guidance
            ce_3d_loss = outputs.loss
            
            # Optional: Add a small guidance loss to help understand 2D reports
            # But keep it minimal to avoid conflicts
            ce_2d_loss = compute_guidance_loss(logits, guidance_tokens, guidance_attention_mask, attention_mask, debug=False, tokenizer=self.tokenizer if hasattr(self, 'tokenizer') else None)
            
            # Simple weighted combination
            outputs.loss = ce_3d_loss + 0.1 * ce_2d_loss
            return outputs


    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        seg_enable: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor, Any]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            print("IMAGE MODE")
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
            )
        else:
            print("TEXT-ONLY MODE")
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if seg_enable:
            outputs = super().generate(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs
            )

            output_hidden_states = outputs.hidden_states
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.config.seg_token_id

            last_tensors = [tuple[-1] for tuple in output_hidden_states]
            last_hidden_state = torch.cat(last_tensors[1:], dim=1)

            seg_prompts = []
            noseg_ids = []
            for i in range(len(seg_token_mask)):
                if torch.sum(seg_token_mask[i]) == 1:
                    seg_token = last_hidden_state[i][seg_token_mask[i]]
                    seg_prompt = self.get_model().seg_projector(seg_token)
                elif torch.sum(seg_token_mask[i]) > 1:
                    seg_tokens = last_hidden_state[i][seg_token_mask[i]]
                    seg_token = torch.mean(seg_tokens, dim=0, keepdim=True)
                    seg_prompt = self.get_model().seg_projector(seg_token)
                else:
                    noseg_ids.append(i)
                    seg_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype,
                                             device=last_hidden_state.device)
                seg_prompts.append(seg_prompt)

            seg_prompts = torch.cat(seg_prompts, dim=0)
            logits = self.get_model().seg_module(images, seg_prompts)
            logits[noseg_ids] = -torch.inf

            return output_ids, logits
        else:
            output_ids = super().generate(
                inputs_embeds=inputs_embeds,
                **kwargs
            )
            return output_ids


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        return inputs


AutoConfig.register("lamed_phi3", LamedPhi3Config)
AutoModelForCausalLM.register(LamedPhi3Config, LamedPhi3ForCausalLMSum)