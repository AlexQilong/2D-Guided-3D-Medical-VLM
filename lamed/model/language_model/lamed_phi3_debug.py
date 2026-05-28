from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import numpy as np
import random

from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM, \
                         Phi3Config, Phi3Model, Phi3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..lamed_arch import LamedMetaModel, LamedMetaForCausalLM
import torch.nn.functional as F

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
    loss_per_token = F.cross_entropy(loss_logits, loss_labels, label_smoothing=0.1)  # [B * (seq_len - 1)]

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


def entropy_loss(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = - (probs * log_probs).sum(dim=-1)  # [B, T]
    return - entropy.mean()  # negative to penalize low entropy (i.e., overconfidence)


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


class LamedPhi3ForCausalLM(LamedMetaForCausalLM, Phi3ForCausalLM):
    config_class = LamedPhi3Config

    def __init__(self, config):
        super(LamedPhi3ForCausalLM, self).__init__(config)
        self.model = LamedPhi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(
                            'LaMed/output/LaMed-Phi3-4B-merge-0000',
                            model_max_length=2048,
                            padding_side="right",
                            use_fast=False,
                            trust_remote_code=True
                        )

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
            count: Optional[int] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        input_ids_pre = input_ids

        if inputs_embeds is None:
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

        try:
            seg_ids = torch.nonzero(torch.sum(segs, dim=(1, 2, 3, 4))).flatten().tolist()
        except:
            seg_ids = []

        if self.get_model().seg_enable and seg_ids:
            outputs = super().forward(
                                    input_ids=input_ids,
                                    inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                    output_hidden_states=True,

                                    position_ids=position_ids,
                                    past_key_values=past_key_values,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    return_dict=return_dict
                                )

            output_hidden_states = outputs.hidden_states

            last_hidden_state = output_hidden_states[-1]

            seg_token_mask = input_ids_pre[:, 1:] == self.config.seg_token_id
            seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1), dtype=seg_token_mask.dtype).cuda(),
                ],
                dim=1,
            )

            seg_prompts = []
            for i in seg_ids:
                if torch.sum(seg_token_mask[i]) == 1:
                    seg_token = last_hidden_state[i][seg_token_mask[i]]
                    seg_prompt = self.get_model().seg_projector(seg_token)
                elif torch.sum(seg_token_mask[i]) > 1:
                    seg_tokens = last_hidden_state[i][seg_token_mask[i]]
                    seg_token = torch.mean(seg_tokens, dim=0, keepdim=True)
                    seg_prompt = self.get_model().seg_projector(seg_token)
                else:
                    seg_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype,
                                             device=last_hidden_state.device)
                seg_prompts.append(seg_prompt)

            seg_prompts = torch.cat(seg_prompts, dim=0)

            D = segs.shape[2]  # Get the depth dimension (D)

            slice_intervals = np.linspace(0, D - 1, num=5, dtype=int)  # Get equally spaced points (n-1 slices)
            slice_indices = list(slice_intervals)
            slice_indices = sorted(random.sample(range(D), 10))

            logits_3D = self.get_model().seg_module(images[seg_ids], text_emb=seg_prompts)
            logits_2D = torch.stack([logits_3D[:, :, idx, :, :] for idx in slice_indices], dim=2)

            loss_dice_3D = self.get_model().dice_loss(logits_3D, segs[seg_ids])
            loss_bce_3D = self.get_model().bce_loss(logits_3D, segs[seg_ids])
            loss_1 = loss_dice_3D + loss_bce_3D
            
            loss_dice_2D = self.get_model().dice_loss(logits_2D, segs[seg_ids][:, :, slice_indices, :, :])
            loss_bce_2D = self.get_model().bce_loss(logits_2D, segs[seg_ids][:, :, slice_indices, :, :])
            loss_2 = loss_dice_2D + loss_bce_2D

            seg_loss = loss_1 + loss_2

            # # Extract slices from predicted 3D segmentation logits
            # axial_pred, sagittal_pred, coronal_pred = get_3D_slices(logits_3D)

            # # Extract ground truth 2D masks (assuming same slice indices)
            # axial_gt, sagittal_gt, coronal_gt = get_3D_slices(segs[seg_ids])

            # # Compute 2D supervision loss
            # loss_axial = self.get_model().dice_loss(axial_pred, axial_gt) + self.get_model().bce_loss(axial_pred, axial_gt)
            # loss_sagittal = self.get_model().dice_loss(sagittal_pred, sagittal_gt) + self.get_model().bce_loss(sagittal_pred, sagittal_gt)
            # loss_coronal = self.get_model().dice_loss(coronal_pred, coronal_gt) + self.get_model().bce_loss(coronal_pred, coronal_gt)

            # Total 2D loss
            # multi_view_loss = (loss_axial + loss_sagittal + loss_coronal) / 3
            # seg_loss = loss_1 + multi_view_loss
            outputs.loss = outputs.loss + seg_loss

            return outputs
        
        elif summarization_tokens is not None:
            """
            train VLM using summarization
            """
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
            ce_loss = compute_cross_entropy_guidance_loss(logits, summarization_tokens, summarization_attention_mask)
            kl_loss = compute_kl_loss(logits, guidance_tokens, guidance_attention_mask)
            outputs.loss = outputs.loss + 0.1 * kl_loss

            return outputs

        else:
            """
            Report summarization
            """
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

            ce_3d_loss = outputs.loss
            # Add safeguard for empty guidance
            if guidance_tokens is not None and guidance_tokens.size(1) > 1:
                ce_2d_loss = compute_cross_entropy_guidance_loss(logits, guidance_tokens, guidance_attention_mask)
                kl_2d_loss = compute_kl_loss(logits, guidance_tokens, guidance_attention_mask)
            else:
                ce_2d_loss = torch.tensor(0.0, device=logits.device)
                kl_2d_loss = torch.tensor(0.0, device=logits.device)
            rep_loss = repetition_penalty_loss(logits)
            entropy = entropy_loss(logits)

            # print(count, f"Loss breakdown: 3D_CE={ce_3d_loss.item():.4f}, 2D_CE={ce_2d_loss.item():.4f}, KL={kl_2d_loss.item():.4f}, REP={rep_loss.item():.4f}, ENT={entropy.item():.4f}")
            # outputs.loss = ce_3d_loss + 0.1 * ce_2d_loss + 0.1 * kl_2d_loss + 0.01 * rep_loss - 0.01 * entropy

            return {
                "ce_3d_loss": ce_3d_loss,
                "ce_2d_loss": ce_2d_loss,
                "kl_2d_loss": kl_2d_loss,
                "rep_loss": rep_loss,
                "entropy": entropy,
                "logits": logits
            }

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
AutoModelForCausalLM.register(LamedPhi3Config, LamedPhi3ForCausalLM)