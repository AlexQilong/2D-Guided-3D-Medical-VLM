import os
import torch
from transformers import Trainer
from transformers.utils import logging, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from typing import Optional

logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"

class LaMedTrainer(Trainer):
    def __init__(self, sample_limit=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_limit = int(sample_limit) if sample_limit is not None else None
        self.sample_counter = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Count samples in this batch
        
        batch_size = inputs["input_ids"].size(0)
        self.sample_counter += batch_size
        
        # Debug first few steps
        if self.state.global_step < 5:
            print(f"DEBUG TRAINER: Step {self.state.global_step}, batch_size={batch_size}, total_samples_seen={self.sample_counter}")

        # Stop if sample limit reached
        # if self.sample_limit is not None and self.sample_counter >= self.sample_limit:
        #     print(f"Reached sample limit of {self.sample_limit}. Stopping training.")
        #     self.control.should_training_stop = True  # graceful stop
 
        # Debug print for truncation and quick loss
        try:
            if self.state.global_step % 100 == 0:
                attn = inputs.get("attention_mask")
                if attn is not None:
                    used = int(attn[0].sum().item())
                    total = attn.shape[1]
                    print(f"[DEBUG] step {self.state.global_step} tokens used: {used}/{total}")
        except Exception:
            pass

        if num_items_in_batch is not None:
            out = super().training_step(model, inputs, num_items_in_batch)
        else:
            out = super().training_step(model, inputs)

        # Print the loss value periodically
        try:
            if self.state.global_step % 10 == 0:  # More frequent for debugging
                loss_val = out.detach().float().item() if hasattr(out, 'detach') else float(out)
                print(f"[DEBUG TRAINER] step {self.state.global_step} loss: {loss_val:.4f}")
                
                # Additional debug info about the batch
                print(f"[DEBUG TRAINER] batch_size: {batch_size}, samples_seen: {self.sample_counter}")
        except Exception as e:
            print(f"[DEBUG TRAINER] Error logging loss: {e}")

        return out

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()

        logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        # In transformers >= 5.0, tokenizer is stored as processing_class
        tokenizer = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
