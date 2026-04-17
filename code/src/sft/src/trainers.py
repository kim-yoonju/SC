import torch
from torch import nn
from transformers import Trainer, TrainerCallback, EvalPrediction, PreTrainedModel, PreTrainedTokenizerBase
from torch.utils.data import Dataset
from utils import print_rank_0
from typing import Union, Optional, List, Tuple, Callable, Dict
from arguments import SFTWeightedWithKLTrainingArguments
from copy import deepcopy
import deepspeed
from base import BaseTrainer
import re
from utils import print_object_on_main_process, print_rank_0, getDataset, set_special_tokens

def compute_lm_loglikeli(logits, labels, chunk_size=256):
    """Chunked CE loss to avoid materializing the full (B, S, V) fp32 tensor."""
    import torch.nn.functional as F
    batch_size, seq_length, vocab_size = logits.shape

    shift_logits = logits[:, :-1, :]          # (B, S-1, V) bf16 — view, no copy
    shift_labels = labels[:, 1:].contiguous().to(logits.device)  # (B, S-1)
    S = shift_logits.shape[1]

    loss = torch.zeros(batch_size, S, device=logits.device)
    for start in range(0, S, chunk_size):
        end = min(start + chunk_size, S)
        chunk = shift_logits[:, start:end, :].float()           # (B, chunk, V) fp32
        chunk_labels = shift_labels[:, start:end].clone()
        mask = chunk_labels == -100
        chunk_labels[mask] = 0
        chunk_loss = F.cross_entropy(
            chunk.view(-1, vocab_size), chunk_labels.view(-1), reduction='none'
        ).view(batch_size, end - start)
        chunk_loss[mask] = 0.0
        loss[:, start:end] = chunk_loss
        del chunk, chunk_loss

    ignore_mask = labels[:, 1:] != -100
    mean_loss = loss.sum(dim=-1) / ignore_mask.sum(dim=-1).clamp(min=1)

    return -mean_loss, loss.view(-1)  # keep same return signature



class SFTWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.debug_mode:
            print_rank_0(f"check inputs :{inputs}")
            
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        logits = model_outputs.logits  

        batch_size, seq_length, vocab_size = logits.shape
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs['labels'][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)  
        weighted_loss = (loss.reshape(batch_size, -1).mean(dim=1) * inputs['weights']).mean()  

        if self.args.debug_mode:
            print_rank_0(f"check logits : {logits}")
            print_rank_0(f"check loss : {loss}")
            print_rank_0(f"check weighted loss : {weighted_loss}")

        return (weighted_loss, logits) if return_outputs else weighted_loss


class SFTWeightedWithKLTrainer(BaseTrainer):
    def _is_create_ref_model(self) -> bool:
        return True
    
    @staticmethod
    def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor = None, gather: bool = True) -> torch.Tensor:
        """
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)
        """
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:] if labels is not None else None

        return BaseTrainer.logprobs_from_logits(shift_logits, shift_labels, gather)
    
    
    def compute_loss(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False, num_items_in_batch=None):
        import torch.nn.functional as F

        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            # labels omitted — HF ForCausalLMLoss does logits.float() on the full
            # (B, S, V) tensor (~39 GiB at batch=16/seq=8000). Use chunked CE below.
        )

        # Chunked CE loss — avoids the full fp32 (B, S, V) allocation
        _, token_loss_flat = compute_lm_loglikeli(model_outputs.logits, inputs['labels'])
        batch_size = model_outputs.logits.shape[0]
        ce_loss_2d = token_loss_flat.view(batch_size, -1)  # (B, S-1)

        if self.args.lm_kl_coeff is not None:
            shift_labels = inputs['labels'][:, 1:]
            mask = torch.not_equal(shift_labels, -100)

            if 'ref_logprobs' in inputs:
                ref_lp = inputs['ref_logprobs'].to(model_outputs.logits.device)  # (B, S-1)
                shift_logits = model_outputs.logits[:, :-1, :]
                B_kl, S_kl, V_kl = shift_logits.shape
                # Chunked log_softmax + gather — peak ~2.5 GiB per chunk vs ~39 GiB full
                model_lp = torch.zeros(B_kl, S_kl, dtype=shift_logits.dtype,
                                       device=shift_logits.device)
                for _s in range(0, S_kl, 256):
                    _e = min(_s + 256, S_kl)
                    _chunk = shift_logits[:, _s:_e, :].float()
                    _chunk_lp = F.log_softmax(_chunk, dim=-1)
                    del _chunk
                    _idx = shift_labels[:, _s:_e].clamp(min=0).unsqueeze(-1)
                    model_lp[:, _s:_e] = torch.gather(_chunk_lp, 2, _idx).squeeze(-1).to(shift_logits.dtype)
                    del _chunk_lp
                kl_divergence = (torch.exp(model_lp) * (model_lp - ref_lp))  # (B, S-1)
            else:
                with torch.no_grad():
                    ref_model_outputs = self.ref_model(
                        input_ids=inputs['input_ids'].cpu(),
                        attention_mask=inputs['attention_mask'].cpu()
                    )
                kl_divergence = self._chunked_full_kl(
                    model_outputs.logits, ref_model_outputs.logits, chunk_size=64
                )

            kl_divergence = (kl_divergence * mask).sum() / mask.sum()
            self.store_metrics({"kl": kl_divergence}, 'train')
            ce_loss = ce_loss_2d.sum() / mask.sum().clamp(min=1)
            loss = ce_loss + self.args.lm_kl_coeff * kl_divergence
            self.store_metrics({"step_loss": loss})

        else:
            mask = inputs['labels'][:, 1:] != -100
            loss = ce_loss_2d.sum() / mask.sum().clamp(min=1)

        return (loss, model_outputs['logits']) if return_outputs else loss

     
     
class OfflineWeightedPolicyTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        args: SFTWeightedWithKLTrainingArguments = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        self.ref_model = ref_model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        if self.is_deepspeed_enabled:
            self.ref_model = self._prepare_deepspeed(self.ref_model)
        else:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
       

    def _prepare_deepspeed(self, model: PreTrainedModel):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
        del config_kwargs['optimizer']
        del config_kwargs['scheduler']

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size" : 0.9 * hidden_size * hidden_size
                        }
                    )
        
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model = deepspeed.initialize(model=model, config=config_kwargs)[0].module
        model.eval()
        return model

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.debug_mode:
            print_rank_0(f"check inputs :{inputs}")
            
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        with torch.no_grad():
             
            ref_model_outputs = self.ref_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            ref_logprob = compute_lm_loglikeli(ref_model_outputs.logits, inputs['labels']).detach()  

        if self.args.debug_mode:
            print_rank_0(f"check ref_model output: {ref_logprob}")

        logprob = compute_lm_loglikeli(model_outputs.logits, inputs['labels'])        
        kl_div = (logprob - ref_logprob)
        
        importance_ratio = (logprob - ref_logprob).exp()
        importance_ratio_clipped = torch.clip(importance_ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)

        advantages = inputs['rewards'] - self.args.lm_kl_coeff * kl_div
        ppo_loss = - torch.minimum(advantages * importance_ratio, advantages * importance_ratio_clipped)

        sample_size, sft_size = (1-inputs['sft_mask']).sum(), (inputs['sft_mask']).sum()
        sft_loss = (- logprob * inputs['sft_mask']).sum() / sft_size if sft_size > 0 else sft_size
        ppo_loss = (ppo_loss * (1 - inputs['sft_mask'])).sum() / sample_size if sample_size > 0 else sample_size
        
        total_loss = self.args.lm_sft_coeff * sft_loss + ppo_loss                
        
        weighted_loss = (total_loss * inputs['weights']).mean()  

         
         
         
         

        return (weighted_loss, model_outputs.logits) if return_outputs else weighted_loss


class SFTWeightedWithKLTrainer_with_verification(BaseTrainer):
    '''
    r1+\n\nWait, let me recheck my solution.\n\n+v1+\n\nLet me try again.\n\n+r2+\n\nWait, let me recheck my solution.\n\n
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.special_tokens = [
            "\n\nWait, let me recheck my answer.\n\n",
            "\n\nLet me try again.\n\n",

        ]

    def _is_create_ref_model(self) -> bool:
        return True
    
    @staticmethod
    def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor = None, gather: bool = True) -> torch.Tensor:
        """
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)
        """
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:] if labels is not None else None

        return BaseTrainer.logprobs_from_logits(shift_logits, shift_labels, gather)
    
    
    def compute_loss(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False,num_items_in_batch=None):
        import torch.nn.functional as F
        # ── hidden states only: never materialise the (B, S, V) bf16 logit tensor ──
        # model.module is the unwrapped HF model inside a DeepSpeed/DDP wrapper.
        base_model = model.module if hasattr(model, 'module') else model
        hidden = base_model.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        ).last_hidden_state                                  # (B, S, H) bf16
        lm_head = base_model.lm_head                         # keep module ref so AccumulateGrad fires
        B, S, _ = hidden.shape
        device   = hidden.device
        conjunction_mask = torch.zeros_like(inputs['labels'], dtype=torch.bool)
         
        for i in range(len(inputs['labels'])):
             
            valid_positions = (inputs['labels'][i] != -100).nonzero(as_tuple=True)[0]
            if len(valid_positions) == 0:
                continue
                
             
            valid_ids = inputs['labels'][i][valid_positions]
             

             
            text = self.processing_class.decode(valid_ids, skip_special_tokens=False)
             
             
             
             
            
             
            splits = re.split("(Wait,|Let me try again.\n\n)", text)
            if len(splits) ==3 :
                conjunction_mask[i, valid_positions[0]:valid_positions[-1]+1] = True
                continue
                
            if len(splits) <= 1:
                continue
                
            current_text = splits[0]
             
            current_tokens = len(self.processing_class(current_text, add_special_tokens=False)["input_ids"])
            current_position = valid_positions[current_tokens-1] if current_tokens > 0 else valid_positions[0]
            
            last_split_token = None
            for idx, split in enumerate(splits[1:]):
                 
                if split in ["Wait,", "Let me try again.\n\n"]:
                    last_split_token = split
                    continue
                    
                if last_split_token == "Wait,":
                     
                    full_text = current_text + last_split_token + split

                     
                    tokenizer_output = self.processing_class(full_text, add_special_tokens=False)["input_ids"]
                     
                     
                    full_tokens = len(tokenizer_output)
                     
                     
                    full_position = valid_positions[full_tokens-1]
                    
                     
                     
                     
                    start_pos = max(0, current_position)   
                     
                    conjunction_mask[i, start_pos:full_position+1] = True
                    
                         
                    current_masked_ids = inputs['labels'][i][start_pos:full_position+1]
                    masked_text = self.processing_class.decode(current_masked_ids[current_masked_ids != -100], skip_special_tokens=False)
                     
                        

                    current_text = full_text
                    current_position = full_position
                    
                else:   
                    if idx == len(splits[1:]) - 3:
                        full_text = current_text + last_split_token + split
                         
                        tokenizer_output = self.processing_class(full_text, add_special_tokens=False)["input_ids"]
                         
                         
                        full_tokens = len(tokenizer_output)                        
                        full_position = valid_positions[full_tokens-1]
                        
                         
                         
                         
                        start_pos = max(0, current_position)   
                        conjunction_mask[i, start_pos:full_position+1] = True
                        
                                                 
                         
                        current_masked_ids = inputs['labels'][i][start_pos:full_position+1]
                        masked_text = self.processing_class.decode(current_masked_ids[current_masked_ids != -100], skip_special_tokens=False)
                         
                        
                        
                        current_text = full_text
                        current_position = full_position
                    else:
                         
                        full_text = current_text + last_split_token + split
                        current_text = full_text
                        current_position = valid_positions[len(self.processing_class(full_text, add_special_tokens=False)["input_ids"])-1]
         
        for i in range(len(inputs['labels'])):
             
            mask_positions = conjunction_mask[i].nonzero(as_tuple=True)[0]
            if len(mask_positions) > 0:
                 
                segments = []
                start = mask_positions[0]
                prev = mask_positions[0]
                
                for pos in mask_positions[1:]:
                    if pos != prev + 1:
                         
                        segments.append((start, prev + 1))
                        start = pos
                    prev = pos
                
                 
                segments.append((start, prev + 1))
                
                 
                for seg_start, seg_end in segments:
                    segment_ids = inputs['labels'][i][seg_start:seg_end]
                    segment_text = self.processing_class.decode(segment_ids[segment_ids != -100], skip_special_tokens=False)
                    print_rank_0(f"Masked text for sample {i}, segment {seg_start}-{seg_end}: {segment_text}")
                    print_rank_0(f"check eos_token: {self.processing_class.eos_token}")
        # ── chunked lm_head: CE + KL in one pass, peak = (B, 256, V) fp32 ≈ 2.5 GiB ──
        shift_hidden = hidden[:, :-1, :].contiguous()        # (B, S-1, H)
        shift_labels = inputs['labels'][:, 1:].contiguous()  # (B, S-1)
        label_mask   = shift_labels != -100
        CHUNK = 256
        V = lm_head.weight.shape[0]

        ce_loss_2d = torch.zeros(B, S - 1, device=device, dtype=hidden.dtype)
        has_ref    = 'ref_logprobs' in inputs and self.args.lm_kl_coeff is not None
        if has_ref:
            ref_lp   = inputs['ref_logprobs'].to(device)     # (B, S-1)
            model_lp = torch.zeros(B, S - 1, device=device, dtype=hidden.dtype)

        acc_correct = torch.zeros(1, device=device, dtype=torch.long)
        acc_total   = torch.zeros(1, device=device, dtype=torch.long)

        for s in range(0, S - 1, CHUNK):
            e           = min(s + CHUNK, S - 1)
            chunk_lbl   = shift_labels[:, s:e].clone()
            chunk_pad   = chunk_lbl == -100
            chunk_lbl[chunk_pad] = 0

            chunk_logits = lm_head(shift_hidden[:, s:e]).float()  # (B, chunk, V) fp32; grad flows through AccumulateGrad

            chunk_ce = F.cross_entropy(
                chunk_logits.view(-1, V), chunk_lbl.view(-1), reduction='none',
            ).view(B, e - s)
            chunk_ce[chunk_pad] = 0.0
            ce_loss_2d[:, s:e]  = chunk_ce.to(hidden.dtype)

            if has_ref:
                chunk_lp  = F.log_softmax(chunk_logits, dim=-1)
                chunk_mlp = torch.gather(chunk_lp, 2,
                                         chunk_lbl.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                chunk_mlp[chunk_pad] = 0.0
                model_lp[:, s:e] = chunk_mlp.to(hidden.dtype)
                del chunk_lp, chunk_mlp

            with torch.no_grad():
                chunk_preds = chunk_logits.argmax(dim=-1)       # (B, chunk)
                acc_correct += ((chunk_preds == chunk_lbl) & ~chunk_pad).sum()
                acc_total   += (~chunk_pad).sum()

            del chunk_logits, chunk_ce

        shift_conjunction_mask = conjunction_mask[:, 1:]

        # ── token accuracy (train or eval) ────────────────────────────────────
        if acc_total.item() > 0:
            acc = acc_correct.float() / acc_total.float()
            if model.training:
                self.store_metrics({"train_acc": acc.item()}, "train")
            else:
                self.store_metrics({"eval_acc": acc.item()}, "eval")

        # ── loss ──────────────────────────────────────────────────────────────
        use_conj = shift_conjunction_mask.any()  # False for eval data without S2R format
        loss_mask = shift_conjunction_mask if use_conj else label_mask

        if self.args.lm_kl_coeff is not None:
            combined_mask = label_mask * shift_conjunction_mask

            if has_ref:
                kl_divergence = torch.exp(model_lp) * (model_lp - ref_lp)
                kl_scalar = (kl_divergence * combined_mask).sum() / combined_mask.sum().clamp(min=1)
                self.store_metrics({"kl": kl_scalar}, "train")
                loss = (ce_loss_2d * loss_mask).sum() / loss_mask.sum().clamp(min=1)
                loss = loss + self.args.lm_kl_coeff * kl_scalar
                self.store_metrics({"step_loss": loss})
            elif self.ref_model is not None:
                # fallback: ref model on CPU (only hit when ref_logprobs not provided)
                with torch.no_grad():
                    ref_out = self.ref_model(
                        input_ids=inputs['input_ids'].cpu(),
                        attention_mask=inputs['attention_mask'].cpu(),
                    )
                # reconstruct full policy logits once for the fallback KL
                policy_logits = torch.cat([
                    lm_head(shift_hidden[:, s:min(s+CHUNK, S-1)]).float()
                    for s in range(0, S - 1, CHUNK)
                ], dim=1)
                kl_divergence = self._chunked_full_kl(policy_logits, ref_out.logits[:, :-1], chunk_size=64)
                del policy_logits
                kl_scalar = (kl_divergence * combined_mask).sum() / combined_mask.sum().clamp(min=1)
                self.store_metrics({"kl": kl_scalar}, "train")
                loss = (ce_loss_2d * loss_mask).sum() / loss_mask.sum().clamp(min=1)
                loss = loss + self.args.lm_kl_coeff * kl_scalar
                self.store_metrics({"step_loss": loss})
            else:
                # eval without ref_logprobs and without ref_model: CE only
                loss = ce_loss_2d.sum() / label_mask.sum().clamp(min=1)
        else:
            loss = (ce_loss_2d * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        return (loss, ce_loss_2d) if return_outputs else loss