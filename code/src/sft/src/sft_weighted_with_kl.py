from arguments import SFTWeightedWithKLTrainingArguments, CustomTrainingArguments, SFTWeightedTrainingArugments
from transformers import (HfArgumentParser, PreTrainedModel, PreTrainedTokenizer, PretrainedConfig, AutoConfig,  Qwen2Config, Qwen2Tokenizer, Qwen2ForCausalLM, AutoTokenizer, AutoModelForCausalLM) 
from utils import print_object_on_main_process, print_rank_0, getDataset, set_special_tokens
from typing import Dict, List, Any, Tuple, Union
from collators import sft_weighted_data_collator
from trainers import SFTWeightedWithKLTrainer, SFTWeightedWithKLTrainer_with_verification
from transformers import (HfArgumentParser, PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizer, Qwen2Config, Qwen2ForCausalLM,
                          Qwen2Tokenizer)
from utils import (getDataset, print_object_on_main_process, print_rank_0,
                   set_special_tokens)

 
import os

os.environ["WANDB_PROJECT"] = "YOUR_WANDB_PROJECT" 
def load_math500_eval_dataset(tokenizer, parquet_path: str):
    """Load math500.parquet and convert to SFT format for eval."""
    import ast
    from datasets import load_dataset as _load_dataset
    ds = _load_dataset('parquet', data_files=parquet_path, split='train')
    samples = []
    for item in ds:
        # prompt field is stored as a Python repr of a list-of-dicts
        messages = ast.literal_eval(item['prompt']) if isinstance(item['prompt'], str) else item['prompt']
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        samples.append({
            'prompt': prompt_str,
            'answer': item['solution'],
            'weight': 1.0,
        })
    from datasets import Dataset as _Dataset
    return _Dataset.from_list(samples)


def data_transform(data_list: List[Dict[str, List]], args: SFTWeightedTrainingArugments) -> List[Dict[str, Any]]:
    new_data_list = []
    if args.data_prompt_name != 'prompt' or args.data_answer_name != 'answer' or args.data_weight_name != 'weight':
        for data in data_list:
            new_data = {
                "prompt": data[args.data_prompt_name],
                "answer": data[args.data_answer_name],
                "weight": data.get('weight', 1.0)
            }
            new_data_list.append(new_data)
    else:
        new_data_list = data_list

    if args.debug_mode:
        new_data_list = new_data_list[:100]

    return new_data_list


def loadTokenizerAndModel(args: CustomTrainingArguments) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    CLASS_MAP: Dict[str, Dict[str, Union[PreTrainedModel, PreTrainedTokenizer, PretrainedConfig]]] = {
        "qwen": {"config": Qwen2Config, "tokenizer": Qwen2Tokenizer, "model": Qwen2ForCausalLM},
        "llama3_1": {"config": AutoConfig, "tokenizer": AutoTokenizer, "model": AutoModelForCausalLM},
        "mistral": {"config": AutoConfig, "tokenizer": AutoTokenizer, "model": AutoModelForCausalLM}
    }
    import transformers 
    print_object_on_main_process("transformers version",transformers.__version__)
    try:
        CONFIG_CLASS, TOKENIZER_CLASS, MODEL_CLASS = CLASS_MAP[args.model_type].values()
    except:
        raise ValueError(f"Do not support model type '{args.model_type}'")
    
    def _load_tokenizer(TOKENIZER_CLASS: PreTrainedTokenizer) -> PreTrainedTokenizer:
        tokenizer = TOKENIZER_CLASS.from_pretrained(
                args.model_name_or_path,
                truncation_side=args.truncation_side,
                padding_side=args.padding_side,
                trust_remote_code=True
            )
        tokenizer.model_max_length = args.model_max_length
        return tokenizer
    config = CONFIG_CLASS.from_pretrained(args.model_name_or_path)
    config.use_cache = False
    tokenizer = _load_tokenizer(TOKENIZER_CLASS)
    model = MODEL_CLASS.from_pretrained(args.model_name_or_path, config=config, attn_implementation="sdpa")
    if args.model_type == "llama3_1":
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
        tokenizer.pad_token = "<pad>"
    
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        model.get_input_embeddings().padding_idx = tokenizer.pad_token_id
        model.get_input_embeddings()._fill_padding_idx_with_zero()
        print_object_on_main_process("Tokenizer has been expanded", model, split_line_color="green", object_color="cyan")

    else:
        set_special_tokens(tokenizer, model)
        print_object_on_main_process("eos_token", tokenizer.eos_token)

    return tokenizer, model


def main():
    parser = HfArgumentParser((SFTWeightedWithKLTrainingArguments,))
    args: SFTWeightedWithKLTrainingArguments = parser.parse_args_into_dataclasses()[0]
    
    os.environ["WANDB_PROJECT"] = "s2r-sft"
    print_object_on_main_process("Arguments", args)

    print_rank_0("Loading data>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    train_dataset = getDataset(args, data_transform=data_transform, type='train')

    # Attach precomputed ref logprobs to each sample (avoids ref model forward at every step)
    if args.ref_logprobs_path is not None:
        import torch as _torch
        print_rank_0(f"Loading precomputed ref logprobs from {args.ref_logprobs_path}")
        ref_logprobs_list = _torch.load(args.ref_logprobs_path, map_location="cpu")
        assert len(ref_logprobs_list) == len(train_dataset), (
            f"ref_logprobs length {len(ref_logprobs_list)} != dataset length {len(train_dataset)}"
        )
        train_dataset = train_dataset.map(
            lambda example, idx: {"ref_logprobs": ref_logprobs_list[idx].tolist()},
            with_indices=True,
        )
    if args.ref_logprobs_path is None:
        _, ref_model = loadTokenizerAndModel(args)
    else:
        ref_model = None

    tokenizer, model = loadTokenizerAndModel(args)

    MATH500_PATH = './data/train_data/math500.parquet'
    if os.path.exists(MATH500_PATH):
        print_rank_0(f"Loading math500 eval dataset from {MATH500_PATH}")
        eval_dataset = load_math500_eval_dataset(tokenizer, MATH500_PATH)
    else:
        eval_dataset = None
    print_object_on_main_process("training set", train_dataset, split_line_color="green", object_color="cyan")
    print_object_on_main_process("evaluation set", eval_dataset, split_line_color="green", object_color="cyan")
    print_object_on_main_process("tokenizer", tokenizer, split_line_color="green", object_color="cyan")
    print_object_on_main_process("model", model, split_line_color="green", object_color="cyan")    

    

    
 
    trainer = SFTWeightedWithKLTrainer_with_verification(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=sft_weighted_data_collator(tokenizer, args)
        )

    if args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        if args.save_training_states:
            trainer.save_state()
        trainer.save_model(output_dir=args.output_dir)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_model(output_dir=args.output_dir)
    
    if args.do_eval:
        trainer.evaluate()
        
         
         
if __name__ == '__main__':
    main()