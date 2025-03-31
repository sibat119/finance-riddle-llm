from datasets import Dataset
from tqdm import tqdm
import torch
import gc
from typing import Dict, List, Tuple, Any, Optional
from torch.distributed import destroy_process_group

from src.llm_inference import (
    selector
)
from src import cfg_reader
from src.utils import (
    files,
    strings,
    display
)

def cleanup_resources(session: Any, verbose: bool = True):
    """Clean up resources and memory."""
    try:
        del session
        gc.collect()
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            destroy_process_group()
    except Exception as e:
        if verbose:
            print(str(e))
            
def main():
    # input_ds_path = f'{files.get_project_root()}/data/input/question_w_type.csv'
    input_ds_path = f'{files.get_project_root()}/data/input/riddles-v1.csv'
    out_dir = f'{files.get_project_root()}/data/output/riddles/'
    ds = Dataset.from_csv(input_ds_path)
    cfg = cfg_reader.primary.load("conf/config.yaml")
    prompts = cfg_reader.primary.load("data/prompt/fin.yml")

    prompt_category_map = {
        "Factual Questions": "factual",
        "Reasoning Questions": "reasoning",
        "Following Instruction Questions": "following_instructions",
        "Context Reasoning Questions": "context_reasoning",
        "Short Term Prediction Questions": "short_term_prediction",
        "Volatility Risk Questions": "volatility_risk",
    }
    
    llms = [
        'BanglaLLM/bangla-llama-7b-instruct-v0.1',
        'Qwen/Qwen2.5-7B-Instruct',
        # 'meta-llama/Llama-3.2-1B-Instruct',
        'meta-llama/Llama-3.1-8B-Instruct',
        # "Qwen/QwQ-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        # 'Qwen/Qwen2.5-14B-Instruct',
        # 'allenai/OLMo-7B-Instruct',
        # 'mistralai/Mistral-7B-Instruct-v0.3',
        ]
    instruct_models = selector.get_instruct_models()

    results = []
    # system_information = "You are a seasoned financial analyst with expertise in global markets, equity analysis, and financial modeling. Provide detailed and data-driven responses including technical metrics tailored to financial experts."
                
    # system_information = "You are a financial analyst who explains topics such as global markets, stocks, and financial models in a simple and easy-to-understand manner. Provide clear, concise answers with relatable examples, avoiding technical jargon, to help a layperson easily grasp the concepts."
    system_information = "Solve these bengali riddles also provide your reasoning for it."

    # Loop through LLMs and generate responses
    for llm in tqdm(llms, desc='Evaluating LLMs'):
        display.in_progress(f'Generating response from {llm}')
        session = selector.select_chat_model(cfg, llm)

        # Process in batches of 16
        batch_size = 16
        for batch_start in tqdm(range(0, len(ds), batch_size), desc=f'Processing batches for {llm}'):
            batch_end = min(batch_start + batch_size, len(ds))
            batch_examples = ds[batch_start:batch_end]
            
            batch_inputs = batch_examples["Questions"]
            batch_ids = list(range(batch_start, batch_start + len(batch_examples)))
            system_information_batched = [system_information] * len(batch_inputs)
            batch_responses = []
            
            # Generate responses for the batch
            if llm in instruct_models:
                if llm in selector.get_gpt_models():
                    batch_responses = []
                    for input_text in batch_inputs:
                        response = session.get_response(input_text[-4000:], sysrole=None)
                        batch_responses.append(response if response is not None else '')
                else:
                    # Use batch inference
                    batch_responses = session.get_response(
                        user_message=batch_inputs,
                        system_message=system_information_batched,
                        clean_output=True)
            else:
                # Use batch inference with lists
                batch_responses = session.get_response(
                    user_message=batch_inputs, 
                    system_message=system_information_batched, 
                    clean_output=True
                    )
            
            # Add responses to results
            for i, (input_id, input_text, response) in enumerate(zip(batch_ids, batch_inputs, batch_responses)):
                existing_entry = next((item for item in results if item["id"] == input_id), None)
                if existing_entry:
                    existing_entry[f"{llm}"] = response
                    existing_entry["prompt"] = system_information
                else:
                    new_entry = {"id": input_id, "input": input_text, "prompt": system_information, f"{llm}": response}
                    results.append(new_entry)

        # clear resources
        cleanup_resources(session=session)

    # Convert to a pandas DataFrame for easier CSV export
    output_dataset = Dataset.from_list(results)
    df = output_dataset.to_pandas()

    # Save to CSV
    output_file = "llm_responses.csv"
    df.to_csv(f"{out_dir}/{output_file}", index=False)

    print(f"Results saved to {output_file}")
