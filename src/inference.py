from datasets import Dataset
from tqdm import tqdm
import torch
import gc
from typing import Dict, List, Tuple, Any, Optional
from torch.distributed import destroy_process_group

from src.chat_session import (
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
    input_ds_path = f'{files.get_project_root()}/data/input/questions.csv'
    out_dir = f'{files.get_project_root()}/data/output'
    ds = Dataset.from_csv(input_ds_path)
    cfg = cfg_reader.primary.load("conf/config.yaml")

    llms = [
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'OLMo-7B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
        'Qwen/Qwen2.5-7B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct'
        ]
    instruct_models = selector.get_instruct_models()

    results = []

    # Loop through LLMs and generate responses
    for llm in tqdm(llms, desc='Evaluating LLMs'):
        display.in_progress(f'Generating response from {llm}')
        session = selector.select_chat_model(cfg, llm)

        # Generate responses for each input
        for example in tqdm(ds):
            input_text = example["question"]
            input_id = example["id"]
            response = ""
            

            # Generate response for the input
            if llm in instruct_models:
                if llm in selector.get_gpt_models():
                    response = session.get_response(input_text[-4000:], sysrole=None)
                    if response is None:
                        response = ''
                else:
                    response = session.get_response(input_text)
            else:
                response = session.get_response([input_text], clean_output=True)

            # Add response as a new field for this LLM
            existing_entry = next((item for item in results if item["id"] == input_id), None)
            if existing_entry:
                existing_entry[f"{llm}"] = response
            else:
                new_entry = {"id": input_id, "input": input_text, f"{llm}": response}
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