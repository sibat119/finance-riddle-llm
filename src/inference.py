from datasets import Dataset
from tqdm import tqdm
import os
import torch
import gc
import argparse
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run inference with specified parameters")
    parser.add_argument("--project", type=str, choices=["fin", "puzzle"], default="puzzle", 
                        help="Project to run inference on (fin/puzzle)")
    parser.add_argument("--model_name", type=str, default="", 
                        help="Model name to use for inference")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="batch size")
    args = parser.parse_args()
    
    # Set paths based on project
    prompt_structure = ""
    if args.project == "fin":
        input_ds_path = f'{files.get_project_root()}/data/input/fin/question_set_fin.csv'
        out_dir = f'{files.get_project_root()}/data/output/financial'
        system_information = "You are a financial analyst who explains topics such as global markets, stocks, and financial models in a simple and easy-to-understand manner. Provide clear, concise answers with relatable examples, avoiding technical jargon, to help a layperson easily grasp the concepts."
        prompt_structure= "You are a financial expert. Gather relevant and reliable information for the question and provide a detailed answer. Give proper explanation/reasoning for the answer. Don't include any unreliable information. Don't include any unreliable information. Do not include any unexplained jargon or technical terms, if you need to include them, please explain them in a way that is easy to understand. Do not say 'according to the sources' or 'mentioned in the <source>' or similar. Your response should be correct/true. Please verify your answer before responding.Question: {}"
    else:  # puzzle
        input_ds_path = f'{files.get_project_root()}/data/input/riddles-v1.csv'
        out_dir = f'{files.get_project_root()}/data/output/riddles/'
        system_information = "Solve these bengali riddles also provide your reasoning for it."
    
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
    
    # Use the specified model if provided, otherwise use the default list
    if args.model_name:
        llms = [args.model_name]
    else:
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
    
    # Loop through LLMs and generate responses
    for llm in tqdm(llms, desc='Evaluating LLMs'):
        display.in_progress(f'Generating response from {llm}')
        # session = None
        session = selector.select_chat_model(cfg, llm)

        # Process in batches of 16
        for batch_start in tqdm(range(1, len(ds), args.batch_size), desc=f'Processing batches for {llm}'):
            batch_end = min(batch_start + args.batch_size, len(ds))
            batch_examples = ds[batch_start:batch_end]
            
            if args.project == "fin":
                questions = batch_examples['Question']
                batch_inputs = [prompt_structure.format(q) for q in questions]
            else:
                batch_inputs = batch_examples["Questions"]
                questions = batch_examples["Questions"]
                
            batch_ids = list(range(batch_start, batch_start + len(batch_examples)))
            system_information_batched = [system_information] * len(batch_inputs)
            batch_responses = []
            
            batch_responses = session.get_response(
                user_message=batch_inputs, 
                system_message=system_information_batched, 
                clean_output=True
                )
            # batch_responses = batch_ids
            
            # Add responses to results
            for i, (input_id, question, input_text, response) in enumerate(zip(batch_ids, questions, batch_inputs, batch_responses)):
                existing_entry = next((item for item in results if item["id"] == input_id), None)
                if existing_entry:
                    existing_entry[f"{llm}"] = response
                else:
                    new_entry = {"id": input_id, "question": question, "prompt": input_text, f"{llm}": response}
                    results.append(new_entry)

        # clear resources
        cleanup_resources(session=session)

    # Check if output file already exists and load it
    if args.model_name:
        mname = args.model_name
        mname = mname.replace("/", "_")
        output_file = f"without_rag_responses{mname}.csv"
    else:
        output_file = f"without_rag_responses.csv"
        
    os.makedirs(out_dir, exist_ok=True)
    output_path = f"{out_dir}/{output_file}"
    
    try:
        # Try to load existing results
        existing_df = None
        try:
            if os.path.exists(output_path):
                existing_df = Dataset.from_csv(output_path).to_pandas()
                print(f"Loaded existing results from {output_path}")
        except Exception as e:
            print(f"Error loading existing file: {str(e)}")
            existing_df = None
            
        # Convert current results to DataFrame
        new_df = Dataset.from_list(results).to_pandas()
        
        if existing_df is not None:
            # Merge with existing results
            for llm in llms:
                if llm in new_df.columns:
                    # Add only the new model results to the existing DataFrame
                    existing_df[llm] = new_df[llm]
            df = existing_df
        else:
            # Use new results if no existing file
            df = new_df
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_file} with {len(llms)} model(s): {', '.join(llms)}")
        
    except Exception as e:
        print(f"Error processing results: {str(e)}")
        # Fallback to saving just the new results
        output_dataset = Dataset.from_list(results)
        df = output_dataset.to_pandas()
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_file}")

# if __name__ == "__main__":
#     main()
