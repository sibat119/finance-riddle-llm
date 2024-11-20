from datasets import Dataset
from tqdm import tqdm

from src.chat_session import (
    selector
)
from src import cfg_reader
from src.utils import (
    files,
    strings,
    display
)
def main():
    input_ds_path = f'{files.get_project_root()}/data/input/questions.csv'
    out_dir = f'{files.get_project_root()}/data/output'
    ds = Dataset.from_csv(input_ds_path)
    cfg = cfg_reader.primary.load("/home/sibat/repoes/LLM-Inference/conf/config.yaml")

    llms = ['meta-llama/Meta-Llama-3-8B-Instruct']
    instruct_models = selector.get_instruct_models()

    results = []

    # Loop through LLMs and generate responses
    for llm in tqdm(llms, desc='Evaluating LLMs'):
        display.in_progress(f'Generating response from {llm}')
        session = selector.select_chat_model(cfg, llm)

        # Generate responses for each input
        def generate_response(example):
            breakpoint()
            input_text = example["question"]

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
            example[f"{llm}-response"] = response
            return example

        # Use `map` to apply response generation
        output_dataset = ds.map(generate_response)

    # Convert to a pandas DataFrame for easier CSV export
    df = output_dataset.to_pandas()

    # Save to CSV
    output_file = "llm_responses.csv"
    df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")