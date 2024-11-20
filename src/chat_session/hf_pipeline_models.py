#external imports
import os
import yaml
import torch
import transformers
from transformers import AutoTokenizer

# local imports
from .chat_session import ChatSession
from src.utils import display

class PipelineSession(ChatSession):
    """
    A subclass of ChatSession specifically for the LLAMA2 language model, using YAML configuration.
    """

    def __init__(self, config, model_name, temperature=0.1):
        """
        Initializes the LLAMA2 chat session.

        :param config_path: Path to the YAML configuration file.
        :param model_name: The name of the model to be used.
        """
        
        super().__init__(config, model_name, temperature)  # Initialize the base class with the loaded configuration

        # Set up LLAMA2 API credentials and model
        # Assuming LLAMA2 uses an API key and has a similar setup to OpenAI
        self.max_length = config.get('max_length', 4096)
        self.num_output_tokens = config.get('num_output_tokens', 512)
        self.temperature = config.get('temperature', .1)
        self.batch_size_dict=config.get('batch_size', None)
        self.batch_size = self.batch_size_dict[model_name] if model_name in self.batch_size_dict.keys() else self.batch_size_dict['default']
        # self.is_generation_model = is_generation_model
        # Initialize usage statistics
        self.usage = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
        }
        #model = "codellama/CodeLlama-13b-hf"

        # import here because i can't see a better way to set the cache directory
        #os.environ['TRANSFORMERS_CACHE'] = config['model_cache']

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = transformers.pipeline(
            'text-generation',
            model=model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            model_kwargs = {
                'cache_dir': config['model_cache'],
                'load_in_8bit': self.use_8bit, # do not set to true if use_4bit is set to true
                'load_in_4bit': self.use_4bit, # do not set to true if use_8bit is set to true
            }
        )
        self.pipeline.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def get_session_type(self) -> str:
        return 'pipeline'

    def get_response(self,
                     user_message:   str | list,
                     system_message: str | list=None,
                     clean_output:   bool = True):
        """
        Retrieves a response from the language model.
        """
        msg, return_str = self._prepare_batch(user_message, system_message)
        sequences = self.pipeline(
            msg,
            do_sample=True,
            top_k=10,
            temperature=self.temperature,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            #max_length=4096,
            max_new_tokens=self.num_output_tokens,
            batch_size=self.batch_size,
        )
        # Implement the logic to interact with the LLAMA2 model's API
        # This is a placeholder implementation
        response = [sample[0]['generated_text'] for sample in sequences]  # Replace this with actual API call logic
        if return_str:
            return response[0]
        else:
            return response

        # Update history and usage statistics
        # [Rest of the method should handle response parsing and updating the session similar to OpenAISession]

def get_gen_params():
    pass
