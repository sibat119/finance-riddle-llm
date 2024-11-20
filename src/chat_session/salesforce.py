#external imports
import os
import yaml
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

# local imports
from .chat_session import ChatSession
from src.utils import display

class SalesforceSession(ChatSession):

    def __init__(self, config, model_name, temperature=0.1):
        """
        Initializes the huggingface generation chat session. This differs from the PipelineSession class by calling model.generate() instead of pipeline() to run inference.

        :param config_path: Path to the YAML configuration file.
        :param model_name: The name of the model to be used.
        """
        super().__init__(config, model_name, temperature)  # Initialize the base class with the loaded configuration

        self.model_name = model_name  # Model name is directly used
        self.max_length = config.get('max_length', 4096)
        self.num_output_tokens = config.get('num_output_tokens', 512)
        self.temperature = config.get('temperature', .1)
        # import here because i can't see a better way to set the cache directory
        #os.environ['TRANSFORMERS_CACHE'] = config['model_cache']

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if 'codet5' in model_name:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map='auto',
                trust_remote_code=True,
                cache_dir=config['model_cache'],
                load_in_8bit=self.use_8bit,
                do_sample=self.do_sample,
            )

        elif 'codegen' in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True,
                cache_dir=config['model_cache'],
                do_sample=self.do_sample,
            )
        else:
            display.error(f'model \'{model_name}\' is not a supported model for the SalesforceSession class')
            raise ValueError()
        self.model.eval()

    def get_session_type(self) -> str:
        return 'salesforce'

    def get_response(self,
                     user_message:   str | list,
                     system_message: str | list = None,
                     clean_output:   bool = True):
        """
        Retrieves a response from the generation model.
        """
        #msg = self._preprocess_msg(user_message, system_message)
        msg, return_str = self._prepare_batch(user_message, system_message)

        if 'codet5' in self.model_name:
            response = self.get_t5_response(msg)
        elif 'codegen' in self.model_name:
            response = self.get_codegen_response(msg)

        if clean_output:
            output = self._clean_output(response, msg)

        if return_str:
            return response[0]
        return response

    def get_t5_response(self, msg):

        max_len = self.max_length-self.num_output_tokens
        encoding = self.tokenizer(
            msg,
            max_length=max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            encoding = encoding.to('cuda')

        encoding['decoder_input_ids'] = encoding['input_ids'].clone()

        """
        outputs = self.model.generate(**encoding, max_length=self.max_length)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        """
        response = []
        for i in tqdm(range(0, len(msg), self.batch_size)):
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=encoding['input_ids'][i:i+self.batch_size],
                    attention_mask=encoding['attention_mask'][i:i+self.batch_size],
                    decoder_input_ids=encoding['decoder_input_ids'][i:i+self.batch_size],
                    #max_length=self.max_length,
                    max_new_tokens=self.num_output_tokens,
                    do_sample=self.do_sample,
                )
                response += self.tokenizer.batch_decode(output, skip_special_tokens=True)


        return response

    def get_codegen_response(self, msg):

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        #input_ids = self.tokenizer(
        tokens = self.tokenizer(
            msg,
            max_length=4096-self.num_output_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )#.input_ids
        if torch.cuda.is_available():
            tokens = tokens.to('cuda')

        response = []
        for i in tqdm(range(0, len(msg), self.batch_size)):
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=tokens['input_ids'][i:i+self.batch_size],
                    attention_mask=tokens['attention_mask'][i:i+self.batch_size],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    #max_length=self.max_length,
                    max_new_tokens=self.num_output_tokens,
                    do_sample=self.do_sample,
                )
                response += self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return response


