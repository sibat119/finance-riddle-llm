#external imports
import os
import yaml
import torch
import transformers
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

# local imports
from .chat_session import ChatSession

class HFGenerateSession(ChatSession):

    def __init__(self, config, model_name, temperature=0.1):
        """
        Initializes the huggingface generation chat session. This differs from the PipelineSession class by calling model.generate() instead of pipeline() to run inference.

        :param config_path: Path to the YAML configuration file.
        :param model_name: The name of the model to be used.
        """
        super().__init__(config, model_name, temperature)  # Initialize the base class with the loaded configuration

        # import here because i can't see a better way to set the cache directory
        #os.environ['TRANSFORMERS_CACHE'] = config['model_cache']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        self.trust_remote_code = False if 'phi-1' in model_name else True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype='auto',
            device_map='auto',
            trust_remote_code=self.trust_remote_code,
            # cache_dir=config['model_cache'],
        )

        self.generation_config = GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            #top_k=top_k,
            #num_beams=num_beams,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def get_session_type(self) -> str:
        return 'hf_generate'
              
    def get_response(self,
                     user_message:   str | list,
                     system_message: str | list = None,
                     clean_output:   bool = True):
        #input_ids = self.tokenizer(

        msg, return_str = self._prepare_batch(user_message, system_message)
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
            generated_ids = self.model.generate( 
                                                input_ids=tokens['input_ids'][i:i+self.batch_size], 
                                                attention_mask=tokens['attention_mask'][i:i+self.batch_size], 
                                                temperature=self.temperature, 
                                                top_p=self.top_p, 
                                                max_new_tokens=self.num_output_tokens, 
                                                do_sample=self.do_sample, 
                                                )
            response += self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        if clean_output:
            response = self._clean_output(response, msg)

        if return_str:
            return response[0]
        return response




