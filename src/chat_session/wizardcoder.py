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
from torch.nn.utils.rnn import pad_sequence

# local imports
from .chat_session import ChatSession

class WizardCoderSession(ChatSession):

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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
            cache_dir=config['model_cache'],
        )

        self.generation_config = GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=self.num_beams,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=self.do_sample,
        )

    def get_session_type(self) -> str:
        return 'wizardcoder'

    def get_response(self,
                     user_message:   str | list,
                     system_message: str | list = None,
                     clean_output:   bool = True):
        """
        Retrieves a response from the generation model.
        """
        msg, return_str = self._prepare_batch(user_message, system_message)

        #msg = self._preprocess_msg(user_message, system_message)
        tokens = self.tokenizer(
            msg, 
            max_length=4096-self.num_output_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            tokens = tokens.to('cuda')
        input_ids = tokens.input_ids
        seq = []
        for i in tqdm(range(0, len(input_ids), self.batch_size)):
            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids[i:i+self.batch_size],
                    max_new_tokens=self.num_output_tokens,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            seq.append(generation_output.sequences)
        #seq = torch.vstack(seq)

        response = []
        for group in seq:
            response += self.tokenizer.batch_decode(group, skip_special_tokens=True)

        assert len(input_ids) == len(response)

        if clean_output:
            output = self._clean_output(response, msg)

        if return_str:
            return output[0]
        else:
            return output


