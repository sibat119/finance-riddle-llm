import torch
from typing import List

from src.utils import (
    display
)


class ChatSession:
    """
    A generic class to manage chat sessions with different language models.
    This class can be used as a base class for specific implementations for
    different LLMs, including open-source models and API-only models.
    """

    def __init__(self, config, model_name, temperature=0.1):
        """
        Initializes the chat session with a configuration.

        :param config: A dictionary containing configuration settings.
        """
        self.config = config
        self.msg_history = []
        self.usage = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
        }

        # set model name
        self.model_name = model_name
        
        # set values from config
        self.max_length = config.get('max_length', 4096)
        self.num_output_tokens = config.get('num_output_tokens', 1024)
        self.dtype = config.get('dtype', 'auto')

        # set temp and top_p
        default_temp = temperature
        default_top_p = .95
        default_top_k = 1
        default_num_beams = 1

        self.temperature = default_temp
        self.top_p = default_top_p
        self.top_k = default_top_k
        self.num_beams = default_num_beams

        self.do_sample = config.get('use_default_sampling_params', True)
        
        if self.do_sample:
            self.temperature = temperature
            self.top_p = config.get('top_p', .95)
            self.top_k = config.get('top_k', 1)
            self.num_beams = config.get('num_beams', 1)

        # set bach size specified in config
        if model_name in config['batch_size'].keys():
            self.batch_size = config['batch_size'][model_name]
        else:
            self.batch_size = config['batch_size']['default']

        # set datatype for huggingface/vllm models
        if self.dtype == 'float16':
            self.dtype = torch.float16

        # set whether to use the quantized version of a given LLM
        models_8bit = config.get('8bit_models', [])
        self.use_8bit = self.model_name in models_8bit

        models_4bit = config.get('4bit_models', [])
        self.use_4bit = self.model_name in models_4bit

    def get_session_type(self) -> str:
        display.error('This class needs to implement the get_session_type() function')
        raise NotImplementedError()
        
    def get_response(self,
                     user_message:   str | list,
                     system_message: str | list = None,
                     clean_output:   bool = True):
        """
        Retrieves a response from the language model.
        This method should be overridden in subclasses.

        :param message: The message to be sent to the model.
        :return: The response from the model.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    def update_history(self, role, content):
        """
        Updates the message history.

        :param role: The role of the message sender ('user' or 'assistant').
        :param content: The content of the message.
        """
        self.msg_history.append({
            'role': role,
            'content': content
        })

    def get_history(self):
        """
        Returns the message history.

        :return: A list of message dictionaries.
        """
        return self.msg_history

    def get_usage(self):
        """
        Returns the usage statistics of the session.

        :return: A dictionary containing usage statistics.
        """
        return self.usage

    def __call__(self, message):
        """
        Shortcut for get_response.

        :param message: The message to be sent to the model.
        :return: The response from the model.
        """
        return self.get_response(message)

    def __str__(self):
        """
        Returns the message history as a JSON formatted string.
        """
        import json
        return json.dumps(self.msg_history, indent=4)

    def _prepare_batch(self, usr_msg, sys_msg=None, is_generation_model=False):

        # convert string input to list
        return_str=False
        if type(usr_msg) == str:
            msg = [self._preprocess_msg(usr_msg, sys_msg, is_generation_model)]
            return_str=True
            return msg, return_str

        if sys_msg is None:
            sys_msg = [None]*len(usr_msg)

        # ensure length of usr_msg and sys_msg match
        if len(usr_msg) != len(sys_msg):
            display.error('length of usr_msg does not match length of sys_msg')
            raise ValueError()

        msg = [self._preprocess_msg(prompt, sys) 
               for prompt, sys in zip(usr_msg, sys_msg)]
        
        return msg, return_str

    def _preprocess_instruct_model_msg(self, usr_msg, sys_msg=None):
        
        if self.model_name == 'Phind/Phind-CodeLlama-34B-v2':
            msg = f'### User Message\n{usr_msg}\n\n### Assistant\n'
            if sys_msg is not None:
                msg = f'### System Prompt\n{sys_msg}\n\n' + msg
            return msg
        elif 'WizardLM' in self.model_name:
            msg = f'### Instruction:\n{usr_msg}\n\n### Response:'
            if sys_msg is not None:
                msg = f'{sys_msg.strip()}\n\n' + msg
            else:
                msg = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n' + msg
            return msg
        elif 'codellama' in self.model_name and 'Instruct' in self.model_name:
            msg = f'[INST]{usr_msg.strip()}[/INST]'
            if sys_msg is not None:
                msg = f'<<SYS>>{sys_msg.strip()}<</SYS>>' + msg
            return msg
        elif 'Salesforce' in self.model_name and 'instruct' in self.model_name:
            msg = f'### Instruction:\n{usr_msg.strip()}\n\n### Response:\n'
            if sys_msg is not None:
                msg = f'{sys_msg.strip()}\n\n' + msg
            else:
                msg = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n' + msg

            return msg
        elif self.model_name == 'mistralai/Mistral-7B-Instruct-v0.1':
            msg = f'[INST] {usr_msg} [/INST]'
            return msg 
        elif 'lmsys/vicuna' in self.model_name:
            msg = f'USER: {usr_msg.strip()}\nASSISTANT: '
            if sys_msg is not None:
                msg = sys_msg.strip() + '\n\n' + msg
            return msg
        elif self._needs_chat_template():
            return self._apply_chat_template(usr_msg, sys_msg)
        else:
            return usr_msg
    
    def _needs_chat_template(self):
        return self.model_name in [
            "google/codegemma-7b-it",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "THUDM/codegeex4-all-9b",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            'Qwen/Qwen2.5-7B-Instruct',
        ]
        
    def _apply_chat_template(self, usr_msg, sys_msg=None):
        msg = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg},
            ]
        # breakpoint()
        if ('Meta-Llama-3-8B-Instruct' in self.model_name 
            or 'open_llama_13b' in self.model_name
            ):
            message = self.tokenizer.apply_chat_template(
                msg,
                tokenize=False,
            )
            
        elif 'OLMo-7B-Instruct' in self.model_name:
            message = self.tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True
            )
        elif (
            'Qwen' in self.model_name  or
            "codegeex4" in self.model_name):
            message = self.tokenizer.apply_chat_template(
                msg, 
                tokenize=False, 
                add_generation_prompt=True
                )
        elif ("codegemma" in self.model_name):
            msg = [
                {"role": "user", "content": f"{sys_msg}\n\n{usr_msg}"},
            ]
            message = self.tokenizer.apply_chat_template(
                msg, 
                tokenize=False, 
                add_generation_prompt=True
                )
            
        return message
    def _preprocess_generation_model_msg(self, usr_msg, sys_msg=None):
        # TODO: Finish implementation.
        breakpoint()
        return usr_msg
    
    def _preprocess_msg(self, usr_msg, sys_msg=None, is_generation_model=False):
        
        if is_generation_model:
            return self._preprocess_generation_model_msg(usr_msg, sys_msg)
        else:
            return self._preprocess_instruct_model_msg(usr_msg, sys_msg)
        
        

    def _clean_output(self, 
                      output: List[str],
                      prompts: List[str]) -> List[str]:
        """
        huggingface generate and pipeline models include the prompt in the response so this function filters the
        original prompt from the output messages
        """

        cleaned_output = []
        for base_msg, out_msg in zip(prompts, output):
            cleaned_output.append(out_msg.replace(base_msg, ''))
        return cleaned_output
