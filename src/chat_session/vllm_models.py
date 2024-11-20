#external imports
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '4'
#os.environ['WORLD_SIZE'] = '1'
import yaml
import torch
import transformers
from vllm import LLM, SamplingParams
from vllm.transformers_utils.config import get_config


# local imports
from .chat_session import ChatSession

class VllmSession(ChatSession):
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

        # set number of devices to use
        num_devices = config.get(
            'num_devices',
            torch.cuda.device_count()
        )
        
        # self.is_generation_model = is_generation_model
        tensor_parallel_size = self._set_tensor_parallel(num_devices)

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.num_output_tokens,
        )
        self.model = LLM(
            model_name,
            trust_remote_code=True,
            download_dir=config['model_cache'],
            dtype=self.dtype,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=self.max_length,
            # These options are for vllm==0.2.7
            max_context_len_to_capture=self.max_length,
            enforce_eager=True,
            worker_use_ray=True,
        )
    def get_session_type(self) -> str:
        return 'vllm'

    def get_response(self,
                     user_message:   str | list,
                     system_message: str | list = None,
                     clean_output:   bool = True):
        """
        Retrieves a response from the vLLM language model.
        """

        msg, return_str = self._prepare_batch(user_message, system_message)
        # Implement the logic to interact with the LLAMA2 model's API
        # This is a placeholder implementation

        # generate response
        #bsize=3
        #for i in range(0, len(msg), bsize):
        seqs = self.model.generate(
            msg,
            sampling_params=self.sampling_params,
        )

        # vLLM automatically removes the prompt from the output
        # so if clean_output is set to False then we need to add the prompt back in
        seqs = [seq.outputs[0].text for seq in seqs]
        if not clean_output:
            seqs = [prompt + seq for prompt, seq in zip(msg, seqs)]


        # Update history and usage statistics
        # [Rest of the method should handle response parsing and updating the session similar to OpenAISession]
        if return_str:
            return seqs[0]#.outputs[0].text
        else:
            #response = [seq.outputs[0].text for seq in seqs]
            return seqs

    def _set_tensor_parallel(self, num_devices):

        # get number of attention heads for the model
        n_head = self._get_num_attn_heads()

        tensor_parallel_size = num_devices
        while n_head%tensor_parallel_size != 0:
            tensor_parallel_size -= 1

        return tensor_parallel_size

    def _get_num_attn_heads(self):
        
        llm_cfg = get_config(self.model_name, trust_remote_code=True)

        # run through possible names for the number of attention heads in llm_cfg
        # this is necessary because the configs for each LLM are not standardized
        n_head = getattr(llm_cfg, 'num_attention_heads', None)
        n_head = getattr(llm_cfg, 'n_head', n_head)
        n_head = getattr(llm_cfg, 'num_heads', n_head)

        if n_head is None:
            print('n_head not set')
            breakpoint()
            raise ValueError()

        return n_head
