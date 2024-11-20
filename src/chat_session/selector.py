from .chat_session import ChatSession
from .openai_gpt import OpenAISession
from .hf_pipeline_models import PipelineSession
from .hf_generate_models import HFGenerateSession
from .wizardcoder import WizardCoderSession
from .salesforce import SalesforceSession
# from .vllm_models import VllmSession
from .incoder import IncoderSession

from src.utils import (
    display
)

def select_chat_model(cfg: dict, model_name: str, temperature: float = 0.1) -> ChatSession:
    """
    returns a ChatSession object given the model name and config
    Input:
        cfg[dict]: the config to initialize the object
        model_name[str]: name of the model

    Output:
        ChatSession object
    """
    if model_name in get_gpt_models():
        return  OpenAISession(cfg, model_name, temperature)
    # elif model_name in get_vllm_models():
    #     return VllmSession(cfg, model_name, temperature)
    elif model_name in get_pipeline_models():
        return PipelineSession(cfg, model_name, temperature)
    elif model_name in get_incoder_models():
        return IncoderSession(cfg, model_name, temperature)
    elif model_name in get_wizardlm_models():
        return WizardCoderSession(cfg, model_name, temperature)
    elif model_name in get_salesforce_models():
        return SalesforceSession(cfg, model_name, temperature)
    elif model_name in get_hf_generate_models() or get_vllm_models():
        return HFGenerateSession(cfg, model_name, temperature)
    else:
        display.error(f'model: {model_name} is an unsupported option')
        raise ValueError()

def get_all_models() -> list:
    return sorted(
        get_instruct_models()
        + get_generation_models()
    )

def get_gpt_models() -> list:
    return [
        'gpt-3.5-turbo',
        'gpt-4',
        # 'gpt-4-32',
    ]

def get_instruct_models() -> list:
    return sorted([
        'gpt-3.5-turbo',
        'gpt-4',
        'Phind/Phind-CodeLlama-34B-v2',
        'WizardLM/WizardCoder-1B-V1.0',
        'WizardLM/WizardCoder-3B-V1.0',
        'WizardLM/WizardCoder-15B-V1.0',
        'WizardLMTeam/WizardCoder-15B-V1.0',
        'codellama/CodeLlama-7b-Instruct-hf',
        'codellama/CodeLlama-13b-Instruct-hf',
        'codellama/CodeLlama-34b-Instruct-hf',
        'Salesforce/codegen25-7b-instruct',
        'Salesforce/instructcodet5p-16b',
        'mistralai/Mistral-7B-Instruct-v0.1',
        'mistralai/Mistral-7B-Instruct-v0.2',
        'lmsys/vicuna-7b-v1.5',
        'lmsys/vicuna-7b-v1.5-16k',
        'lmsys/vicuna-13b-v1.5',
        'lmsys/vicuna-13b-v1.5-16k',
        'meta-llama/Llama-3.2-1B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'Qwen/Qwen2.5-Coder-1.5B-Instruct',
        'Qwen/Qwen2.5-Coder-7B-Instruct',
        'THUDM/chatglm2-6b',
        'microsoft/Phi-3.5-mini-instruct',
        'microsoft/Phi-3-mini-4k-instruct',
        'THUDM/codegeex4-all-9b',
        'google/codegemma-7b-it',
        'mistralai/Codestral-22B-v0.1',
        'mistralai/Mamba-Codestral-7B-v0.1',
        'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
        'deepseek-ai/DeepSeek-Coder-V2-Instruct',
        'deepseek-ai/deepseek-coder-6.7b-instruct',
        'stabilityai/stable-code-instruct-3b',
        'allenai/OLMo-7B-Instruct-hf',
        'allenai/OLMo-7B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
        'Qwen/Qwen2.5-7B-Instruct',
        'bavest/fin-llama-33b-merged',
    ])

def get_pipeline_models() -> list:
    return [
        # 'codellama/CodeLlama-7b-hf',
        # 'codellama/CodeLlama-13b-hf',
        'codellama/CodeLlama-34b-hf',
        'EleutherAI/gpt-neo-2.7B',
        'microsoft/Phi-3.5-mini-instruct',
        'microsoft/Phi-3-mini-4k-instruct',
        'meta-llama/Llama-3.2-1B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
    ]

def get_wizardlm_models() -> list:
    models = get_instruct_models()
    models = list(filter(lambda x: 'WizardLM' in x, models))

    return models

def get_salesforce_models() -> list:
    #models = get_instruct_models()
    models = get_all_models()
    models = list(filter(lambda x: 'Salesforce' in x, models))
    
    if 'Salesforce/codegen25-7b-instruct' not in models:
        models.append('Salesforce/codegen25-7b-instruct')

    return models

def get_vllm_models() -> list:
    return sorted([
        'mistralai/Mistral-7B-v0.1',
        'mistralai/Mistral-7B-Instruct-v0.1',
        'mistralai/Mistral-7B-Instruct-v0.2',
        'lmsys/vicuna-13b-v1.3',
        'Phind/Phind-CodeLlama-34B-v2',
        'codellama/CodeLlama-7b-Instruct-hf',
        'codellama/CodeLlama-13b-Instruct-hf',
        'codellama/CodeLlama-34b-Instruct-hf',
        'codellama/CodeLlama-7b-hf',
        'codellama/CodeLlama-13b-hf',
        'codellama/CodeLlama-34b-hf',
        'lmsys/vicuna-7b-v1.5',
        'lmsys/vicuna-7b-v1.5-16k',
        'lmsys/vicuna-13b-v1.5',
        'lmsys/vicuna-13b-v1.5-16k',
        'bigcode/starcoder',
        'bigcode/gpt_bigcode-santacoder',
        'bigcode/santacoder',
        'bigcode/starcoder',
        'bigcode/starcoderplus',
        'EleutherAI/gpt-j-6b',
        #'EleutherAI/gpt-neo-2.7B',
        'EleutherAI/gpt-neox-20b',
        #'stabilityai/stablelm-base-alpha-7b-v2',
        'meta-llama/Llama-3.2-1B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
    ])

def get_generation_models() -> list:
    return sorted([
        'bigcode/santacoder',
        'bigcode/gpt_bigcode-santacoder',
        'bigcode/starcoder',
        'bigcode/starcoderplus',
        'bigcode/starcoder2-7b',
        'bigcode/starcoder2-3b',
        'codellama/CodeLlama-7b-hf',
        'codellama/CodeLlama-13b-hf',
        'codellama/CodeLlama-34b-hf',
        'Salesforce/codegen-2B-multi',
        'Salesforce/codegen-2B-nl',
        'Salesforce/codegen-6B-multi',
        'Salesforce/codegen-6B-nl',
        'Salesforce/codegen-16B-multi',
        'Salesforce/codegen-16B-nl',
        'Salesforce/codegen2-1B',
        'Salesforce/codegen2-3_7B',
        'Salesforce/codegen2-7B',
        'Salesforce/codegen2-16B',
        'Salesforce/codet5p-2b',
        'Salesforce/codet5p-6b',
        'Salesforce/codet5p-16b',
        'Salesforce/codegen-350M-mono'
        'mistralai/Mistral-7B-v0.1',
        'facebook/incoder-1B',
        'facebook/incoder-6B',
        'EleutherAI/gpt-j-6b',
        'EleutherAI/gpt-neo-2.7B',
        'EleutherAI/gpt-neox-20b',
        'NinedayWang/PolyCoder-2.7B',
        'stabilityai/stablelm-base-alpha-7b-v2',
        'microsoft/phi-1', 
        'microsoft/phi-1_5',
        'microsoft/phi-2',
        'google/codegemma-2b',
        'google/codegemma-7b',
        'Salesforce/codegen25-7b-multi_P',
        'Salesforce/codegen-350M-multi',
        'replit/replit-code-v1_5-3b',
        'replit/replit-code-v1-3b',
        'stabilityai/stable-code-3b',
        
    ])

def get_hf_generate_models() -> list:
    models = sorted([
        'bigcode/santacoder',
        'bigcode/starcoder',
        'bigcode/starcoderplus',
        'bigcode/starcoder2-7b',
        'bigcode/starcoder2-3b',
        'EleutherAI/gpt-neo-2.7B',
        'NinedayWang/PolyCoder-2.7B',
        'stabilityai/stablelm-base-alpha-7b-v2',
        'microsoft/phi-1', 
        'microsoft/phi-1_5',
        'microsoft/phi-2',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'Qwen/Qwen2.5-Coder-1.5B-Instruct',
        'Qwen/Qwen2.5-Coder-7B-Instruct',
        'THUDM/chatglm2-6b',
        'google/codegemma-2b',
        'google/codegemma-7b',
        'google/codegemma-7b-it',
        'THUDM/codegeex4-all-9b',
        'mistralai/Codestral-22B-v0.1',
        'mistralai/Mamba-Codestral-7B-v0.1',
        'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
        'deepseek-ai/DeepSeek-Coder-V2-Instruct',
        'deepseek-ai/deepseek-coder-6.7b-instruct',
        'replit/replit-code-v1_5-3b',
        'replit/replit-code-v1-3b',
        'stabilityai/stable-code-3b',
        'stabilityai/stable-code-instruct-3b',
        'allenai/OLMo-7B-Instruct-hf',
        'allenai/OLMo-7B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
        'Qwen/Qwen2.5-7B-Instruct',
        'bavest/fin-llama-33b-merged'
    ])
    return models

def get_incoder_models() -> list:
    models = [
        'facebook/incoder-1B',
        'facebook/incoder-6B',
    ]
    return models



