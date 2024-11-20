import psutil
import openai
import os
from openai import OpenAI

# local imports
from .chat_session import ChatSession
from src.utils import (
    display,
)

class OpenAISession(ChatSession):
    """
    A subclass of ChatSession specifically for OpenAI's language models, using YAML configuration.
    """

    def __init__(self, config, model_name, temperature=0.1):
        """
        Initializes the OpenAI chat session.

        :param config_path: Path to the YAML configuration file.
        :param model_name: The name of the model to be used (e.g., '3.5' or '4').
        """
        # # Load configuration from YAML file
        # with open(config_path, 'r') as file:
        #     full_config = yaml.safe_load(file)
        
        # config = full_config['data_path']
        
        super().__init__(config, model_name, temperature)  # Initialize the base class with the loaded configuration

        # Set up OpenAI API credentials and model
        self.api_key = self.config.get('openai', None).get('key', None)
        # self.organization = self.config.get('openai', None).get('organization', None)
        
        # Determine the model based on the provided model name
        self.model_name = model_name
        
        os.environ['OPENAI_API_KEY'] = self.api_key

        
        # Initialize usage statistics
        self.usage = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
        }
    def get_session_type(self) -> str:
        return 'openai'

    def get_response(self, user_message, system_message=None):
        """
        Retrieves a response from OpenAI's language model.

        :param user_message: The user message to be sent to the model.
        :param system_message: An optional system message.
        :return: The response from the model.
        """
        
        
        """
        user_message = user_message[0] if isinstance(user_message, list) else user_message
        system_message = system_message[0] if isinstance(system_message, list) else system_message
        
        if system_message:
            self.update_history('system', system_message)
            self.usage['prompt_tokens'] += len(system_message.split())

        if isinstance(user_message, str):
            user_message = [user_message]
        """

        responses = []
        if type(user_message) == str:
            user_message = [user_message]
        # breakpoint()
        self.update_history('system', system_message)
        for msg in user_message:
            self.update_history('user', msg)

            try:
                # client = OpenAI()
                with OpenAI() as client:
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=self.msg_history,
                    )
            except KeyboardInterrupt as e:
                print(e)
                import sys; sys.exit(0)
            except openai.OpenAIError as e:
                display.error(f'Error occurred during Openai API call.\n\tError message: {e}')
                return None
            except Exception as e:
                display.error(f"An error unrelated to the OpenAI API occurred: {e}")
                return None

            response_content = response.choices[0].message.content
            responses.append(response_content)
            self.update_history('assistant', response_content)

            # Update usage statistics
            self.usage['completion_tokens'] += len(response_content.split())
            self.usage['prompt_tokens'] += len(msg.split())
            self.usage['total_tokens'] = self.usage['completion_tokens'] + self.usage['prompt_tokens']
            print(self.usage['total_tokens'])

        return responses if len(responses) > 1 else [responses[0]]
