"""
Client factory for creating configured AI model clients
"""

from typing import Union, Dict, Any
import asyncio
from openai import AsyncAzureOpenAI, AzureOpenAI
from azure.ai.inference.aio import ChatCompletionsClient as AsyncChatCompletionsClient
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import TokenCredential, AzureKeyCredential

from .models import ModelConfig, RoutingResult
from .auth import AuthenticationManager
from .exceptions import AuthenticationError, ConfigurationError


class ClientFactory:
    """Factory for creating configured AI model clients"""
    
    @staticmethod
    def create_openai_client(
        model_config: ModelConfig, 
        async_client: bool = True
    ) -> Union[AsyncAzureOpenAI, AzureOpenAI]:
        """Create an Azure OpenAI client"""
        
        auth_manager = AuthenticationManager(model_config.auth)
        credential = auth_manager.get_credential()
        
        client_kwargs = {
            "azure_endpoint": model_config.endpoint,
            "api_version": model_config.api_version,
        }
        
        if isinstance(credential, str):
            # API key authentication
            client_kwargs["api_key"] = credential
        else:
            # Token-based authentication
            client_kwargs["azure_ad_token_provider"] = lambda: credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            ).token
        
        if async_client:
            return AsyncAzureOpenAI(**client_kwargs)
        else:
            return AzureOpenAI(**client_kwargs)
    
    @staticmethod
    def create_inference_client(
        model_config: ModelConfig,
        async_client: bool = True
    ) -> Union[AsyncChatCompletionsClient, ChatCompletionsClient]:
        """Create an Azure AI Inference client"""
        
        auth_manager = AuthenticationManager(model_config.auth)
        credential = auth_manager.get_credential()
        
        if isinstance(credential, str):
            # API key authentication
            azure_credential = AzureKeyCredential(credential)
        elif isinstance(credential, TokenCredential):
            azure_credential = credential
        else:
            raise AuthenticationError("Invalid credential type for Azure AI Inference client")
        
        if async_client:
            return AsyncChatCompletionsClient(
                endpoint=model_config.endpoint,
                credential=azure_credential
            )
        else:
            return ChatCompletionsClient(
                endpoint=model_config.endpoint,
                credential=azure_credential
            )
    
    @staticmethod
    def create_client_from_result(
        result: RoutingResult,
        client_type: str = "openai",
        async_client: bool = True
    ) -> Union[AsyncAzureOpenAI, AzureOpenAI, AsyncChatCompletionsClient, ChatCompletionsClient]:
        """Create a client from a routing result"""
        
        if client_type.lower() == "openai":
            return ClientFactory.create_openai_client(result.model_config, async_client)
        elif client_type.lower() == "inference":
            return ClientFactory.create_inference_client(result.model_config, async_client)
        else:
            raise ConfigurationError(f"Unsupported client type: {client_type}")


class ConfiguredClient:
    """Wrapper that combines client with model configuration"""
    
    def __init__(
        self,
        client: Union[AsyncAzureOpenAI, AzureOpenAI, AsyncChatCompletionsClient, ChatCompletionsClient],
        model_config: ModelConfig,
        routing_result: RoutingResult
    ):
        self.client = client
        self.model_config = model_config
        self.routing_result = routing_result
    
    @property
    def deployment_name(self) -> str:
        """Get the deployment name for API calls"""
        return self.model_config.deployment_name
    
    @property
    def is_async(self) -> bool:
        """Check if this is an async client"""
        return isinstance(self.client, (AsyncAzureOpenAI, AsyncChatCompletionsClient))
    
    async def chat_completion(self, messages: list, **kwargs) -> Any:
        """Create a chat completion with appropriate parameters"""
        
        # Set default parameters from model config
        if "model" not in kwargs:
            kwargs["model"] = self.deployment_name
        
        if "temperature" not in kwargs and self.model_config.temperature is not None:
            kwargs["temperature"] = self.model_config.temperature
        
        if "max_tokens" not in kwargs and self.model_config.max_tokens is not None:
            kwargs["max_tokens"] = self.model_config.max_tokens
        
        # Call the appropriate method based on client type
        if isinstance(self.client, (AsyncAzureOpenAI, AzureOpenAI)):
            if self.is_async:
                return await self.client.chat.completions.create(messages=messages, **kwargs)
            else:
                return self.client.chat.completions.create(messages=messages, **kwargs)
        
        elif isinstance(self.client, (AsyncChatCompletionsClient, ChatCompletionsClient)):
            if self.is_async:
                return await self.client.complete(messages=messages, **kwargs)
            else:
                return self.client.complete(messages=messages, **kwargs)
        
        else:
            raise ConfigurationError("Unknown client type")
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get information about the configured client"""
        return {
            "model_name": self.routing_result.model_name,
            "deployment_name": self.deployment_name,
            "endpoint": self.model_config.endpoint,
            "client_type": type(self.client).__name__,
            "is_async": self.is_async,
            "use_case": self.routing_result.use_case.name if self.routing_result.use_case else None,
            "confidence": self.routing_result.confidence,
            "reasoning": self.routing_result.reasoning,
        }
