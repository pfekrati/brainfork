"""Main router implementation for Brainfork."""

import json
from typing import Dict, List, Optional
from openai import AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential, ClientSecretCredential

from .models import (
    ModelConfig, 
    AuthConfig, 
    UseCase, 
    RoutingResult, 
    ConfiguredClient
)


class ModelRouter:
    """Intelligent AI model router for Azure OpenAI."""
    
    def __init__(
        self,
        models: Dict[str, ModelConfig],
        use_cases: List[UseCase],
        default_model: str,
        routing_model: Optional[str] = None,
        routing_temperature: float = 0
    ):
        """Initialize the ModelRouter.
        
        Args:
            models: Dictionary of model configurations
            use_cases: List of use case definitions
            default_model: Default model to use when no use case matches
            routing_model: Model to use for routing decisions (defaults to default_model)
            routing_temperature: Temperature for routing model decisions
        """
        self.models = models
        self.use_cases = use_cases
        self.default_model = default_model
        self.routing_model = routing_model or default_model
        self.routing_temperature = routing_temperature
        
        # Validate configurations
        self._validate_config()
    
    def _validate_config(self):
        """Validate the router configuration."""
        if self.default_model not in self.models:
            raise ValueError(f"Default model '{self.default_model}' not found in models")
        
        if self.routing_model not in self.models:
            raise ValueError(f"Routing model '{self.routing_model}' not found in models")
        
        for use_case in self.use_cases:
            if use_case.model_name not in self.models:
                raise ValueError(f"Model '{use_case.model_name}' in use case '{use_case.name}' not found in models")
    
    async def _get_client(self, model_configuration: ModelConfig) -> AsyncAzureOpenAI:
        """Get an authenticated Azure OpenAI client."""
        auth = model_configuration.auth
        
        if auth.api_key:
            return AsyncAzureOpenAI(
                api_key=auth.api_key,
                azure_endpoint=model_configuration.endpoint,
                api_version=model_configuration.api_version
            )
        elif auth.use_managed_identity:
            credential = DefaultAzureCredential()
            token = await credential.get_token("https://cognitiveservices.azure.com/.default")
            return AsyncAzureOpenAI(
                azure_ad_token=token.token,
                azure_endpoint=model_configuration.endpoint,
                api_version=model_configuration.api_version
            )
        elif auth.client_id and auth.client_secret and auth.tenant_id:
            credential = ClientSecretCredential(
                tenant_id=auth.tenant_id,
                client_id=auth.client_id,
                client_secret=auth.client_secret
            )
            token = await credential.get_token("https://cognitiveservices.azure.com/.default")
            return AsyncAzureOpenAI(
                azure_ad_token=token.token,
                azure_endpoint=model_configuration.endpoint,
                api_version=model_configuration.api_version
            )
        else:
            raise ValueError("No valid authentication method provided")
    
    async def route_conversation(self, messages: List[Dict]) -> RoutingResult:
        """Route a conversation to the most appropriate model.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            RoutingResult with selected model and reasoning
        """
        # Simple keyword-based routing for now
        # In a real implementation, you'd use the routing model to make decisions
        conversation_text = " ".join([msg.get("content", "") for msg in messages]).lower()
        
        best_match = None
        best_confidence = 0.0
        
        for use_case in self.use_cases:
            confidence = self._calculate_confidence(conversation_text, use_case)
            if confidence >= use_case.min_confidence and confidence > best_confidence:
                best_match = use_case
                best_confidence = confidence
        
        if best_match:
            return RoutingResult(
                model_name=best_match.model_name,
                selected_model=self.models[best_match.model_name],
                use_case=best_match,
                confidence=best_confidence,
                reasoning=f"Matched use case '{best_match.name}' with {best_confidence:.2f} confidence"
            )
        else:
            return RoutingResult(
                model_name=self.default_model,
                selected_model=self.models[self.default_model],
                use_case=None,
                confidence=1.0,
                reasoning="No specific use case matched, using default model"
            )
    
    def _calculate_confidence(self, text: str, use_case: UseCase) -> float:
        """Calculate confidence score for a use case match."""
        keyword_matches = sum(1 for keyword in use_case.keywords if keyword.lower() in text)
        if not use_case.keywords:
            return 0.0
        
        confidence = keyword_matches / len(use_case.keywords)
        return min(confidence, 1.0)
    
    async def get_configured_client(self, messages: List[Dict]) -> ConfiguredClient:
        """Get a configured client for the conversation.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            ConfiguredClient with the appropriate model
        """
        routing_result = await self.route_conversation(messages)
        client = await self._get_client(routing_result.selected_model)
        
        return ConfiguredClient(
            client=client,
            model_configuration=routing_result.selected_model
        )
    
    def get_model_info(self) -> Dict:
        """Get information about configured models and use cases."""
        return {
            "models": list(self.models.keys()),
            "use_cases": [{"name": uc.name, "model": uc.model_name} for uc in self.use_cases],
            "default_model": self.default_model,
            "routing_model": self.routing_model
        }