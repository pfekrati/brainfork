"""
Main router class that orchestrates model selection and client creation
"""

import asyncio
from typing import Dict, List, Optional, Union, Any

from .models import (
    ModelConfig,
    UseCase, 
    RoutingResult,
    ConversationMessage
)
from .routing_engine import RoutingEngine
from .client_factory import ClientFactory, ConfiguredClient
from .exceptions import ModelNotFoundError, ConfigurationError


class ModelRouter:
    """Main router class for intelligent AI model selection"""
    
    def __init__(
        self,
        models: Dict[str, ModelConfig],
        use_cases: List[UseCase],
        default_model: str,
        routing_model: Optional[str] = None,
        routing_temperature: float = 0.1,
        enable_caching: bool = True,
        client_type: str = "openai"
    ):
        """
        Initialize the model router
        
        Args:
            models: Dictionary of available models {name: config}
            use_cases: List of use case definitions
            default_model: Name of default model to use when no use case matches
            routing_model: Name of model to use for routing decisions (defaults to default_model)
            routing_temperature: Temperature for routing model (lower = more deterministic)
            enable_caching: Whether to cache routing decisions
            client_type: Type of client to create ("openai" or "inference")
        """
        
        self.models = models
        self.use_cases = use_cases
        self.default_model = default_model
        self.routing_model = routing_model or default_model
        self.client_type = client_type
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize routing engine
        self.routing_engine = RoutingEngine(
            routing_model_config=self.models[self.routing_model],
            use_cases=use_cases,
            routing_temperature=routing_temperature,
            enable_caching=enable_caching
        )
    
    def _validate_configuration(self) -> None:
        """Validate the router configuration"""
        
        if not self.models:
            raise ConfigurationError("At least one model must be configured")
        
        if self.default_model not in self.models:
            raise ModelNotFoundError(f"Default model '{self.default_model}' not found in models")
        
        if self.routing_model not in self.models:
            raise ModelNotFoundError(f"Routing model '{self.routing_model}' not found in models")
        
        # Validate use cases reference existing models
        for use_case in self.use_cases:
            if use_case.model_name not in self.models:
                raise ModelNotFoundError(
                    f"Use case '{use_case.name}' references unknown model '{use_case.model_name}'"
                )
        
        # Check for duplicate use case names
        use_case_names = [uc.name for uc in self.use_cases]
        if len(use_case_names) != len(set(use_case_names)):
            raise ConfigurationError("Use case names must be unique")
    
    async def route_conversation(
        self,
        messages: List[Union[Dict[str, Any], ConversationMessage]]
    ) -> RoutingResult:
        """
        Route a conversation to the most appropriate model
        
        Args:
            messages: List of conversation messages
            override_model: Optional model name to force use (skips routing)
            
        Returns:
            RoutingResult with selected model and reasoning
        """
        
        if not messages:
            raise ConfigurationError("Messages list cannot be empty")
    
        
        # Use routing engine to analyze conversation
        return await self.routing_engine.analyze_conversation(
            messages=messages,
            available_models=self.models,
            default_model=self.default_model
        )
    
    async def get_configured_client(
        self,
        messages: List[Union[Dict[str, Any], ConversationMessage]],
        override_model: Optional[str] = None,
        async_client: bool = True
    ) -> ConfiguredClient:
        """
        Get a configured client for the conversation
        
        Args:
            messages: List of conversation messages
            override_model: Optional model name to force use
            async_client: Whether to create an async client
            
        Returns:
            ConfiguredClient ready for use
        """
        
        # Get routing result
        routing_result = await self.route_conversation(messages, override_model)
        
        # Create appropriate client
        if self.client_type.lower() == "openai":
            client = ClientFactory.create_openai_client(
                routing_result.model_config,
                async_client=async_client
            )
        elif self.client_type.lower() == "inference":
            client = ClientFactory.create_inference_client(
                routing_result.model_config,
                async_client=async_client
            )
        else:
            raise ConfigurationError(f"Unsupported client type: {self.client_type}")
        
        return ConfiguredClient(
            client=client,
            model_config=routing_result.model_config,
            routing_result=routing_result
        )
    
    def route_conversation_sync(
        self,
        messages: List[Union[Dict[str, Any], ConversationMessage]],
        override_model: Optional[str] = None
    ) -> RoutingResult:
        """
        Synchronous version of route_conversation
        
        Args:
            messages: List of conversation messages
            override_model: Optional model name to force use
            
        Returns:
            RoutingResult with selected model and reasoning
        """
        
        return asyncio.run(self.route_conversation(messages, override_model))
    
    def get_configured_client_sync(
        self,
        messages: List[Union[Dict[str, Any], ConversationMessage]],
        override_model: Optional[str] = None
    ) -> ConfiguredClient:
        """
        Synchronous version of get_configured_client
        
        Args:
            messages: List of conversation messages
            override_model: Optional model name to force use
            
        Returns:
            ConfiguredClient ready for use (sync client)
        """
        
        return asyncio.run(
            self.get_configured_client(messages, override_model, async_client=False)
        )
    
    def add_model(self, name: str, config: ModelConfig) -> None:
        """Add a new model to the router"""
        self.models[name] = config
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the router"""
        if name == self.default_model:
            raise ConfigurationError("Cannot remove default model")
        if name == self.routing_model:
            raise ConfigurationError("Cannot remove routing model")
        
        # Check if any use cases depend on this model
        dependent_use_cases = [uc.name for uc in self.use_cases if uc.model_name == name]
        if dependent_use_cases:
            raise ConfigurationError(
                f"Cannot remove model '{name}' - used by use cases: {dependent_use_cases}"
            )
        
        del self.models[name]
    
    def add_use_case(self, use_case: UseCase) -> None:
        """Add a new use case to the router"""
        if use_case.model_name not in self.models:
            raise ModelNotFoundError(f"Use case references unknown model '{use_case.model_name}'")
        
        # Check for duplicate names
        existing_names = [uc.name for uc in self.use_cases]
        if use_case.name in existing_names:
            raise ConfigurationError(f"Use case '{use_case.name}' already exists")
        
        self.use_cases.append(use_case)
        
        # Update routing engine
        self.routing_engine.use_cases = self.use_cases
    
    def remove_use_case(self, name: str) -> None:
        """Remove a use case from the router"""
        self.use_cases = [uc for uc in self.use_cases if uc.name != name]
        self.routing_engine.use_cases = self.use_cases
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about configured models and use cases"""
        return {
            "models": {
                name: {
                    "endpoint": config.endpoint,
                    "deployment_name": config.deployment_name,
                    "api_version": config.api_version,
                    "model_type": config.model_type,
                    "auth_type": config.auth.auth_type.value
                }
                for name, config in self.models.items()
            },
            "use_cases": [
                {
                    "name": uc.name,
                    "description": uc.description,
                    "model_name": uc.model_name,
                    "priority": uc.priority,
                    "keywords": uc.keywords,
                    "min_confidence": uc.min_confidence
                }
                for uc in self.use_cases
            ],
            "default_model": self.default_model,
            "routing_model": self.routing_model,
            "client_type": self.client_type
        }
    
    def clear_cache(self) -> None:
        """Clear the routing cache"""
        self.routing_engine._routing_cache.clear()
