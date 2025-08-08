"""
Brainfork - Intelligent model routing for Azure AI Foundry
"""

from .router import ModelRouter
from .models import ModelConfig, AuthConfig, UseCase, RoutingResult, ConversationMessage
from .auth import AuthenticationManager
from .client_factory import ClientFactory, ConfiguredClient
from .routing_engine import RoutingEngine
from .exceptions import (
    BrainforkError,
    ModelNotFoundError,
    AuthenticationError,
    RoutingError,
    ConfigurationError,
)

__version__ = "0.1.0"
__all__ = [
    "ModelRouter",
    "ModelConfig", 
    "AuthConfig",
    "UseCase",
    "RoutingResult",
    "ConversationMessage",
    "AuthenticationManager",
    "ClientFactory",
    "ConfiguredClient",
    "RoutingEngine",
    "BrainforkError",
    "ModelNotFoundError", 
    "AuthenticationError",
    "RoutingError",
    "ConfigurationError",
]
