"""
Brainfork - An intelligent AI model router for Azure OpenAI

Automatically selects the best model for your specific use case based on 
conversation context, keywords, and predefined routing rules.
"""

from .router import ModelRouter
from .models import ModelConfig, AuthConfig, UseCase, RoutingResult, ConfiguredClient

__version__ = "0.1.0"
__author__ = "Pooyan Fekrati"
__email__ = "p.fekrati@hotmail.com"

__all__ = [
    "ModelRouter",
    "ModelConfig", 
    "AuthConfig",
    "UseCase",
    "RoutingResult",
    "ConfiguredClient",
]