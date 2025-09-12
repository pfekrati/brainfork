"""Data models for Brainfork."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from openai import AsyncAzureOpenAI


class AuthConfig(BaseModel):
    """Authentication configuration for Azure OpenAI."""
    api_key: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    tenant_id: Optional[str] = None
    use_managed_identity: bool = False

    def validate_auth(self) -> bool:
        """Validate that at least one authentication method is provided."""
        if self.use_managed_identity:
            return True
        if self.api_key:
            return True
        if self.client_id and self.client_secret and self.tenant_id:
            return True
        return False


class ModelConfig(BaseModel):
    """Configuration for an Azure OpenAI model."""
    endpoint: str
    deployment_name: str
    api_version: str = "2025-01-01-preview"
    auth: AuthConfig
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class UseCase(BaseModel):
    """Definition of a use case for model routing."""
    name: str
    description: str
    model_name: str
    keywords: List[str] = Field(default_factory=list)
    context_requirements: List[str] = Field(default_factory=list)
    min_confidence: float = 0.7


class RoutingResult(BaseModel):
    """Result of model routing decision."""
    model_name: str
    selected_model: ModelConfig
    use_case: Optional[UseCase] = None
    confidence: float
    reasoning: str


class ConfiguredClient(BaseModel):
    """A configured Azure OpenAI client with its model config."""
    client: AsyncAzureOpenAI
    model_configuration: ModelConfig

    class Config:
        arbitrary_types_allowed = True