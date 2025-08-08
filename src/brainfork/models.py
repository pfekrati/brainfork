"""
Data models for Azure AI Router configuration
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class AuthType(str, Enum):
    """Authentication types supported"""
    API_KEY = "api_key"
    ENTRA_ID = "entra_id"
    MANAGED_IDENTITY = "managed_identity"


class AuthConfig(BaseModel):
    """Authentication configuration for a model"""
    
    api_key: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    tenant_id: Optional[str] = None
    use_managed_identity: bool = False
    
    @validator('api_key', 'client_id', 'client_secret', 'tenant_id')
    def validate_strings(cls, v):
        if v is not None and not isinstance(v, str):
            raise ValueError("Must be a string")
        return v
    
    @property
    def auth_type(self) -> AuthType:
        """Determine the authentication type based on provided credentials"""
        if self.use_managed_identity:
            return AuthType.MANAGED_IDENTITY
        elif self.client_id and self.tenant_id:
            return AuthType.ENTRA_ID
        elif self.api_key:
            return AuthType.API_KEY
        else:
            raise ValueError("No valid authentication method configured")


class ModelConfig(BaseModel):
    """Configuration for an AI model"""
    
    endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    deployment_name: str = Field(..., description="Model deployment name")
    api_version: str = Field(default=None, description="API version")
    auth: AuthConfig = Field(..., description="Authentication configuration")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens for this model")
    temperature: Optional[float] = Field(default=None, description="Default temperature")
    
    @validator('endpoint')
    def validate_endpoint(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Endpoint must be a valid URL")
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return v


class UseCase(BaseModel):
    """Definition of a use case for model routing"""
    
    name: str = Field(..., description="Unique name for the use case")
    description: str = Field(..., description="Description of when to use this model")
    model_name: str = Field(..., description="Name of the model to use for this case")
    priority: int = Field(default=1, description="Priority level (1=highest)")
    keywords: List[str] = Field(default_factory=list, description="Keywords that trigger this use case")
    context_requirements: List[str] = Field(default_factory=list, description="Context requirements")
    min_confidence: float = Field(default=0.7, description="Minimum confidence score to trigger")
    
    @validator('priority')
    def validate_priority(cls, v):
        if v < 1:
            raise ValueError("Priority must be 1 or higher")
        return v
    
    @validator('min_confidence')
    def validate_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class RoutingResult(BaseModel):
    """Result of model routing decision"""
    
    model_name: str = Field(..., description="Selected model name")
    model_config: ModelConfig = Field(..., description="Configuration of selected model")
    use_case: Optional[UseCase] = Field(None, description="Matched use case")
    confidence: float = Field(..., description="Confidence in routing decision")
    reasoning: str = Field(..., description="Explanation of routing decision")
    
    @property
    def endpoint(self) -> str:
        """Get the model endpoint"""
        return self.model_config.endpoint
    
    @property
    def deployment_name(self) -> str:
        """Get the deployment name"""
        return self.model_config.deployment_name
    
    @property
    def api_version(self) -> str:
        """Get the API version"""
        return self.model_config.api_version
    
    def get_client_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for creating a client"""
        return {
            "endpoint": self.model_config.endpoint,
            "deployment_name": self.model_config.deployment_name,
            "api_version": self.model_config.api_version,
            "auth_config": self.model_config.auth,
        }


class ConversationMessage(BaseModel):
    """Represents a message in a conversation"""
    
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name for the message sender")
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant', 'system', 'function', 'tool']:
            raise ValueError("Role must be one of: user, assistant, system, function, tool")
        return v
