"""
Authentication management for Azure AI models
"""

import os
from typing import Optional, Union
from azure.identity import (
    DefaultAzureCredential,
    ClientSecretCredential,
    ManagedIdentityCredential,
)
from azure.core.credentials import TokenCredential

from .models import AuthConfig, AuthType
from .exceptions import AuthenticationError


class AuthenticationManager:
    """Manages authentication for Azure AI models"""
    
    def __init__(self, auth_config: AuthConfig):
        self.auth_config = auth_config
        self._credential: Optional[TokenCredential] = None
    
    def get_credential(self) -> Union[str, TokenCredential]:
        """Get the appropriate credential based on auth configuration"""
        
        auth_type = self.auth_config.auth_type
        
        if auth_type == AuthType.API_KEY:
            if not self.auth_config.api_key:
                raise AuthenticationError("API key is required but not provided")
            return self.auth_config.api_key
        
        elif auth_type == AuthType.ENTRA_ID:
            return self._get_entra_id_credential()
        
        elif auth_type == AuthType.MANAGED_IDENTITY:
            return self._get_managed_identity_credential()
        
        else:
            raise AuthenticationError(f"Unsupported authentication type: {auth_type}")
    
    def _get_entra_id_credential(self) -> TokenCredential:
        """Get Entra ID (Azure AD) credential"""
        
        if not self.auth_config.client_id or not self.auth_config.tenant_id:
            raise AuthenticationError(
                "Client ID and Tenant ID are required for Entra ID authentication"
            )
        
        try:
            if self.auth_config.client_secret:
                # Client credentials flow
                credential = ClientSecretCredential(
                    tenant_id=self.auth_config.tenant_id,
                    client_id=self.auth_config.client_id,
                    client_secret=self.auth_config.client_secret
                )
            else:
                # Use default credential (could be managed identity, CLI, etc.)
                credential = DefaultAzureCredential()
            
            return credential
            
        except Exception as e:
            raise AuthenticationError(f"Failed to create Entra ID credential: {str(e)}")
    
    def _get_managed_identity_credential(self) -> TokenCredential:
        """Get managed identity credential"""
        
        try:
            # Use system-assigned managed identity by default
            credential = ManagedIdentityCredential()
            return credential
            
        except Exception as e:
            raise AuthenticationError(f"Failed to create managed identity credential: {str(e)}")
    
    @staticmethod
    def from_environment(model_name: str) -> 'AuthenticationManager':
        """Create AuthenticationManager from environment variables"""
        
        # Try to load configuration from environment variables
        # Pattern: AZURE_AI_ROUTER_{MODEL_NAME}_{SETTING}
        prefix = f"AZURE_AI_ROUTER_{model_name.upper().replace('-', '_')}"
        
        api_key = os.getenv(f"{prefix}_API_KEY")
        client_id = os.getenv(f"{prefix}_CLIENT_ID")
        client_secret = os.getenv(f"{prefix}_CLIENT_SECRET")
        tenant_id = os.getenv(f"{prefix}_TENANT_ID")
        use_managed_identity = os.getenv(f"{prefix}_USE_MANAGED_IDENTITY", "false").lower() == "true"
        
        # Fallback to generic environment variables
        if not any([api_key, client_id, use_managed_identity]):
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            client_id = os.getenv("AZURE_CLIENT_ID")
            client_secret = os.getenv("AZURE_CLIENT_SECRET")
            tenant_id = os.getenv("AZURE_TENANT_ID")
        
        auth_config = AuthConfig(
            api_key=api_key,
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            use_managed_identity=use_managed_identity
        )
        
        return AuthenticationManager(auth_config)
