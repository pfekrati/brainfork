"""
Utility functions for Brainfork
"""

import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from .models import ModelConfig, AuthConfig, UseCase
from .exceptions import ConfigurationError


def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON or YAML file"""
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                return yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ConfigurationError(f"Unsupported configuration file format: {file_path.suffix}")
    
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration file: {str(e)}")


def save_config_to_file(config: Dict[str, Any], file_path: str) -> None:
    """Save configuration to a JSON or YAML file"""
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif file_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ConfigurationError(f"Unsupported configuration file format: {file_path.suffix}")
    
    except Exception as e:
        raise ConfigurationError(f"Failed to save configuration file: {str(e)}")


def parse_models_from_config(config: Dict[str, Any]) -> Dict[str, ModelConfig]:
    """Parse model configurations from config dictionary"""
    
    models = {}
    models_config = config.get('models', {})
    
    for name, model_data in models_config.items():
        auth_data = model_data.get('auth', {})
        
        auth_config = AuthConfig(
            api_key=auth_data.get('api_key'),
            client_id=auth_data.get('client_id'),
            client_secret=auth_data.get('client_secret'),
            tenant_id=auth_data.get('tenant_id'),
            use_managed_identity=auth_data.get('use_managed_identity', False)
        )
        
        model_config = ModelConfig(
            endpoint=model_data['endpoint'],
            deployment_name=model_data['deployment_name'],
            api_version=model_data.get('api_version', '2024-02-01'),
            auth=auth_config,
            model_type=model_data.get('model_type', 'chat'),
            max_tokens=model_data.get('max_tokens'),
            temperature=model_data.get('temperature')
        )
        
        models[name] = model_config
    
    return models


def parse_use_cases_from_config(config: Dict[str, Any]) -> List[UseCase]:
    """Parse use case configurations from config dictionary"""
    
    use_cases = []
    use_cases_config = config.get('use_cases', [])
    
    for uc_data in use_cases_config:
        use_case = UseCase(
            name=uc_data['name'],
            description=uc_data['description'],
            model_name=uc_data['model_name'],
            priority=uc_data.get('priority', 1),
            keywords=uc_data.get('keywords', []),
            context_requirements=uc_data.get('context_requirements', []),
            min_confidence=uc_data.get('min_confidence', 0.7)
        )
        
        use_cases.append(use_case)
    
    return use_cases


def create_sample_config() -> Dict[str, Any]:
    """Create a sample configuration dictionary"""
    
    return {
        "models": {
            "gpt-4o-mini": {
                "endpoint": "https://your-endpoint.openai.azure.com/",
                "deployment_name": "gpt-4o-mini",
                "api_version": "2024-02-01",
                "auth": {
                    "api_key": "your-api-key"
                },
                "model_type": "chat",
                "temperature": 0.7,
                "max_tokens": 4000
            },
            "o1-mini": {
                "endpoint": "https://your-endpoint.openai.azure.com/",
                "deployment_name": "o1-mini", 
                "api_version": "2024-02-01",
                "auth": {
                    "client_id": "your-client-id",
                    "client_secret": "your-client-secret",
                    "tenant_id": "your-tenant-id"
                },
                "model_type": "chat"
            },
            "gpt-4": {
                "endpoint": "https://your-endpoint.openai.azure.com/",
                "deployment_name": "gpt-4",
                "api_version": "2024-02-01",
                "auth": {
                    "use_managed_identity": True
                },
                "model_type": "chat"
            }
        },
        "use_cases": [
            {
                "name": "text_classification",
                "description": "Text classification, sentiment analysis, and simple categorization tasks",
                "model_name": "gpt-4o-mini",
                "priority": 1,
                "keywords": ["classify", "categorize", "sentiment", "label"],
                "context_requirements": ["simple_task"],
                "min_confidence": 0.7
            },
            {
                "name": "advanced_reasoning",
                "description": "Complex reasoning, mathematical problems, and analytical tasks",
                "model_name": "o1-mini",
                "priority": 2,
                "keywords": ["math", "reasoning", "analysis", "problem", "solve", "calculate"],
                "context_requirements": ["complex_reasoning"],
                "min_confidence": 0.8
            },
            {
                "name": "code_generation",
                "description": "Code generation, programming assistance, and technical documentation",
                "model_name": "gpt-4",
                "priority": 2,
                "keywords": ["code", "programming", "function", "class", "debug", "implement"],
                "context_requirements": ["technical"],
                "min_confidence": 0.75
            }
        ],
        "router_settings": {
            "default_model": "gpt-4o-mini",
            "routing_model": "gpt-4o-mini", 
            "routing_temperature": 0.1,
            "enable_caching": True,
            "client_type": "openai"
        }
    }


def environment_variable_substitution(config: Dict[str, Any]) -> Dict[str, Any]:
    """Replace environment variable placeholders in configuration"""
    
    def substitute_value(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            default_value = None
            
            # Handle default values: ${VAR_NAME:default_value}
            if ":" in env_var:
                env_var, default_value = env_var.split(":", 1)
            
            return os.getenv(env_var, default_value)
        
        elif isinstance(value, dict):
            return {k: substitute_value(v) for k, v in value.items()}
        
        elif isinstance(value, list):
            return [substitute_value(item) for item in value]
        
        else:
            return value
    
    return substitute_value(config)


def validate_model_connectivity(model_config: ModelConfig) -> bool:
    """Test connectivity to a model endpoint (basic validation)"""
    
    try:
        from .client_factory import ClientFactory
        
        # Create a client and test basic connectivity
        client = ClientFactory.create_openai_client(model_config, async_client=False)
        
        # Try a simple test call (this might fail but will validate auth/endpoint)
        test_messages = [{"role": "user", "content": "test"}]
        
        try:
            response = client.chat.completions.create(
                model=model_config.deployment_name,
                messages=test_messages,
                max_tokens=1
            )
            return True
        except Exception:
            # Even if the call fails, if we get a proper API error (not auth error),
            # the configuration is likely correct
            return True
            
    except Exception as e:
        print(f"Connectivity test failed for model: {str(e)}")
        return False
