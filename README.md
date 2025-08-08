# Brainfork

An intelligent AI model routing library for Azure AI Foundry that automatically selects the best model based on predefined use cases and conversation context.

## Features

- **Intelligent Model Routing**: Automatically selects the best AI model based on conversation analysis
- **Multiple Authentication Methods**: Supports API keys, Entra ID authentication, and managed identity
- **Flexible Use Case Configuration**: Define custom routing rules for different scenarios
- **Azure AI Foundry Integration**: Native support for Azure AI Foundry models
- **Client Factory**: Seamlessly generates configured clients for selected models
- **Extensible Architecture**: Easy to extend with custom routing logic

## Installation

```bash
pip install brainfork
```

## Quick Start

```python
from brainfork import ModelRouter, ModelConfig, UseCase, AuthConfig

# Configure your models
models = {
    "gpt-4o-mini": ModelConfig(
        endpoint="https://your-endpoint.openai.azure.com/",
        deployment_name="gpt-4o-mini",
        api_version="2024-02-01",
        auth=AuthConfig(api_key="your-api-key")
    ),
    "o1-mini": ModelConfig(
        endpoint="https://your-endpoint.openai.azure.com/",
        deployment_name="o1-mini", 
        api_version="2024-02-01",
        auth=AuthConfig(api_key="your-api-key")
    )
}

# Define use cases
use_cases = [
    UseCase(
        name="text_classification",
        description="Text classification and sentiment analysis tasks",
        model_name="gpt-4o-mini",
        priority=1
    ),
    UseCase(
        name="advanced_reasoning", 
        description="Complex reasoning, math, and analytical tasks",
        model_name="o1-mini",
        priority=2
    )
]

# Initialize router
router = ModelRouter(
    models=models,
    use_cases=use_cases,
    default_model="gpt-4o-mini",
    routing_model="gpt-4o-mini"  # Model used for routing decisions
)

# Route a conversation
messages = [
    {"role": "user", "content": "Solve this complex math problem: ..."}
]

result = await router.route_conversation(messages)
print(f"Selected model: {result.model_name}")

# Get configured client
configured_client = await router.get_configured_client(messages)
response = await configured_client.client.chat.completions.create(
    model=configured_client.model_config.deployment_name,
    messages=messages
)
```

## Authentication Options

### API Key Authentication
```python
auth = AuthConfig(api_key="your-api-key")
```

### Entra ID Authentication  
```python
auth = AuthConfig(
    client_id="your-client-id",
    client_secret="your-client-secret",
    tenant_id="your-tenant-id"
)
```

### Managed Identity
```python
auth = AuthConfig(use_managed_identity=True)
```

## Advanced Configuration

```python
# Custom use case with detailed criteria
use_case = UseCase(
    name="code_generation",
    description="Code generation and programming tasks",
    model_name="gpt-4o",
    priority=1,
    keywords=["code", "programming", "function", "class"],
    context_requirements=["technical", "development"]
)

# Router with custom routing logic
router = ModelRouter(
    models=models,
    use_cases=use_cases,
    default_model="gpt-4o-mini",
    routing_model="gpt-4o-mini",
    routing_temperature=0.1,
    enable_caching=True
)
```

## License

MIT License
