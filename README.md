# Brainfork

**Brainfork** is an intelligent AI model router for Azure OpenAI that automatically selects the best model for your specific use case. It analyzes your conversations and routes them to the most appropriate AI model based on predefined use cases, keywords, and context requirements.

## Features

- 🎯 **Intelligent Routing**: Automatically selects the best AI model based on conversation context
- 🔧 **Flexible Configuration**: Support for multiple authentication methods (API key, Entra ID, Managed Identity)
- 📊 **Confidence Scoring**: Get confidence scores for routing decisions with detailed reasoning
- 🔄 **Multiple Use Cases**: Define custom use cases with keywords and context requirements
- 🚀 **Async Support**: Built with async/await for high-performance applications
- 📝 **Type Safety**: Full Pydantic model validation and type hints

## Installation

```bash
pip install brainfork
```

## Quick Start

Here's a complete example showing how to configure and use Brainfork:

```python
import asyncio
from brainfork import ModelRouter, ModelConfig, AuthConfig, UseCase

async def main():
    # Configure your Azure OpenAI models
    models = {
        "gpt-4.1": ModelConfig(
            endpoint="https://your-openai-endpoint.openai.azure.com/",
            deployment_name="gpt-4.1",
            api_version="2025-01-01-preview",
            auth=AuthConfig(
                client_id="your-client-id",
                client_secret="your-client-secret",
                tenant_id="your-tenant-id"
            )
        ),
        "gpt-5-mini": ModelConfig(
            endpoint="https://your-openai-endpoint.openai.azure.com/",
            deployment_name="gpt-5-mini",
            api_version="2025-04-01-preview",
            auth=AuthConfig(api_key="your-api-key") 
        ),
        "o3-mini": ModelConfig(
            endpoint="https://your-openai-endpoint.openai.azure.com/",
            deployment_name="o3-mini",
            api_version="2025-01-01-preview",
            auth=AuthConfig(api_key="your-api-key")
        )
    }

    # Define use cases for intelligent routing
    use_cases = [
        UseCase(
            name="math related questions",
            description="when user is asking about math related questions or questions that require complex reasoning and analysis or coding",
            model_name="o3-mini",
            keywords=["math", "reasoning", "analysis", "problem", "solve", "calculate", "code", "programming", "function", "class", "debug"],
            min_confidence=0.8
        ),
        UseCase(
            name="text summarization",
            description="when user is asking for a summary of a text or document",
            model_name="gpt-5-mini",
            keywords=["summarize", "summary", "text", "document"],
            min_confidence=0.8
        ),
        UseCase(
            name="new content generation",
            description="when user is asking for new content generation like writing an article, story, or creative content",
            model_name="gpt-4.1",
            keywords=["create", "generate", "write", "article", "story", "content"],
            min_confidence=0.75
        )
    ]

    # Initialize the router
    router = ModelRouter(
        models=models,
        use_cases=use_cases,
        default_model="gpt-5-mini",
        routing_model="gpt-4.1-mini",
        routing_temperature=0
    )

    # Example conversation
    messages = [
        {"role": "user", "content": "Write a Python function to implement binary search"}
    ]

    # Route the conversation
    result = await router.route_conversation(messages)
    
    print(f"Selected Model: {result.model_name}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Use Case: {result.use_case.name if result.use_case else 'None (default)'}")
    print(f"Reasoning: {result.reasoning}")

    # Get a configured client and make API call
    configured_client = await router.get_configured_client(messages)
    response = await configured_client.client.chat.completions.create(
        model=configured_client.model_config.deployment_name,
        messages=messages,
        max_completion_tokens=1000
    )
    
    print(f"Response: {response.choices[0].message.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Routing Examples

Based on the conversation content, Brainfork intelligently routes to different models:

### 🧮 Mathematical/Coding Questions → o3-mini
```python
messages = [
    {"role": "user", "content": "Solve this complex mathematical equation: x^3 + 2x^2 - 5x + 3 = 0"}
]
# Routes to o3-mini (specialized for reasoning and analysis)
```

### 📝 Text Summarization → gpt-5-mini
```python
messages = [
    {"role": "user", "content": "Can you provide a summary of this news article: [long article text]"}
]
# Routes to gpt-5-mini (optimized for text processing)
```

### ✨ Creative Content → gpt-4.1
```python
messages = [
    {"role": "user", "content": "Can you generate a creative story about friendship and adventure?"}
]
# Routes to gpt-4.1 (best for creative content generation)
```

### 🤔 General Questions → Default Model
```python
messages = [
    {"role": "user", "content": "What's the reason we have different seasons on earth?"}
]
# Routes to gpt-5-mini (default model for general queries)
```

## Authentication Methods

Brainfork supports multiple authentication methods:

### API Key Authentication
```python
auth=AuthConfig(api_key="your-api-key")
```

### Entra ID Authentication
```python
auth=AuthConfig(
    client_id="your-client-id",
    client_secret="your-client-secret",
    tenant_id="your-tenant-id"
)
```

### Managed Identity Authentication
```python
auth=AuthConfig(use_managed_identity=True)
```

## Configuration

### ModelConfig
Configure individual AI models with their endpoints and authentication:

```python
ModelConfig(
    endpoint="https://your-endpoint.openai.azure.com/",
    deployment_name="your-deployment",
    api_version="2025-01-01-preview",
    auth=AuthConfig(...),
    max_tokens=4000,  # Optional
    temperature=0.7   # Optional
)
```

### UseCase
Define routing rules based on conversation context:

```python
UseCase(
    name="use_case_name",
    description="Detailed description of when to use this model",
    model_name="target_model",
    keywords=["keyword1", "keyword2"],
    context_requirements=["requirement1"],
    min_confidence=0.8
)
```

### ModelRouter
Initialize the router with your configuration:

```python
ModelRouter(
    models=models_dict,
    use_cases=use_cases_list,
    default_model="fallback_model",
    routing_model="model_for_routing_decisions",
    routing_temperature=0
)
```

## API Reference

### Main Methods

#### `route_conversation(messages: List[Dict]) -> RoutingResult`
Routes a conversation to the most appropriate model.

#### `get_configured_client(messages: List[Dict]) -> ConfiguredClient`
Returns a configured Azure OpenAI client for the selected model.

#### `get_model_info() -> Dict`
Returns information about configured models and use cases.

### Response Objects

#### `RoutingResult`
- `model_name`: Selected model name
- `selected_model`: ModelConfig of selected model
- `use_case`: Matched UseCase (if any)
- `confidence`: Confidence score (0.0-1.0)
- `reasoning`: Explanation of routing decision

## Best Practices

1. **Define Clear Use Cases**: Create specific use cases with relevant keywords
2. **Set Appropriate Confidence Thresholds**: Balance between accuracy and fallback frequency
3. **Use Descriptive Model Names**: Make it easy to understand each model's purpose
4. **Monitor Routing Decisions**: Review confidence scores and reasoning for optimization
5. **Secure Your Credentials**: Use environment variables or Azure Key Vault for sensitive data

## Example Output

When running the example, you'll see output like:

```
🔄 Processing: Math Problem
User: Solve this complex mathematical equation: x^3 + 2x^2 - 5x + 3 = 0
✅ Selected Model: o3-mini
📊 Confidence: 0.95
🎯 Use Case: math related questions
💭 Reasoning: The user is asking to solve a complex mathematical equation, which requires mathematical reasoning and analysis. This matches the 'math related questions' use case with high confidence due to keywords like 'solve', 'mathematical', and 'equation'.
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
