# Azure AI Router

## Requirements

- Python 3.9+
- Azure OpenAI or Azure AI Foundry access
- Required Python packages (see pyproject.toml)

## Installation

### From Source

```bash
git clone <repository-url>
cd azure-ai-router
pip install -e .
```

### Development Installation

```bash
git clone <repository-url>
cd azure-ai-router
pip install -e ".[dev]"
```

## Quick Setup

1. **Configure your Azure OpenAI credentials**:
   ```bash
   # Option 1: Environment variables
   export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
   export AZURE_OPENAI_API_KEY="your-api-key"
   
   # Option 2: For Entra ID authentication
   export AZURE_CLIENT_ID="your-client-id"
   export AZURE_CLIENT_SECRET="your-client-secret"
   export AZURE_TENANT_ID="your-tenant-id"
   ```

2. **Run the basic example**:
   ```bash
   cd examples
   python basic_example.py
   ```

3. **Create a configuration file**:
   ```bash
   cd examples
   python config_example.py
   ```
   This will create a sample `router_config.json` file that you can customize.

## Configuration

### Environment Variables

The library supports environment variable substitution in configuration files:

```json
{
  "models": {
    "gpt-4o-mini": {
      "endpoint": "${AZURE_OPENAI_ENDPOINT}",
      "deployment_name": "gpt-4o-mini",
      "auth": {
        "api_key": "${AZURE_OPENAI_API_KEY}"
      }
    }
  }
}
```

### Authentication Methods

#### API Key
```python
auth = AuthConfig(api_key="your-api-key")
```

#### Entra ID (Azure AD)
```python
auth = AuthConfig(
    client_id="your-client-id",
    client_secret="your-client-secret",
    tenant_id="your-tenant-id"
)
```

#### Managed Identity
```python
auth = AuthConfig(use_managed_identity=True)
```

## Examples

### Basic Usage

```python
import asyncio
from azure_ai_router import ModelRouter, ModelConfig, AuthConfig, UseCase

# Configure models
models = {
    "gpt-4o-mini": ModelConfig(
        endpoint="https://your-endpoint.openai.azure.com/",
        deployment_name="gpt-4o-mini",
        api_version="2024-02-01",
        auth=AuthConfig(api_key="your-api-key")
    )
}

# Define use cases
use_cases = [
    UseCase(
        name="classification",
        description="Text classification tasks",
        model_name="gpt-4o-mini",
        keywords=["classify", "categorize"]
    )
]

# Initialize router
router = ModelRouter(
    models=models,
    use_cases=use_cases,
    default_model="gpt-4o-mini"
)

# Route conversation
async def main():
    messages = [{"role": "user", "content": "Classify this text..."}]
    result = await router.route_conversation(messages)
    print(f"Selected model: {result.model_name}")

asyncio.run(main())
```

### Using Configuration Files

```python
from azure_ai_router.utils import (
    load_config_from_file,
    parse_models_from_config,
    parse_use_cases_from_config
)

# Load from file
config = load_config_from_file("router_config.json")
models = parse_models_from_config(config)
use_cases = parse_use_cases_from_config(config)

router = ModelRouter(models=models, use_cases=use_cases, default_model="gpt-4o-mini")
```

## Testing

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=azure_ai_router
```

## Development

### Project Structure

```
azure-ai-router/
├── src/azure_ai_router/
│   ├── __init__.py          # Package exports
│   ├── router.py            # Main router class
│   ├── models.py            # Data models
│   ├── auth.py              # Authentication management
│   ├── routing_engine.py    # Routing logic
│   ├── client_factory.py    # Client creation
│   ├── utils.py             # Utility functions
│   └── exceptions.py        # Custom exceptions
├── examples/                # Usage examples
├── tests/                   # Test suite
└── pyproject.toml           # Project configuration
```

### Code Style

The project uses:
- **Black** for code formatting
- **isort** for import sorting
- **mypy** for type checking
- **flake8** for linting

```bash
# Format code
black src/ tests/ examples/

# Sort imports
isort src/ tests/ examples/

# Type checking
mypy src/

# Linting
flake8 src/ tests/ examples/
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you've installed the package with `pip install -e .`

2. **Authentication errors**: 
   - Check your Azure OpenAI endpoint and credentials
   - Ensure your deployment names match your Azure configuration
   - Verify your API version is supported

3. **Model not found errors**: 
   - Ensure model names in use cases match model dictionary keys
   - Check that default_model exists in models dictionary

4. **Routing errors**: 
   - Verify the routing model is properly configured
   - Check that the routing model has appropriate permissions
   - Review use case definitions for conflicts

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your router code here
```

### Connectivity Testing

Test model connectivity:

```python
from azure_ai_router.utils import validate_model_connectivity

# Test a model configuration
is_connected = validate_model_connectivity(your_model_config)
print(f"Model connectivity: {is_connected}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.
