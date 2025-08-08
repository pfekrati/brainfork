#!/usr/bin/env python3
"""
Azure AI Router - Demonstration Script

This script demonstrates the key features of the Azure AI Router library.
Make sure to set your Azure OpenAI credentials before running.
"""

import asyncio
import json
import os
from pathlib import Path

# Import the Azure AI Router components
from azure_ai_router import ModelRouter, ModelConfig, AuthConfig, UseCase
from azure_ai_router.utils import create_sample_config, save_config_to_file


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🤖 {title}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n📋 {title}")
    print("-" * 40)


async def demo_basic_routing():
    """Demonstrate basic model routing"""
    print_header("Azure AI Router - Basic Routing Demo")
    
    # Configure models (replace with your actual endpoints and credentials)
    models = {
        "gpt-4o-mini": ModelConfig(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/"),
            deployment_name="gpt-4o-mini",
            api_version="2024-02-01",
            auth=AuthConfig(api_key=os.getenv("AZURE_OPENAI_API_KEY", "your-api-key")),
            temperature=0.3
        ),
        "o1-mini": ModelConfig(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/"),
            deployment_name="o1-mini",
            api_version="2024-02-01",
            auth=AuthConfig(api_key=os.getenv("AZURE_OPENAI_API_KEY", "your-api-key"))
        ),
        "gpt-4": ModelConfig(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/"),
            deployment_name="gpt-4",
            api_version="2024-02-01",
            auth=AuthConfig(
                client_id=os.getenv("AZURE_CLIENT_ID"),
                client_secret=os.getenv("AZURE_CLIENT_SECRET"),
                tenant_id=os.getenv("AZURE_TENANT_ID")
            ) if os.getenv("AZURE_CLIENT_ID") else AuthConfig(api_key=os.getenv("AZURE_OPENAI_API_KEY", "your-api-key"))
        )
    }
    
    # Define use cases for different types of tasks
    use_cases = [
        UseCase(
            name="text_classification",
            description="Text classification, sentiment analysis, and simple categorization tasks",
            model_name="gpt-4o-mini",
            priority=1,
            keywords=["classify", "categorize", "sentiment", "label", "tag"],
            min_confidence=0.7
        ),
        UseCase(
            name="advanced_reasoning",
            description="Complex reasoning, mathematical problems, and analytical tasks",
            model_name="o1-mini",
            priority=2,
            keywords=["math", "reasoning", "analysis", "problem", "solve", "calculate", "proof"],
            min_confidence=0.8
        ),
        UseCase(
            name="code_generation",
            description="Code generation, programming assistance, and technical documentation",
            model_name="gpt-4",
            priority=2,
            keywords=["code", "programming", "function", "class", "debug", "implement"],
            min_confidence=0.75
        )
    ]
    
    print_section("Initializing Router")
    
    # Initialize the router
    router = ModelRouter(
        models=models,
        use_cases=use_cases,
        default_model="gpt-4o-mini",
        routing_model="gpt-4o-mini",  # Model used for making routing decisions
        routing_temperature=0.1,      # Low temperature for consistent routing
        enable_caching=True
    )
    
    print(f"✅ Router initialized with {len(models)} models and {len(use_cases)} use cases")
    print(f"🎯 Default model: {router.default_model}")
    print(f"🧠 Routing model: {router.routing_model}")
    
    print_section("Test Conversations")
    
    # Test different types of conversations
    test_conversations = [
        {
            "name": "Sentiment Analysis",
            "messages": [
                {"role": "user", "content": "Can you classify the sentiment of this customer review: 'This product exceeded my expectations! Great quality and fast shipping.'"}
            ],
            "expected_model": "gpt-4o-mini"
        },
        {
            "name": "Mathematical Problem",
            "messages": [
                {"role": "user", "content": "I need to solve this complex calculus problem: Find the integral of x^2 * e^(x^3) dx using substitution method"}
            ],
            "expected_model": "o1-mini"
        },
        {
            "name": "Code Implementation",
            "messages": [
                {"role": "user", "content": "Write a Python function that implements a binary search tree with insert, delete, and search operations"}
            ],
            "expected_model": "gpt-4"
        },
        {
            "name": "General Question",
            "messages": [
                {"role": "user", "content": "What's the weather like in Paris today?"}
            ],
            "expected_model": "gpt-4o-mini"  # Should use default
        }
    ]
    
    for i, conversation in enumerate(test_conversations, 1):
        print(f"\n🔄 Test {i}: {conversation['name']}")
        print(f"💬 User: {conversation['messages'][0]['content']}")
        
        try:
            # Route the conversation
            result = await router.route_conversation(conversation['messages'])
            
            print(f"🤖 Selected Model: {result.model_name}")
            print(f"📊 Confidence: {result.confidence:.3f}")
            print(f"🎯 Use Case: {result.use_case.name if result.use_case else 'Default (no specific use case)'}")
            print(f"🔐 Auth Type: {result.model_config.auth.auth_type.value}")
            print(f"💭 Reasoning: {result.reasoning}")
            
            # Check if routing matched expectation
            if result.model_name == conversation["expected_model"]:
                print("✅ Routing matched expectation!")
            else:
                print(f"⚠️  Expected {conversation['expected_model']}, got {result.model_name}")
                
        except Exception as e:
            print(f"❌ Error routing conversation: {str(e)}")
        
        print("-" * 50)


def demo_configuration_management():
    """Demonstrate configuration file management"""
    print_header("Configuration Management Demo")
    
    print_section("Creating Sample Configuration")
    
    # Create sample configuration
    config = create_sample_config()
    config_file = Path("demo_router_config.json")
    
    # Save to file
    save_config_to_file(config, str(config_file))
    print(f"✅ Sample configuration saved to {config_file}")
    
    # Show configuration structure
    print_section("Configuration Structure")
    print("📋 Models configured:")
    for model_name, model_config in config["models"].items():
        auth_type = "api_key" if "api_key" in model_config["auth"] else "entra_id" if "client_id" in model_config["auth"] else "managed_identity"
        print(f"  • {model_name}: {model_config['deployment_name']} ({auth_type})")
    
    print("\n🎯 Use cases configured:")
    for use_case in config["use_cases"]:
        print(f"  • {use_case['name']}: {use_case['model_name']} (priority: {use_case['priority']})")
    
    print(f"\n⚙️  Router settings:")
    settings = config["router_settings"]
    for key, value in settings.items():
        print(f"  • {key}: {value}")
    
    # Clean up
    if config_file.exists():
        config_file.unlink()
        print(f"\n🧹 Cleaned up {config_file}")


def demo_authentication_options():
    """Demonstrate different authentication options"""
    print_header("Authentication Options Demo")
    
    print_section("API Key Authentication")
    auth_api_key = AuthConfig(api_key="your-api-key")
    print(f"✅ Auth Type: {auth_api_key.auth_type.value}")
    print("📝 Use case: Simple, direct authentication with API key")
    
    print_section("Entra ID Authentication")
    auth_entra_id = AuthConfig(
        client_id="your-client-id",
        client_secret="your-client-secret",
        tenant_id="your-tenant-id"
    )
    print(f"✅ Auth Type: {auth_entra_id.auth_type.value}")
    print("📝 Use case: Enterprise authentication with Azure AD")
    
    print_section("Managed Identity Authentication")
    auth_managed_identity = AuthConfig(use_managed_identity=True)
    print(f"✅ Auth Type: {auth_managed_identity.auth_type.value}")
    print("📝 Use case: Secure authentication in Azure environments")


async def main():
    """Main demonstration function"""
    print("🚀 Starting Azure AI Router Demonstration")
    print("=" * 60)
    
    # Check for environment variables
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("⚠️  Note: Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables")
        print("   for full functionality. Using placeholder values for demonstration.")
    
    try:
        # Run demonstrations
        await demo_basic_routing()
        demo_configuration_management()
        demo_authentication_options()
        
        print_header("Demonstration Complete")
        print("🎉 Azure AI Router demonstration completed successfully!")
        print("\n📚 Next steps:")
        print("  • Check out the examples/ directory for more detailed examples")
        print("  • Read SETUP.md for installation and configuration instructions")
        print("  • Explore the source code in src/azure_ai_router/")
        print("  • Run the test suite with: pytest tests/")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install -r requirements.txt")


if __name__ == "__main__":
    asyncio.run(main())
