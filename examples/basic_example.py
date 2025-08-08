"""
Basic example of using Brainfork
"""

import asyncio
from brainfork import ModelRouter, ModelConfig, AuthConfig, UseCase


async def main():
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
        ),
        "gpt-4": ModelConfig(
            endpoint="https://your-endpoint.openai.azure.com/",
            deployment_name="gpt-4",
            api_version="2024-02-01",
            auth=AuthConfig(
                client_id="your-client-id",
                client_secret="your-client-secret",
                tenant_id="your-tenant-id"
            )
        )
    }

    # Define use cases
    use_cases = [
        UseCase(
            name="text_classification",
            description="Text classification and sentiment analysis tasks",
            model_name="gpt-4o-mini",
            priority=1,
            keywords=["classify", "categorize", "sentiment", "label"],
            min_confidence=0.7
        ),
        UseCase(
            name="advanced_reasoning",
            description="Complex reasoning, mathematical problems, and analytical tasks",
            model_name="o1-mini",
            priority=2,
            keywords=["math", "reasoning", "analysis", "problem", "solve", "calculate"],
            min_confidence=0.8
        ),
        UseCase(
            name="code_generation",
            description="Code generation and programming assistance",
            model_name="gpt-4",
            priority=2,
            keywords=["code", "programming", "function", "class", "debug"],
            min_confidence=0.75
        )
    ]

    # Initialize router
    router = ModelRouter(
        models=models,
        use_cases=use_cases,
        default_model="gpt-4o-mini",
        routing_model="gpt-4o-mini"
    )

    # Example conversations
    test_conversations = [
        {
            "name": "Text Classification",
            "messages": [
                {"role": "user", "content": "Can you classify the sentiment of this text: 'I love this product!'"}
            ]
        },
        {
            "name": "Math Problem",
            "messages": [
                {"role": "user", "content": "Solve this complex mathematical equation: x^3 + 2x^2 - 5x + 3 = 0"}
            ]
        },
        {
            "name": "Code Generation",
            "messages": [
                {"role": "user", "content": "Write a Python function to implement binary search"}
            ]
        },
        {
            "name": "General Question",
            "messages": [
                {"role": "user", "content": "What's the weather like today?"}
            ]
        }
    ]

    print("Brainfork - Basic Example")
    print("=" * 50)

    for conversation in test_conversations:
        print(f"\n🔄 Processing: {conversation['name']}")
        print(f"User: {conversation['messages'][0]['content']}")
        
        # Route the conversation
        result = await router.route_conversation(conversation['messages'])
        
        print(f"✅ Selected Model: {result.model_name}")
        print(f"📊 Confidence: {result.confidence:.2f}")
        print(f"🎯 Use Case: {result.use_case.name if result.use_case else 'None (default)'}")
        print(f"💭 Reasoning: {result.reasoning}")
        
        # Optionally, get a configured client and make a real API call
        # (Uncomment the lines below if you have valid credentials)
        """
        try:
            configured_client = await router.get_configured_client(conversation['messages'])
            
            # Make API call
            response = await configured_client.chat_completion(
                messages=conversation['messages'],
                max_tokens=100
            )
            
            print(f"🤖 Response: {response.choices[0].message.content[:100]}...")
            
        except Exception as e:
            print(f"❌ API call failed: {str(e)}")
        """
        
        print("-" * 50)

    # Display router information
    print(f"\n📋 Router Configuration:")
    router_info = router.get_model_info()
    print(f"Models: {list(router_info['models'].keys())}")
    print(f"Use Cases: {[uc['name'] for uc in router_info['use_cases']]}")
    print(f"Default Model: {router_info['default_model']}")


if __name__ == "__main__":
    asyncio.run(main())
