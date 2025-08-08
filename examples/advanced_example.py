"""
Advanced example showing custom routing logic and model management
"""

import asyncio
from brainfork import ModelRouter, ModelConfig, AuthConfig, UseCase


class CustomModelRouter(ModelRouter):
    """Extended router with custom logic"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage_stats = {}
    
    async def route_conversation(self, messages, override_model=None):
        """Override to add usage tracking"""
        result = await super().route_conversation(messages, override_model)
        
        # Track usage
        model_name = result.model_name
        if model_name not in self.usage_stats:
            self.usage_stats[model_name] = 0
        self.usage_stats[model_name] += 1
        
        return result
    
    def get_usage_stats(self):
        """Get model usage statistics"""
        return self.usage_stats
    
    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.usage_stats = {}


async def main():
    print("Brainfork - Advanced Example")
    print("=" * 50)

    # Configure models with different authentication methods
    models = {
        "gpt-4o-mini-api": ModelConfig(
            endpoint="https://your-endpoint.openai.azure.com/",
            deployment_name="gpt-4o-mini",
            api_version="2024-02-01",
            auth=AuthConfig(api_key="your-api-key"),
            temperature=0.3,
            max_tokens=2000
        ),
        "gpt-4o-mini-entra": ModelConfig(
            endpoint="https://your-endpoint.openai.azure.com/",
            deployment_name="gpt-4o-mini",
            api_version="2024-02-01",
            auth=AuthConfig(
                client_id="your-client-id",
                client_secret="your-client-secret",
                tenant_id="your-tenant-id"
            ),
            temperature=0.5
        ),
        "o1-mini": ModelConfig(
            endpoint="https://your-endpoint.openai.azure.com/",
            deployment_name="o1-mini",
            api_version="2024-02-01",
            auth=AuthConfig(api_key="your-api-key")
        ),
        "gpt-4-managed": ModelConfig(
            endpoint="https://your-endpoint.openai.azure.com/",
            deployment_name="gpt-4",
            api_version="2024-02-01",
            auth=AuthConfig(use_managed_identity=True)
        )
    }

    # Define comprehensive use cases
    use_cases = [
        UseCase(
            name="quick_classification",
            description="Fast text classification and simple categorization",
            model_name="gpt-4o-mini-api",
            priority=1,
            keywords=["classify", "categorize", "label", "tag", "sentiment"],
            min_confidence=0.8
        ),
        UseCase(
            name="enterprise_classification",
            description="Enterprise-grade classification with Entra ID auth",
            model_name="gpt-4o-mini-entra", 
            priority=1,
            keywords=["enterprise", "business", "corporate"],
            min_confidence=0.7
        ),
        UseCase(
            name="complex_reasoning",
            description="Advanced reasoning and mathematical problem solving",
            model_name="o1-mini",
            priority=2,
            keywords=["reasoning", "math", "logic", "proof", "theorem", "analysis"],
            min_confidence=0.85
        ),
        UseCase(
            name="secure_operations",
            description="Secure operations using managed identity",
            model_name="gpt-4-managed",
            priority=2,
            keywords=["secure", "confidential", "sensitive", "compliance"],
            min_confidence=0.8
        ),
        UseCase(
            name="code_review",
            description="Code review and technical analysis",
            model_name="gpt-4-managed",
            priority=2,
            keywords=["code", "review", "bug", "security", "performance"],
            min_confidence=0.75
        )
    ]

    # Initialize custom router
    router = CustomModelRouter(
        models=models,
        use_cases=use_cases,
        default_model="gpt-4o-mini-api",
        routing_model="gpt-4o-mini-api",
        routing_temperature=0.05  # Very deterministic routing
    )

    # Test various conversation types
    test_scenarios = [
        {
            "name": "Simple Classification",
            "messages": [
                {"role": "user", "content": "Classify this email as spam or not spam: 'Get rich quick!'"}
            ]
        },
        {
            "name": "Enterprise Document Analysis",
            "messages": [
                {"role": "user", "content": "Analyze this corporate document for business intelligence insights and categorize key findings"}
            ]
        },
        {
            "name": "Mathematical Proof",
            "messages": [
                {"role": "user", "content": "Prove that the square root of 2 is irrational using proof by contradiction"}
            ]
        },
        {
            "name": "Security Code Review",
            "messages": [
                {"role": "user", "content": "Review this authentication code for security vulnerabilities and suggest improvements"}
            ]
        },
        {
            "name": "Confidential Analysis",
            "messages": [
                {"role": "user", "content": "Analyze this sensitive financial data while maintaining strict compliance"}
            ]
        },
        {
            "name": "General Question",
            "messages": [
                {"role": "user", "content": "What's the capital of France?"}
            ]
        }
    ]

    print("\n🧪 Running test scenarios...")

    for scenario in test_scenarios:
        print(f"\n🔄 Scenario: {scenario['name']}")
        print(f"📝 Input: {scenario['messages'][0]['content'][:80]}...")
        
        # Route the conversation
        result = await router.route_conversation(scenario['messages'])
        
        print(f"🤖 Selected Model: {result.model_name}")
        print(f"📊 Confidence: {result.confidence:.3f}")
        print(f"🎯 Use Case: {result.use_case.name if result.use_case else 'Default'}")
        print(f"🔐 Auth Type: {result.model_config.auth.auth_type.value}")
        print(f"💭 Reasoning: {result.reasoning[:100]}...")
        
        # Test different override scenarios
        if scenario['name'] == "General Question":
            print("\n🔧 Testing model override...")
            override_result = await router.route_conversation(
                scenario['messages'], 
                override_model="o1-mini"
            )
            print(f"   Override Model: {override_result.model_name}")
        
        print("-" * 60)

    # Dynamic model management
    print("\n🔧 Dynamic Model Management Demo")
    print("=" * 40)
    
    # Add a new model
    print("➕ Adding new model...")
    new_model = ModelConfig(
        endpoint="https://new-endpoint.openai.azure.com/",
        deployment_name="gpt-4-turbo",
        api_version="2024-02-01",
        auth=AuthConfig(api_key="new-api-key")
    )
    router.add_model("gpt-4-turbo", new_model)
    
    # Add a new use case
    print("➕ Adding new use case...")
    new_use_case = UseCase(
        name="high_performance",
        description="High-performance tasks requiring latest model",
        model_name="gpt-4-turbo",
        priority=1,
        keywords=["performance", "speed", "latest", "turbo"],
        min_confidence=0.7
    )
    router.add_use_case(new_use_case)
    
    # Test the new configuration
    test_message = [{"role": "user", "content": "I need the highest performance model for this complex task"}]
    result = await router.route_conversation(test_message)
    print(f"✅ New routing result: {result.model_name} (confidence: {result.confidence:.3f})")

    # Show usage statistics
    print(f"\n📈 Usage Statistics:")
    usage_stats = router.get_usage_stats()
    for model, count in usage_stats.items():
        print(f"  • {model}: {count} requests")

    # Show final router configuration
    print(f"\n📋 Final Router Configuration:")
    router_info = router.get_model_info()
    print(f"  Models: {len(router_info['models'])}")
    print(f"  Use Cases: {len(router_info['use_cases'])}")
    print(f"  Default Model: {router_info['default_model']}")

    # Clear cache and reset stats
    router.clear_cache()
    router.reset_usage_stats()
    print(f"\n🧹 Cache and statistics cleared")


if __name__ == "__main__":
    asyncio.run(main())
