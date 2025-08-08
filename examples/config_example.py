"""
Example using configuration files with Brainfork
"""

import asyncio
from pathlib import Path
from brainfork import ModelRouter
from brainfork.utils import (
    load_config_from_file,
    parse_models_from_config,
    parse_use_cases_from_config,
    create_sample_config,
    save_config_to_file,
    environment_variable_substitution
)


async def main():
    print("Brainfork - Configuration File Example")
    print("=" * 50)

    # Create sample configuration file
    config_file = Path("router_config.json")
    
    if not config_file.exists():
        print("📝 Creating sample configuration file...")
        sample_config = create_sample_config()
        save_config_to_file(sample_config, str(config_file))
        print(f"✅ Sample configuration saved to {config_file}")
        print("⚠️  Please update the configuration with your actual Azure OpenAI credentials")
        return

    # Load configuration from file
    print(f"📂 Loading configuration from {config_file}...")
    config = load_config_from_file(str(config_file))
    
    # Apply environment variable substitution
    config = environment_variable_substitution(config)
    
    # Parse models and use cases
    models = parse_models_from_config(config)
    use_cases = parse_use_cases_from_config(config)
    
    # Get router settings
    router_settings = config.get('router_settings', {})
    
    print(f"✅ Loaded {len(models)} models and {len(use_cases)} use cases")

    # Initialize router
    router = ModelRouter(
        models=models,
        use_cases=use_cases,
        default_model=router_settings.get('default_model', 'gpt-4o-mini'),
        routing_model=router_settings.get('routing_model', 'gpt-4o-mini'),
        routing_temperature=router_settings.get('routing_temperature', 0.1),
        enable_caching=router_settings.get('enable_caching', True),
        client_type=router_settings.get('client_type', 'openai')
    )

    # Test conversations
    test_conversations = [
        {
            "name": "Sentiment Analysis",
            "messages": [
                {"role": "user", "content": "Analyze the sentiment of customer reviews for our new product launch"}
            ]
        },
        {
            "name": "Complex Math", 
            "messages": [
                {"role": "user", "content": "I need help solving a system of differential equations for my physics research"}
            ]
        },
        {
            "name": "Python Development",
            "messages": [
                {"role": "user", "content": "Can you help me implement a REST API with FastAPI and PostgreSQL?"}
            ]
        }
    ]

    for conversation in test_conversations:
        print(f"\n🔄 Processing: {conversation['name']}")
        print(f"User: {conversation['messages'][0]['content']}")
        
        # Route the conversation
        result = await router.route_conversation(conversation['messages'])
        
        print(f"✅ Selected Model: {result.model_name}")
        print(f"📊 Confidence: {result.confidence:.2f}")
        print(f"🎯 Use Case: {result.use_case.name if result.use_case else 'None (default)'}")
        print(f"💭 Reasoning: {result.reasoning}")
        print("-" * 50)

    # Show router information
    print(f"\n📋 Router Configuration Summary:")
    router_info = router.get_model_info()
    
    print(f"\n🤖 Models ({len(router_info['models'])}):")
    for model_name, model_info in router_info['models'].items():
        print(f"  • {model_name}: {model_info['deployment_name']} ({model_info['auth_type']})")
    
    print(f"\n🎯 Use Cases ({len(router_info['use_cases'])}):")
    for use_case in router_info['use_cases']:
        print(f"  • {use_case['name']}: {use_case['model_name']} (priority: {use_case['priority']})")
    
    print(f"\n⚙️  Settings:")
    print(f"  • Default Model: {router_info['default_model']}")
    print(f"  • Routing Model: {router_info['routing_model']}")
    print(f"  • Client Type: {router_info['client_type']}")


if __name__ == "__main__":
    asyncio.run(main())
