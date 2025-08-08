"""
Tests for Brainfork
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from brainfork import (
    ModelRouter,
    ModelConfig,
    AuthConfig, 
    UseCase,
    RoutingResult
)
from brainfork.exceptions import (
    ModelNotFoundError,
    ConfigurationError,
    RoutingError
)


class TestAuthConfig:
    """Test AuthConfig functionality"""
    
    def test_api_key_auth(self):
        auth = AuthConfig(api_key="test-key")
        assert auth.auth_type.value == "api_key"
    
    def test_entra_id_auth(self):
        auth = AuthConfig(
            client_id="test-client-id",
            tenant_id="test-tenant-id"
        )
        assert auth.auth_type.value == "entra_id"
    
    def test_managed_identity_auth(self):
        auth = AuthConfig(use_managed_identity=True)
        assert auth.auth_type.value == "managed_identity"
    
    def test_invalid_auth(self):
        with pytest.raises(ValueError):
            auth = AuthConfig()
            _ = auth.auth_type


class TestModelConfig:
    """Test ModelConfig functionality"""
    
    def test_valid_model_config(self):
        auth = AuthConfig(api_key="test-key")
        config = ModelConfig(
            endpoint="https://test.openai.azure.com/",
            deployment_name="test-deployment",
            auth=auth
        )
        
        assert config.endpoint == "https://test.openai.azure.com/"
        assert config.deployment_name == "test-deployment"
        assert config.api_version == "2024-02-01"  # default
    
    def test_invalid_endpoint(self):
        auth = AuthConfig(api_key="test-key")
        
        with pytest.raises(ValueError):
            ModelConfig(
                endpoint="invalid-url",
                deployment_name="test",
                auth=auth
            )
    
    def test_invalid_temperature(self):
        auth = AuthConfig(api_key="test-key")
        
        with pytest.raises(ValueError):
            ModelConfig(
                endpoint="https://test.openai.azure.com/",
                deployment_name="test",
                auth=auth,
                temperature=3.0  # Invalid, should be 0-2
            )


class TestUseCase:
    """Test UseCase functionality"""
    
    def test_valid_use_case(self):
        use_case = UseCase(
            name="test_case",
            description="Test use case",
            model_name="test-model"
        )
        
        assert use_case.name == "test_case"
        assert use_case.priority == 1  # default
        assert use_case.min_confidence == 0.7  # default
    
    def test_invalid_priority(self):
        with pytest.raises(ValueError):
            UseCase(
                name="test_case",
                description="Test use case", 
                model_name="test-model",
                priority=0  # Invalid, should be >= 1
            )
    
    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            UseCase(
                name="test_case",
                description="Test use case",
                model_name="test-model",
                min_confidence=1.5  # Invalid, should be 0-1
            )


class TestModelRouter:
    """Test ModelRouter functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth = AuthConfig(api_key="test-key")
        
        self.models = {
            "gpt-4o-mini": ModelConfig(
                endpoint="https://test.openai.azure.com/",
                deployment_name="gpt-4o-mini",
                auth=self.auth
            ),
            "o1-mini": ModelConfig(
                endpoint="https://test.openai.azure.com/",
                deployment_name="o1-mini",
                auth=self.auth
            )
        }
        
        self.use_cases = [
            UseCase(
                name="classification",
                description="Text classification",
                model_name="gpt-4o-mini",
                keywords=["classify", "categorize"]
            ),
            UseCase(
                name="reasoning",
                description="Complex reasoning",
                model_name="o1-mini",
                keywords=["math", "reasoning"]
            )
        ]
    
    def test_router_initialization(self):
        """Test router initialization"""
        router = ModelRouter(
            models=self.models,
            use_cases=self.use_cases,
            default_model="gpt-4o-mini"
        )
        
        assert router.default_model == "gpt-4o-mini"
        assert router.routing_model == "gpt-4o-mini"  # Should default to default_model
        assert len(router.models) == 2
        assert len(router.use_cases) == 2
    
    def test_invalid_default_model(self):
        """Test error when default model doesn't exist"""
        with pytest.raises(ModelNotFoundError):
            ModelRouter(
                models=self.models,
                use_cases=self.use_cases,
                default_model="nonexistent-model"
            )
    
    def test_invalid_use_case_model(self):
        """Test error when use case references nonexistent model"""
        invalid_use_cases = [
            UseCase(
                name="invalid",
                description="Invalid use case",
                model_name="nonexistent-model"
            )
        ]
        
        with pytest.raises(ModelNotFoundError):
            ModelRouter(
                models=self.models,
                use_cases=invalid_use_cases,
                default_model="gpt-4o-mini"
            )
    
    def test_duplicate_use_case_names(self):
        """Test error when use case names are not unique"""
        duplicate_use_cases = [
            UseCase(name="same", description="First", model_name="gpt-4o-mini"),
            UseCase(name="same", description="Second", model_name="o1-mini")
        ]
        
        with pytest.raises(ConfigurationError):
            ModelRouter(
                models=self.models,
                use_cases=duplicate_use_cases,
                default_model="gpt-4o-mini"
            )
    
    def test_add_model(self):
        """Test adding a new model"""
        router = ModelRouter(
            models=self.models,
            use_cases=self.use_cases,
            default_model="gpt-4o-mini"
        )
        
        new_model = ModelConfig(
            endpoint="https://new.openai.azure.com/",
            deployment_name="new-model",
            auth=self.auth
        )
        
        router.add_model("new-model", new_model)
        assert "new-model" in router.models
    
    def test_remove_model(self):
        """Test removing a model"""
        router = ModelRouter(
            models=self.models,
            use_cases=self.use_cases,
            default_model="gpt-4o-mini"
        )
        
        router.remove_model("o1-mini")
        assert "o1-mini" not in router.models
    
    def test_cannot_remove_default_model(self):
        """Test that default model cannot be removed"""
        router = ModelRouter(
            models=self.models,
            use_cases=self.use_cases,
            default_model="gpt-4o-mini"
        )
        
        with pytest.raises(ConfigurationError):
            router.remove_model("gpt-4o-mini")
    
    def test_add_use_case(self):
        """Test adding a new use case"""
        router = ModelRouter(
            models=self.models,
            use_cases=self.use_cases,
            default_model="gpt-4o-mini"
        )
        
        new_use_case = UseCase(
            name="new_case",
            description="New use case",
            model_name="o1-mini"
        )
        
        router.add_use_case(new_use_case)
        assert len(router.use_cases) == 3
        assert any(uc.name == "new_case" for uc in router.use_cases)
    
    def test_duplicate_use_case_name(self):
        """Test error when adding duplicate use case name"""
        router = ModelRouter(
            models=self.models,
            use_cases=self.use_cases,
            default_model="gpt-4o-mini"
        )
        
        duplicate_use_case = UseCase(
            name="classification",  # Already exists
            description="Duplicate",
            model_name="o1-mini"
        )
        
        with pytest.raises(ConfigurationError):
            router.add_use_case(duplicate_use_case)
    
    def test_remove_use_case(self):
        """Test removing a use case"""
        router = ModelRouter(
            models=self.models,
            use_cases=self.use_cases,
            default_model="gpt-4o-mini"
        )
        
        router.remove_use_case("classification")
        assert len(router.use_cases) == 1
        assert not any(uc.name == "classification" for uc in router.use_cases)
    
    @pytest.mark.asyncio
    async def test_model_override(self):
        """Test manual model override"""
        router = ModelRouter(
            models=self.models,
            use_cases=self.use_cases,
            default_model="gpt-4o-mini"
        )
        
        messages = [{"role": "user", "content": "Test message"}]
        
        result = await router.route_conversation(
            messages,
            override_model="o1-mini"
        )
        
        assert result.model_name == "o1-mini"
        assert result.confidence == 1.0
        assert "manually overridden" in result.reasoning
    
    @pytest.mark.asyncio
    async def test_invalid_override_model(self):
        """Test error with invalid override model"""
        router = ModelRouter(
            models=self.models,
            use_cases=self.use_cases,
            default_model="gpt-4o-mini"
        )
        
        messages = [{"role": "user", "content": "Test message"}]
        
        with pytest.raises(ModelNotFoundError):
            await router.route_conversation(
                messages,
                override_model="nonexistent-model"
            )
    
    def test_get_model_info(self):
        """Test getting model information"""
        router = ModelRouter(
            models=self.models,
            use_cases=self.use_cases,
            default_model="gpt-4o-mini"
        )
        
        info = router.get_model_info()
        
        assert "models" in info
        assert "use_cases" in info
        assert info["default_model"] == "gpt-4o-mini"
        assert len(info["models"]) == 2
        assert len(info["use_cases"]) == 2


@pytest.mark.asyncio 
async def test_routing_with_mock():
    """Test routing with mocked AI response"""
    
    auth = AuthConfig(api_key="test-key")
    models = {
        "gpt-4o-mini": ModelConfig(
            endpoint="https://test.openai.azure.com/",
            deployment_name="gpt-4o-mini",
            auth=auth
        ),
        "o1-mini": ModelConfig(
            endpoint="https://test.openai.azure.com/",
            deployment_name="o1-mini",
            auth=auth
        )
    }
    
    use_cases = [
        UseCase(
            name="math",
            description="Mathematical problems",
            model_name="o1-mini",
            keywords=["math", "calculate"]
        )
    ]
    
    router = ModelRouter(
        models=models,
        use_cases=use_cases,
        default_model="gpt-4o-mini"
    )
    
    # Mock the routing response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '''
    {
        "use_case": "math",
        "confidence": 0.9,
        "reasoning": "The user is asking for mathematical calculation"
    }
    '''
    
    with patch('azure_ai_router.client_factory.ClientFactory.create_openai_client') as mock_client_factory:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_factory.return_value = mock_client
        
        messages = [{"role": "user", "content": "Calculate 2 + 2"}]
        result = await router.route_conversation(messages)
        
        assert result.model_name == "o1-mini"
        assert result.confidence == 0.9
        assert result.use_case.name == "math"


if __name__ == "__main__":
    pytest.main([__file__])
