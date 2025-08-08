"""
Core routing engine that analyzes conversations and selects appropriate models
"""

import json
from typing import Dict, List, Optional, Any, Union

from .models import (
    ModelConfig, 
    UseCase, 
    RoutingResult, 
    ConversationMessage, 
    AuthConfig
)
from .client_factory import ClientFactory, ConfiguredClient
from .exceptions import RoutingError, ModelNotFoundError, ConfigurationError
import math
from structured_logprobs import add_logprobs


class RoutingEngine:
    """Engine that analyzes conversations and determines the best model to use"""
    
    def __init__(
        self,
        routing_model_config: ModelConfig,
        use_cases: List[UseCase],
        routing_temperature: float = 0.1
    ):
        self.routing_model_config = routing_model_config
        self.use_cases = use_cases
        self.routing_temperature = routing_temperature
        self._routing_cache: Dict[str, RoutingResult] = {}
    
    async def analyze_conversation(
        self,
        messages: List[Union[Dict[str, Any], ConversationMessage]],
        available_models: Dict[str, ModelConfig],
        default_model: str
    ) -> RoutingResult:
        """Analyze conversation and determine the best model to use"""
    
        try:
            # Create routing prompt
            routing_prompt = self._create_routing_prompt(messages, self.use_cases)
            
            # Get routing decision from AI model
            routing_response = await self._get_routing_decision(routing_prompt)
            
            # Parse response and create result
            result = self._parse_routing_response(
                routing_response, 
                available_models, 
                default_model
            )
            
            return result
            
        except Exception as e:
            # Fallback to default model if routing fails
            return RoutingResult(
                model_name=default_model,
                selected_model=available_models[default_model],
                use_case=None,
                confidence=0.5,
                reasoning=f"Fallback to default model due to routing error: {str(e)}"
            )
    
    def _format_messages(
        self, 
        messages: List[Union[Dict[str, Any], ConversationMessage]]
    ) -> List[Dict[str, Any]]:
        """Convert messages to standard dictionary format"""
        
        formatted = []
        for msg in messages:
            if isinstance(msg, ConversationMessage):
                formatted.append(msg.dict())
            elif isinstance(msg, dict):
                formatted.append(msg)
            else:
                raise ConfigurationError(f"Invalid message type: {type(msg)}")
        
        return formatted
    
    def _generate_cache_key(self, messages: List[Dict[str, Any]]) -> str:
        """Generate a cache key for the conversation"""
        
        # Create a simplified representation for caching
        key_data = {
            "messages": [
                {"role": msg.get("role"), "content": msg.get("content", "")[:100]}
                for msg in messages[-3:]  # Only use last 3 messages for key
            ],
            "use_cases": [uc.name for uc in self.use_cases]
        }
        
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    def _create_routing_prompt(
        self, 
        messages: List[Dict[str, Any]], 
        use_cases: List[UseCase]
    ) -> List[Dict[str, str]]:
        """Create the prompt for routing decision"""
        
        # Build use case descriptions
        use_case_descriptions = []
        for uc in use_cases:
            description = f"""
**{uc.name}** (Model: {uc.model_name}, Priority: {uc.priority})
- Description: {uc.description}
- Keywords: {', '.join(uc.keywords) if uc.keywords else 'None'}
- Context Requirements: {', '.join(uc.context_requirements) if uc.context_requirements else 'None'}
- Min Confidence: {uc.min_confidence}
"""
            use_case_descriptions.append(description)
        
        # Format conversation for analysis
        conversation_text = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages
        ])
        
        system_prompt = f"""You are an AI model routing assistant. Your job is to analyze conversations and determine which AI model is best suited for the task.

Available Use Cases:
{chr(10).join(use_case_descriptions)}

Please analyze the following conversation and determine:
1. Which use case best matches the conversation (if any)
2. Your confidence level (0.0 to 1.0) in this decision
3. A brief explanation of your reasoning

Respond in valid JSON format with this structure:
{{
    "use_case": "use_case_name or null",
    "confidence": 0.85,
    "reasoning": "Brief explanation of why this use case was selected"
}}

Consider:
- The topic and complexity of the conversation
- Keywords and context that match use case criteria
- The type of task being requested
- The level of reasoning or analysis required

If no use case clearly matches, respond with "use_case": null and explain why."""
        
        routing_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Conversation to analyze:\n\n{conversation_text}"}
        ]
        
        return routing_messages
    
    async def _get_routing_decision(self, routing_prompt: List[Dict[str, str]]) -> Any:
        """Get routing decision from the AI model"""
        
        try:
            # Create client for routing model
            client = ClientFactory.create_openai_client(
                self.routing_model_config, 
                async_client=True
            )
            
            # Make API call
            response = await client.chat.completions.create(
                model=self.routing_model_config.deployment_name,
                messages=routing_prompt,
                temperature=self.routing_temperature,
                logprobs=True,
                response_format={"type": "json_object"}
            )

            logprobs = add_logprobs(response)

            data = json.loads(response.choices[0].message.content)

            # Set the confidence value in the data object using logprobs
            data["confidence"] = math.exp(logprobs.log_probs[0].get("use_case", -100.0))

            return data

        except Exception as e:
            raise RoutingError(f"Failed to get routing decision: {str(e)}")
    
    def _parse_routing_response(
        self,
        response: str,
        available_models: Dict[str, ModelConfig],
        default_model: str
    ) -> RoutingResult:
        """Parse the routing response and create a RoutingResult"""
        
        try:
            # data = json.loads(response)
            data = response

            use_case_name = data.get("use_case")
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "No reasoning provided")
            
            # Find the matching use case
            selected_use_case = None
            selected_model = default_model
            
            if use_case_name:
                for uc in self.use_cases:
                    if uc.name == use_case_name:
                        selected_use_case = uc
                        selected_model = uc.model_name
                        break
            
            # Validate model exists
            if selected_model not in available_models:
                selected_model = default_model
                reasoning += f" (Model {selected_model} not found, using default)"
            
            # Check confidence threshold
            if selected_use_case and confidence < selected_use_case.min_confidence:
                selected_use_case = None
                selected_model = default_model
                reasoning += f" (Confidence {confidence} below threshold, using default)"
            
            return RoutingResult(
                model_name=selected_model,
                selected_model=available_models[selected_model],
                use_case=selected_use_case,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to default model
            return RoutingResult(
                model_name=default_model,
                model_config=available_models[default_model],
                use_case=None,
                confidence=0.5,
                reasoning=f"Failed to parse routing response: {str(e)}"
            )
