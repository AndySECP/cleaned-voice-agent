# core/agent/llm_agent.py
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import json
import logging
from datetime import datetime
import httpx

from src.core.agent.tools import AVAILABLE_TOOLS
from src.core.agent.memory import MemoryManager #, Message

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an AI assistant specializing in analyzing an hallucination detection experiments. 
Speak naturally as if we're having a friendly conversation. Your responses should be easy to listen to and understand. Response should be concise and just respond to the question.

You have access to these tools:
1. get_best_models: Find top models by metric
2. compare_hyperparams: Compare different settings
3. get_experiment_details: Get specific experiment info

Guidelines for speaking:
- Use conversational language like "The model uses..." or "It's configured with..."
- Break information into digestible chunks
- Pause naturally using commas and periods
- Avoid listing technical details unless specifically asked, when you do, do it in a conversational way
- When mentioning numbers, round them for easier speech
- Offer to provide more specific details if needed

Remember to:
- Keep it concise and clear
- Focus on the most relevant information
- Use natural transitions between ideas
- Speak as if you're explaining to a colleague

Example response style:
Instead of listing parameters, say something like:
"This model uses a Llama architecture with 30 layers. The key settings are a learning rate of 0.0003 and a hidden size of 576. Would you like me to go into more detail about any specific aspects?"

The experiments use SmolLM2 models trained for hallucination detection with various settings."""


class HallucinationAnalysisAgent:
    def __init__(
        self,
        openai_api_key: str,
        data_manager,
        query_optimizer,
        tools,
        memory_manager: Optional[MemoryManager] = None,
        use_realtime: bool = False
    ):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.data_manager = data_manager
        self.query_optimizer = query_optimizer
        self.tools = tools
        self.memory = memory_manager or MemoryManager()
        self.api_key = openai_api_key
        self.use_realtime = use_realtime
        
        # Store tools as both dict and list for different use cases
        self.available_functions = {
            "get_best_models": tools.get_best_models,
            "compare_hyperparams": tools.compare_hyperparams,
            "get_experiment_details": tools.get_experiment_details,
            "analyze_by_model_type": tools.analyze_by_model_type,
            "analyze_config_impact": tools.analyze_config_impact,
            "get_performance_distribution": tools.get_performance_distribution,
            "compare_architectures": tools.compare_architectures,
        }
        self.tool_descriptions = AVAILABLE_TOOLS
        
    async def get_realtime_session(self) -> Dict[str, Any]:
        """Generate an ephemeral token for WebRTC connection"""
        async with httpx.AsyncClient() as client:
            try:
                # Format tools for realtime API format
                formatted_tools = []
                for tool in self.tool_descriptions:
                    if tool["type"] == "function":
                        formatted_tool = {
                            "name": tool["function"]["name"],
                            "type": "function",
                            "description": tool["function"]["description"],
                            "parameters": tool["function"]["parameters"]
                        }
                        formatted_tools.append(formatted_tool)

                session_config = {
                    "model": "gpt-4o-realtime-preview-2024-12-17",
                    "voice": "alloy",
                    "tools": formatted_tools,
                    "temperature": 0.7,
                    "instructions": SYSTEM_PROMPT
                }
                
                logger.info(f"Sending realtime session request with config: {json.dumps(session_config, indent=2)}")
                
                response = await client.post(
                    "https://api.openai.com/v1/realtime/sessions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=session_config,
                    timeout=30.0
                )

                if response.status_code != 200:
                    logger.error(f"OpenAI response status: {response.status_code}")
                    logger.error(f"OpenAI response text: {response.text}")
                
                response.raise_for_status()
                return response.json()

            except Exception as e:
                logger.error(f"Error generating realtime session: {str(e)}")
                raise

    async def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a user query using either realtime or standard mode"""
        if self.use_realtime:
            # Check if this is a session initialization request
            is_init = context and context.get('mode') == 'realtime'
            if is_init:
                return await self._process_realtime_query(query, context)
            else:
                return await self._process_standard_query(query, context)
        else:
            return await self._process_standard_query(query, context)

    async def _process_realtime_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a query using the realtime API"""
        try:
            # Get session details with properly formatted tools
            session = await self.get_realtime_session()
            
            # Store query in memory
            await self.memory.add_message("user", query)

            # If this is a function execution request from a previous realtime call
            if context and context.get('function_name'):
                function_name = context['function_name']
                arguments = context['arguments']
                
                # Execute the function if it exists
                if function_name in self.available_functions:
                    try:
                        result = await self.available_functions[function_name](**arguments)
                        return {
                            "response": "Function executed successfully",
                            "metadata": {
                                "function_name": function_name,
                                "success": True
                            },
                            "function_result": result
                        }
                    except Exception as e:
                        logger.error(f"Error executing function {function_name}: {e}")
                        return {
                            "response": f"Error executing function: {str(e)}",
                            "metadata": {
                                "function_name": function_name,
                                "success": False,
                                "error": str(e)
                            }
                        }

            # Return session info for regular realtime initialization
            return {
                "response": "Realtime session initialized",
                "metadata": {
                    "session_id": session["id"],
                    "client_secret": session["client_secret"],
                    "voice": session["voice"],
                    "expires_at": session["expires_at"]
                },
                "session": session  # Full session details
            }

        except Exception as e:
            logger.error(f"Error in realtime processing: {str(e)}")
            raise

    async def _process_standard_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a user query using the LLM agent with simplified conversation handling"""
        try:
            # Get conversation history
            conversation_history = self.memory.get_recent_context()
            
            # Initialize messages array
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": query})

            # Store user query
            await self.memory.add_message("user", query)

            # Get initial response
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                tools=AVAILABLE_TOOLS,
                tool_choice="auto",
                temperature=0.1
            )

            message = response.choices[0].message

            # If no tool calls, store and return direct response
            if not message.tool_calls:
                await self.memory.add_message("assistant", message.content)
                return {
                    "response": message.content,
                    "metadata": {
                        "tools_used": [],
                        "timestamp": datetime.now().isoformat()
                    }
                }

            # Handle tool calls
            tools_used = []
            tool_results = []

            # Process each tool call
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                if tool_name in self.available_functions:
                    try:
                        result = await self.available_functions[tool_name](**tool_args)
                        tool_results.append(result)
                        tools_used.append(tool_name)
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
                        tool_results.append({"error": str(e)})

            # Get final response using tool results
            final_messages = messages.copy()
            if tool_results:
                final_messages.append({
                    "role": "assistant",
                    "content": "I've gathered the requested information."
                })
                final_messages.append({
                    "role": "user",
                    "content": f"Here are the results: {json.dumps(tool_results)}"
                })

            final_response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=final_messages,
                temperature=0.1
            )

            final_message = final_response.choices[0].message.content
            
            # Store final response
            await self.memory.add_message("assistant", final_message)

            return {
                "response": final_message,
                "metadata": {
                    "tools_used": tools_used,
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise

    async def _get_filtered_context(self) -> List[Dict[str, Any]]:
        """Get conversation history ensuring proper tool call handling"""
        try:
            # Get recent messages from memory
            messages = self.memory.get_recent_context()
            
            # Filter and transform messages for the conversation
            filtered_messages = []
            
            for msg in messages:
                if msg["role"] not in {"system", "user", "assistant", "tool"}:
                    continue
                    
                message = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                
                # Handle tool calls in assistant messages
                if msg["role"] == "assistant" and "tool_calls" in msg:
                    message["tool_calls"] = msg["tool_calls"]
                
                # Handle tool response messages
                if msg["role"] == "tool":
                    if "tool_call_id" not in msg or "name" not in msg:
                        continue
                    message["tool_call_id"] = msg["tool_call_id"]
                    message["name"] = msg["name"]
                
                filtered_messages.append(message)
            
            return filtered_messages
        except Exception as e:
            logger.error(f"Error getting filtered context: {e}")
            return []

    async def _create_assistant_message(self, message) -> Dict[str, Any]:
        """Create and store assistant message with tool calls"""
        assistant_message = {
            "role": "assistant",
            "content": message.content or None
        }
        
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in message.tool_calls
            ]
        
        # Store assistant's tool request as essential
        await self.memory.add_message(Message(
            role="assistant",
            content=message.content or "",
            timestamp=datetime.now().isoformat(),
            metadata={},
            tool_calls=assistant_message.get("tool_calls", []),
            essential=True  # Mark tool requests as essential
        ))
        
        return assistant_message

    async def _process_tool_calls(
        self,
        tool_calls: List[Any],
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process tool calls and manage their memory storage"""
        tools_used = []
        tool_messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            if tool_name in self.available_functions:
                # Execute tool
                result = await self.available_functions[tool_name](**tool_args)
                tools_used.append(tool_name)
                
                # Create tool message
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(result)
                }
                tool_messages.append(tool_message)
                
                # Store tool result as non-essential
                await self.memory.add_message(Message(
                    role="tool",
                    content=json.dumps(result),
                    timestamp=datetime.now().isoformat(),
                    metadata={"tool_name": tool_name},
                    tool_call_id=tool_call.id,
                    name=tool_name,
                    essential=False  # Mark tool responses as non-essential
                ))
        
        return {
            "messages": tool_messages,
            "tools_used": tools_used
        }

    async def _handle_response(self, response: Any, original_query: str) -> Dict[str, Any]:
        """Handle LLM response and execute tool calls with memory management"""
        try:
            message = response.choices[0].message
            
            if not message.tool_calls:
                return {
                    "response": message.content,
                    "metadata": {
                        "tools_used": [],
                        "timestamp": datetime.now().isoformat()
                    }
                }

            # Execute tool calls
            tools_used = []
            results = []
            
            for tool_call in message.tool_calls:
                result = await self._execute_tool(tool_call)
                tools_used.append(tool_call.function.name)
                results.append(result)

            # Get final response incorporating tool results
            final_response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": original_query},
                    message,
                    {"role": "function", "name": "function_results", "content": json.dumps(results)},
                ],
                temperature=0.1
            )

            return {
                "response": final_response.choices[0].message.content,
                "raw_results": results,
                "metadata": {
                    "tools_used": tools_used,
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Error handling response: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _execute_tool(self, tool_call: Any) -> Dict[str, Any]:
        """Execute a tool call and return its result"""
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        # Check if tool exists in available functions
        if tool_name in self.available_functions:
            try:
                result = await self.available_functions[tool_name](**tool_args)
                return result
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {str(e)}")
                return {"error": f"Tool execution failed: {str(e)}"}
        else:
            logger.warning(f"Unknown tool requested: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}

    async def _analyze_metric(self, metric_name: str) -> Dict[str, Any]:
        """Analyze a specific metric across experiments"""
        optimization_plan = self.query_optimizer.optimize_query(f"analyze {metric_name}")
        return await self.data_manager.get_metric_analysis(
            metric_name,
            optimization_plan
        )

    async def _compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple experiments"""
        return await self.data_manager.compare_experiments(
            experiment_ids,
            metrics
        )

    async def _get_experiment_details(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific experiment"""
        return await self.data_manager.get_experiment_details(experiment_id)
