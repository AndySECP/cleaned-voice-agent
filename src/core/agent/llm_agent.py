# core/agent/llm_agent.py
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import json
import logging
from datetime import datetime

from src.core.agent.tools import AVAILABLE_TOOLS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""You are an AI assistant specialized in analyzing hallucination detection experiments. 
Your role is to help users understand and analyze experimental results from various model runs.

You have access to the following tools:
1. get_best_models: Get top performing models ranked by a specific metric
2. compare_hyperparams: Compare performance across different hyperparameter settings
3. get_experiment_details: Get detailed information about specific experiments

Available metrics for analysis:
- accuracy (model's accuracy score)
- f1 (F1 score balancing precision and recall)
- precision (true positives / predicted positives)
- recall (true positives / actual positives)
- hallucination_rate (rate of hallucinated content)
- train_loss (training loss)
- eval_loss (evaluation loss)

Guidelines:
- Always provide context and explain your findings
- If data is missing, explain what you found and suggest alternatives
- Use natural language to explain technical concepts
- When comparing models, consider trade-offs between metrics
- Provide specific examples and numbers to support your analysis

The experiments primarily focus on SmolLM2 models trained for hallucination detection, with various 
learning rates and warmup ratios."""


class HallucinationAnalysisAgent:
    def __init__(
        self,
        openai_api_key: str,
        data_manager,
        query_optimizer,
        tools
    ):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.data_manager = data_manager
        self.query_optimizer = query_optimizer
        self.tools = tools
        self.available_functions = {
            "get_best_models": tools.get_best_models,
            "compare_hyperparams": tools.compare_hyperparams,
            "get_experiment_details": tools.get_experiment_details
        }
        
    async def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a user query using the LLM agent"""
        try:
            # Initialize conversation
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]

            if context:
                messages.append({
                    "role": "assistant",
                    "content": f"Previous context: {json.dumps(context)}"
                })

            # Get initial response
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                tools=AVAILABLE_TOOLS,
                tool_choice="auto",
                temperature=0.1
            )

            message = response.choices[0].message
            messages.append(message)  # Add assistant's message to conversation

            # If no tool calls, return direct response
            if not message.tool_calls:
                return {
                    "response": message.content,
                    "metadata": {
                        "tools_used": [],
                        "timestamp": datetime.now().isoformat()
                    }
                }

            # Execute tool calls and add results to conversation
            tools_used = []
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                if tool_name in self.available_functions:
                    # Execute the tool
                    result = await self.available_functions[tool_name](**tool_args)
                    tools_used.append(tool_name)
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(result)
                    })

            # Get final response with tool results
            final_response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=0.1
            )

            return {
                "response": final_response.choices[0].message.content,
                "metadata": {
                    "tools_used": tools_used,
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": "I encountered an error while processing your request. Please try rephrasing your question.",
                "error": str(e),
                "metadata": {
                    "timestamp": datetime.now().isoformat()
                }
            }

    async def _handle_response(self, response: Any, original_query: str) -> Dict[str, Any]:
        """Handle LLM response and execute any tool calls"""
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
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # Execute the appropriate tool
                if tool_name == "analyze_metric":
                    result = await self._analyze_metric(tool_args["metric_name"])
                elif tool_name == "compare_experiments":
                    result = await self._compare_experiments(
                        tool_args["experiment_ids"],
                        tool_args.get("metrics", ["accuracy", "hallucination_rate"])
                    )
                elif tool_name == "get_experiment_details":
                    result = await self._get_experiment_details(tool_args["experiment_id"])
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                tools_used.append(tool_name)
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
