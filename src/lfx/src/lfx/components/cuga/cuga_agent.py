import asyncio
import json
import traceback
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

from langchain_core.agents import AgentFinish
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.tools import StructuredTool, BaseTool

from lfx.base.agents.agent import LCToolsAgentComponent
from lfx.base.models.model_input_constants import (
    ALL_PROVIDER_FIELDS,
    MODEL_DYNAMIC_UPDATE_FIELDS,
    MODEL_PROVIDERS,
    MODEL_PROVIDERS_DICT,
    MODELS_METADATA,
)
from lfx.base.models.model_utils import get_model_name
from lfx.components.helpers import CurrentDateComponent
from lfx.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from lfx.components.models_and_agents.memory import MemoryComponent
from lfx.custom.custom_component.component import _get_component_toolkit
from lfx.custom.utils import update_component_build_config
from lfx.field_typing import Tool
from lfx.io import BoolInput, DropdownInput, IntInput, MultilineInput, Output
from lfx.log.logger import logger
from lfx.schema.dotdict import dotdict
from lfx.schema.message import Message

if TYPE_CHECKING:
    from lfx.schema.log import SendMessageFunctionType


def set_advanced_true(component_input):
    """Set the advanced flag to True for a component input.

    Args:
        component_input: The component input to modify

    Returns:
        The modified component input with advanced=True
    """
    component_input.advanced = True
    return component_input


MODEL_PROVIDERS_LIST = ["OpenAI"]


class CugaComponent(ToolCallingAgentComponent):
    """Cuga Agent Component for advanced AI task execution.

    The Cuga component is an advanced AI agent that can execute complex tasks using
    various tools and browser automation. It supports custom instructions, web applications,
    and API interactions.

    This component uses the new Cuga SDK (v2) which provides a simplified interface
    for agent creation and execution.

    Attributes:
        display_name: Human-readable name for the component
        description: Brief description of the component's purpose
        documentation: URL to component documentation
        icon: Icon identifier for the UI
        name: Internal component name

    Note:
        Future enhancements can include policy features:
        - Intent Guards: Block/allow based on user intent
        - Playbooks: Predefined workflows for specific tasks
        - Tool Guides: Provide guidance on tool usage
        - Tool Approval: Require approval before tool execution
        - Output Formatters: Format agent responses
    """

    display_name: str = "Cuga"
    description: str = "Define the Cuga agent's instructions, then assign it a task."
    documentation: str = "https://docs.langflow.org/bundles-cuga"
    icon = "bot"
    name = "Cuga"

    memory_inputs = [set_advanced_true(component_input) for component_input in MemoryComponent().inputs]

    inputs = [
        DropdownInput(
            name="agent_llm",
            display_name="Model Provider",
            info="The provider of the language model that the agent will use to generate responses.",
            options=[*MODEL_PROVIDERS_LIST, "Custom"],
            value="OpenAI",
            real_time_refresh=True,
            input_types=[],
            options_metadata=[MODELS_METADATA[key] for key in MODEL_PROVIDERS_LIST] + [{"icon": "brain"}],
        ),
        *MODEL_PROVIDERS_DICT["OpenAI"]["inputs"],
        MultilineInput(
            name="instructions",
            display_name="Instructions",
            info=(
                "Custom instructions for the agent to adhere to during its operation.\n"
                "Example:\n"
                "## Plan\n"
                "< planning instructions e.g. which tools and when to use>\n"
                "## Answer\n"
                "< final answer instructions how to answer>"
            ),
            value="",
            advanced=False,
        ),
        IntInput(
            name="n_messages",
            display_name="Number of Chat History Messages",
            value=100,
            info="Number of chat history messages to retrieve.",
            advanced=True,
            show=True,
        ),
        *LCToolsAgentComponent.get_base_inputs(),
        BoolInput(
            name="add_current_date_tool",
            display_name="Current Date",
            advanced=True,
            info="If true, will add a tool to the agent that returns the current date.",
            value=True,
        ),
        BoolInput(
            name="lite_mode",
            display_name="Enable CugaLite",
            info="Faster reasoning for simple tasks. Enable CugaLite for simple API tasks.",
            value=True,
            advanced=True,
        ),
        IntInput(
            name="lite_mode_tool_threshold",
            display_name="CugaLite Tool Threshold",
            info="Route to CugaLite if app has fewer than this many tools.",
            value=25,
            advanced=True,
        ),
        DropdownInput(
            name="decomposition_strategy",
            display_name="Decomposition Strategy",
            info="Strategy for task decomposition: 'flexible' allows multiple subtasks per app,\n"
            " 'exact' enforces one subtask per app.",
            options=["flexible", "exact"],
            value="flexible",
            advanced=True,
        ),
        BoolInput(
            name="browser_enabled",
            display_name="Enable Browser",
            info="Toggle to enable a built-in browser tool for web scraping and searching.",
            value=False,
            advanced=True,
        ),
        MultilineInput(
            name="web_apps",
            display_name="Web applications",
            info=(
                "Cuga will automatically start this web application when Enable Browser is true. "
                "Currently only supports one web application. Example: https://example.com"
            ),
            value="",
            advanced=True,
        ),
    ]
    outputs = [
        Output(name="response", display_name="Response", method="message_response"),
    ]

    def _convert_tools_to_langchain(self, lfx_tools: list[Tool]) -> list[BaseTool]:
        """Convert LFX tools to LangChain BaseTool format.

        Args:
            lfx_tools: List of LFX Tool objects

        Returns:
            list[BaseTool]: List of LangChain BaseTool objects
        """
        langchain_tools = []
        for tool in lfx_tools:
            if isinstance(tool, (StructuredTool, BaseTool)):
                langchain_tools.append(tool)
            else:
                # If tool has a different format, try to convert it
                logger.warning(f"[CUGA] Tool {getattr(tool, 'name', 'unknown')} may need conversion")
                langchain_tools.append(tool)
        return langchain_tools

    def _convert_state_to_event(self, state_update: tuple | dict, node_name: str | None = None) -> dict[str, Any]:
        """Convert LangGraph state update to LFX event format.

        Args:
            state_update: Tuple of (node_name, state_dict) from LangGraph stream or dict
            node_name: Optional node name override

        Returns:
            dict: LFX-formatted event dictionary
        """
        try:
            # LangGraph stream yields tuples of (node_name, state_dict)
            if isinstance(state_update, tuple) and len(state_update) == 2:
                node, state = state_update
                node_name = node_name or node
            else:
                # Fallback for different formats
                state = state_update if isinstance(state_update, dict) else {}
                node_name = node_name or "unknown"

            # Extract relevant information from state
            if isinstance(state, dict):
                logger.debug(f"[CUGA] State dict keys: {state.keys()}")
                
                # Check for final answer in various possible fields
                # The Cuga SDK v2 may use different state structures
                
                # Try to extract final answer from messages (LangGraph pattern)
                if "messages" in state:
                    messages = state.get("messages", [])
                    logger.debug(f"[CUGA] Found messages field with {len(messages)} messages")
                    if messages and len(messages) > 0:
                        last_message = messages[-1]
                        logger.debug(f"[CUGA] Last message type: {type(last_message)}")
                        # Check if this is an AI message (final answer)
                        if isinstance(last_message, AIMessage):
                            if hasattr(last_message, "content"):
                                final_answer = last_message.content
                                logger.debug(f"[CUGA] Last AI message content: '{final_answer}'")
                                if final_answer and str(final_answer).strip():
                                    logger.debug(f"[CUGA] Yielding on_chain_end with final answer from messages")
                                    return {
                                        "event": "on_chain_end",
                                        "run_id": str(uuid.uuid4()),
                                        "name": "CugaAgent",
                                        "data": {"output": AgentFinish(return_values={"output": str(final_answer)}, log="")},
                                    }
                
                # Check for final_answer field (legacy pattern)
                if "final_answer" in state:
                    final_answer = state.get("final_answer", "")
                    logger.debug(f"[CUGA] Found final_answer field in state: '{final_answer}'")
                    if final_answer and str(final_answer).strip():
                        logger.debug(f"[CUGA] Yielding on_chain_end with final_answer field")
                        return {
                            "event": "on_chain_end",
                            "run_id": str(uuid.uuid4()),
                            "name": "CugaAgent",
                            "data": {"output": AgentFinish(return_values={"output": str(final_answer)}, log="")},
                        }

                # Check for error
                if "error" in state:
                    error_msg = state.get("error", "Unknown error")
                    return {
                        "event": "on_chain_error",
                        "run_id": str(uuid.uuid4()),
                        "name": node_name,
                        "data": {"error": error_msg},
                    }

                # Check for tool execution (script field indicates code execution)
                if "script" in state:
                    script_content = state.get("script", "")
                    return {
                        "event": "on_tool_start",
                        "run_id": str(uuid.uuid4()),
                        "name": "CodeAgent",
                        "data": {"input": {"code": script_content}},
                    }

                # Generic state update - treat as thinking/processing
                return {
                    "event": "on_chain_start",
                    "run_id": str(uuid.uuid4()),
                    "name": node_name,
                    "data": {"input": state},
                }
            
            # If state is not a dict, return a generic event
            return {
                "event": "on_chain_start",
                "run_id": str(uuid.uuid4()),
                "name": node_name or "unknown",
                "data": {"input": {}},
            }

        except Exception as e:
            logger.error(f"[CUGA] Error converting state to event: {e}")
            logger.error(f"[CUGA] Traceback: {traceback.format_exc()}")
            return {
                "event": "on_chain_error",
                "run_id": str(uuid.uuid4()),
                "name": "StateConverter",
                "data": {"error": f"State conversion error: {str(e)}"},
            }

    async def call_agent(
        self, current_input: str, tools: list[Tool], history_messages: list[Message], llm
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute the Cuga agent with the given input and tools.

        This method initializes and runs the Cuga agent using the new SDK,
        processing the input through the agent's workflow and yielding events
        for real-time monitoring.

        Args:
            current_input: The user input to process
            tools: List of available tools for the agent
            history_messages: Previous conversation history
            llm: The language model instance to use

        Yields:
            dict: Agent events including tool usage, thinking, and final results

        Raises:
            Exception: All exceptions are caught and converted to error events
        """
        # Initial event
        yield {
            "event": "on_chain_start",
            "run_id": str(uuid.uuid4()),
            "name": "CUGA_initializing",
            "data": {"input": {"input": current_input, "chat_history": []}},
        }

        logger.debug(f"[CUGA] LLM MODEL TYPE: {type(llm)}")

        if not current_input:
            error_msg = "Input cannot be empty"
            logger.error(f"[CUGA] {error_msg}")
            yield {
                "event": "on_chain_error",
                "run_id": str(uuid.uuid4()),
                "name": "CugaAgent",
                "data": {"error": error_msg},
            }
            return

        try:
            # Configure Cuga settings
            from cuga.config import settings

            logger.debug("[CUGA] Updating CUGA settings")

            settings.advanced_features.lite_mode = self.lite_mode
            settings.advanced_features.lite_mode_tool_threshold = self.lite_mode_tool_threshold
            settings.advanced_features.decomposition_strategy = self.decomposition_strategy

            if self.browser_enabled:
                logger.debug("[CUGA] browser_enabled is true, setting mode to hybrid")
                settings.advanced_features.mode = "hybrid"
                settings.advanced_features.use_vision = False
            else:
                logger.debug("[CUGA] browser_enabled is false, setting mode to api")
                settings.advanced_features.mode = "api"

            # Import new SDK
            from cuga.sdk import CugaAgent

            # Convert history messages to LangChain format
            logger.debug(f"[CUGA] Converting {len(history_messages)} history messages to LangChain format")
            lc_messages = []
            for i, msg in enumerate(history_messages):
                msg_text = getattr(msg, "text", "N/A")[:50] if hasattr(msg, "text") else "N/A"
                logger.debug(
                    f"[CUGA] Message {i}: type={type(msg)}, sender={getattr(msg, 'sender', 'N/A')}, "
                    f"text={msg_text}..."
                )
                # Ensure text is a string
                msg_content = str(msg.text) if hasattr(msg, "text") else ""
                if hasattr(msg, "sender") and msg.sender == "Human":
                    lc_messages.append(HumanMessage(content=msg_content))
                else:
                    lc_messages.append(AIMessage(content=msg_content))

            logger.debug(f"[CUGA] Converted to {len(lc_messages)} LangChain messages")

            # Convert tools to LangChain format
            langchain_tools = self._convert_tools_to_langchain(tools)
            logger.debug(f"[CUGA] Converted {len(langchain_tools)} tools to LangChain format")

            # Get instructions
            instructions_to_use = self.instructions or ""
            logger.debug(f"[CUGA] Using instructions: {instructions_to_use[:100]}...")

            # Get Langfuse callbacks for tracing
            # Debug tracing service state
            logger.debug(f"[CUGA] Tracing service exists: {hasattr(self, 'tracing_service')}")
            if hasattr(self, 'tracing_service') and self.tracing_service:
                logger.debug(f"[CUGA] Tracing service type: {type(self.tracing_service).__name__}")
                logger.debug(f"[CUGA] Tracing service deactivated: {getattr(self.tracing_service, 'deactivated', 'N/A')}")
                
                # Check trace context
                try:
                    from langflow.services.tracing.service import trace_context_var
                    trace_context = trace_context_var.get()
                    logger.debug(f"[CUGA] Trace context exists: {trace_context is not None}")
                    if trace_context:
                        logger.debug(f"[CUGA] Trace context tracers: {list(trace_context.tracers.keys())}")
                        for name, tracer in trace_context.tracers.items():
                            logger.debug(f"[CUGA] Tracer '{name}' ready: {getattr(tracer, 'ready', 'N/A')}")
                except Exception as e:
                    logger.debug(f"[CUGA] Error checking trace context: {e}")
            else:
                logger.warning("[CUGA] No tracing service available")
                
            langchain_callbacks = self.get_langchain_callbacks()
            logger.debug(f"[CUGA] Got {len(langchain_callbacks)} Langchain callbacks for tracing")
            logger.debug(f"[CUGA] Callbacks: {[type(cb).__name__ for cb in langchain_callbacks]}")
            
            if len(langchain_callbacks) == 0:
                logger.warning("[CUGA] No Langfuse callbacks available - check environment variables:")
                logger.warning("[CUGA]   LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST")

            # Create Cuga agent with new SDK
            # Note: We pass callbacks to the agent, not to the LLM
            # The CUGA SDK expects callbacks to be set on the agent instance
            logger.debug("[CUGA] Creating CugaAgent with new SDK")
            cuga_agent = CugaAgent(
                tools=langchain_tools,
                model=llm,
                special_instructions=instructions_to_use if instructions_to_use else None,
                callbacks=langchain_callbacks,  # Pass callbacks to agent constructor
            )
            logger.debug(f"[CUGA] Created agent with {len(langchain_callbacks)} callbacks")

            # Get thread_id for conversation continuity
            thread_id = str(self.graph.session_id)
            logger.debug(f"[CUGA] Using thread_id (session_id): {thread_id}")

            # Yield thinking event
            yield {
                "event": "on_chain_start",
                "run_id": str(uuid.uuid4()),
                "name": "CUGA_thinking",
                "data": {"input": {"input": current_input, "chat_history": lc_messages}},
            }

            # Check if this is a continuation (has history)
            if lc_messages:
                # Add current input to history
                lc_messages.append(HumanMessage(content=current_input))
                message_to_send = lc_messages
                logger.debug(f"[CUGA] Continuing conversation with {len(lc_messages)} total messages")
            else:
                message_to_send = current_input
                logger.debug("[CUGA] Starting new conversation")

            # Stream agent execution
            logger.debug("[CUGA] Starting agent stream")
            last_state = None
            tool_events = {}  # Track tool events by run_id
            received_final_answer = False  # Track if we got a final answer in the stream

            async for state_update in cuga_agent.stream(
                message=message_to_send,
                thread_id=thread_id,
            ):
                logger.debug(f"[CUGA] Received state update: {type(state_update)}")
                logger.debug(f"[CUGA] State update content: {state_update}")
                last_state = state_update

                # Convert state update to LFX event format
                event = self._convert_state_to_event(state_update)

                # Track tool events to emit tool_end when we get results
                if event["event"] == "on_tool_start":
                    tool_events[event["run_id"]] = event
                    yield event
                elif event["event"] == "on_tool_end":
                    yield event
                elif event["event"] == "on_chain_end":
                    # Final answer received in stream
                    received_final_answer = True
                    yield event
                elif event["event"] == "on_chain_error":
                    yield event
                else:
                    # Other events (thinking, processing)
                    yield event

            # Only get final state if we didn't receive a final answer in the stream
            # This avoids unnecessary blocking calls and potential timeouts
            if not received_final_answer:
                logger.debug("[CUGA] No final answer in stream, getting final state from graph")
                
                try:
                    # Get the final state from the graph using the thread_id
                    final_state = cuga_agent.graph.get_state({"configurable": {"thread_id": thread_id}})
                    logger.debug(f"[CUGA] Final state type: {type(final_state)}")
                    logger.debug(f"[CUGA] Final state: {final_state}")
                    
                    # Extract the state values
                    if hasattr(final_state, "values"):
                        state_values = final_state.values
                        logger.debug(f"[CUGA] State values keys: {state_values.keys() if isinstance(state_values, dict) else 'N/A'}")
                        
                        # Try to extract final answer from various possible fields
                        final_answer = None
                        
                        # Check for messages field (LangGraph pattern)
                        if isinstance(state_values, dict) and "messages" in state_values:
                            messages = state_values.get("messages", [])
                            logger.debug(f"[CUGA] Found {len(messages)} messages in final state")
                            if messages and len(messages) > 0:
                                last_message = messages[-1]
                                logger.debug(f"[CUGA] Last message type: {type(last_message)}")
                                if isinstance(last_message, AIMessage):
                                    final_answer = last_message.content
                                    logger.debug(f"[CUGA] Extracted answer from last AI message: '{final_answer}'")
                        
                        # Check for final_answer field
                        if not final_answer and isinstance(state_values, dict) and "final_answer" in state_values:
                            final_answer = state_values.get("final_answer", "")
                            logger.debug(f"[CUGA] Found final_answer field: '{final_answer}'")
                        
                        # Check for output field
                        if not final_answer and isinstance(state_values, dict) and "output" in state_values:
                            final_answer = state_values.get("output", "")
                            logger.debug(f"[CUGA] Found output field: '{final_answer}'")
                        
                        if final_answer and str(final_answer).strip():
                            logger.debug(f"[CUGA] Yielding final on_chain_end event with answer: '{final_answer}'")
                            yield {
                                "event": "on_chain_end",
                                "run_id": str(uuid.uuid4()),
                                "name": "CugaAgent",
                                "data": {"output": AgentFinish(return_values={"output": str(final_answer)}, log="")},
                            }
                        else:
                            logger.warning("[CUGA] No final answer found in state values")
                    else:
                        logger.warning(f"[CUGA] Final state has no 'values' attribute: {dir(final_state)}")
                except Exception as e:
                    logger.error(f"[CUGA] Error getting final state: {e}")
                    logger.error(f"[CUGA] Traceback: {traceback.format_exc()}")
            else:
                logger.debug("[CUGA] Final answer already received in stream, skipping get_state()")

        except Exception as e:
            # Comprehensive error handling - catch ALL exceptions
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(f"[CUGA] {error_type} occurred: {error_msg}")
            logger.error(f"[CUGA] Traceback: {traceback.format_exc()}")

            # Check for specific error types and provide helpful messages
            if "playwright" in error_msg.lower():
                error_msg = (
                    "Playwright is not installed. Please install Playwright Chromium using: "
                    "uv run -m playwright install chromium"
                )
            elif "connection" in error_msg.lower():
                error_msg = f"Connection error: {error_msg}"
            elif "timeout" in error_msg.lower():
                error_msg = f"Timeout error: {error_msg}"

            # Always yield error event
            yield {
                "event": "on_chain_error",
                "run_id": str(uuid.uuid4()),
                "name": "CugaAgent",
                "data": {"error": f"{error_type}: {error_msg}"},
            }

    async def message_response(self) -> Message:
        """Generate a message response using the Cuga agent.

        This method processes the input through the Cuga agent and returns a structured
        message response. It handles agent initialization, tool setup, and event processing.

        Returns:
            Message: The agent's response message

        Raises:
            ValueError: If input is empty or invalid
            Exception: If there's an error during agent execution
        """
        logger.debug("[CUGA] Starting Cuga agent run for message_response.")
        logger.debug(f"[CUGA] Agent input value: {self.input_value}")

        # Validate input is not empty
        if not self.input_value or not str(self.input_value).strip():
            msg = "Message cannot be empty. Please provide a valid message."
            raise ValueError(msg)

        try:
            from lfx.schema.content_block import ContentBlock
            from lfx.schema.message import MESSAGE_SENDER_AI

            llm_model, self.chat_history, self.tools = await self.get_agent_requirements()

            # Create agent message for event processing
            agent_message = Message(
                sender=MESSAGE_SENDER_AI,
                sender_name="Cuga",
                properties={"icon": "Bot", "state": "partial"},
                content_blocks=[ContentBlock(title="Agent Steps", contents=[])],
                session_id=self.graph.session_id,
            )

            # Pre-assign an ID for event processing, following the base agent pattern
            # This ensures streaming works even when not connected to ChatOutput
            if not self.is_connected_to_chat_output():
                # When not connected to ChatOutput, assign ID upfront for streaming support
                agent_message.data["id"] = uuid.uuid4()

            # Get input text
            input_text = self.input_value.text if hasattr(self.input_value, "text") else str(self.input_value)

            # Create event iterator from call_agent
            event_iterator = self.call_agent(
                current_input=input_text, tools=self.tools or [], history_messages=self.chat_history, llm=llm_model
            )

            # Process events using the existing event processing system
            from lfx.base.agents.events import process_agent_events

            # Create a wrapper that forces DB updates for event handlers
            # This ensures the UI can see loading steps in real-time via polling
            async def force_db_update_send_message(message, id_=None, *, skip_db_update=False):  # noqa: ARG001
                # Always persist to DB so polling-based UI shows loading steps in real-time
                content_blocks_len = len(message.content_blocks[0].contents) if message.content_blocks else 0
                logger.debug(
                    f"[CUGA] Sending message update - state: {message.properties.state}, "
                    f"content_blocks: {content_blocks_len}"
                )

                result = await self.send_message(message, id_=id_, skip_db_update=False)

                logger.debug(f"[CUGA] Message processed with ID: {result.id}")
                return result

            result = await process_agent_events(
                event_iterator, agent_message, cast("SendMessageFunctionType", force_db_update_send_message)
            )

            logger.debug("[CUGA] Agent run finished successfully.")
            logger.debug(f"[CUGA] Agent output: {result}")

            return result

        except Exception as e:
            logger.error(f"[CUGA] Error in message_response: {e}")
            logger.error(f"[CUGA] Traceback: {traceback.format_exc()}")

            # Check if error is related to Playwright installation
            error_str = str(e).lower()
            if "playwright install" in error_str or "playwright" in error_str:
                msg = (
                    "Playwright is not installed. Please install Playwright Chromium using: "
                    "uv run -m playwright install chromium"
                )
                raise ValueError(msg) from e

            # Re-raise the exception with context
            raise

    async def get_agent_requirements(self):
        """Get the agent requirements for the Cuga agent.

        This method retrieves and configures all necessary components for the agent
        including the language model, chat history, and tools.

        Returns:
            tuple: A tuple containing (llm_model, chat_history, tools)

        Raises:
            ValueError: If no language model is selected or if there's an error
                in model initialization
        """
        llm_model, display_name = await self.get_llm()
        if llm_model is None:
            msg = "No language model selected. Please choose a model to proceed."
            raise ValueError(msg)
        self.model_name = get_model_name(llm_model, display_name=display_name)

        # Get memory data
        self.chat_history = await self.get_memory_data()
        if isinstance(self.chat_history, Message):
            self.chat_history = [self.chat_history]

        # Add current date tool if enabled
        if self.add_current_date_tool:
            if not isinstance(self.tools, list):
                self.tools = []
            current_date_tool = (await CurrentDateComponent(**self.get_base_args()).to_toolkit()).pop(0)
            if not isinstance(current_date_tool, StructuredTool):
                msg = "CurrentDateComponent must be converted to a StructuredTool"
                raise TypeError(msg)
            self.tools.append(current_date_tool)

        logger.debug("[CUGA] Retrieved agent requirements: LLM, chat history, and tools.")
        logger.debug(f"[CUGA] LLM model: {self.model_name}")
        logger.debug(f"[CUGA] Number of chat history messages: {len(self.chat_history)}")
        logger.debug(f"[CUGA] Tools available: {[tool.name for tool in self.tools]}")
        logger.debug(f"[CUGA] metadata: {[tool.metadata for tool in self.tools]}")

        return llm_model, self.chat_history, self.tools

    async def get_memory_data(self):
        """Retrieve chat history messages.

        This method fetches the conversation history from memory, excluding the current
        input message to avoid duplication.

        Returns:
            list: List of Message objects representing the chat history
        """
        logger.debug("[CUGA] Retrieving chat history messages.")
        logger.debug(f"[CUGA] Session ID: {self.graph.session_id}")
        logger.debug(f"[CUGA] n_messages: {self.n_messages}")
        logger.debug(f"[CUGA] input_value: {self.input_value}")
        logger.debug(f"[CUGA] input_value type: {type(self.input_value)}")
        logger.debug(f"[CUGA] input_value id: {getattr(self.input_value, 'id', None)}")

        messages = (
            await MemoryComponent(**self.get_base_args())
            .set(session_id=str(self.graph.session_id), order="Ascending", n_messages=self.n_messages)
            .retrieve_messages()
        )
        # Handle case where messages might not be a list
        if not isinstance(messages, list):
            messages = []
        logger.debug(f"[CUGA] Retrieved {len(messages)} messages from memory")
        return [
            message for message in messages if getattr(message, "id", None) != getattr(self.input_value, "id", None)
        ]

    async def get_llm(self):
        """Get language model for the Cuga agent.

        This method initializes and configures the language model based on the
        selected provider and parameters.

        Returns:
            tuple: A tuple containing (llm_model, display_name)

        Raises:
            ValueError: If the model provider is invalid or model initialization fails
        """
        logger.debug("[CUGA] Getting language model for the agent.")
        logger.debug(f"[CUGA] Requested LLM provider: {self.agent_llm}")

        if not isinstance(self.agent_llm, str):
            logger.debug("[CUGA] Agent LLM is already a model instance.")
            return self.agent_llm, None

        try:
            provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
            if not provider_info:
                msg = f"Invalid model provider: {self.agent_llm}"
                raise ValueError(msg)

            component_class = provider_info.get("component_class")
            display_name = component_class.display_name
            inputs = provider_info.get("inputs")
            prefix = provider_info.get("prefix", "")
            logger.debug(f"[CUGA] Successfully built LLM model from provider: {self.agent_llm}")
            return self._build_llm_model(component_class, inputs, prefix), display_name

        except (AttributeError, ValueError, TypeError, RuntimeError) as e:
            await logger.aerror(f"[CUGA] Error building {self.agent_llm} language model: {e!s}")
            msg = f"Failed to initialize language model: {e!s}"
            raise ValueError(msg) from e

    def _build_llm_model(self, component, inputs, prefix=""):
        """Build LLM model with parameters.

        This method constructs a language model instance using the provided component
        class and input parameters.

        Args:
            component: The LLM component class to instantiate
            inputs: List of input field definitions
            prefix: Optional prefix for parameter names

        Returns:
            The configured LLM model instance
        """
        model_kwargs = {}
        for input_ in inputs:
            if hasattr(self, f"{prefix}{input_.name}"):
                model_kwargs[input_.name] = getattr(self, f"{prefix}{input_.name}")
        return component.set(**model_kwargs).build_model()

    def set_component_params(self, component):
        """Set component parameters based on provider.

        This method configures component parameters according to the selected
        model provider's requirements.

        Args:
            component: The component to configure

        Returns:
            The configured component
        """
        provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
        if provider_info:
            inputs = provider_info.get("inputs")
            prefix = provider_info.get("prefix")
            model_kwargs = {}
            for input_ in inputs:
                if hasattr(self, f"{prefix}{input_.name}"):
                    model_kwargs[input_.name] = getattr(self, f"{prefix}{input_.name}")
            return component.set(**model_kwargs)
        return component

    def delete_fields(self, build_config: dotdict, fields: dict | list[str]) -> None:
        """Delete specified fields from build_config.

        This method removes unwanted fields from the build configuration.

        Args:
            build_config: The build configuration dictionary
            fields: Fields to remove (can be dict or list of strings)
        """
        for field in fields:
            build_config.pop(field, None)

    def update_input_types(self, build_config: dotdict) -> dotdict:
        """Update input types for all fields in build_config.

        This method ensures all fields in the build configuration have proper
        input types defined.

        Args:
            build_config: The build configuration to update

        Returns:
            dotdict: Updated build configuration with input types
        """
        for key, value in build_config.items():
            if isinstance(value, dict):
                if value.get("input_types") is None:
                    build_config[key]["input_types"] = []
            elif hasattr(value, "input_types") and value.input_types is None:
                value.input_types = []
        return build_config

    async def update_build_config(
        self, build_config: dotdict, field_value: str, field_name: str | None = None
    ) -> dotdict:
        """Update build configuration based on field changes.

        This method dynamically updates the component's build configuration when
        certain fields change, particularly the model provider selection.

        Args:
            build_config: The current build configuration
            field_value: The new value for the field
            field_name: The name of the field being changed

        Returns:
            dotdict: Updated build configuration

        Raises:
            ValueError: If required keys are missing from the configuration
        """
        if field_name in ("agent_llm",):
            build_config["agent_llm"]["value"] = field_value
            provider_info = MODEL_PROVIDERS_DICT.get(field_value)
            if provider_info:
                component_class = provider_info.get("component_class")
                if component_class and hasattr(component_class, "update_build_config"):
                    build_config = await update_component_build_config(
                        component_class, build_config, field_value, "model_name"
                    )

            provider_configs: dict[str, tuple[dict, list[dict]]] = {
                provider: (
                    MODEL_PROVIDERS_DICT[provider]["fields"],
                    [
                        MODEL_PROVIDERS_DICT[other_provider]["fields"]
                        for other_provider in MODEL_PROVIDERS_DICT
                        if other_provider != provider
                    ],
                )
                for provider in MODEL_PROVIDERS_DICT
            }
            if field_value in provider_configs:
                fields_to_add, fields_to_delete = provider_configs[field_value]

                # Delete fields from other providers
                for fields in fields_to_delete:
                    self.delete_fields(build_config, fields)

                # Add provider-specific fields
                if field_value == "OpenAI" and not any(field in build_config for field in fields_to_add):
                    build_config.update(fields_to_add)
                else:
                    build_config.update(fields_to_add)
                build_config["agent_llm"]["input_types"] = []
            elif field_value == "Custom":
                # Delete all provider fields
                self.delete_fields(build_config, ALL_PROVIDER_FIELDS)
                # Update with custom component
                custom_component = DropdownInput(
                    name="agent_llm",
                    display_name="Language Model",
                    options=[*sorted(MODEL_PROVIDERS), "Custom"],
                    value="Custom",
                    real_time_refresh=True,
                    input_types=["LanguageModel"],
                    options_metadata=[MODELS_METADATA[key] for key in sorted(MODELS_METADATA.keys())]
                    + [{"icon": "brain"}],
                )
                build_config.update({"agent_llm": custom_component.to_dict()})

            # Update input types for all fields
            build_config = self.update_input_types(build_config)

            # Validate required keys
            default_keys = [
                "code",
                "_type",
                "agent_llm",
                "tools",
                "input_value",
                "add_current_date_tool",
                "instructions",
                "agent_description",
                "max_iterations",
                "handle_parsing_errors",
                "verbose",
            ]
            missing_keys = [key for key in default_keys if key not in build_config]
            if missing_keys:
                msg = f"Missing required keys in build_config: {missing_keys}"
                raise ValueError(msg)

        if (
            isinstance(self.agent_llm, str)
            and self.agent_llm in MODEL_PROVIDERS_DICT
            and field_name in MODEL_DYNAMIC_UPDATE_FIELDS
        ):
            provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
            if provider_info:
                component_class = provider_info.get("component_class")
                component_class = self.set_component_params(component_class)
                prefix = provider_info.get("prefix")
                if component_class and hasattr(component_class, "update_build_config"):
                    if isinstance(field_name, str) and isinstance(prefix, str):
                        field_name = field_name.replace(prefix, "")
                    build_config = await update_component_build_config(
                        component_class, build_config, field_value, "model_name"
                    )
        return dotdict({k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in build_config.items()})

    async def _get_tools(self) -> list[Tool]:
        """Build agent tools.

        This method constructs the list of tools available to the Cuga agent,
        including component tools and any additional configured tools.

        Returns:
            list[Tool]: List of available tools for the agent
        """
        logger.debug("[CUGA] Building agent tools.")
        component_toolkit = _get_component_toolkit()
        tools_names = self._build_tools_names()
        agent_description = self.get_tool_description()
        description = f"{agent_description}{tools_names}"
        tools = component_toolkit(component=self).get_tools(
            tool_name="Call_CugaAgent", tool_description=description, callbacks=self.get_langchain_callbacks()
        )
        if hasattr(self, "tools_metadata"):
            tools = component_toolkit(component=self, metadata=self.tools_metadata).update_tools_metadata(tools=tools)
        logger.debug(f"[CUGA] Tools built: {[tool.name for tool in tools]}")
        return tools

# Made with Bob
