import streamlit as st
from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
from haystack.dataclasses import ChatMessage
import os
import nest_asyncio
import logging
import json
import traceback
import pandas as pd
import re

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Haystack loggers to ERROR level to reduce noise
logging.getLogger('haystack').setLevel(logging.ERROR)
logging.getLogger('haystack_integrations').setLevel(logging.ERROR)

# Apply nest_asyncio to allow nested asyncio event loops
nest_asyncio.apply()

# Set page config and initialize chat history
st.set_page_config(page_title="Chat with OnSIDES Database", layout="wide")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for tool invocations
if "tool_invocations" not in st.session_state:
    st.session_state.tool_invocations = []

# Function to initialize the Haystack pipeline
@st.cache_resource
def initialize_pipeline():
    # Check for API key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        st.error("ANTHROPIC_API_KEY environment variable not set.")
        st.stop()
    
    # Initialize MCP tools
    current_env = os.environ.copy()
    server_info = StdioServerInfo(
        command="uv", 
        args=["run","mcp-alchemy", "--with", "psycopg2-binary", "--refresh-package", "mcp-alchemy", "mcp-alchemy"],
        env=current_env
    )
    mcp_toolset = MCPToolset(server_info)
    
    # Initialize the agent and pipeline
    agent = Agent(
        chat_generator=AnthropicChatGenerator(model="claude-3-7-sonnet-latest"), 
        tools=mcp_toolset
    )
    
    pipeline = Pipeline()
    pipeline.add_component("agent", agent)
    return pipeline, mcp_toolset

# Initialize pipeline once
pipeline, mcp_toolset = initialize_pipeline()

# Add sidebar with tool information
with st.sidebar:
    st.title("Available Tools")
    # Display tools in sidebar
    for tool in mcp_toolset.tools:
        with st.expander(f"🔧 {tool.name}"):
            st.markdown(tool.description)
            st.markdown("**Parameters:**")
            st.json(tool.parameters)
    
    # Add database schema information
    st.title("OnSIDES Database")
    st.markdown("""
    Database containing 3.6 million drug-adverse event pairs across 2,793 drug ingredients. See the complete [table descriptions and schema](https://github.com/tatonetti-lab/onsides?tab=readme-ov-file#table-descriptions) for more details.
    """)
    
    st.markdown("---")

# Main chat interface
st.title("MedOryx - Adverse Drug Events")
st.info("""**Chat with OnSIDES database powered by [Haystack](https://github.com/deepset-ai/haystack) & [MCP-Alchemy](https://github.com/runekaagaard/mcp-alchemy).**

[OnSIDES](https://onsidesdb.org) is a comprehensive resource containing over 3.6 million drug-ADE pairs extracted from manually curated drug labels. With this app, you can:

- Query adverse drug events (ADEs) for specific medications
- Find drugs that cause particular side effects
- Compare adverse events across drug classes
- Explore serious ADEs from drug labels' Boxed Warnings section

Reference: [OnSIDES: Extracting adverse drug events from drug labels using natural language processing models](https://pubmed.ncbi.nlm.nih.gov/40179876/)

Example Queries:
-  What are the most common side effects of metformin?
-  Show me all cardiovascular adverse events associated with statins.
-  Which antidepressants have the fewest reported gastrointestinal side effects?
-  List all drugs that can cause severe liver toxicity.
-  Compare the adverse events of amoxicillin and azithromycin.
-  What pediatric-specific adverse events are reported for methylphenidate?
-  Which antipsychotics have the highest risk of weight gain?
""")

# Display chat history with tool invocations
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display tool invocations for this assistant message if they exist
        invocations = st.session_state.tool_invocations[idx] if idx < len(st.session_state.tool_invocations) else None
        if message["role"] == "assistant" and invocations:
            with st.expander("🔧 View Tool Invocations", expanded=False):
                for tool_invocation in invocations:
                    tool_name = tool_invocation.get("name", "Unknown Tool")
                    tool_args = tool_invocation.get("args", {})
                    tool_result = tool_invocation.get("result", "No result")
                    
                    # Display using Streamlit columns
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"🔧 **{tool_name}**")
                    with col2:
                        st.markdown("**Input:**")
                        st.json(tool_args)
                        st.markdown("**Output:**")
                        # Handle potential non-JSON results gracefully
                        try:
                            # Special handling for the specific output format with meta and content
                            if isinstance(tool_result, str) and "meta=" in tool_result and "content=[TextContent" in tool_result:
                                # Extract the actual text content from the formatted output
                                text_match = re.search(r"text='([^']*)'", tool_result)
                                if text_match:
                                    # Get the text and handle escaped newlines
                                    extracted_text = text_match.group(1)
                                    # Replace literal \n with actual newlines
                                    extracted_text = extracted_text.replace('\\n', '\n')
                                    
                                    # Check if this is SQL result format with rows
                                    if "row" in extracted_text and any(f"{i}. row" in extracted_text for i in range(1, 11)):
                                        # Display SQL results in a structured table
                                        # Split into rows
                                        rows = re.split(r'\d+\. row', extracted_text)
                                        if len(rows) > 1:
                                            rows = rows[1:]  # Skip the first split which is empty
                                            
                                            data_list = []
                                            for row in rows:
                                                row_data = {}
                                                for line in row.strip().split('\n'):
                                                    if ':' in line:
                                                        key, value = line.split(':', 1)
                                                        row_data[key.strip()] = value.strip()
                                                data_list.append(row_data)
                                            
                                            if data_list:
                                                # Create dataframe from all rows
                                                df = pd.DataFrame(data_list)
                                                st.dataframe(df)
                                        else:
                                            st.text(extracted_text)
                                    elif "," in extracted_text and len(extracted_text.split(",")) > 1:
                                        # Handle comma-separated lists
                                        items = [item.strip() for item in extracted_text.split(",")]
                                        for item in items:
                                            st.write(f"- {item}")
                                    else:
                                        # Display as is for other text
                                        st.write(extracted_text)
                                else:
                                    # Fallback if regex fails
                                    st.code(tool_result)
                            # Direct output format with \n characters 
                            elif isinstance(tool_result, str) and "\\n" in tool_result and any(f"{i}. row" in tool_result for i in range(1, 11)):
                                # This looks like SQL results with escaped newlines
                                # Replace escaped newlines
                                clean_text = tool_result.replace('\\n', '\n')
                                
                                # Split into rows
                                rows = re.split(r'\d+\. row', clean_text)
                                if len(rows) > 1:
                                    rows = rows[1:]  # Skip the first split which is empty
                                    
                                    data_list = []
                                    for row in rows:
                                        row_data = {}
                                        for line in row.strip().split('\n'):
                                            if ':' in line:
                                                key, value = line.split(':', 1)
                                                row_data[key.strip()] = value.strip()
                                        data_list.append(row_data)
                                    
                                    if data_list:
                                        # Create dataframe from all rows
                                        df = pd.DataFrame(data_list)
                                        st.dataframe(df)
                                else:
                                    st.text(clean_text)
                            # Existing logic for other formats
                            elif isinstance(tool_result, str):
                                try:
                                    parsed_result = json.loads(tool_result)
                                    st.json(parsed_result)
                                except json.JSONDecodeError:
                                    # Check if this is SQL-like result format
                                    if "row" in tool_result and (": " in tool_result or ":" in tool_result):
                                        # Display in a more structured format for SQL results
                                        rows = tool_result.split("\n\n")
                                        for row in rows:
                                            if row.strip() and not row.startswith("Result:"):
                                                lines = row.split("\n")
                                                st.markdown(f"**{lines[0]}**")
                                                # Create a dataframe for the row data
                                                data = {}
                                                for line in lines[1:]:
                                                    if ":" in line:
                                                        key, value = line.split(":", 1)
                                                        data[key.strip()] = value.strip()
                                                if data:
                                                    st.table(data)
                                    else:
                                        # If not JSON string, display as preformatted text/markdown
                                        st.markdown(f"```\n{tool_result}\n```")
                                except Exception as json_ex:
                                    logger.error(f"Error displaying tool result: {json_ex}")
                                    st.text(str(tool_result)) # Fallback to text
                            else:
                                st.json(tool_result) # If already object/dict, display as JSON
                        except Exception as json_ex:
                            logger.error(f"Error displaying tool result: {json_ex}")
                            st.text(str(tool_result)) # Fallback to text
                    st.markdown("---")

# Process new messages
if prompt := st.chat_input("What would you like to know?"):
    # Display and store user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.tool_invocations.append(None)  # Add placeholder for user message tools
    
    # Process with agent and display response
    with st.chat_message("assistant"):
        with st.spinner(""):
            try:
                # Convert chat history to ChatMessage objects
                chat_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        chat_messages.append(ChatMessage.from_user(msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_messages.append(ChatMessage.from_assistant(msg["content"]))
                
                # Remove the last message (current prompt) that we just added
                chat_messages = chat_messages[:-1]
                
                # Add the current prompt
                chat_messages.append(ChatMessage.from_user(prompt))
                
                # Run agent pipeline with full chat history
                result = pipeline.run({"agent": {"messages": chat_messages}})
                
                # Extract response from assistant message
                messages = result.get("agent", {}).get("messages", [])
                
                # Extract tool invocations
                current_tool_invocations = []
                for msg in messages:
                    if hasattr(msg, 'role') and msg.role == "tool" and hasattr(msg, 'tool_call_results'):
                        for tool_call_result in msg.tool_call_results:
                            tool_origin = tool_call_result.origin  # This is the original ToolCall
                            if tool_origin:
                                current_tool_invocations.append({
                                    "name": tool_origin.tool_name,
                                    "args": tool_origin.arguments,
                                    "result": tool_call_result.result  # The actual string/object result
                                })
                
                # Find assistant response
                response_text = ""
                for msg in reversed(messages):
                    if hasattr(msg, 'role') and msg.role == "assistant":
                        response_text = msg.text
                        break
                
                # Display and store response
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.session_state.tool_invocations.append(current_tool_invocations)
                
                # Display current tool invocations
                if current_tool_invocations:
                    with st.expander("🔧 View Tool Invocations", expanded=False):
                        for tool_invocation in current_tool_invocations:
                            tool_name = tool_invocation.get("name", "Unknown Tool")
                            tool_args = tool_invocation.get("args", {})
                            tool_result = tool_invocation.get("result", "No result")
                            
                            # Display using Streamlit columns
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.markdown(f"🔧 **{tool_name}**")
                            with col2:
                                st.markdown("**Input:**")
                                st.json(tool_args)
                                st.markdown("**Output:**")
                                # Handle potential non-JSON results gracefully
                                try:
                                    # Special handling for the specific output format with meta and content
                                    if isinstance(tool_result, str) and "meta=" in tool_result and "content=[TextContent" in tool_result:
                                        # Extract the actual text content from the formatted output
                                        text_match = re.search(r"text='([^']*)'", tool_result)
                                        if text_match:
                                            # Get the text and handle escaped newlines
                                            extracted_text = text_match.group(1)
                                            # Replace literal \n with actual newlines
                                            extracted_text = extracted_text.replace('\\n', '\n')
                                            
                                            # Check if this is SQL result format with rows
                                            if "row" in extracted_text and any(f"{i}. row" in extracted_text for i in range(1, 11)):
                                                # Display SQL results in a structured table
                                                # Split into rows
                                                rows = re.split(r'\d+\. row', extracted_text)
                                                if len(rows) > 1:
                                                    rows = rows[1:]  # Skip the first split which is empty
                                                    
                                                    data_list = []
                                                    for row in rows:
                                                        row_data = {}
                                                        for line in row.strip().split('\n'):
                                                            if ':' in line:
                                                                key, value = line.split(':', 1)
                                                                row_data[key.strip()] = value.strip()
                                                        data_list.append(row_data)
                                                    
                                                    if data_list:
                                                        # Create dataframe from all rows
                                                        df = pd.DataFrame(data_list)
                                                        st.dataframe(df)
                                                else:
                                                    st.text(extracted_text)
                                            elif "," in extracted_text and len(extracted_text.split(",")) > 1:
                                                # Handle comma-separated lists
                                                items = [item.strip() for item in extracted_text.split(",")]
                                                for item in items:
                                                    st.write(f"- {item}")
                                            else:
                                                # Display as is for other text
                                                st.write(extracted_text)
                                        else:
                                            # Fallback if regex fails
                                            st.code(tool_result)
                                    # Direct output format with \n characters 
                                    elif isinstance(tool_result, str) and "\\n" in tool_result and any(f"{i}. row" in tool_result for i in range(1, 11)):
                                        # This looks like SQL results with escaped newlines
                                        # Replace escaped newlines
                                        clean_text = tool_result.replace('\\n', '\n')
                                        
                                        # Split into rows
                                        rows = re.split(r'\d+\. row', clean_text)
                                        if len(rows) > 1:
                                            rows = rows[1:]  # Skip the first split which is empty
                                            
                                            data_list = []
                                            for row in rows:
                                                row_data = {}
                                                for line in row.strip().split('\n'):
                                                    if ':' in line:
                                                        key, value = line.split(':', 1)
                                                        row_data[key.strip()] = value.strip()
                                                data_list.append(row_data)
                                            
                                            if data_list:
                                                # Create dataframe from all rows
                                                df = pd.DataFrame(data_list)
                                                st.dataframe(df)
                                        else:
                                            st.text(clean_text)
                                    # Existing logic for other formats
                                    elif isinstance(tool_result, str):
                                        try:
                                            parsed_result = json.loads(tool_result)
                                            st.json(parsed_result)
                                        except json.JSONDecodeError:
                                            # Check if this is SQL-like result format
                                            if "row" in tool_result and (": " in tool_result or ":" in tool_result):
                                                # Display in a more structured format for SQL results
                                                rows = tool_result.split("\n\n")
                                                for row in rows:
                                                    if row.strip() and not row.startswith("Result:"):
                                                        lines = row.split("\n")
                                                        st.markdown(f"**{lines[0]}**")
                                                        # Create a dataframe for the row data
                                                        data = {}
                                                        for line in lines[1:]:
                                                            if ":" in line:
                                                                key, value = line.split(":", 1)
                                                                data[key.strip()] = value.strip()
                                                        if data:
                                                            st.table(data)
                                            else:
                                                # If not JSON string, display as preformatted text/markdown
                                                st.markdown(f"```\n{tool_result}\n```")
                                        except Exception as json_ex:
                                            logger.error(f"Error displaying tool result: {json_ex}")
                                            st.text(str(tool_result)) # Fallback to text
                                    else:
                                        st.json(tool_result) # If already object/dict, display as JSON
                                except Exception as json_ex:
                                    logger.error(f"Error displaying tool result: {json_ex}")
                                    st.text(str(tool_result)) # Fallback to text
                            st.markdown("---")
                            
            except Exception as e:
                error_msg = f"Error processing your request: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.session_state.tool_invocations.append([])  # Add empty tools for error message 