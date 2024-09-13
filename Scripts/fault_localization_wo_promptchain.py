# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import tool
from langchain_community.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.agent import AgentFinish
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser
import json
import sys
# from langchain.agents import AgentType, initialize_agent, load_tools
import os
# from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import AIMessage, HumanMessage
import pdb
from langchain.agents.agent_toolkits.conversational_retrieval.tool import (
    create_retriever_tool,
)
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.agents import AgentActionMessageLog, AgentFinish
import re

MEMORY_KEY = "chat_history"

chat_history = []


@tool
def get_test_body_stacktrace() -> list:
    """Returns the test body and stack trace"""
    test_body = data['test_body']
    stacktrace = data['stack_trace']
    return [test_body, stacktrace]

@tool
def get_covered_methods_by_failedTest() -> list:
    """Returns the covered methods by a failed test."""
    covered_methods = data['covered_methods']
    # return covered_methods[:100]
    return covered_methods



def extract_json_info(text):
    # Define the regular expression pattern to match the start and end of a JSON object
    pattern = r'\{[\s\S]*?\}'

    # Initialize an empty list to hold all found JSON objects
    json_objects = []

    # Find all matches of the pattern in the text
    for match in re.finditer(pattern, text):
        # Extract the matched JSON-like string
        json_str = match.group()

        # Attempt to convert the matched string into a Python dictionary
        try:
            data = json.loads(json_str)
            json_objects.append(data)
        except json.JSONDecodeError:
            # If it's not valid JSON, continue to the next match
            continue

    # Check if any valid JSON objects were found
    if json_objects:
        return json_objects
    else:
        return "No valid JSON-like structure found."

def is_duplicate(new_obj, existing_objects):
    # Check if new_obj is in existing_objects based on 'test_id' and 'method_signatures'
    for obj in existing_objects:
        if obj['test_id'] == new_obj['test_id'] and set(obj['method_signatures']) == set(new_obj['method_signatures']):
            return True
    return False

def count_files_in_directory(directory_path):
    file_count = 0
    for root, dirs, files in os.walk(directory_path):
        file_count += len(files)
    return file_count

def parse_and_save_methodsig_json(contents, project_name, bug_id, path):
    # Initialize a list to store JSON objects found
    json_objects = []

    # Loop through each content string in the contents list
    # Extract JSON strings from code blocks marked with triple backticks specifically for JSON
    code_blocks = re.findall(r'```json\n([\s\S]*?)\n```', contents)
    for block in code_blocks:
        try:
            json_obj = json.loads(block)
            # Check for duplicates before appending
            if not is_duplicate(json_obj, json_objects):
                json_objects.append(json_obj)
        except json.JSONDecodeError:
            continue  # Skip blocks that cannot be parsed as JSON

    # Prepare the final JSON structure
    # If there's only one object, use it directly; otherwise, use the whole list or None if it's empty
    final_json = json_objects[0] if len(json_objects) == 1 else (json_objects if json_objects else None)

    # Define the output file path
    file_path = path

    # Create the directory if it doesn't exist
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    # Write the combined data to the file
    with open(file_path, "w") as json_file:
        json.dump(final_json, json_file, indent=4)
    
    print(f"Data saved to {file_path}")
    return file_path

def parse_and_save_finalansjson(contents, project_name, bug_id, test_id, path):
    json_objects = []
    # Updated regex to match JSON objects or arrays enclosed in triple backticks
    json_block_pattern = re.compile(r'```json\n\{[\s\S]*?\}\n```|```json\n\[[\s\S]*?\]\n```', re.MULTILINE)

    code_blocks = json_block_pattern.findall(contents)
    for block in code_blocks:
        try:
            # Clean up the block to remove markdown code block syntax
            clean_block = block.replace('```json\n', '').replace('\n```', '').strip()
            # Normalize JSON structure (ensure proper JSON formatting)
            clean_block = re.sub(r'^\{\n\s*\{', '{', clean_block)
            clean_block = re.sub(r'\}\n\s*\}$', '}', clean_block)
            clean_block = re.sub(r'([\{\s,])(\w+)(:)', r'\1"\2"\3', clean_block)
            # Parse the JSON data
            json_obj = json.loads(clean_block)
            if isinstance(json_obj, dict):
                json_objects.append(json_obj)  # Append single object
            elif isinstance(json_obj, list):
                json_objects.extend(json_obj)  # Extend list of objects
        except json.JSONDecodeError as e:
            print("JSON decode error:", e, "in block:", block)
            continue

    # Final JSON structure
    final_json = {
        "project_name": project_name,
        "bug_id": bug_id,
        "test_id": test_id,
        "ans": json_objects,
        "final_full_answer": contents
    }

    # File and directory handling
    file_path = path
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    # Write JSON data to file
    with open(file_path, "w") as json_file:
        json.dump(final_json, json_file, indent=4)
    
    print(f"Data saved to {file_path}")
    return file_path

def parse_and_save_methodsig_json_2(contents, project_name, bug_id, test_id, path):
    json_objects = []
    code_blocks = re.findall(r'```json\n([\s\S]*?)\n```', contents)
    
    for block in code_blocks:
        try:
            json_obj = json.loads(block)
            if not is_duplicate(json_obj, json_objects):
                json_objects.append(json_obj)
        except json.JSONDecodeError:
            continue  # Skip blocks that cannot be parsed as JSON

    # Handle the case where no valid JSON objects are found
    if json_objects:
        # If there's only one object, use it directly; otherwise, use the whole list
        final_json = json_objects[0] if len(json_objects) == 1 else json_objects
    else:
        # Initialize as an empty dict or with default structure when no data is found
        final_json = {
            "project_name": project_name,
            "bug_id": bug_id,
            "test_id": test_id,
            "method_signatures": [],
            "final_ans": contents
        }

    # Assign additional properties to the final_json
    if isinstance(final_json, dict):
        final_json['project_name'] = project_name
        final_json['bug_id'] = bug_id
        final_json['test_id'] = test_id
        final_json['final_ans'] = contents

    # Define the output file path
    file_path = path

    # Create the directory if it doesn't exist
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    # Write the combined data to the file
    with open(file_path, "w") as json_file:
        json.dump(final_json, json_file, indent=4)
    
    print(f"Data saved to {file_path}")
    return file_path



def save_raw_output(output, file_path):
    # Create the directory if it doesn't exist
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(output, f, indent=4)

def condense_prompt(prompt: ChatPromptValue) -> ChatPromptValue:
    '''
    This function is used to condense the prompt to a maximum of 4,000 tokens.
    '''
    messages = prompt.to_messages()
    num_tokens = llm.get_num_tokens_from_messages(messages)
    ai_function_messages = messages[2:]
    while num_tokens > 4000:
        ai_function_messages = ai_function_messages[2:]
        num_tokens = llm.get_num_tokens_from_messages(
            messages[:2] + ai_function_messages
        )
    messages = messages[:2] + ai_function_messages
    return ChatPromptValue(messages=messages)



project_name = sys.argv[1]
bug_id = sys.argv[2]
tech = sys.argv[3]
model_name = sys.argv[4]

# llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
# llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# tools = [get_covered_method_by_failedTest, get_stacktrace, get_test_body, get_method_body]
tools = [get_test_body_stacktrace, get_covered_methods_by_failedTest]





# File path to your JSON file
file_path = f'/Users/user/Desktop/llmfl/llama-index-test/data/{project_name}/processed_by_{tech.lower()}_withoutline/{bug_id}/'
total_tests = count_files_in_directory(file_path)

for test_id in range(total_tests):
    original_data_path = f'/Users/user/Desktop/llmfl/llama-index-test/data/{project_name}/processed_by_{tech.lower()}_withoutline/{bug_id}/test_{test_id}.json'
    # Load JSON data
    with open(original_data_path, 'r') as file:
        data = json.load(file)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
            """
                You will act as two agents, Debugger and Tester. You will be presented with a Java project's failing test, and tools (functions) to access the source code of the system under test (SUT). Your task is list all the suspicious methods which containts the fault by ranking them from most to least suspicious. You will be given 4 chances to interact with functions to gather relevant information.
                """, 
            ),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )


    llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])


    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        # | condense_prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()

        # | output_parser

    )


    user_input = """
    ```json
    [
        {
            "method_id": int  // The most suspicious method's id
            "reasoning": string  // The analysis of the method about what it does and the reason for the method being suspicious
            "rank": int   // The rank of suspiciousness
        }
    ]
    ```
    """


    intermediate_steps = []

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


    result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
    fault_result_wo_pormptchain = f"/Users/user/Desktop/llmfl/llama-index-test/SingleAgent/data/{model_name}/{project_name}/{tech}/FaultLocalization_wo_pormptchain/{project_name}_{bug_id}/test_{test_id}.json"
    raw_output_path = f"/Users/user/Desktop/llmfl/llama-index-test/SingleAgent/data//{model_name}/{project_name}/{tech}/Raw_outputs/FaultLocalization_wo_pormptchain/{project_name}_{bug_id}/test_{test_id}.json"
    save_raw_output(result['output'], raw_output_path)

    # parse_and_save_methodsig_json_2(result['output'], project_name, bug_id, test_id, candidates_path)
    parse_and_save_finalansjson(result['output'], project_name, bug_id, test_id, fault_result_wo_pormptchain)