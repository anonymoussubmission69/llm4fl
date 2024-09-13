# from langchain.chat_models import ChatOpenAI
from collections import defaultdict
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_community.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
# from langchain.schema.agent import AgentFinish
from langchain.agents import AgentExecutor
# from langchain.agents.output_parsers import ReActSingleInputOutputParser
# from langchain_core.pydantic_v1 import BaseModel, Field, validator
# from langchain.output_parsers import PydanticOutputParser
import json
import sys
# from langchain.agents import AgentType, initialize_agent, load_tools
import os
# from langchain.prompts import MessagesPlaceholder
# from langchain.schema.messages import AIMessage, HumanMessage
import pdb
from langchain.agents.agent_toolkits.conversational_retrieval.tool import (
    create_retriever_tool,
)
# from langchain.utils.openai_functions import convert_pydantic_to_openai_function
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompt_values import ChatPromptValue
# from langchain_core.agents import AgentActionMessageLog, AgentFinish
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

def is_duplicate(new_item, existing_items):
    return new_item in existing_items

def count_files_in_directory(directory_path):
    file_count = 0
    for root, dirs, files in os.walk(directory_path):
        file_count += len(files)
    return file_count

def parse_and_save_methodsig_json_2(contents, project_name, bug_id, test_id, path):
    method_ids = []
    code_blocks = re.findall(r'```json\n\{[\s\S]*?\}\n```', contents)  # Adjusted to capture full JSON blocks

    # Process each JSON block found
    for block in code_blocks:
        try:
            json_obj = json.loads(block.replace('```json\n', '').replace('\n```', ''))
            # Check if it has the expected structure and keys
            if isinstance(json_obj, dict) and 'method_ids' in json_obj:
                for method_id in json_obj['method_ids']:
                    if isinstance(method_id, int) and method_id not in method_ids:
                        method_ids.append(method_id)
            else:
                print("Invalid JSON object structure.")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e} in block: {block}")
            continue

    # Read existing data if the file already exists
    existing_data = None
    file_path = path
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            existing_data = json.load(json_file)

    if existing_data:
        # Merge new data into existing while preventing duplicates
        existing_method_ids = existing_data.get('method_ids', [])
        all_method_ids = existing_method_ids[:]
        for method_id in method_ids:
            if method_id not in all_method_ids:
                all_method_ids.append(method_id)
        existing_data['method_ids'] = all_method_ids
        final_json = existing_data
    else:
        # Prepare new data with unique method IDs
        final_json = {
            "test_id": test_id,
            "method_ids": method_ids,
            "project_name": project_name,
            "bug_id": bug_id,
            "final_ans": contents
        }

    # Ensure the directory exists before saving
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    # Write the combined or updated data to the file
    with open(file_path, "w") as json_file:
        json.dump(final_json, json_file, indent=4)
    
    print(f"Data saved to {file_path}")
    return file_path


def count_split_tests(folder_path):
    test_files = defaultdict(list)
    
    # Walk through all files in the specified folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                # Extract test name and its part index from the file name
                parts = file.split('_')
                if len(parts) > 2 and parts[-2].isdigit() and parts[-1].rstrip('.json').isdigit():
                    test_id = int(parts[-2])  # This assumes the format test_ID_PART.json
                    part_id = int(parts[-1].rstrip('.json'))
                    test_files[test_id].append(part_id)
    
    # Sort the part indices for each test
    for key in test_files:
        test_files[key].sort()

    return dict(test_files)

def save_raw_output(output, base_directory, test_id, split_id):
    # Define the file path
    file_path = os.path.join(base_directory)
    
    # Ensure the base directory exists
    os.makedirs(base_directory, exist_ok=True)

    # Structure to hold or update the file contents
    data_to_save = {
        "test_id": test_id,
        "raw_outputs": []
    }
    file_path = os.path.join(base_directory, f"test_{test_id}.json")
    # If file exists, read its content first
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data_to_save = json.load(file)

    # Append the new output with the split_id
    # Check if split_id already exists to avoid duplication
    existing_splits = {item['split_id'] for item in data_to_save['raw_outputs']}
    if split_id not in existing_splits:
        data_to_save['raw_outputs'].append({
            "split_id": split_id,
            "output": output
        })
    else:
        # Update the existing entry for this split_id
        for item in data_to_save['raw_outputs']:
            if item['split_id'] == split_id:
                item['output'] = output

    # Save the updated content back to the file
    with open(file_path, "w") as f:
        json.dump(data_to_save, f, indent=4)

    print(f"Output saved to {file_path}")

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
model = sys.argv[4]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# tools = [get_covered_method_by_failedTest, get_stacktrace, get_test_body, get_method_body]
tools = [get_test_body_stacktrace, get_covered_methods_by_failedTest]

# print(count_split_tests("/Users/user/Desktop/llmfl/llama-index-test/data/Time/processed_by_sbfl_withoutline_split/7"))



# File path to your JSON file
file_path = f'/Users/user/Desktop/llmfl/llama-index-test/data/{project_name}/processed_by_{tech.lower()}_withoutline_split/{bug_id}/'
total_tests = count_split_tests(file_path)
print(total_tests)

for test_id in total_tests.keys():
    for split_id in total_tests[test_id]:
        original_data_path = f'/Users/user/Desktop/llmfl/llama-index-test/data/{project_name}/processed_by_{tech.lower()}_withoutline_split/{bug_id}/test_{test_id}_{split_id}.json'
        # print(original_data_path)
        # pdb.set_trace()
        # Load JSON data
        with open(original_data_path, 'r') as file:
            data = json.load(file)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a Tester agent. You will be presented with a failing test, and tools (functions) to access the source code of the system under test (SUT). Your task is list all the suspicious methods which needs to be analyzed to find the fault. You will be given 4 chances to interact with functions to gather relevant information.
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


        if split_id == 0:
            user_input = """
            As a Tester agent, I want you to list the methods which might be suspicious to find the fault in the system under test. First analyze the test body and stack trace and then based on that look for the covered methods by the failed test which might be suspicious or leading to the fault.

            You Must conclude your analysis with a JSON object ranking these methods and summarizing your reasoning, following the specified structure: 
            ```json
            {
                    "method_ids": [1,2,3,4,5]  // The potential suspicious method's ids
            }
            ```
            """
        else:
            user_input = """
            As a Tester agent, I want you to list all the methods which might be suspicious to find the fault in the system under test. First analyze the test body and stack trace and then based on that look for two most suspicious covered methods by the failed test which might be suspicious or leading to the fault.

            You Must conclude your analysis with a JSON object ranking these methods and summarizing your reasoning, following the specified structure: 
            ```json
            {
                    "method_ids": [1]  // The potential suspicious method's ids
            }
            ```
            """

        intermediate_steps = []

        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


        result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
        candidates_path = f"/Users/user/Desktop/llmfl/llama-index-test/SingleAgent/data/{model}/{project_name}/{tech}/Candidates/{project_name}_{bug_id}/test_{test_id}.json"
        raw_output_path = f"/Users/user/Desktop/llmfl/llama-index-test/SingleAgent/data/{model}/{project_name}/{tech}/Raw_outputs/Candidates_split/{project_name}_{bug_id}"
        save_raw_output(result['output'], raw_output_path, test_id, split_id)

        parse_and_save_methodsig_json_2(result['output'], project_name, bug_id, test_id, candidates_path)