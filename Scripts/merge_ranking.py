import json
import os
from collections import defaultdict
import sys

def map_method_ids_to_signatures(processed_dir, bug_id, test_id):
    method_signatures_map = {}
    test_file_path = os.path.join(processed_dir, f"{bug_id}", f"test_{test_id}.json")
    # print(test_file_path)
    if os.path.exists(test_file_path):
        with open(test_file_path, 'r') as json_file:
            data = json.load(json_file)
        for method in data.get('covered_methods', []):
            method_signatures_map[method['method_id']] = method['method_signature']
    # print(method_signatures_map)
    return method_signatures_map

def integrate_additional_signatures(merged_data, methodsig_path, processed_dir):
    for root, dirs, files in os.walk(methodsig_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    bug_id = str(data['bug_id'])
                    test_id = str(data['test_id'])
                    if bug_id in merged_data and test_id in merged_data[bug_id]:
                        existing_ids = set(merged_data[bug_id][test_id]['method_ids'])
                        for method_id in data['method_ids']:
                            if method_id not in existing_ids:
                                merged_data[bug_id][test_id]['method_ids'].append(method_id)
                except Exception as e:
                    print(f"Error processing methodsig file {file_path}: {e}")



def merge_json_files(reasoning_path, txt_files_directory, project_name, methodsig_path, processed_dir):
    merged_data = defaultdict(lambda: defaultdict(lambda: {
        "method_ids": [],
        "method_signatures": [],
        "d4j_groundtruth": []
    }))

    # Integrate additional method IDs first
    integrate_additional_signatures(merged_data, methodsig_path, processed_dir)

    # Process each file in the reasoning path to add initial method IDs
    for root, dirs, files in os.walk(reasoning_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    bug_id = str(data['bug_id'])
                    test_id = str(data['test_id'])
                    for entry in data['ans']:
                        if entry['method_id'] not in merged_data[bug_id][test_id]['method_ids']:
                            merged_data[bug_id][test_id]['method_ids'].append(entry['method_id'])
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Map all method signatures after integrating all method IDs
    for bug_id in merged_data:
        for test_id in merged_data[bug_id]:
            method_signatures_map = map_method_ids_to_signatures(processed_dir, bug_id, test_id)
            for method_id in merged_data[bug_id][test_id]['method_ids']:
                method_signature = method_signatures_map.get(method_id, "Unknown Signature")
                merged_data[bug_id][test_id]['method_signatures'].append(method_signature)

    # Load ground truth signatures
    for bug_id, tests in merged_data.items():
        txt_file_path = os.path.join(txt_files_directory, f"{bug_id}.txt")
        if os.path.exists(txt_file_path):
            try:
                with open(txt_file_path, 'r') as txt_file:
                    ground_truth_signatures = [line.strip() for line in txt_file.readlines() if line.strip()]
                for test_id in tests:
                    merged_data[bug_id][test_id]['d4j_groundtruth'] = ground_truth_signatures
            except Exception as e:
                print(f"Error reading ground truth file {txt_file_path}: {e}")

    # Construct the final output
    final_output = {
        "project_name": project_name,
        "bugs": {}
    }
    for bug_id, tests in merged_data.items():
        final_output['bugs'][bug_id] = {}
        for test_id, data in tests.items():
            final_output['bugs'][bug_id][test_id] = {
                "method_ids": data['method_ids'],
                "method_signatures": data['method_signatures'],
                "d4j_groundtruth": data['d4j_groundtruth']
            }

    return final_output


# Example usage and file writing
project_name = sys.argv[1]
technique = sys.argv[2]
model = sys.argv[3]
# rank_phase = 'ReRank'
rank_phase = sys.argv[4]
reasoning_path = f'data/{model}/{project_name}/{technique}/{rank_phase}'
methodsig_path = f'data/{model}/{project_name}/{technique}/Candidates'
txt_files_directory = f'../data/BuggyMethods/{project_name}'
processed_dir = f'../data/{project_name}/processed_by_{technique.lower()}_withoutline'
result = merge_json_files(reasoning_path, txt_files_directory, project_name, methodsig_path, processed_dir)

# Write the result to a file, if the folder does not exist, create it
if not os.path.exists(f'data/{model}/{project_name}/{technique}/Ranking'):
    os.makedirs(f'data/{model}/{project_name}/{technique}/Ranking')
    
if rank_phase == 'FaultLocalization':
    with open(f'data/{model}/{project_name}/{technique}/Ranking/{project_name}_merged_output_{technique.lower()}.json', 'w') as f:
        json.dump(result, f, indent=4)

elif rank_phase == 'ReRank':
    with open(f'data/{model}/{project_name}/{technique}/Ranking/{project_name}_merged_{rank_phase}_{technique.lower()}.json', 'w') as f:
        json.dump(result, f, indent=4)

else:
    with open(f'data/{model}/{project_name}/{technique}/Ranking/{project_name}_merged_{rank_phase}_{technique.lower()}.json', 'w') as f:
        json.dump(result, f, indent=4)
