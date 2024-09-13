import sys
import json

def calculate_top_results_with_missed_bugs(merged_data):
    results = {
        "top_1": 0,
        "top_3": 0,
        "top_5": 0,
        "top_10": 0
    }
    missed_top_1_bugs = []
    missed_top_3_bugs = []
    missed_top_5_bugs = []
    missed_top_10_bugs = []
    found_top_1_bugs = []
    found_top_3_bugs = []

    for bug_id, tests in merged_data['bugs'].items():
        found_top_1 = False
        found_top_3 = False
        found_top_5 = False
        found_top_10 = False
        
        for test_id, test_data in tests.items():
            method_signatures = test_data['method_signatures']
            ground_truth_methods = set(test_data['d4j_groundtruth'])
            
            # Calculate presence of ground truth methods in top 1, 3, and 5
            top_1_methods = method_signatures[:1]
            top_3_methods = method_signatures[:3]
            top_5_methods = method_signatures[:5]

            if ground_truth_methods.intersection(top_1_methods):
                found_top_1 = True
            if ground_truth_methods.intersection(top_3_methods):
                found_top_3 = True
            if ground_truth_methods.intersection(top_5_methods):
                found_top_5 = True
            if ground_truth_methods.intersection(method_signatures[:10]):
                found_top_10 = True

        if found_top_1:
            # print(bug_id)
            results['top_1'] += 1
            found_top_1_bugs.append(bug_id)
        else:
            missed_top_1_bugs.append(bug_id)

        if found_top_3:
            # print(bug_id)
            results['top_3'] += 1
            found_top_3_bugs.append(bug_id)
        else:
            missed_top_3_bugs.append(bug_id)

        if found_top_5:
            # print(bug_id)
            results['top_5'] += 1
        else:
            missed_top_5_bugs.append(bug_id)

        if found_top_10:
            results['top_10'] += 1
        else:
            missed_top_10_bugs.append(bug_id)

    return results, missed_top_1_bugs, missed_top_3_bugs, missed_top_5_bugs, missed_top_10_bugs, found_top_1_bugs, found_top_3_bugs

project_name = sys.argv[1]
technique = sys.argv[2]

model = sys.argv[3]
rank_phase = sys.argv[4]
print(rank_phase)
if rank_phase == 'FaultLocalization':
    json_file_path = f'data/{model}/{project_name}/{technique}/Ranking/{project_name}_merged_output_{technique.lower()}.json'
else:
    json_file_path = f'data/{model}/{project_name}/{technique}/Ranking/{project_name}_merged_{rank_phase}_{technique.lower()}.json'
# Load merged data from the JSON file (assuming it has already been saved)
with open(json_file_path, 'r') as f:
    merged_data = json.load(f)

# Calculate the results
results, missed_top_1_bugs, missed_top_3_bugs, missed_top_5_bugs, missed_top_10_bugs, found_top_1_bugs, found_top_3_bugs = calculate_top_results_with_missed_bugs(merged_data)


# Output results
print("Top-1 Accuracy:", results['top_1'])
print("Top-3 Accuracy:", results['top_3'])
print("Top-5 Accuracy:", results['top_5'])
print("Top-10 Accuracy:", results['top_10'])
