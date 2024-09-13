#!/bin/bash

# projects=("Math" "Csv" "Codec" "Gson" "JacksonCore" "JacksonXml" "Mockito" "Time" "Lang" "Cli" "Compress" "Jsoup")
# bug_counts=(106 16 18 18 26 6 38 27 65 40 47 93)

projects=("Math" "Csv" "Codec" "Gson" "JacksonCore" "JacksonXml" "Mockito" "Time" "Lang" "Cli" "Compress" "Jsoup")
bug_counts=(106 16 18 18 26 6 38 27 65 40 47 93)


technique="sbfl"
model="gpt4omini"

# Loop through each project index
for i in "${!projects[@]}"
do
    project=${projects[$i]}
    bug_count=${bug_counts[$i]}
    echo "Processing project: $project with bug count up to $bug_count"

    # Initialize the project-specific text file
    output_file="data/RunningTime/SBFL/fl_wo_promptchain/${project}_time_report.txt"
    echo "Project: $project" > $output_file
    
    total_time=0

    # Run the scripts for versions 1 through the bug count for the current project
    for version in $(seq 1 $bug_count)
    do
        start_time=$(date +%s)
        
        # python candidate_selection.py ${project} ${version} ${technique} ${model}
        # python fault_localization.py ${project} ${version} ${technique} ${model}
        python fault_localization_wo_promptchain.py ${project} ${version} ${technique} ${model}
        # python candidate_selection.py ${project} ${version} ${technique} ${model}
        # python fault_localization.py ${project} ${version} ${technique} ${model}
        
        end_time=$(date +%s)
        time_taken=$((end_time - start_time))
        
        # Add the time taken for this version to the total
        total_time=$((total_time + time_taken))
        
        # Record the time taken in the text file
        echo "Bug $version: $time_taken seconds" >> $output_file
    done
    
    # Record the total time for the project
    echo "Total time: $total_time seconds" >> $output_file
done

echo "All projects processed. Time reports saved to text files."
