#!/bin/bash

# Arrays for projects and bug counts
projects=("Math" "Csv" "Codec" "Gson" "JacksonCore" "JacksonXml" "Mockito" "Time" "Lang" "Cli" "Compress" "Jsoup")
bug_counts=(106 16 18 18 26 6 38 27 65 40 47 93)


# projects=("Math" "Csv" "Codec" "Gson" "JacksonCore" "JacksonXml" "Mockito" "Time" "Lang" "Cli" "Compress" "Jsoup")
# bug_counts=(106 16 18 18 26 6 38 27 65 40 47 93)


technique="random"
model="gpt4omini"

# Loop through each project index
for i in "${!projects[@]}"
do
    project=${projects[$i]}
    bug_count=${bug_counts[$i]}
    echo "Processing project: $project with bug count up to $bug_count"

    # Run the scripts for versions 1 through the bug count for the current project
    for version in $(seq 1 $bug_count)
    do
        python candidate_selection.py ${project} ${version} ${technique} ${model}
        python fault_localization.py ${project} ${version} ${technique} ${model}
        # python fault_localization_fix.py ${project} ${version} ${technique} ${model}
        # python fault_localization_wo_promptchain.py ${project} ${version} ${technique} ${model}
    done
done

echo "All projects processed."
