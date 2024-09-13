#!/bin/bash

# Set the project name
project="Cli"
technique="sbfl"
model="gpt4omini"

# Loop through the versions and checkout the buggy version of the project
for version in 2 34 33 35 3 19 21 17 28 10 11 16 20 27 18 39 1 37 36 31 38 22 25 13 14 15 12


# for version in 57

do
    # rm -rf data/${model}/${project}/${technique}/Candidates/${project}_${version}
    # python candidate_selection_split.py ${project} ${version} ${technique} ${model}
    # python candidate_selection.py ${project} ${version} ${technique} ${model}
    # python fault_localization.py ${project} ${version} ${technique} ${model}
    # python fault_localization_fix.py ${project} ${version} ${technique} ${model}
    python fault_localization_wo_promptchain.py ${project} ${version} ${technique} ${model}
done