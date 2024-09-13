# **LLM4FL**

This repository contains all the data, scripts, and instructions needed to run and evaluate the **LLM4FL**.

---

## **Table of Contents**

- [System Requirements](#system-requirements)
- [Repository Structure](#repository-structure)
- [Running the Fault Localization Process](#running-the-fault-localization-process)
- [Merging Rankings](#merging-rankings)
- [Evaluating Top-k Metrics](#evaluating-top-k-metrics)
- [How to Cite](#how-to-cite)

---

## **System Requirements**

To use the scripts in this repository, make sure you have the following dependencies installed:

1. **LangChain v0.2**
2. **LangGraph v0.2.21**
3. **OpenAI API Key**

You can install these dependencies via `pip`:
```bash
pip install langchain==0.2 langgraph==0.2.21
```

Ensure that your OpenAI API key is properly configured. You can set it as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key"
```

---

## **Repository Structure**

- **data/**: Contains all the raw data required for the fault localization process.
- **scripts/**: Contains all the Python and Bash scripts used for running the experiments.
- **agent_runner_batch.sh**: Script to run the fault localization process in batch mode.
- **merge_ranking.py**: Python script to merge the ranking across multiple versions of a project.
- **fault_localization.py**: Python script to evaluate fault localization performance and calculate top-k metrics.

---

## **Running the Fault Localization Process**

To start the fault localization process, execute the following command:

```bash
bash agent_runner_batch.sh
```

This script runs the entire batch of fault localization processes, utilizing the relevant data and scripts within the repository.

---

## **Merging Rankings**

To merge rankings across all versions of a single project, use the following command:

```bash
python3 merge_ranking.py <ProjectName> <RankingMethod> <Model> <Component>
```

- `ProjectName`: Name of the project (e.g., `Lang`)
- `RankingMethod`: The ranking method used (e.g., `sbfl`)
- `Model`: The model used for fault localization (e.g., `gpt4omini`)
- `Component`: The component involved in fault localization (e.g., `FaultLocalization`)

Example:
```bash
python3 merge_ranking.py Lang sbfl gpt4omini FaultLocalization
```

---

## **Evaluating Top-k Metrics**

To compute top-k fault localization metrics, run the following command:

```bash
python3 fault_localization.py <ProjectName> <RankingMethod> <Model> <Component>
```

This script provides metrics on how well the fault localization process performed, such as identifying the top-k most suspicious components.

Example:
```bash
python3 fault_localization.py Lang sbfl gpt4omini FaultLocalization
```

---