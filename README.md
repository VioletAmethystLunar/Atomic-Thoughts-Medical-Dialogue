# Experience is the Teacher: Reusing Atomic Thoughts from LLMs to Improve Medical Dialogue
## 🏗️ Framework Architecture
![Framework Architecture](Assets/framework_01.png)

## 👩‍⚕️ Introduction

We introduce a framework that decomposes medical reasoning into **Atomic Thoughts**, builds an **Action Library** from experience, and reuses these thoughts to generate high-quality, empathetic, and clinically accurate medical dialogues.

This repository contains the code and data processing pipeline for the paper **"Experience is the Teacher: Reusing Atomic Thoughts from LLMs to Improve Medical Dialogue"**.

## 📄 Paper

You can read the full paper here: [Experience is the Teacher: Reusing Atomic Thoughts from LLMs to Improve Medical Dialogue](Assets/Experience_is_the_Teacher__Reusing_Atomic_Thoughts_from_LLMs_to_Improve_Medical_Dialogue.pdf)

## 📂 Repository Structure

The project is organized as follows:

```text
Atomic-Thoughts-Medical-Dialogue/
├── .idea/                 # Project configuration
├── Annotate/              # Core processing pipeline
│   ├── data/              # Input raw data (e.g., ReMeDi dataset)
│   ├── library/           # Atomic Action Library storage
│   ├── results/           # Intermediate and final output files
│   ├── annotate.py        # Step 1: Intent & Atomic Action Annotation
│   ├── filter.py          # Step 2: Data Filtering
│   ├── sample.py          # Step 3: Data Sampling
│   ├── filtered_by_struct.py # Step 4: Struct Alignment
│   ├── generate.py        # Step 5: CoT Generation
│   └── refine.py          # Step 6: Response Refinement
├── Assets/                # Image
├── Eval/                  # Evaluation scripts (Metrics calculation)
├── Train/                 # Training scripts (LLaMA-Factory configs)
├── run.sh        # execution script
└── README.md
```

## 🛠️ Setup & Installation

### 1. Clone the repository

```bash
cd Atomic-Thoughts-Medical-Dialogue
```


### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

The pipeline requires access to an LLM API (e.g., Alibaba Cloud DashScope for Qwen models).
You can set your API key via environment variable:

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

> **Note:** The `run.sh` script handles this export for you if configured therein.

## 🚀 Running the Pipeline

We provide a comprehensive shell script `run.sh` to execute the entire data processing and generation pipeline sequentially.

### Quick Start

```bash
chmod +x run.sh
./run.sh
```

### Pipeline Stages Detail

The pipeline executes the following Python scripts in order:

1. **Annotation** (`annotate.py`)
* **Goal:** Annotate raw medical dialogues with structured thought chains.
* **Process:** Uses LLMs to identify **Patient Intent (Level 1)**, **Sub-intent (Level 2)**, and decomposes the doctor's reasoning into **Atomic Medical Actions (Level 3)**.
* **Output:** `results/medical_thought_chains_*.jsonl`


2. **Filtering** (`filter.py`)
* **Goal:** Ensure data quality by removing irrelevant or low-quality turns.
* **Process:** Filters out cases with specific excluded intents or non-clinical "chitchat" turns.
* **Output:** `results/filtered_struct_*.jsonl`


3. **Sampling** (`sample.py`)
* **Goal:** Balance the dataset and select representative dialogue samples.
* **Process:** Performs random sampling on the raw data to prepare for structure matching.
* **Output:** `results/ReMeDi_raw_sampled.jsonl`


4. **Structure Matching** (`filtered_by_struct.py`)
* **Goal:** Align sampled raw data with high-quality structural annotations.
* **Process:** Intersects the sampled dataset with the filtered structural dataset based on Case IDs.
* **Output:** `results/ReMeDi_filtered_by_struct_*.jsonl`


5. **CoT Generation** (`generate.py`)
* **Goal:** Generate explicit Chain-of-Thought (CoT) for doctor responses.
* **Process:** Using the annotated Atomic Actions, the model generates a `<think>` block (reasoning process).
* **Output:** `results/ReMeDi_thoughtchain_generated_*.jsonl`


6. **Refinement** (`refine.py`)
* **Goal:** Polish the final response for empathy and naturalness.
* **Process:** A simulator refines the doctor's reply based on the generated thoughts, removing redundancy while maintaining medical accuracy and a caring tone.
* **Output:** `results/ReMeDi_thoughtchain_refined_*.jsonl`



## 📊 Evaluation

The validation set of the ReMeDi dataset is publicly available in the Eval/directory.


