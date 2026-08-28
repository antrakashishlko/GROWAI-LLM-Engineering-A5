 # GROWAI LLM Engineering - Assignment 5

## Multi-Step Content Pipeline

A multi-step content generation project that builds an Explanation → Quiz → Answer Key pipeline using LangChain and LlamaIndex with Ollama and the Qwen3 0.6B local LLM.

## Features

- Local LLM integration using Ollama
- Explanation generation for a 10-year-old
- Automatic generation of 5 quiz questions
- Automatic answer key generation
- Sequential multi-step pipeline using LangChain
- Same pipeline implemented using LlamaIndex
- LangChain vs LlamaIndex framework comparison
- Reusable pipeline function
- Live output demonstration

## Technologies Used

- Python
- LangChain
- LlamaIndex
- Ollama
- Qwen3 0.6B
- LangChain Core

## Requirements

Install the required Python packages:
```text
pip install -r requirements.txt
```

Make sure Ollama is installed and the Qwen3 0.6B model is available locally:
```text
ollama pull qwen3:0.6b
```

## Setup / Installation

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the required dependencies using <mark>requirements.txt</mark>.
4. Make sure Ollama is running.
5. Make sure the "qwen3:0.6b" model is available locally.

## How to Run

Run the Python file using:
```text
python multi_step_pipeline.py
```

The program generates:

1. A simple explanation of the topic
2. Five quiz questions based on the explanation
3. An answer key for the quiz

It then runs the same workflow using LlamaIndex and displays a framework comparison.

## Project Files

- `pipeline.py` — Main Python implementation containing the LangChain and LlamaIndex pipelines.
- `requirements.txt` — Required Python dependencies.
- `.gitignore` — Files and folders excluded from Git tracking.

## LangChain Pipeline

The LangChain implementation uses three connected chains:

## Topic → Explanation → Quiz Questions → Answer Key

Each step uses the output of the previous step as its input.

## LlamaIndex Pipeline

The same three-step workflow is rebuilt using LlamaIndex and the same local Qwen3 0.6B model for comparison.

## Framework Comparison

## LangChain

- Easy to compose sequential steps using the pipe operator.
- Provides clear separation between prompts, models and output parsers.
- Well suited for structured multi-step LLM workflows.

## LlamaIndex

- Provides a simple interface for LLM-based applications.
- Particularly useful for document, data and retrieval-oriented applications.
- Well suited for data-centric and RAG-based workflows.

## Real-World Relevance

This pipeline can be used in educational applications to automatically generate learning material. For example, a teacher can provide a topic and the system can generate a child-friendly explanation, create a quiz from it and produce an answer key.

The same multi-step pattern can also be applied to content generation, document processing, and LLM-based automation.

## Edge Case / Failure Point

An empty, invalid or unclear topic may result in an irrelevant explanation. Since each stage depends on the previous output, this can affect the quiz and answer key as well.

In a production application, input validation and output validation can be added before passing results between pipeline stages.

## Assignment
GROWAI LLM Engineering & Generative AI - Assignment 5
