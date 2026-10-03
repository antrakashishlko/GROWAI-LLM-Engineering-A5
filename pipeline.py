"""
Multi-Step Content Pipeline
This project demonstrates a three-step content pipeline:
1. Generate a simple explanation of a topic.
2. Generate 5 quiz questions from that explanation.
3. Generate an answer key from the quiz.

The workflow is implemented using LangChain and LlamaIndex.
"""

# ============================================================
# IMPORTS
# ============================================================

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.llms.ollama import Ollama

# ============================================================
# CONFIGURATION
# ============================================================

TOPIC = "photosynthesis"

OLLAMA_URL = "http://localhost:11434"

LANGCHAIN_MODEL = "phi3:mini"
LLAMAINDEX_MODEL = "qwen3:0.6b"

# ============================================================
# LANGCHAIN PIPELINE
# ============================================================

print("=" * 60)
print("LANGCHAIN MULTI-STEP CONTENT PIPELINE")
print("=" * 60)

# Create LangChain Ollama model
model = ChatOllama(
    model=LANGCHAIN_MODEL,
    base_url=OLLAMA_URL,
    temperature=0,
    num_predict=350
)

print("\nLangChain Ollama model connected successfully!")

# ------------------------------------------------------------
# STEP 1 - Explanation
# ------------------------------------------------------------

explanation_prompt = ChatPromptTemplate.from_template(
    """
You are explaining a science topic to a 10-year-old child.

Topic: {topic}

Write approximately 200 words.

Rules:
- Use simple English.
- Use short and clear sentences.
- Explain the important facts accurately.
- Use a simple example.
- Do not use unnecessary technical terms.
- Do not include a title.
- Write between 180 and 210 words.
"""
)

explanation_chain = (
    explanation_prompt
    | model
    | StrOutputParser()
)

# ------------------------------------------------------------
# STEP 2 - Quiz Questions
# ------------------------------------------------------------

quiz_prompt = ChatPromptTemplate.from_template(
    """
Create exactly 5 quiz questions for a 10-year-old.

Use ONLY the information from the explanation below.

Explanation:
{explanation}

Rules:
- Number the questions from 1 to 5.
- Each question must have a clear answer in the explanation.
- Keep questions simple.
- Do not provide answers.
- Do not add extra questions.
"""
)

quiz_chain = (
    quiz_prompt
    | model
    | StrOutputParser()
)

# ------------------------------------------------------------
# STEP 3 - Answer Key
# ------------------------------------------------------------

answer_prompt = ChatPromptTemplate.from_template(
    """
Create the answer key for the quiz below.

Quiz:
{quiz}

Use the explanation below to verify the answers.

Explanation:
{explanation}

Rules:
- Give exactly 5 answers.
- Match answers with questions 1 to 5.
- Use only information supported by the explanation.
- Do not invent facts.
- Keep answers short.

Format:

1. Answer
2. Answer
3. Answer
4. Answer
5. Answer
"""
)

answer_chain = (
    answer_prompt
    | model
    | StrOutputParser()
)

# ------------------------------------------------------------
# Complete LangChain Pipeline
# ------------------------------------------------------------

def run_langchain_pipeline(topic):
    """
    Runs the three LangChain steps in sequence.

    Explanation -> Quiz -> Answer Key
    """

    # Edge case: empty topic
    if not topic or not topic.strip():
        raise ValueError("Topic cannot be empty.")

    # Step 1
    explanation = explanation_chain.invoke({
        "topic": topic
    })

    # Step 2
    quiz = quiz_chain.invoke({
        "explanation": explanation
    })

    # Step 3
    answer_key = answer_chain.invoke({
        "quiz": quiz,
        "explanation": explanation
    })

    return explanation, quiz, answer_key

# Run the pipeline once
explanation, quiz, answer_key = run_langchain_pipeline(TOPIC)

# Display LangChain results
print("\n--- EXPLANATION ---")
print(explanation)

print("\n--- QUIZ QUESTIONS ---")
print(quiz)

print("\n--- ANSWER KEY ---")
print(answer_key)

# ============================================================
# LLAMAINDEX PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("LLAMAINDEX MULTI-STEP CONTENT PIPELINE")
print("=" * 60)

# Create local Ollama model
llm = Ollama(
    model=LLAMAINDEX_MODEL,
    base_url=OLLAMA_URL,
    temperature=0,
    request_timeout=120.0,
    context_window=2048
)

# ------------------------------------------------------------
# Custom LlamaIndex Query Engine
# ------------------------------------------------------------

class LocalOllamaQueryEngine(CustomQueryEngine):
    """
    Custom LlamaIndex query engine that uses
    a local Ollama model.

    It avoids external embedding models and
    OpenAI API requirements.
    """

    llm: Ollama
    prompt: str

    def custom_query(self, query_str: str):
        """
        Combines the stored prompt with the query
        and sends it to the local Ollama model.
        """

        full_prompt = (
            self.prompt
            + "\n\n"
            + query_str
        )

        response = self.llm.complete(full_prompt)

        return response.text

# ------------------------------------------------------------
# STEP 1 - Explanation
# ------------------------------------------------------------

explanation_engine = LocalOllamaQueryEngine(
    llm=llm,
    prompt=f"""
Explain the topic "{TOPIC}" to a 10-year-old.

Write approximately 200 words.

Rules:
- Use simple English.
- Use short and clear sentences.
- Explain important facts accurately.
- Give a simple example.
- Do not use unnecessary technical words.
"""
)

explanation_response = explanation_engine.query(
    "Generate the explanation."
)

explanation_li = str(explanation_response)

# ------------------------------------------------------------
# STEP 2 - Quiz Questions
# ------------------------------------------------------------

quiz_engine = LocalOllamaQueryEngine(
    llm=llm,
    prompt=f"""
Create exactly 5 quiz questions for a 10-year-old.

Use ONLY the explanation below:

{explanation_li}

Rules:
- Number questions from 1 to 5.
- Each question must be answerable from the explanation.
- Keep questions simple and clear.
- Do not provide answers.
- Do not add extra questions.
"""
)

quiz_response = quiz_engine.query(
    "Generate exactly five quiz questions."
)

quiz_li = str(quiz_response)

# ------------------------------------------------------------
# STEP 3 - Answer Key
# ------------------------------------------------------------

answer_engine = LocalOllamaQueryEngine(
    llm=llm,
    prompt=f"""
Create the answer key for these quiz questions:

{quiz_li}

Use the explanation below to verify the answers:

{explanation_li}

Rules:
- Give exactly 5 answers.
- Match each answer with its question.
- Use only information supported by the explanation.
- Do not invent facts.
- Keep answers short.

Format:

1. Answer
2. Answer
3. Answer
4. Answer
5. Answer
"""
)

answer_response = answer_engine.query(
    "Generate the answer key."
)

answer_key_li = str(answer_response)

# ------------------------------------------------------------
# Display LlamaIndex Results
# ------------------------------------------------------------

print("\n--- EXPLANATION ---")
print(explanation_li)

print("\n--- QUIZ QUESTIONS ---")
print(quiz_li)

print("\n--- ANSWER KEY ---")
print(answer_key_li)

# ============================================================
# FRAMEWORK COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("LANGCHAIN vs LLAMAINDEX")
print("=" * 60)

print("\nLangChain:")
print("- Uses the | pipe operator to connect multiple steps.")
print("- Makes the data flow between steps easy to follow.")
print("- Provides direct control over prompts and output parsing.")
print("- Useful for multi-step LLM workflows and chains.")

print("\nLlamaIndex:")
print("- Provides query engine abstractions for LLM applications.")
print("- Can work with custom query logic and local LLMs.")
print("- Provides useful tools for documents, indexes and retrieval.")
print("- Useful for data-centric and RAG-oriented applications.")

print("\nConclusion:")
print(
    "LangChain provides a clear and direct way to compose "
    "multi-step chains, while LlamaIndex provides useful "
    "query-engine abstractions for LLM applications and "
    "data-oriented workflows."
)

print("\n" + "=" * 60)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)
