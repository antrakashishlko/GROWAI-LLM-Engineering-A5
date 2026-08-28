from langchain_ollama import ChatOllama

# Create the local Ollama model
model = ChatOllama(
    model="qwen3:0.6b",
    base_url="http://localhost:11434"
)

print("Ollama model connected successfully!")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Explanation prompt
explanation_prompt = ChatPromptTemplate.from_template(
"""
Explain the topic "{topic}" in about 200 words for a 10-year-old.
Use simple language, clear examples and avoid complicated technical terms.
"""
)

# Explanation chain
explanation_chain = explanation_prompt | model | StrOutputParser()

# Test the first step
topic = "photosynthesis"

explanation = explanation_chain.invoke({
    "topic": topic
})

print("\n" + "=" * 60)
print("EXPLANATION")
print("=" * 60)
print(explanation)

# Quiz prompt
quiz_prompt = ChatPromptTemplate.from_template(
    """
Based on the explanation below, create exactly 5 quiz questions
for a 10-year-old.

Explanation:
{explanation}

Number the questions from 1 to 5.
Do not provide the answers.
"""
)

# Quiz chain
quiz_chain = quiz_prompt | model | StrOutputParser()

# Generate quiz questions using the explanation
quiz = quiz_chain.invoke({
    "explanation": explanation
})

print("\n" + "=" * 60)
print("QUIZ QUESTIONS")
print("=" * 60)
print(quiz)

# Answer key prompt
answer_prompt = ChatPromptTemplate.from_template(
    """
Based on the quiz questions below, provide the correct answer
for each question.

Quiz:
{quiz}

Write the answers clearly as:
1. Answer
2. Answer
3. Answer
4. Answer
5. Answer
"""
)

# Answer key chain
answer_chain = answer_prompt | model | StrOutputParser()

# Generate answer key using the quiz
answer_key = answer_chain.invoke({
    "quiz": quiz
})

print("\n" + "=" * 60)
print("ANSWER KEY")
print("=" * 60)
print(answer_key)

# Complete multi-step pipeline

def run_pipeline(topic):
    explanation = explanation_chain.invoke({
        "topic": topic
    })

    quiz = quiz_chain.invoke({
        "explanation": explanation
    })

    answer_key = answer_chain.invoke({
        "quiz": quiz
    })

    return explanation, quiz, answer_key


# Run the complete pipeline
explanation, quiz, answer_key = run_pipeline("photosynthesis")

print("\n" + "=" * 60)
print("MULTI-STEP CONTENT PIPELINE")
print("=" * 60)

print("\n--- EXPLANATION ---")
print(explanation)

print("\n--- QUIZ QUESTIONS ---")
print(quiz)

print("\n--- ANSWER KEY ---")
print(answer_key)

# ==============================
# LlamaIndex Pipeline
# ==============================

from llama_index.llms.ollama import Ollama

llm = Ollama(
    model="qwen3:0.6b",
    base_url="http://localhost:11434",
    request_timeout=120.0
)

# Step 1: Generate explanation
explanation_response = llm.complete(
    """
Explain the topic "photosynthesis" in about 200 words
for a 10-year-old. Use simple language and examples.
"""
)

explanation_li = explanation_response.text

# Step 2: Generate quiz from explanation
quiz_response = llm.complete(
    f"""
Based on the explanation below, create exactly 5 quiz questions
for a 10-year-old. Do not provide the answers.

Explanation:
{explanation_li}
"""
)

quiz_li = quiz_response.text

# Step 3: Generate answer key from quiz
answer_response = llm.complete(
    f"""
Based on these quiz questions, provide the correct answer
for each question.

Quiz:
{quiz_li}

Format:
1. Answer
2. Answer
3. Answer
4. Answer
5. Answer
"""
)

answer_key_li = answer_response.text

print("\n" + "=" * 60)
print("LLAMAINDEX PIPELINE")
print("=" * 60)

print("\n--- EXPLANATION ---")
print(explanation_li)

print("\n--- QUIZ QUESTIONS ---")
print(quiz_li)

print("\n--- ANSWER KEY ---")
print(answer_key_li)

# ==============================
# Framework Comparison
# ==============================

print("\n" + "=" * 60)
print("LANGCHAIN vs LLAMAINDEX")
print("=" * 60)

print("\nLangChain:")
print("- Easy to compose multiple steps using the | pipe operator.")
print("- Clear separation of prompts, models and output parsers.")
print("- Good choice for multi-step LLM workflows and chains.")

print("\nLlamaIndex:")
print("- Provides a simple interface for calling LLMs and building LLM applications.")
print("- Particularly useful when working with documents, data and retrieval.")
print("- Good choice for data-centric and RAG-oriented applications.")