from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
import sys


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_path = "faiss_index"

vector_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

def retrive_context(query, k=3, score_threshold=0.8):
    retrieved_context = vector_db.similarity_search(
        query,
        k = k,
        score_threshold = score_threshold
    )
    return retrieved_context


llm = OllamaLLM(model='llama3')

def answer_question(question, context, llm):
    # Reformat context
    formatted_context = "\n".join([doc.page_content for doc in context])

    # Prompt Template
    prompt = f"""
    You are an expert research assistant specializing in answering questions about research papers.

    Task: Answer the question based on the provided context, with detail explaination and reasoning.

    Instructions:
    * Be concise and accurate.
    * If the context does not contain the answer, say EXACTLY "I cannot answer confidently"
    * If the question is unrelated to the context, say EXACTLY "NA"
    * If the question asks for a yes/no answer, provide it and explain your reasoning shortly.

    Context:
    {formatted_context}

    Question:
    {question}

    Answer:
    """

    # Generate answer using the LLM
    try:
        response = llm.invoke(prompt)  # Use the llm object directly
        return response.strip() # Remove leading/trailing whitespace
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return "Error processing the request."

def response(query, k=10, score_threshold=0.8):
    retrieved_context = retrive_context(query, k=k, score_threshold=score_threshold)
    if not retrieved_context:
        return "No relevant context found."
    
    # Answer the question using the LLM
    response = answer_question(query, retrieved_context, llm)
    return response


if __name__ == "__main__":
    query = sys.argv[1]
    print("Your Question is : ", query)
    answer = response(query, k=10, score_threshold=0.8)
    print("The answer is : ", answer)

