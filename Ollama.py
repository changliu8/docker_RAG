from langchain_community.document_loaders import PyPDFLoader

# splitting & embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# vector db
from langchain.vectorstores import FAISS

from langchain_ollama.llms import OllamaLLM


import os
import re
import sys


pdf_folder_path = "content/"

pdf_loaders = []
for file in os.listdir(pdf_folder_path):
    if file.endswith(".pdf"):
        print("Reading file: ", file)
        pdf_loaders.append(PyPDFLoader(os.path.join(pdf_folder_path, file)))
    
#loading all pdf, and convert the context to documents.
def load_pdf(loaders):
    full_documents = []
    for loader in loaders:
        print("Converting file: ", loader.file_path)
        documents = loader.load()
        full_documents.extend(documents)
    return full_documents


#convert the documents to text , i.e. string
def convert_to_text(documents):
    full_text = ""
    for document in documents:
        if len(document.page_content) > 20:
            full_text += document.page_content
    return full_text
    


full_documents = load_pdf(pdf_loaders)
print("The total page of the textbooks are : ", len(full_documents))
full_text = convert_to_text(full_documents)
print("The total number of words are : ", len(full_text))


#remove extra spaces, such that multiple white space.
def remove_extra_spaces(text):
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

#reformat them to sentences.
def clean_text(text):
    cleaned_lines = []
    lines = text.split("\n")
    for line in lines:
        line = remove_extra_spaces(line)
        cleaned_lines.append(line)
    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text

clean_texted_text = clean_text(full_text)
print("The total number of words after cleaning are : ", len(clean_texted_text))


#write out
file_name = "cleaned_text.txt"
with open(file_name, 'w', encoding='utf-8') as f:
    f.write(clean_texted_text)


#save it
with open('cleaned_text.txt', 'r', encoding='utf-8') as f:
    clean_texted_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100
)

chunks = text_splitter.split_text(clean_texted_text)
print("The total number of chunks are : ", len(chunks))

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = FAISS.from_texts(
    texts = [chunk for chunk in chunks],
    embedding = embeddings,
)

vector_db.save_local("faiss_index")
print("The vector db is saved in the faiss_index folder")

#retrieve the context.
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

