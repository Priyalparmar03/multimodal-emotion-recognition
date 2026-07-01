from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.load_local(
    "rag/knowledge_base/faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

def get_context_for_emotion(predicted_emotion):
    query = f"How to respond to someone feeling {predicted_emotion}"
    results = vector_store.similarity_search(query, k=2)  # no retriever object at all
    return [doc.page_content for doc in results]

def generate_response(predicted_emotion):
    context = get_context_for_emotion(predicted_emotion)
    response = f"Detected emotion: '{predicted_emotion}'\nSuggested response:\n"
    response += "\n".join(f"- {c}" for c in context)
    return response

if __name__ == "__main__":
    print(generate_response("sad"))