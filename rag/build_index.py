from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

knowledge_base = [
    "If you are feeling sad, try talking to a friend or writing down your thoughts.",
    "Feeling angry is normal, take a few deep breaths before reacting.",
    "When you feel happy, it's a great time to note what caused it so you can repeat it.",
    "Feeling anxious? Try the 5-4-3-2-1 grounding technique.",
    "Sarcastic or humorous content is often used to cope with stress.",
    "Offensive content should be flagged and handled with care.",
    "Motivational content can be a great mood booster.",
    "Neutral content usually doesn't need a strong emotional response.",
]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docs = [Document(page_content=text) for text in knowledge_base]

vector_store = FAISS.from_documents(docs, embedding_model)
vector_store.save_local("rag/knowledge_base/faiss_index")
print("FAISS index built and saved")


