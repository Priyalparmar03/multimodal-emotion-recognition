from rag.retriever import get_context_for_emotion, generate_response

def test_retriever_returns_results():
    context = get_context_for_emotion("sad")
    assert isinstance(context, list)
    assert len(context) > 0

def test_generate_response_contains_emotion():
    response = generate_response("angry")
    assert "angry" in response.lower()