from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="meta/llama2-70b")  # any placeholder
print([m.id for m in llm.available_models][:50])