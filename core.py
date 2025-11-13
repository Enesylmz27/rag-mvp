# core.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# !!! DEĞİŞİKLİK BURADA !!!
# Eski (Hatalı): from langchain_ollama import Ollama
# Yeni (Doğru):
from langchain_ollama.chat_models import ChatOllama

DB_DIR = "db"


def get_retriever(k=5):
    """Chroma veritabanından retriever nesnesini döndürür."""
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vect_db = Chroma(persist_directory=DB_DIR, embedding_function=emb)
    return vect_db.as_retriever(search_kwargs={"k": k})


def get_llm():
    """Arka planda çalışan OLLAMA sunucusuna bağlanır."""

    # !!! DEĞİŞİKLİK BURADA !!!
    # Eski (Hatalı): return Ollama(
    # Yeni (Doğru):
    return ChatOllama(
        model="llama3",  # 'ollama pull llama3' ile indirdiğiniz model
        temperature=0.1
    )


# Llama 3 için kullandığımız Prompt Şablonu aynı kalıyor
# core.py içindeki PROMPT_TEMPLATE değişkenini güncelleyin:

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Sen yardımcı bir asistansın. Aşağıdaki "Bağlam" metnini kullanarak "Soru"ya Türkçe ve kısa yanıt ver.

Kurallar:
1. Sadece verilen bağlamdaki bilgileri kullan.
2. Eğer sorunun cevabı bağlamda kesin olarak yoksa, yorum yapma veya açıklama getirme. Sadece ve sadece şu cümleyi yaz: "Bu belgeden çıkaramıyorum."
3. Cevabı uydurma.<|eot_id|><|start_header_id|>user<|end_header_id|>

Bağlam:
{context}

Soru: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def answer_question(question: str, retriever):
    """Retriever ile alakalı parçaları bulur ve Ollama LLM ile cevap üretir."""

    retriever.search_kwargs["k"] = 5
    docs = retriever.invoke(question)

    context = "\n\n".join(d.page_content for d in docs)[:7000]

    final_prompt = PROMPT_TEMPLATE.format(question=question, context=context)

    llm = get_llm()

    # .invoke() metodunu çağırıyoruz
    result = llm.invoke(final_prompt)

    # ChatOllama nesnesi, 'result' olarak bir AIMessage nesnesi döndürür.
    # Sadece metin içeriğini almak için .content kullanmalıyız.
    clean_result = result.content.strip()

    srcs = []
    for d in docs:
        meta = d.metadata or {}
        s = meta.get("source", "?")
        if "page" in meta:
            s += f" (sayfa {meta['page'] + 1})"
        srcs.append(s)

    uniq_srcs = list(dict.fromkeys(srcs))

    return clean_result, uniq_srcs