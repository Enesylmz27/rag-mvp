# core.py
import random
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

# ==========================================
# 🛠️ AYARLAR
# ==========================================
TEST_K = 3
TEST_MMR = True
DB_DIR = "db"


# ==========================================

def get_mock_weather(city="Elazığ"):
    """
    Gerçek bir API yerine rastgele hava durumu döndüren MOCK (Taklit) fonksiyon.
    """
    if not city:
        city = "Elazığ"

    conditions = [
        "Güneşli ☀️, 25°C",
        "Sağanak Yağışlı 🌧️, 10°C",
        "Karlı ❄️, -2°C",
        "Rüzgarlı 💨, 18°C"
    ]
    forecast = random.choice(conditions)
    return f"{city} şehri için hava durumu şu an: {forecast}."


def get_retriever():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vect_db = Chroma(persist_directory=DB_DIR, embedding_function=emb)
    return vect_db.as_retriever(search_kwargs={"k": TEST_K})


def get_llm():
    return ChatOllama(model="llama3", temperature=0.1)


# !!! GÜNCELLENMİŞ PROMPT !!!
# Hem belgeyi (Document) hem de API bilgisini (External Info) içeriyor.
# core.py içindeki PROMPT_TEMPLATE kısmını tamamen bununla değiştirin:

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Sen SADECE TÜRKÇE konuşan yardımcı bir asistansın. Asla İngilizce cevap verme. Aşağıdaki "Bağlam" ve "Dış Bilgi" kısımlarını kullanarak soruya cevap ver.

Kurallar:
1. CEVABI KESİNLİKLE TÜRKÇE OLARAK VER.
2. Öncelikle sorunun cevabını belgelerdeki (Bağlam) kurallara göre ver.
3. Ardından, verilen "Dış Bilgi"ye (Hava Durumu) dayanarak kullanıcıya kısa bir tavsiye ekle.
4. Cevabı uydurma.<|eot_id|><|start_header_id|>user<|end_header_id|>

Dış Bilgi (API):
{api_context}

Bağlam (Belgeler):
{context}

Soru: {question} (Cevabı Türkçe ver)<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def answer_question(question: str, retriever, city_name="Elazığ"):
    """
    Parametreler:
    - city_name: Arayüzden gelen şehir bilgisi (Hava durumu için)
    """

    # 1. API Verisini Çek (Mock)
    api_data = get_mock_weather(city_name)
    print(f"🌍 API Çağrısı Yapıldı: {api_data}")

    # 2. Belgeleri Getir (RAG)
    if TEST_MMR:
        retriever.search_type = "mmr"
        retriever.search_kwargs = {"k": TEST_K, "fetch_k": TEST_K * 4}
    else:
        retriever.search_type = "similarity"
        retriever.search_kwargs = {"k": TEST_K}

    docs = retriever.invoke(question)

    # 3. Prompt'u Hazırla (Belge + API)
    context_text = "\n\n".join(d.page_content for d in docs)[:7000]

    # Prompt'a hem context'i hem api_context'i gönderiyoruz
    final_prompt = PROMPT_TEMPLATE.format(
        question=question,
        context=context_text,
        api_context=api_data
    )

    llm = get_llm()
    result = llm.invoke(final_prompt)

    clean_result = result.content.strip()

    # Kaynakları Listele
    srcs = []
    for d in docs:
        meta = d.metadata or {}
        s = meta.get("source", "?")
        if "page" in meta:
            s += f" (sayfa {meta['page'] + 1})"
        srcs.append(s)
    uniq_srcs = list(dict.fromkeys(srcs))

    # Cevabın altına API bilgisini de ekleyelim ki kullanıcı neye göre tavsiye verdiğimizi görsün
    clean_result += f"\n\n🌤️ (Referans alınan hava durumu: {api_data})"

    return clean_result, uniq_srcs