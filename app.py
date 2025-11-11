# app.py
import gradio as gr

# Artık dosya yüklemeyeceğimiz için 'tempfile', 'PyPDFLoader',
# 'TextLoader', 'RecursiveCharacterTextSplitter', 'HuggingFaceEmbeddings',
# 'Chroma' importlarına bu dosyada gerek kalmadı.

# Çekirdek mantığımızı import et
from core import get_retriever, answer_question, DB_DIR


# 'index_uploaded' fonksiyonu tamamen kaldırıldı.

def chat_fn(message, history):  # 'retriever' parametresi kaldırıldı
    """Chat arayüzünden gelen mesajı cevaplar (Sadece varsayılan index)."""

    if not (message or "").strip():
        history.append({"role": "assistant", "content": "⚠️ Lütfen bir soru yazın."})
        return history, ""

        # 'retriever is None' kontrolü kaldırıldı.
    # Her zaman varsayılan 'db/' index'ini yüklemeyi dene:
    try:
        retriever = get_retriever()
    except Exception as e:
        # Bu hata genellikle 'python index_build.py' çalıştırılmadığında olur
        err_msg = f"⚠️ Varsayılan index ({DB_DIR}/) yüklenemedi. Lütfen önce `python index_build.py` çalıştırın. Hata: {e}"
        history.append({"role": "assistant", "content": err_msg})
        return history, ""

    # Soruyu cevapla
    answer, sources = answer_question(message, retriever)

    out = answer
    if sources:
        out += "\n\n📚 Kaynaklar:\n- " + "\n- ".join(sources)

    history.append({"role": "assistant", "content": out})

    return history, ""


# Gradio Arayüz Tanımı
with gr.Blocks() as demo:
    gr.Markdown("## Belge Tabanlı Q&A (LangChain + Chroma + Gradio)")

    # 'upload = gr.File(...)' bileşeni kaldırıldı.
    # 'status = gr.Markdown(...)' bileşeni kaldırıldı.
    # 'retriever_state = gr.State(None)' kaldırıldı.

    chatbot = gr.Chatbot(height=350, label="RAG Chatbot", type="messages")
    msg = gr.Textbox(placeholder="Sorunu yaz ve Enter'a bas")

    # 'upload.upload(...)' olayı kaldırıldı.

    # 'msg.submit' güncellendi: 'retriever_state' input'lardan kaldırıldı.
    msg.submit(
        chat_fn,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)