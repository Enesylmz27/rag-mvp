# app.py
import gradio as gr
from core import get_retriever, answer_question, DB_DIR


def chat_fn(message, history, city_val):
    """
    city_val: Arayüzdeki şehir kutusundan gelen veri.
    """
    history = history or []
    if not (message or "").strip():
        return history, ""

    history.append({"role": "user", "content": message})

    try:
        retriever = get_retriever()

        # city_val'i core.py'ye gönderiyoruz
        answer, sources = answer_question(message, retriever, city_name=city_val)

        out = answer
        if sources:
            out += "\n\n📚 Kaynaklar:\n- " + "\n- ".join(sources)

        history.append({"role": "assistant", "content": out})

    except Exception as e:
        history.append({"role": "assistant", "content": f"🚨 Hata: {str(e)}"})
        print(f"HATA: {e}")

    return history, ""


# Arayüz
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## RAG Projesi: 4. Hafta - API Entegrasyonu (Hava Durumu)")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, label="Asistan", type="messages")
            msg = gr.Textbox(placeholder="Sorunu yaz...", label="Soru")

        with gr.Column(scale=1):
            # YENİ: Şehir Seçimi
            city_box = gr.Textbox(
                label="Şehir Seçimi",
                placeholder="Örn: Elazığ, İstanbul",
                value="Elazığ"
            )
            gr.Markdown("ℹ️ Asistan, vereceği cevaplarda yukarıdaki şehrin anlık hava durumunu dikkate alacaktır.")

            clear_btn = gr.ClearButton([msg, chatbot])

    msg.submit(chat_fn, inputs=[msg, chatbot, city_box], outputs=[chatbot, msg])

demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)