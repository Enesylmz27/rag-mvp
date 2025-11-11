# index_build.py
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# DEĞİŞEN İMPORTLAR:
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("data")
DB_DIR = "db"


def load_docs(data_dir: Path):
    """Verilen klasördeki PDF, TXT ve MD dosyalarını yükler."""
    docs = []
    print("🔎 Desteklenen dosyalar aranıyor...")
    for p in data_dir.glob("**/*"):
        p_str = str(p)
        if p.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(p_str).load())
        elif p.suffix.lower() in {".txt", ".md"}:
            docs.extend(TextLoader(p_str, encoding="utf-8").load())
    return docs


def build_index():
    print("🔎 Belgeler yükleniyor...")
    raw_docs = load_docs(DATA_DIR)

    if not raw_docs:
        print(f"⚠️ {DATA_DIR} klasöründe desteklenen (PDF, TXT, MD) belge bulunamadı. Index oluşturulmadı.")
        return

    print(f"➡️ {len(raw_docs)} doküman/sayfa bulundu.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(raw_docs)
    print(f"🧩 {len(chunks)} parça (chunk) oluşturuldu.")

    # DEĞİŞEN SINIF ADI: HuggingFaceBgeEmbeddings -> HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Chroma Veritabanı Oluşturma
    vectordb = Chroma.from_documents(chunks, emb, persist_directory=DB_DIR)

    vectordb.persist()

    print(f"✅ Chroma index oluşturuldu: {DB_DIR}/")


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    build_index()