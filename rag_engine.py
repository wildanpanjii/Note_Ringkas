import os
import warnings

# Mengabaikan pesan warning dari library pihak ketiga agar output terminal bersih
warnings.filterwarnings("ignore")

# --- PERBAIKAN KOMPATIBILITAS ---
# langchain v1.4.2 tidak memiliki atribut 'debug', 'verbose', dan 'llm_cache',
# tetapi langchain-core v0.2.43 membutuhkannya di globals.py dan callbacks/manager.py.
# Monkey-patch ini HARUS dilakukan SEBELUM import apapun dari langchain.
import langchain
if not hasattr(langchain, "debug"):
    langchain.debug = False
if not hasattr(langchain, "verbose"):
    langchain.verbose = False
if not hasattr(langchain, "llm_cache"):
    langchain.llm_cache = None
# --- AKHIR PERBAIKAN ---

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# Memuat kunci API dari file .env
load_dotenv()

# Pastikan variabel lingkungan GOOGLE_API_KEY dan GEMINI_API_KEY sinkron
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
elif os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def buat_vector_db():
    print("Membaca dan memotong dokumen PDF...")
    loader = DirectoryLoader("data/", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    print("Membuat embedding dan menyimpan ke ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="chroma_db")
    print("Database vektor berhasil dibuat!")

def tanya_ai(pertanyaan):
    print(f"\nMencari jawaban untuk: '{pertanyaan}'...")
    
    # 1. Buka Database Vektor
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 2. Ambil Dokumen Secara Manual (Tanpa modul chains!)
    dokumen_relevan = retriever.invoke(pertanyaan)
    teks_konteks = "\n\n".join([doc.page_content for doc in dokumen_relevan])
    
    # 3. Rakit Prompt Sendiri
    prompt_manual = f"""Anda adalah asisten AI cerdas. Jawablah pertanyaan pengguna HANYA berdasarkan konteks dokumen berikut. Jika tidak ada informasi di dalam konteks, katakan bahwa Anda tidak tahu.

    KONTEKS DOKUMEN:
    {teks_konteks}

    PERTANYAAN PENGGUNA: 
    {pertanyaan}
    """
    
    # 4. Panggil Gemini Secara Langsung
    # Gunakan gemini-3.8-flash (model terbaru), dengan fallback ke gemini-3.6-flash jika ada kendala
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    fallback_model = "gemini-3.6-flash"
    try:
        llm = ChatGoogleGenerativeAI(model=model_name)
        hasil = llm.invoke(prompt_manual)
    except Exception as e:
        error_str = str(e)
        # Jika rate limit / quota habis, langsung coba fallback model (quota per-model terpisah)
        if "429" in error_str or "quota" in error_str.lower() or "ResourceExhausted" in error_str:
            print(f"Catatan: Model {model_name} menemui rate limit. Mencoba model cadangan ({fallback_model})...")
        else:
            print(f"Catatan: Model {model_name} menemui kendala ({e}). Mencoba model cadangan ({fallback_model})...")
        try:
            llm = ChatGoogleGenerativeAI(model=fallback_model)
            hasil = llm.invoke(prompt_manual)
        except Exception as e2:
            print(f"\nERROR: Kedua model gagal.")
            print(f"  - {model_name}: {e}")
            print(f"  - {fallback_model}: {e2}")
            print(f"\nKemungkinan penyebab:")
            print(f"  1. Quota API harian habis (free tier: 20 request/hari per model)")
            print(f"  2. API key tidak valid")
            print(f"  3. Tidak ada koneksi internet")
            print(f"\nSilakan coba lagi besok atau upgrade API key Anda.")
            return
    
    # 5. Tampilkan Output
    print("\n=== JAWABAN AI ===")
    jawaban = getattr(hasil, "text", None)
    if not jawaban:
        if isinstance(hasil.content, str):
            jawaban = hasil.content
        elif isinstance(hasil.content, list):
            jawaban = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in hasil.content)
        else:
            jawaban = str(hasil.content)
    print(jawaban)
    
    print("\n=== SUMBER REFERENSI ===")
    for doc in dokumen_relevan:
        sumber = doc.metadata.get('source', 'Tidak diketahui')
        halaman = doc.metadata.get('page', 'Tidak diketahui')
        print(f"- {sumber} (Halaman {halaman})")

    return jawaban

if __name__ == "__main__":
    # buat_vector_db() # Biarkan di-comment
    tanya_ai("Apa saja syarat pendaftaran yang disebutkan dalam dokumen?")