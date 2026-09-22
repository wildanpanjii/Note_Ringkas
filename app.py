import streamlit as st
import rag_engine  # Mengimpor mesin AI yang sudah Anda buat

st.set_page_config(page_title="RAG AI Chatbot", page_icon="🤖")
st.title("🤖 Chatbot AI Berbasis Dokumen")
st.write("Tanyakan apa saja berdasarkan dokumen PDF yang telah diproses.")

# Komponen File Uploader
st.sidebar.header("📂 Pengaturan Dokumen")
if st.sidebar.button("Proses Ulang Dokumen (Buat Database)"):
    with st.spinner("Sedang membaca dan memotong PDF..."):
        rag_engine.buat_vector_db()
        st.sidebar.success("Database vektor berhasil diperbarui!")

# Tampilan Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan histori obrolan
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kolom input pertanyaan
if pertanyaan := st.chat_input("Ketikkan pertanyaan Anda di sini..."):
    # Tampilkan pertanyaan pengguna
    with st.chat_message("user"):
        st.markdown(pertanyaan)
    st.session_state.messages.append({"role": "user", "content": pertanyaan})

    # Panggil fungsi tanya_ai dari rag_engine.py
    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban di dalam dokumen..."):
            try:
                jawaban_ai = rag_engine.tanya_ai(pertanyaan)
                st.markdown(jawaban_ai)
                st.session_state.messages.append({"role": "assistant", "content": jawaban_ai})
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menghubungi model AI: {e}")