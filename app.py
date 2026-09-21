"""
app.py
======
Tampilan web chatbot "KoncoTani" menggunakan Streamlit + Gemini API.

Jalankan dari terminal (di folder project ini):

    streamlit run app.py

Browser akan otomatis terbuka di http://localhost:8501
"""

import json
from pathlib import Path

import streamlit as st

from chatbot_core import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING,
    HARGA_PER_JUTA,
    MODEL_DEFAULT,
    PILIHAN_MODEL,
    PILIHAN_THINKING,
    SYSTEM_PROMPT,
    ApiKeyTidakDitemukan,
    ambil_teks,
    buat_client,
    buat_giliran,
    catatan_hasil,
    normalisasi_riwayat,
    pesan_error,
    riwayat_ke_json,
    simpan_riwayat,
    stream_jawaban,
)

st.set_page_config(page_title="KoncoTani", page_icon="🌾", layout="centered")


def pasang_gaya() -> None:
    """
    Memuat tema visual (nuansa sawah hijau) dari file gaya.css.

    Warna dasar diatur di .streamlit/config.toml, sedangkan hiasan latar dan
    gaya gelembung chat ada di gaya.css. Dipisah ke file sendiri supaya mudah
    diutak-atik tanpa menyentuh logika chatbot.
    """
    berkas = Path(__file__).parent / "gaya.css"
    if berkas.exists():
        st.markdown(f"<style>{berkas.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)


pasang_gaya()


# ==========================================
# 1. CLIENT & STATE
# ==========================================


@st.cache_resource
def ambil_client():
    """
    Client dibuat sekali saja dan dipakai ulang di setiap interaksi.
    (Streamlit menjalankan ulang seluruh script setiap kali ada input baru,
    sehingga tanpa cache client akan dibuat berulang kali secara sia-sia.)
    """
    return buat_client()


try:
    client = ambil_client()
except ApiKeyTidakDitemukan as e:
    st.title("🌾 KoncoTani")
    st.error(str(e))
    st.stop()

# `riwayat` menyimpan percakapan sekaligus ringkasan proses berpikir model.
# Field "berpikir" hanya untuk ditampilkan, tidak ikut dikirim ke API.
# Role mengikuti istilah Gemini: "user" (pengguna) dan "model" (balasan AI).
if "riwayat" not in st.session_state:
    st.session_state.riwayat = []
    st.session_state.token_masuk = 0
    st.session_state.token_keluar = 0


def contents_untuk_api() -> list[dict]:
    """Menyiapkan `contents` sesuai format Gemini: role + parts."""
    return [buat_giliran(m["role"], m["content"]) for m in st.session_state.riwayat]


def percakapan_baru() -> None:
    st.session_state.riwayat = []
    st.session_state.token_masuk = 0
    st.session_state.token_keluar = 0


# ==========================================
# 2. SIDEBAR: PENGATURAN & RIWAYAT
# ==========================================

with st.sidebar:
    st.header("⚙️ Pengaturan")

    model = st.selectbox(
        "Model Gemini",
        options=list(PILIHAN_MODEL),
        index=list(PILIHAN_MODEL).index(MODEL_DEFAULT),
        format_func=lambda m: f"{m} - {PILIHAN_MODEL[m]}",
    )

    thinking_level = st.select_slider(
        "Kedalaman berpikir",
        options=PILIHAN_THINKING,
        value=DEFAULT_THINKING,
        help=(
            "Seberapa lama model berpikir sebelum menjawab. Semakin tinggi, "
            "semakin teliti - tapi lebih lambat dan lebih boros token."
        ),
    )

    temperature = st.slider(
        "Temperature (tingkat keacakan)",
        min_value=0.0,
        max_value=2.0,
        value=DEFAULT_TEMPERATURE,
        step=0.1,
        help=(
            "Makin tinggi, makin kreatif/acak jawabannya. Makin rendah, makin konsisten. "
            "Untuk Gemini 3, Google menyarankan membiarkannya di 1.0."
        ),
    )
    if temperature != DEFAULT_TEMPERATURE:
        st.caption(
            "⚠️ Nilai selain 1.0 tidak disarankan untuk Gemini 3 — jawaban bisa "
            "berputar-putar pada pertanyaan yang butuh penalaran."
        )

    max_output_tokens = st.slider(
        "Panjang jawaban maksimal (token)",
        min_value=2000,
        max_value=32000,
        value=DEFAULT_MAX_TOKENS,
        step=1000,
        help="Batas atas satu jawaban. Token proses berpikir ikut dihitung di sini.",
    )

    tampilkan_proses = st.toggle(
        "Tampilkan proses berpikir",
        value=False,
        help="Menampilkan ringkasan alur berpikir model sebelum ia menjawab.",
    )

    with st.expander("Lihat system prompt"):
        st.code(SYSTEM_PROMPT, language="text")

    st.divider()
    st.header("💾 Riwayat percakapan")

    st.button("🧹 Percakapan baru", use_container_width=True, on_click=percakapan_baru)

    ada_isi = len(st.session_state.riwayat) > 0

    if st.button("📁 Simpan ke folder riwayat/", use_container_width=True, disabled=not ada_isi):
        path = simpan_riwayat(contents_untuk_api(), model=model)
        st.success(f"Tersimpan: `{path.name}`")

    st.download_button(
        "⬇️ Unduh sebagai JSON",
        data=riwayat_ke_json(contents_untuk_api(), model) if ada_isi else "{}",
        file_name="riwayat_chat.json",
        mime="application/json",
        use_container_width=True,
        disabled=not ada_isi,
    )

    berkas = st.file_uploader("Muat riwayat dari file JSON", type="json")
    if berkas is not None and st.button("📂 Muat percakapan ini", use_container_width=True):
        try:
            dimuat = normalisasi_riwayat(json.loads(berkas.getvalue().decode("utf-8")))
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            st.error(f"File riwayat tidak bisa dibaca: {e}")
        else:
            st.session_state.riwayat = [
                {"role": g["role"], "content": ambil_teks(g), "berpikir": ""} for g in dimuat
            ]
            st.success(f"{len(dimuat)} pesan dimuat.")
            st.rerun()

    st.divider()
    st.header("📊 Statistik sesi")
    jumlah_tanya = sum(1 for m in st.session_state.riwayat if m["role"] == "user")
    harga_masuk, harga_keluar = HARGA_PER_JUTA.get(model, (0.0, 0.0))
    biaya = (
        st.session_state.token_masuk / 1_000_000 * harga_masuk
        + st.session_state.token_keluar / 1_000_000 * harga_keluar
    )
    kolom_kiri, kolom_kanan = st.columns(2)
    kolom_kiri.metric("Pertanyaan", jumlah_tanya)
    kolom_kanan.metric("Token keluar", f"{st.session_state.token_keluar:,}")
    st.caption(
        f"Token masuk: {st.session_state.token_masuk:,} · "
        f"Di free tier sesi ini **gratis**; bila pakai tier berbayar ≈ ${biaya:.4f}"
    )


# ==========================================
# 3. AREA PERCAKAPAN
# ==========================================

st.title("🌾 KoncoTani")
st.caption(
    "Konco-nya petani untuk padi, jagung, kedelai, dan umbi-umbian — "
    f"konsultasi hama, penyakit, pupuk, dan tanah. Ditenagai Gemini API (`{model}`)."
)

if not st.session_state.riwayat:
    st.info(
        "Ceritakan kondisi tanaman atau lahanmu, misalnya: "
        "*daun padi saya menguning dari ujung, umur 45 hari, luas 1/2 hektar. Kenapa ya?*"
    )

# Menampilkan ulang seluruh percakapan setiap kali halaman dijalankan ulang.
# Role "model" milik Gemini dipetakan ke "assistant" supaya dikenali Streamlit.
for pesan in st.session_state.riwayat:
    peran_tampilan = "assistant" if pesan["role"] == "model" else "user"
    with st.chat_message(peran_tampilan, avatar="🌾" if peran_tampilan == "assistant" else None):
        if pesan.get("berpikir"):
            with st.expander("🧠 Proses berpikir"):
                st.markdown(pesan["berpikir"])
        st.markdown(pesan["content"])


# ==========================================
# 4. MENGIRIM PERTANYAAN BARU
# ==========================================

pertanyaan = st.chat_input("Ceritakan masalah tanaman atau lahanmu...")

if pertanyaan:
    st.session_state.riwayat.append({"role": "user", "content": pertanyaan, "berpikir": ""})
    with st.chat_message("user"):
        st.markdown(pertanyaan)

    with st.chat_message("assistant", avatar="🌾"):
        kotak_berpikir = None
        if tampilkan_proses:
            with st.expander("🧠 Proses berpikir", expanded=True):
                kotak_berpikir = st.empty()
        kotak_teks = st.empty()

        teks, berpikir, hasil = "", "", None

        try:
            with st.spinner("KoncoTani sedang menganalisis..."):
                for jenis, isi in stream_jawaban(
                    client,
                    contents_untuk_api(),
                    model=model,
                    max_output_tokens=max_output_tokens,
                    thinking_level=thinking_level,
                    temperature=temperature,
                    tampilkan_proses_berpikir=tampilkan_proses,
                ):
                    if jenis == "berpikir" and kotak_berpikir is not None:
                        berpikir += isi
                        kotak_berpikir.markdown(berpikir)
                    elif jenis == "teks":
                        teks += isi
                        kotak_teks.markdown(teks + "▌")  # kursor berkedip
                    elif jenis == "selesai":
                        hasil = isi
        except Exception as e:  # noqa: BLE001 - semua error API diterjemahkan di satu tempat
            kotak_teks.empty()
            st.error(pesan_error(e))
            # Request gagal -> buang pertanyaan tadi supaya history tidak rusak
            st.session_state.riwayat.pop()
            st.stop()

        kotak_teks.markdown(teks)

    catatan = catatan_hasil(hasil)

    if teks:
        st.session_state.riwayat.append(
            {"role": "model", "content": teks, "berpikir": berpikir}
        )
    else:
        # Tidak ada teks jawaban (mis. diblokir filter) -> buang pertanyaan tadi
        st.session_state.riwayat.pop()
        st.warning(catatan or "Model tidak mengembalikan jawaban. Coba ulangi pertanyaannya.")
        st.stop()

    st.session_state.token_masuk += hasil.token_masuk
    st.session_state.token_keluar += hasil.token_keluar

    if catatan:
        st.warning(catatan)

    # Jalankan ulang halaman agar tombol & statistik di sidebar ikut diperbarui.
    st.rerun()
