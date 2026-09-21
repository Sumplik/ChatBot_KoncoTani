"""
chatbot_core.py
===============
Logika inti chatbot "KoncoTani" berbasis Gemini API (Google AI Studio).

File ini sengaja dipisah dari tampilan supaya bisa dipakai ulang oleh:
  - app.py       -> tampilan web (Streamlit)
  - chat_cli.py  -> versi console/terminal

Isi file ini:
  1. Konfigurasi model & system prompt
  2. Pembuatan client Gemini (API key dibaca dari file .env)
  3. Format conversation history ala Gemini
  4. Fungsi streaming jawaban (jawaban muncul bertahap)
  5. Penanganan error yang ramah dibaca manusia
  6. Simpan & muat riwayat percakapan (JSON)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Membaca file .env di folder project -> mengisi os.environ["GEMINI_API_KEY"]
load_dotenv()

# ==========================================
# 1. KONFIGURASI MODEL
# ==========================================

# Model Gemini yang dipakai. Semuanya tersedia di free tier Google AI Studio dan
# sudah diuji bisa dipanggil dengan API key biasa.
# Daftar lengkap: https://ai.google.dev/gemini-api/docs/models
#
# Catatan: model lama seperti `gemini-2.5-flash` masih muncul di `models.list()`,
# tetapi sudah tidak dibuka untuk pengguna baru - memanggilnya menghasilkan error
# 404 dengan saran berpindah ke seri Gemini 3.
PILIHAN_MODEL = {
    "gemini-3.8-flash": "Paling baru & paling pintar (disarankan)",
    "gemini-3.6-flash": "Generasi sebelumnya, biasanya lebih lega saat server ramai",
    "gemini-3.5-flash-lite": "Ringan & cepat untuk tanya-jawab singkat",
    "gemini-3.1-flash-lite": "Paling hemat, untuk percakapan sederhana",
}
MODEL_DEFAULT = "gemini-3.8-flash"

# Batas maksimal panjang JAWABAN. Perlu diingat: token "berpikir" model juga
# dihitung di sini, jadi jangan diset terlalu kecil.
DEFAULT_MAX_TOKENS = 8000

# Seberapa dalam model berpikir sebelum menjawab.
#   low    -> cepat & hemat, cocok untuk obrolan ringan
#   medium -> seimbang (default di sini)
#   high   -> paling teliti, cocok untuk soal/penjelasan rumit
DEFAULT_THINKING = "medium"
PILIHAN_THINKING = ["low", "medium", "high"]

# Temperature = tingkat keacakan pemilihan kata (0 = paling konsisten,
# 2 = paling liar). Masih diterima model Gemini 3, tetapi Google menyarankan
# membiarkannya di nilai bawaan 1.0 karena kemampuan bernalar model dikalibrasi
# di nilai tersebut - lihat penjelasan di README.
DEFAULT_TEMPERATURE = 1.0

# Harga tier berbayar (USD per 1 juta token) - hanya untuk estimasi di sidebar.
# Di free tier, semua pemakaian ini tidak ditagih.
HARGA_PER_JUTA = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}

# Folder tempat file riwayat percakapan disimpan
RIWAYAT_DIR = Path(__file__).parent / "riwayat"


# ==========================================
# 2. SYSTEM PROMPT (kepribadian chatbot)
# ==========================================
# System prompt = instruksi "di belakang layar" yang menentukan peran dan gaya
# bicara model. Pengguna tidak melihatnya, tapi model selalu mengikutinya.
# Ubah teks di bawah ini kalau ingin mengganti tema chatbot.

SYSTEM_PROMPT = """Kamu adalah KoncoTani, asisten penyuluh pertanian digital untuk petani Indonesia.

## RUANG LINGKUP (wajib dipatuhi)
Kamu HANYA membahas tanaman pangan utama Indonesia:
- Padi
- Jagung
- Kedelai
- Umbi-umbian (singkong, ubi jalar, talas, dan sejenisnya)

Tolak dengan sopan bila ditanya tanaman hias, tanaman perkebunan besar (sawit,
karet, kopi, tebu, kakao), tanaman kehutanan, atau topik di luar pertanian pangan.
Cara menolak: sebut singkat bahwa itu di luar keahlianmu, lalu tawarkan bantuan
untuk padi, jagung, kedelai, atau umbi-umbian.

## PERAN 1 - DOKTER TANAMAN (hama & penyakit)
- Dengarkan keluhan petani, lalu diagnosis gejalanya (misal: daun padi menguning,
  batang jagung busuk, daun kedelai bolong-bolong).
- Sebutkan dugaan hama/penyakitnya (misal wereng batang cokelat, ulat grayak,
  penyakit blas, bulai) beserta ciri pembeda, supaya petani bisa memastikan
  sendiri di lapangan.
- Beri penanganan bertingkat: perbaikan cara budidaya dan pengendalian
  organik/hayati lebih dulu, pestisida kimia sebagai jalan terakhir - lengkap
  dengan bahan aktif, dosis anjuran, dan waktu aplikasi yang tepat.

## PERAN 2 - PAKAR PUPUK & TANAH (nutrisi & kesuburan)
- Hitung kebutuhan pupuk (Urea, NPK, SP-36, KCl, pupuk organik) berdasarkan luas
  lahan dan fase tumbuh (vegetatif vs generatif). Tampilkan cara hitungnya,
  jangan hanya hasil akhirnya.
- Bantu masalah tanah: tanah masam/pH rendah (dosis kapur pertanian atau
  dolomit), tanah jenuh akibat pupuk kimia berlebihan, serta tanah keras dan
  miskin bahan organik.

## CARA MENJAWAB
- Gunakan Bahasa Indonesia sehari-hari yang mudah dipahami petani. Istilah ilmiah
  boleh dipakai, tapi selalu jelaskan artinya.
- Sebelum memberi resep, tanyakan dulu data yang belum diketahui: jenis tanaman,
  umur atau fase tanaman, luas lahan, gejala yang terlihat, dan tindakan yang
  sudah dilakukan. Tanya seperlunya saja, maksimal 3 pertanyaan sekali jalan.
- Pakai satuan yang akrab bagi petani: hektar, meter persegi, kg per hektar, dan
  gram per tangki semprot 14-16 liter.
- Jawab ringkas dan terstruktur dengan poin-poin. Jangan bertele-tele.

## KEAMANAN
- Selalu ingatkan pemakaian pestisida sesuai dosis pada label, memakai alat
  pelindung (masker, sarung tangan, baju lengan panjang), dan memperhatikan masa
  tunggu sebelum panen.
- Jangan pernah menyarankan menaikkan dosis melebihi anjuran label atau mencampur
  beberapa pestisida tanpa dasar yang jelas.
- Bila gejala tidak bisa dipastikan hanya dari cerita petani, katakan terus terang
  dan sarankan pemeriksaan langsung oleh petugas POPT atau penyuluh setempat.
"""


# ==========================================
# 3. CONVERSATION HISTORY (format Gemini)
# ==========================================
# Di Gemini, riwayat percakapan disebut `contents`: list of dict dengan bentuk
#     {"role": "user" | "model", "parts": [{"text": "..."}]}
#
# Dua hal yang berbeda dari contoh Groq/OpenAI di kelas:
#   1. Role balasan model bernama "model", bukan "assistant".
#   2. Isi pesan dibungkus dalam list `parts`, bukan string `content` langsung.
#      Bentuk ini dipakai karena satu pesan bisa berisi beberapa bagian
#      (teks + gambar + audio), bukan hanya teks.


def reset_history() -> list[dict]:
    """Mengembalikan conversation history ke kondisi awal (kosong)."""
    return []


def buat_giliran(role: str, teks: str) -> dict:
    """Membuat satu giliran percakapan dalam format Gemini."""
    return {"role": role, "parts": [{"text": teks}]}


def ambil_teks(giliran: dict) -> str:
    """Mengambil kembali teks dari satu giliran percakapan."""
    return "".join(bagian.get("text", "") for bagian in giliran.get("parts", []))


# ==========================================
# 4. CLIENT GEMINI
# ==========================================


class ApiKeyTidakDitemukan(RuntimeError):
    """Dilempar saat GEMINI_API_KEY belum diset di .env maupun environment."""


def buat_client() -> genai.Client:
    """
    Membuat client Gemini. Client adalah "gerbang" komunikasi antara kode
    Python kita dengan server Google - semua request dikirim lewat objek ini.

    API key dibaca otomatis dari environment variable GEMINI_API_KEY
    (diisi oleh file .env), jadi key tidak pernah ditulis di dalam kode.
    """
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise ApiKeyTidakDitemukan(
            "GEMINI_API_KEY belum diset.\n"
            "Langkah perbaikan:\n"
            "  1. Salin file .env.example menjadi .env\n"
            "  2. Isi GEMINI_API_KEY dengan key gratis dari https://aistudio.google.com/apikey\n"
            "  3. Jalankan ulang program ini"
        )
    return genai.Client()


# ==========================================
# 5. MENGIRIM PESAN (STREAMING)
# ==========================================


@dataclass
class HasilAkhir:
    """Ringkasan satu balasan: jumlah token terpakai & alasan model berhenti."""

    token_masuk: int = 0
    token_keluar: int = 0
    token_berpikir: int = 0
    alasan_berhenti: str | None = None
    alasan_diblokir: str | None = None


def stream_jawaban(
    client: genai.Client,
    contents: list[dict],
    *,
    system: str = SYSTEM_PROMPT,
    model: str = MODEL_DEFAULT,
    max_output_tokens: int = DEFAULT_MAX_TOKENS,
    thinking_level: str = DEFAULT_THINKING,
    temperature: float | None = None,
    tampilkan_proses_berpikir: bool = False,
) -> Iterator[tuple[str, object]]:
    """
    Mengirim seluruh riwayat percakapan ke Gemini dan mengembalikan jawaban
    secara bertahap (streaming), mirip pengalaman mengetik di ChatGPT.

    Kenapa seluruh riwayat dikirim ulang setiap kali?
    Karena API bersifat *stateless* - model tidak menyimpan percakapan kita.
    Agar terasa "ingat", seluruh `contents` dikirim ulang di tiap request.

    Fungsi ini adalah generator yang menghasilkan pasangan (jenis, isi):
      ("berpikir", str)        -> ringkasan proses berpikir model
      ("teks", str)            -> potongan jawaban untuk pengguna
      ("selesai", HasilAkhir)  -> ringkasan token & alasan berhenti
    """
    config = types.GenerateContentConfig(
        # System prompt di Gemini dikirim lewat `system_instruction`,
        # terpisah dari `contents`.
        system_instruction=system,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(
            thinking_level=thinking_level,
            include_thoughts=tampilkan_proses_berpikir,
        ),
        # Chatbot ini murni tanya-jawab teks (tidak memakai tool/function calling),
        # jadi fitur automatic function calling dimatikan supaya SDK tidak
        # memunculkan peringatan yang tidak relevan.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Temperature hanya dikirim bila pengguna memang mengaturnya; kalau None,
    # model memakai nilai bawaannya sendiri.
    if temperature is not None:
        config.temperature = temperature

    hasil = HasilAkhir()

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        # Kalau pertanyaan diblokir filter keamanan, tidak ada kandidat jawaban.
        if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
            hasil.alasan_diblokir = str(chunk.prompt_feedback.block_reason)

        for kandidat in chunk.candidates or []:
            isi = kandidat.content
            for bagian in (isi.parts if isi and isi.parts else []):
                if not bagian.text:
                    continue
                # `bagian.thought = True` menandai potongan proses berpikir,
                # bukan jawaban yang ditujukan untuk pengguna.
                yield ("berpikir" if bagian.thought else "teks", bagian.text)

            if kandidat.finish_reason:
                hasil.alasan_berhenti = str(kandidat.finish_reason)

        # Chunk terakhir membawa perhitungan token kumulatif.
        if chunk.usage_metadata:
            pakai = chunk.usage_metadata
            hasil.token_masuk = pakai.prompt_token_count or 0
            hasil.token_keluar = pakai.candidates_token_count or 0
            hasil.token_berpikir = pakai.thoughts_token_count or 0

    yield ("selesai", hasil)


def pesan_error(e: Exception) -> str:
    """
    Menerjemahkan error dari SDK Gemini menjadi kalimat yang mudah dibaca.

    Urutan pengecekan sengaja dari yang paling spesifik ke paling umum,
    supaya penyebab error tidak tertelan oleh kelas error yang terlalu luas.
    """
    if isinstance(e, ApiKeyTidakDitemukan):
        return str(e)

    if isinstance(e, errors.ClientError):  # error 4xx = kesalahan dari sisi kita
        pesan = str(getattr(e, "message", "") or e)
        if e.code == 429:
            return (
                "Kuota habis atau permintaan terlalu cepat (rate limit free tier). "
                "Tunggu sekitar satu menit, lalu coba lagi."
            )
        if e.code in (400, 401) and "api key" in pesan.lower():
            return "API key ditolak. Periksa kembali isi GEMINI_API_KEY di file .env."
        if e.code == 403:
            return "API key tidak punya izin untuk model ini. Cek pengaturan di Google AI Studio."
        if e.code == 404:
            return "Model tidak ditemukan. Periksa kembali nama model di chatbot_core.py."
        return f"Permintaan ditolak server (kode {e.code}): {pesan}"

    if isinstance(e, errors.ServerError):  # error 5xx = masalah di sisi Google
        return f"Server Gemini sedang bermasalah (kode {e.code}). Coba lagi sebentar lagi."

    if isinstance(e, errors.APIError):
        return f"Error dari API Gemini: {e}"

    if isinstance(e, (ConnectionError, TimeoutError)):
        return "Gagal terhubung ke server. Periksa koneksi internet kamu."

    return f"Terjadi error tak terduga: {type(e).__name__}: {e}"


def catatan_hasil(hasil: HasilAkhir) -> str | None:
    """
    Mengembalikan catatan bila model berhenti karena alasan yang perlu
    diketahui pengguna. Mengembalikan None bila jawaban selesai normal.
    """
    if hasil.alasan_diblokir:
        return (
            f"Pertanyaan diblokir filter keamanan Gemini ({hasil.alasan_diblokir}). "
            "Coba tulis ulang pertanyaannya."
        )

    alasan = (hasil.alasan_berhenti or "").upper()

    if "MAX_TOKENS" in alasan:
        return (
            "Jawaban terpotong karena mencapai batas panjang. Naikkan nilai "
            "'Panjang jawaban maksimal', atau turunkan 'Kedalaman berpikir' "
            "(token berpikir ikut memakan jatah panjang jawaban)."
        )
    if any(k in alasan for k in ("SAFETY", "PROHIBITED", "BLOCKLIST", "SPII")):
        return f"Jawaban dihentikan filter keamanan Gemini ({alasan})."
    if "RECITATION" in alasan:
        return "Jawaban dihentikan karena terlalu mirip dengan sumber berhak cipta."
    return None


# ==========================================
# 6. SIMPAN & MUAT RIWAYAT PERCAKAPAN
# ==========================================


def riwayat_ke_json(contents: list[dict], model: str = MODEL_DEFAULT) -> str:
    """Mengubah riwayat percakapan menjadi teks JSON (siap ditulis/diunduh)."""
    data = {
        "disimpan_pada": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "system_prompt": SYSTEM_PROMPT,
        "contents": contents,
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def simpan_riwayat(
    contents: list[dict],
    filename: str | Path | None = None,
    model: str = MODEL_DEFAULT,
) -> Path:
    """Menyimpan riwayat percakapan ke file JSON di folder `riwayat/`."""
    RIWAYAT_DIR.mkdir(exist_ok=True)

    if filename is None:
        stempel = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = RIWAYAT_DIR / f"riwayat_chat_{stempel}.json"

    path = Path(filename)
    path.write_text(riwayat_ke_json(contents, model), encoding="utf-8")
    return path


def normalisasi_riwayat(data) -> list[dict]:
    """
    Mengambil riwayat percakapan dari isi file JSON dan menyeragamkannya ke
    format Gemini.

    Mendukung beberapa bentuk file:
      - format Gemini  : {"contents": [{"role": "model", "parts": [...]}]}
      - format lama    : {"messages": [{"role": "assistant", "content": "..."}]}
      - list telanjang : [{"role": "user", "content": "..."}]
    Pesan ber-role "system" dibuang, karena di Gemini system prompt dikirim
    terpisah lewat `system_instruction`.
    """
    if isinstance(data, dict):
        mentah = data.get("contents") or data.get("messages") or []
    else:
        mentah = data

    hasil = []
    for pesan in mentah:
        role = pesan.get("role")
        if role == "system":
            continue
        if role == "assistant":  # format Groq/OpenAI/Claude -> Gemini
            role = "model"
        if role not in ("user", "model"):
            continue

        teks = ambil_teks(pesan) if "parts" in pesan else pesan.get("content", "")
        if teks:
            hasil.append(buat_giliran(role, teks))

    return hasil


def muat_riwayat(filename: str | Path) -> list[dict]:
    """Memuat kembali riwayat percakapan dari sebuah file JSON."""
    return normalisasi_riwayat(json.loads(Path(filename).read_text(encoding="utf-8")))
