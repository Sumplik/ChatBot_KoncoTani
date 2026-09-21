"""
chat_cli.py
===========
Versi console/terminal dari chatbot "KoncoTani" (Gemini API).

Jalankan dari terminal (di folder project ini):

    python chat_cli.py

Perintah khusus di dalam chat:
    exit   -> keluar dari chatbot
    clear  -> menghapus conversation history (mulai obrolan baru)
    save   -> menyimpan riwayat percakapan ke file JSON
    help   -> menampilkan daftar perintah
"""

from chatbot_core import (
    MODEL_DEFAULT,
    ApiKeyTidakDitemukan,
    buat_client,
    buat_giliran,
    catatan_hasil,
    pesan_error,
    reset_history,
    simpan_riwayat,
    stream_jawaban,
)

GARIS = "=" * 60


def tampilkan_bantuan() -> None:
    print("Perintah khusus:")
    print("  exit   -> keluar dari chatbot")
    print("  clear  -> menghapus conversation history")
    print("  save   -> menyimpan riwayat percakapan ke file JSON")
    print("  help   -> menampilkan bantuan ini")
    print()


def jawab(client, contents) -> str | None:
    """
    Menampilkan jawaban Gemini secara streaming ke terminal.
    Mengembalikan teks jawaban, atau None bila gagal/kosong.
    """
    teks = ""
    hasil = None
    print("KoncoTani : ", end="", flush=True)

    try:
        for jenis, isi in stream_jawaban(client, contents):
            if jenis == "teks":
                print(isi, end="", flush=True)
                teks += isi
            elif jenis == "selesai":
                hasil = isi
    except Exception as e:  # noqa: BLE001 - semua error API diterjemahkan di satu tempat
        print("\n\n[!] " + pesan_error(e) + "\n")
        return None

    print("\n")

    catatan = catatan_hasil(hasil)
    if catatan:
        print(f"[i] {catatan}\n")

    if not teks:
        return None

    return teks


def main() -> None:
    try:
        client = buat_client()
    except ApiKeyTidakDitemukan as e:
        print("[!] " + str(e))
        return

    print(GARIS)
    print("      KONCOTANI - ASISTEN PERTANIAN TANAMAN PANGAN")
    print("      (padi, jagung, kedelai, dan umbi-umbian)")
    print(GARIS)
    print(f"Model: {MODEL_DEFAULT}")
    tampilkan_bantuan()

    contents = reset_history()

    while True:
        try:
            user_input = input("Petani    : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nChatbot selesai. Sampai jumpa!")
            break

        if not user_input:
            print("Silakan masukkan pertanyaan.\n")
            continue

        perintah = user_input.lower()

        if perintah == "exit":
            if contents:
                path = simpan_riwayat(contents)
                print(f"Riwayat percakapan disimpan ke: {path}")
            print("\nChatbot selesai. Sampai jumpa!")
            break

        if perintah == "clear":
            contents = reset_history()
            print("\nConversation history telah dihapus.\n")
            continue

        if perintah == "save":
            if not contents:
                print("Belum ada percakapan untuk disimpan.\n")
            else:
                path = simpan_riwayat(contents)
                print(f"Riwayat percakapan disimpan ke: {path}\n")
            continue

        if perintah == "help":
            tampilkan_bantuan()
            continue

        # Tambahkan pertanyaan user ke history, lalu kirim SELURUH history ke API
        # (API bersifat stateless - model hanya "ingat" apa yang kita kirim ulang).
        contents.append(buat_giliran("user", user_input))

        teks = jawab(client, contents)

        if teks is not None:
            # Simpan jawaban ke history supaya jadi konteks giliran berikutnya.
            # Perhatikan: role balasan model di Gemini bernama "model".
            contents.append(buat_giliran("model", teks))
        else:
            # Request gagal -> buang pertanyaan tadi supaya history tidak rusak
            contents.pop()


if __name__ == "__main__":
    main()
