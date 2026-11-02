# Cara menyamarkan data pribadi sebelum dikirim ke API ChatGPT, Claude, atau Gemini

> **Draf.** Untuk blog, Medium, atau LinkedIn saat lindungidatamu dirilis. Contoh kode memakai
> versi pengembangan; `pip install lindungidatamu` baru berlaku setelah rilis di PyPI.

Banyak tim di Indonesia sekarang memakai model AI lewat API: chatbot layanan pelanggan, ringkasan
keluhan nasabah, analisis transkrip call center, sampai otomatisasi dokumen. Hampir semuanya
mengirim teks yang berisi data pribadi: nama, NIK, nomor rekening, alamat, nomor HP.

Di bawah UU Pelindungan Data Pribadi (UU No. 27 Tahun 2022), mengirim data pribadi pelanggan ke
layanan AI termasuk pemrosesan data, dan perusahaan Anda yang bertanggung jawab atasnya. Cara paling
aman bukan melarang AI, tetapi memastikan model tidak pernah melihat data aslinya.

## Idenya: samarkan, kirim, kembalikan

```
Kirim kontrak ke Budi Santoso, NIK 3174091203880004, rek. BCA 1234567890
                 ↓ disamarkan di server Anda
Kirim kontrak ke [PERSON_1], NIK [NIK_1], rek. BCA [BANK_1]
                 ↓ dikirim ke model AI
Kontrak untuk [PERSON_1] sudah disiapkan; dana akan dikirim ke [BANK_1].
                 ↓ dikembalikan di server Anda
Kontrak untuk Budi Santoso sudah disiapkan; dana akan dikirim ke 1234567890.
```

Model tetap bisa bekerja (ia tahu ada satu orang, satu NIK, satu rekening), tetapi tidak pernah
menerima nilai aslinya. Tabel penanda ke nilai asli tetap di server Anda.

## Kenapa regex sederhana tidak cukup

Mencari "16 angka" untuk NIK terdengar mudah, sampai Anda bertemu data sungguhan:

- NIK ditulis `3174 0912 0388 0004`, `317409-120388-0004`, atau salah ketik satu angka.
- Nomor invoice, resi JNE, dan nomor pesanan juga panjang, tetapi bukan data pribadi.
- NPWP punya format lama (`01.234.567.8-901.000`) dan, sejak 2024, format 16 digit.
- Nomor rekening jarang punya format tetap; yang jelas hanya labelnya: "no. rek.", "norek",
  "rekening BSI nomor …", atau langsung setelah nama bank.
- Alamat Indonesia panjang: jalan, RT/RW, kelurahan, kecamatan, kota, kode pos. Menyamarkan
  jalannya saja masih membocorkan sisanya.
- Nama orang tidak punya pola sama sekali; perlu model.

## Pakai lindungidatamu

[lindungidatamu](https://github.com/kfhanson/lindungidatamu) adalah library Python open source
(Apache-2.0) yang menangani semua hal di atas, berjalan di server Anda sendiri, tanpa mengirim data
ke mana pun.

```bash
pip install lindungidatamu              # pola dan penyamaran, tanpa dependensi
pip install "lindungidatamu[names]"     # + nama orang, organisasi, alamat (model GLiNER, lokal)
```

```python
import lindungidatamu as ld
from lindungidatamu.names import GlinerNames

nama = GlinerNames()            # unduh model sekali (sekitar 1 GB), lalu berjalan lokal

pesan = "Tolong ringkas komplain Budi Santoso, NIK 3174091203880004, HP 0812-3456-7890"
m = ld.mask(pesan, names=nama)
print(m.text)
# Tolong ringkas komplain [PERSON_1], NIK [NIK_1], HP [PHONE_1]

jawaban = panggil_model_ai(m.text)      # ChatGPT, Claude, Gemini: yang terkirim hanya penanda
print(m.restore(jawaban))               # nilai asli kembali, di server Anda
```

Untuk percakapan, pakai satu `Vault` agar orang yang sama selalu mendapat penanda yang sama ("Budi"
setelah "Budi Santoso" tetap `[PERSON_1]`):

```python
v = ld.Vault()
ld.mask("Budi Santoso minta refund", vault=v, names=nama)
ld.mask("Tolong balas ke Budi", vault=v, names=nama)      # [PERSON_1] lagi
```

Data milik perusahaan Anda sendiri juga bisa:

```python
ld.mask(teks,
        terms=[ld.Term("CLIENT", "PT Sinar Abadi")],                # nama klien
        formats=[ld.Format("POLICY", "POL-########")])              # nomor polis: # angka, @ huruf
```

Sudah memakai Microsoft Presidio? Tambahkan pengenal khusus Indonesia:

```python
from presidio_analyzer import AnalyzerEngine
from lindungidatamu.presidio import add_to
analyzer = add_to(AnalyzerEngine())
```

## Seberapa akurat?

Diukur terbuka di [benchmark](https://github.com/kfhanson/lindungidatamu/tree/main/bench):

- **Nama orang pada teks Indonesia sungguhan** (data uji publik IndoNLU, berita dan ensiklopedia):
  terlindungi **90–93%** dengan model nama. Lapisan khusus Indonesia memangkas sepertiga sampai
  separuh salah tandai model (kata ganti seperti *dia*, *ia*; jabatan seperti *presiden*; bank,
  partai, dan kementerian yang bukan data pribadi).
- **NIK, NPWP, KK, BPJS, SIM, paspor, rekening, kartu, HP, email, alamat**: 100% pada dua set uji
  sintetis kami (651 teks: chat, formulir, kontrak, layanan pelanggan, log). Set itu kami tulis
  sendiri, jadi artinya "format-format ini ditangani", bukan janji untuk semua data nyata.

Artinya: untuk nomor identitas dan alamat, pola sudah sangat andal; untuk nama, sekitar 1 dari 10
nama bisa lolos. Kalau risiko itu terlalu tinggi untuk kasus Anda, kombinasikan dengan kebijakan:
jangan kirim dokumen utuh berisi daftar nama ke AI, atau tinjau ulang keluaran sebelum dipakai.

## Tips praktis

1. **Samarkan di server, bukan di browser pengguna**, agar tabel nilai asli tidak pernah keluar.
2. **Simpan `Vault` hanya selama percakapan berlangsung.** Setelah itu hapus.
3. **Samarkan juga log dan dataset**, bukan hanya prompt: log aplikasi sering menjadi tempat
   data pribadi bocor.
4. **Uji dengan data Anda sendiri.** Format nomor pelanggan, polis, atau karyawan tiap perusahaan
   berbeda; tambahkan lewat `Format` dan ukur.

---

lindungidatamu dikembangkan terbuka di GitHub. Masukan, laporan salah deteksi, atau contoh format
yang belum tertangani sangat membantu.
