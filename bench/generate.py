"""Builds the benchmark: Indonesian texts with every sensitive span labelled.

    uv run python bench/generate.py          # -> bench/data/id-pii-v1.jsonl (seeded: same file every time)

Synthetic on purpose (no real people's data), but written like real messages: chats, customer
service, contracts, HR, bank complaints, forms, logs, and Bahasa mixed with English. Values come in
the formats people actually write, including ones lindungidatamu doesn't handle yet, so the
benchmark shows misses instead of hiding them. Hard negatives (invoice and order numbers, waybills,
amounts, dates, public institutions) must not be masked.

Each line: {"id", "genre", "text", "spans": [{"start", "end", "type"}]}. Types: NIK NPWP KK BPJS
SIM PASSPORT BANK CARD PHONE EMAIL ADDRESS IP HOST SECRET PERSON ORG.
"""

import json
import random
import re
from pathlib import Path

SEED = 20261030
OUT = Path(__file__).parent / "data" / "id-pii-v1.jsonl"
rng = random.Random(SEED)

# ---- values -------------------------------------------------------------------------------------

PROVINCES = [11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 31, 32, 33, 34, 35, 36, 51, 52, 53, 61, 62, 63, 64, 65,
             71, 72, 73, 74, 75, 76, 81, 82, 91, 92, 94]
FIRST = ["Budi", "Siti", "Agus", "Dewi", "Rina", "Andi", "Putri", "Ahmad", "Muhammad", "Nur", "Rizky", "Fajar",
         "Dian", "Wahyu", "Yusuf", "Indah", "Sri", "Joko", "Bambang", "Hendra", "Lestari", "Kartika", "Ayu",
         "Bayu", "Dimas", "Galih", "Intan", "Asep", "Ujang", "Cecep", "Euis", "Nenden", "Dadang", "Made",
         "Ketut", "Wayan", "Komang", "Kadek", "Putu", "Yohanes", "Maria", "Stefanus", "Grace", "Kevin",
         "Felicia", "Michael", "Abdullah", "Fauzi", "Hasan", "Zainal", "Rahmat", "Teguh", "Wulan", "Citra"]
LAST = ["Santoso", "Susanto", "Wibowo", "Pratama", "Saputra", "Hidayat", "Kurniawan", "Setiawan",
        "Rahmawati", "Puspitasari", "Siregar", "Nasution", "Simanjuntak", "Hutapea", "Sitompul", "Lubis",
        "Harahap", "Wijaya", "Gunawan", "Hartono", "Halim", "Kusuma", "Tanoto", "Lim", "Tan", "Sihombing",
        "Pangaribuan", "Manurung", "Putra", "Utami", "Permana", "Ramadhan", "Firmansyah", "Nugroho"]
SINGLE = ["Sutrisno", "Mulyono", "Paijo", "Sukarti", "Suparman", "Wagiman", "Tukiyem", "Sumarni"]
TITLES = ["Bapak", "Ibu", "Pak", "Bu", "Sdr.", "Sdri.", "H.", "Hj.", "dr.", "Ir."]
ORGS = ["PT Maju Jaya Abadi", "CV Sumber Rejeki", "PT Sinar Mas Sentosa", "PT Karya Nusantara Teknik",
        "Koperasi Sejahtera Bersama", "PT Global Logistik Indonesia", "CV Berkah Mandiri", "PT Cahaya Timur Lestari",
        "Yayasan Harapan Bangsa", "PT Prima Data Solusi", "Klinik Sehat Sentosa", "PT Agro Lestari Makmur"]
STREETS = ["Kenanga", "Melati", "Sudirman", "Gatot Subroto", "Diponegoro", "Ahmad Yani", "Pahlawan", "Merdeka",
           "Raya Bogor", "Cempaka Putih", "Mawar", "Kebon Jeruk", "Veteran", "Imam Bonjol", "Pemuda", "Gajah Mada"]
KEL = ["Menteng", "Cikini", "Sukajadi", "Lowokwaru", "Tegalsari", "Bojongsari", "Karang Anyar", "Sumur Batu"]
KEC = ["Menteng", "Coblong", "Lowokwaru", "Genteng", "Cilandak", "Tebet", "Sukasari", "Banjarsari"]
CITY = ["Jakarta Selatan", "Bandung", "Surabaya", "Semarang", "Malang", "Medan", "Makassar", "Depok", "Bekasi",
        "Tangerang", "Yogyakarta", "Denpasar", "Palembang", "Bogor"]
BANKS = ["BCA", "Mandiri", "BRI", "BNI", "CIMB Niaga", "Permata", "BSI", "BTN"]
DOMAINS = ["gmail.com", "yahoo.co.id", "yahoo.com", "outlook.com", "majujaya.co.id", "kantor.go.id", "mail.com"]


def d(n):
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def nik_digits(female=None):
    female = rng.random() < 0.5 if female is None else female
    day = rng.randint(1, 28) + (40 if female else 0)
    return (f"{rng.choice(PROVINCES):02d}{rng.randint(1, 79):02d}{rng.randint(1, 40):02d}"
            f"{day:02d}{rng.randint(1, 12):02d}{rng.randint(60, 99):02d}{rng.randint(1, 999):04d}")


def group(s, sizes, sep):
    out, i = [], 0
    for n in sizes:
        out.append(s[i:i + n])
        i += n
    return sep.join(out)


def nik():
    x = nik_digits()
    return rng.choice([x, x, x, group(x, (4, 4, 4, 4), " "), group(x, (6, 6, 4), "-"), group(x, (4, 4, 4, 4), ".")])


def npwp():
    old = f"{d(2)}.{d(3)}.{d(3)}.{d(1)}-{d(3)}.{d(3)}"
    return rng.choice([old, old, re.sub(r"\D", "", old), "0" + re.sub(r"\D", "", old), nik_digits()])


def kk():
    return nik_digits()


def bpjs():
    return rng.choice(["000" + d(10), "0001" + d(9), d(11)])


def sim():
    return d(rng.choice([12, 12, 13, 14]))


def passport():
    return rng.choice("ABCEX") + rng.choice(["", " "]) + d(7)


def bank():
    n = rng.choice([10, 10, 13, 15, 12])
    x = d(n)
    return rng.choice([x, x, group(x, (3, 3, n - 6), "-"), group(x, (4, 4, n - 8), " ")])


def card():
    prefix = rng.choice(["4", "51", "52", "53", "54", "55"])
    body = prefix + d(15 - len(prefix))
    total = 0
    for i, ch in enumerate(reversed(body)):
        n = int(ch)
        if i % 2 == 0:
            n *= 2
            n = n - 9 if n > 9 else n
        total += n
    x = body + str((10 - total % 10) % 10)
    return rng.choice([group(x, (4, 4, 4, 4), " "), group(x, (4, 4, 4, 4), "-"), x])


def phone():
    rest = "8" + rng.choice("1235789") + d(rng.choice([8, 9, 10]))
    n = len(rest)
    return rng.choice([
        "0" + rest, "0" + group(rest, (3, 4, n - 7), "-"), "0" + group(rest, (3, 4, n - 7), " "),
        "+62" + rest, "+62 " + group(rest, (3, 4, n - 7), " "), "62" + rest, "0" + group(rest, (3, 4, n - 7), "."),
    ])


def person():
    style = rng.random()
    if style < 0.15:
        return rng.choice(SINGLE)
    if style < 0.25:
        return rng.choice(FIRST)
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def email(name=None):
    name = (name or person()).lower()
    local = rng.choice([name.replace(" ", "."), name.replace(" ", "_"), name.split()[0] + d(2), name.replace(" ", "")])
    return f"{local}@{rng.choice(DOMAINS)}"


def address():
    street = f"{rng.choice(['Jl.', 'Jalan', 'Jl'])} {rng.choice(STREETS)} No. {rng.randint(1, 250)}"
    parts = [street]
    if rng.random() < 0.7:
        parts.append(f"RT {rng.randint(1, 15):03d}/RW {rng.randint(1, 12):03d}")
    if rng.random() < 0.6:
        parts.append(f"{rng.choice(['Kel.', 'Kelurahan'])} {rng.choice(KEL)}")
    if rng.random() < 0.6:
        parts.append(f"{rng.choice(['Kec.', 'Kecamatan'])} {rng.choice(KEC)}")
    if rng.random() < 0.7:
        parts.append(rng.choice(CITY))
    if rng.random() < 0.4:
        parts.append(d(5))
    if rng.random() < 0.15:
        return f"Perumahan Griya {rng.choice(['Asri', 'Indah', 'Permai'])} Blok {rng.choice('ABCD')}{rng.randint(1, 20)} No. {rng.randint(1, 40)}, {rng.choice(CITY)}"
    return ", ".join(parts)


def ip():
    return rng.choice([f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
                       f"192.168.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
                       f"172.{rng.randint(16, 31)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"])


def host():
    return f"{rng.choice(['db', 'api', 'srv', 'app', 'files', 'vpn', 'erp'])}{rng.randint(1, 9):02d}.{rng.choice(['corp', 'internal', 'local', 'lan'])}"


def secret():
    return rng.choice([
        "AKIA" + "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567") for _ in range(16)),
        "ghp_" + "".join(rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(36)),
        "sk-proj-" + "".join(rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(40)),
    ])


def org():
    return rng.choice(ORGS)


VALUES = {"NIK": nik, "NPWP": npwp, "KK": kk, "BPJS": bpjs, "SIM": sim, "PASSPORT": passport, "BANK": bank,
          "CARD": card, "PHONE": phone, "EMAIL": email, "ADDRESS": address, "IP": ip, "HOST": host,
          "SECRET": secret, "PERSON": person, "ORG": org}

# Hard negatives: look like data, aren't personal.
NEG = {
    "INVOICE": lambda: f"INV/{rng.randint(2024, 2026)}/{rng.randint(1, 12):02d}/{d(5)}",
    "ORDER": lambda: "TRX" + d(14),
    "WAYBILL": lambda: rng.choice(["JNE ", "J&T ", "SiCepat "]) + d(rng.choice([12, 13])),
    "AMOUNT": lambda: f"Rp {rng.randint(1, 999)}.{d(3)}.{d(3)}",
    "DATE": lambda: f"{rng.randint(1, 28):02d}-{rng.randint(1, 12):02d}-{rng.randint(2024, 2026)}",
    "VERSION": lambda: f"{rng.randint(1, 14)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}",
    "PUBLIC": lambda: rng.choice(["Bank Indonesia", "OJK", "Kementerian Keuangan", "BPJS Kesehatan pusat", "Dukcapil"]),
    "PRODUCT": lambda: f"SKU-{d(6)}",
}

# ---- templates ----------------------------------------------------------------------------------
# {TYPE} is a labelled value; {TITLE} a title (not labelled: "Bapak" isn't the name); {NEG:X} a
# hard negative; {ANY} picks one of BANKS as plain text.

TEMPLATES = {
    "chat": [
        "{TITLE} {PERSON}, ini NIK saya {NIK} buat daftar ya",
        "kak tolong transfer ke norek {BANK} a.n. {PERSON}, makasih 🙏",
        "nomor wa aku {PHONE}, chat aja kalau paketnya udah sampe",
        "bro ktp gw {NIK}, alamat {ADDRESS}",
        "udah aku kirim ke email {EMAIL} ya, cek spam juga",
        "rek {ANY} {BANK} atas nama {PERSON}, totalnya {NEG:AMOUNT}",
        "resi {NEG:WAYBILL} udah dikirim tgl {NEG:DATE}, hp kurir {PHONE}",
        "minta tolong cek pesanan {NEG:ORDER} atas nama {PERSON}",
        "Pak {PERSON} hp nya {PHONE} ya, rumahnya di {ADDRESS}",
        "bpjs aku {BPJS} masih aktif ga sih",
    ],
    "customer_service": [
        "Agent: Boleh disebutkan nomor KTP Bapak?\nCustomer: {NIK}\nAgent: Terima kasih, atas nama {PERSON}?",
        "Nasabah {PERSON} melaporkan kartu {CARD} terdebet dua kali sebesar {NEG:AMOUNT} pada {NEG:DATE}.",
        "Pelanggan atas nama {PERSON}, no HP {PHONE}, email {EMAIL}, komplain tagihan {NEG:INVOICE}.",
        "Customer {PERSON} requested a refund to account {BANK} ({ANY}). Ticket ref {NEG:ORDER}.",
        "Mohon verifikasi: nama {PERSON}, NIK {NIK}, nomor rekening {ANY} {BANK}.",
        "Penelepon mengaku sebagai {PERSON} dari {ORG}, minta reset password akun.",
        "Kartu kredit nasabah {CARD} diblokir sementara atas permintaan {TITLE} {PERSON}.",
        "Pemegang polis {PERSON} (paspor {PASSPORT}) mengajukan klaim rawat inap.",
    ],
    "contract": [
        "Yang bertanda tangan di bawah ini, {PERSON}, NIK {NIK}, beralamat di {ADDRESS}, bertindak untuk dan atas nama {ORG}.",
        "PIHAK KEDUA: {ORG}, NPWP {NPWP}, diwakili oleh {TITLE} {PERSON} selaku Direktur.",
        "Pembayaran dilakukan ke rekening {ANY} nomor {BANK} atas nama {ORG} paling lambat {NEG:DATE}.",
        "Penyewa: {PERSON}, pemegang SIM A nomor {SIM}, berdomisili di {ADDRESS}.",
        "Nilai kontrak sebesar {NEG:AMOUNT} sebagaimana tercantum dalam invoice {NEG:INVOICE}.",
        "Surat kuasa ini diberikan oleh {PERSON} (NIK {NIK}, KK {KK}) kepada {PERSON}.",
    ],
    "hr": [
        "Data karyawan baru: {PERSON}, NIK {NIK}, NPWP {NPWP}, BPJS Ketenagakerjaan {BPJS}, rek gaji {ANY} {BANK}.",
        "Hi team, {PERSON} will join {ORG} next month. Personal email: {EMAIL}, phone {PHONE}.",
        "Mohon update alamat {PERSON} menjadi {ADDRESS}.",
        "Slip gaji {PERSON} untuk periode {NEG:DATE} dikirim ke {EMAIL}.",
        "Kandidat {PERSON} melampirkan paspor {PASSPORT} dan Kartu Keluarga nomor {KK}.",
        "Reimbursement {NEG:AMOUNT} untuk {PERSON}, ditransfer ke rekening {BANK}.",
    ],
    "form": [
        "Nama Lengkap: {PERSON}\nNIK: {NIK}\nNo. KK: {KK}\nAlamat: {ADDRESS}\nNo. HP: {PHONE}\nEmail: {EMAIL}",
        "Nama: {PERSON}\nNPWP: {NPWP}\nNo. Rekening: {BANK} ({ANY})\nNo. BPJS Kesehatan: {BPJS}",
        "Nama pemohon: {PERSON}\nNomor SIM: {SIM}\nNomor Paspor: {PASSPORT}\nTelepon: {PHONE}",
        "Full name: {PERSON}\nID card (KTP): {NIK}\nTax ID (NPWP): {NPWP}\nMobile: {PHONE}",
    ],
    "it_log": [
        "ERROR connect to {HOST} ({IP}) failed after 3 retries, user={EMAIL}",
        "deploy v{NEG:VERSION} to {HOST} using key {SECRET}",
        "export AWS_ACCESS_KEY_ID={SECRET} && ssh admin@{IP}",
        "ticket: printer di {IP} offline, laporan dari {PERSON} ext {NEG:PRODUCT}",
        "GITHUB_TOKEN={SECRET} dipakai di pipeline build {NEG:ORDER}",
    ],
    "ai_prompt": [
        "Tolong buatkan surat peringatan untuk karyawan bernama {PERSON} (NIK {NIK}) yang tinggal di {ADDRESS}.",
        "Summarize this complaint from {PERSON}, card {CARD}, phone {PHONE}, about invoice {NEG:INVOICE}.",
        "Ringkas kontrak antara {ORG} dan {ORG}, nilai {NEG:AMOUNT}, ditandatangani {PERSON}.",
        "Bantu balas email {PERSON} <{EMAIL}> yang minta refund ke rekening {ANY} {BANK}.",
        "Analisis data nasabah: {PERSON}, {NIK}, {PHONE}; {PERSON}, {NIK}, {PHONE}.",
        "Cek apakah NPWP {NPWP} milik {ORG} sudah benar formatnya.",
    ],
    "clean": [
        "Tolong buatkan ringkasan rapat evaluasi kuartal tiga dalam lima poin.",
        "Laporan {NEG:PUBLIC} menyebut inflasi turun pada {NEG:DATE}.",
        "Invoice {NEG:INVOICE} sebesar {NEG:AMOUNT} sudah dibayar, pesanan {NEG:ORDER} dikirim via {NEG:WAYBILL}.",
        "Update aplikasi ke versi {NEG:VERSION}, produk {NEG:PRODUCT} habis stok.",
        "Write a Python function that validates an Indonesian phone number format.",
        "Kebijakan {NEG:PUBLIC} terbaru soal suku bunga berlaku mulai {NEG:DATE}.",
        "Simulasi cicilan 12 bulan untuk {NEG:AMOUNT} dengan bunga 1 persen.",
        "Tolong jelaskan perbedaan SIM A dan SIM C untuk pelajar.",
    ],
}

SLOT = re.compile(r"\{(NEG:[A-Z]+|[A-Z]+)\}")


def fill(template: str) -> tuple[str, list[dict]]:
    out, spans, pos, last = [], [], 0, 0
    for m in SLOT.finditer(template):
        out.append(template[last:m.start()])
        pos += m.start() - last
        key = m.group(1)
        if key == "TITLE":
            value = rng.choice(TITLES)
        elif key == "ANY":
            value = rng.choice(BANKS)
        elif key.startswith("NEG:"):
            value = NEG[key[4:]]()
        else:
            value = VALUES[key]()
            spans.append({"start": pos, "end": pos + len(value), "type": key})
        out.append(value)
        pos += len(value)
        last = m.end()
    out.append(template[last:])
    return "".join(out), spans


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for genre, templates in TEMPLATES.items():
        for t_i, template in enumerate(templates):
            for k in range(7):
                text, spans = fill(template)
                for s in spans:            # every label must point at what it says
                    assert text[s["start"]:s["end"]], s
                rows.append({"id": f"{genre}-{t_i:02d}-{k}", "genre": genre, "text": text, "spans": spans})
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for r in rows:
        for s in r["spans"]:
            counts[s["type"]] = counts.get(s["type"], 0) + 1
    print(f"{len(rows)} texts, {sum(counts.values())} labelled spans -> {OUT}")
    print(dict(sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
