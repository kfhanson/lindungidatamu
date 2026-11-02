"""Builds the second benchmark set (v2): new genres and templates, written before any fix that came
out of v1, so improvements are measured on text they weren't tuned against.

    uv run python bench/generate_v2.py       # -> bench/data/id-pii-v2.jsonl (seeded)

Same values and labels as v1 (bench/generate.py), plus address shapes v1 never used: gang (Gg.),
village (Desa, Dusun), regency (Kab.), and kompleks/blok forms.
"""

import json
import re
from pathlib import Path

import generate as g

g.rng.seed(20261031)
rng = g.rng
OUT = Path(__file__).parent / "data" / "id-pii-v2.jsonl"

VILLAGES = ["Sukamaju", "Cibodas", "Karangrejo", "Sidomulyo", "Mekarsari", "Tanjungsari", "Wonosari", "Sumberagung"]
DUSUN = ["Krajan", "Kidul", "Lor", "Tengah", "Wetan"]
REGENCIES = ["Bogor", "Sleman", "Bantul", "Sidoarjo", "Gresik", "Garut", "Klaten", "Banyumas", "Gianyar", "Deli Serdang"]


def address_v2():
    shape = rng.random()
    if shape < 0.35:
        return (f"Gg. {rng.choice(['Mawar', 'Melati', 'Anggrek', 'Damai', 'Kenari'])} {rng.choice(['', 'II ', 'III '])}"
                f"No. {rng.randint(1, 60)}, RT {rng.randint(1, 12):02d} RW {rng.randint(1, 9):02d}, {rng.choice(g.CITY)}").replace("  ", " ")
    if shape < 0.65:
        return (f"Dusun {rng.choice(DUSUN)} RT {rng.randint(1, 9):03d}/RW {rng.randint(1, 9):03d}, Desa {rng.choice(VILLAGES)}, "
                f"Kec. {rng.choice(g.KEC)}, Kab. {rng.choice(REGENCIES)}")
    if shape < 0.85:
        return (f"Komplek {rng.choice(['Bumi', 'Taman', 'Graha'])} {rng.choice(['Asri', 'Indah', 'Lestari'])} Blok {rng.choice('ABCDEF')}{rng.randint(1, 12)}"
                f" No. {rng.randint(1, 30)}, {rng.choice(['Kel.', 'Kelurahan'])} {rng.choice(g.KEL)}, {rng.choice(g.CITY)} {g.d(5)}")
    return g.address()


VALUES = dict(g.VALUES, ADDRESS=address_v2)

TEMPLATES = {
    "online_shop": [
        "Halo kak, paketnya kirim ke {ADDRESS} ya, a.n. {PERSON}, hp {PHONE}",
        "Pembeli {PERSON} komplain barang rusak, minta refund ke rekening {ANY} {BANK}",
        "Pesanan {NEG:ORDER} udah dikirim via {NEG:WAYBILL}, total {NEG:AMOUNT}",
        "Seller {ORG}: mohon konfirmasi alamat {ADDRESS} untuk pesanan {NEG:ORDER}",
        "kak sy udh transfer ke {ANY} norek {BANK} an {PERSON}, bukti nyusul",
    ],
    "clinic": [
        "Pasien {PERSON}, lahir {NEG:DATE}, NIK {NIK}, No. BPJS {BPJS}, keluhan demam 3 hari.",
        "Rujukan untuk {TITLE} {PERSON} (kartu BPJS {BPJS}) ke {ORG}, alamat pasien {ADDRESS}.",
        "Hasil lab atas nama {PERSON} dikirim ke {EMAIL}, kontak keluarga {PHONE}.",
        "Jadwal kontrol {PERSON} dipindah ke {NEG:DATE}, mohon konfirmasi ke {PHONE}.",
    ],
    "school": [
        "Formulir PPDB: Nama siswa {PERSON}, NIK {NIK}, No. KK {KK}, alamat {ADDRESS}, nama ayah {PERSON}.",
        "Orang tua {PERSON} (HP {PHONE}) mengajukan keringanan SPP sebesar {NEG:AMOUNT}.",
        "Data guru: {PERSON}, NPWP {NPWP}, rekening gaji {BANK} ({ANY}).",
    ],
    "loan_app": [
        "Pengajuan pinjaman: {PERSON}, NIK {NIK}, NPWP {NPWP}, penghasilan {NEG:AMOUNT}/bulan, rek pencairan {BANK}",
        "Kontak darurat {PERSON}: {PHONE} ({PERSON}, saudara)",
        "Tagihan {NEG:INVOICE} jatuh tempo {NEG:DATE}, bayar ke VA {ANY} atas nama {PERSON}",
        "Verifikasi gagal: foto KTP {NIK} tidak sesuai dengan nama {PERSON}",
    ],
    "rt_letter": [
        "SURAT KETERANGAN DOMISILI\nYang bertanda tangan di bawah ini Ketua RT menerangkan bahwa {PERSON}, NIK {NIK}, KK {KK}, benar berdomisili di {ADDRESS}.",
        "Warga atas nama {PERSON} (NIK {NIK}) pindah ke {ADDRESS} per {NEG:DATE}.",
    ],
    "rental": [
        "Penyewa motor: {PERSON}, SIM C {SIM}, jaminan KTP {NIK}, HP {PHONE}",
        "Sewa kontrakan di {ADDRESS} atas nama {PERSON}, deposit {NEG:AMOUNT} ditransfer ke {ANY} {BANK}",
        "Tenant {PERSON} (passport {PASSPORT}) checked in, contact {EMAIL}",
    ],
    "insurance": [
        "Klaim nomor {NEG:ORDER}: tertanggung {PERSON}, polis atas nama {ORG}, rekening penerima {BANK}",
        "Nasabah {PERSON} dengan kartu {CARD} meminta pembatalan polis, alamat korespondensi {ADDRESS}",
    ],
    "social_dm": [
        "min, akun aku dibajak, email nya {EMAIL}, no hp {PHONE}, tolong dibantu ya",
        "kak aku {PERSON}, yg kemarin pesen, alamatku {ADDRESS}",
        "gan bisa cod ga? ini wa ku {PHONE}",
    ],
    "business_email": [
        "Dear {PERSON},\nPlease find attached the NDA between {ORG} and {ORG}. Send the signed copy to {EMAIL}.",
        "Hi {PERSON}, payment of {NEG:AMOUNT} for invoice {NEG:INVOICE} goes to {ANY} account {BANK}. Call me at {PHONE}.",
        "FYI: the VPN host is {HOST} ({IP}); {PERSON}'s token {SECRET} was rotated today.",
    ],
    "clean_v2": [
        "Promo akhir tahun: diskon 20% untuk produk {NEG:PRODUCT} sampai {NEG:DATE}.",
        "Rapat koordinasi dengan {NEG:PUBLIC} dijadwalkan ulang, bahan presentasi menyusul.",
        "Tolong buatkan caption Instagram untuk launching versi {NEG:VERSION} aplikasi kami.",
        "Berapa biaya kirim JNE dari Bandung ke Surabaya untuk paket 2 kg?",
        "Jelaskan cara mengurus SIM C yang hilang dan dokumen apa saja yang perlu dibawa.",
        "Bandingkan bunga KPR BCA, Mandiri, dan BRI untuk tenor 15 tahun.",
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
            value = rng.choice(g.TITLES)
        elif key == "ANY":
            value = rng.choice(g.BANKS)
        elif key.startswith("NEG:"):
            value = g.NEG[key[4:]]()
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
            for k in range(8):
                text, spans = fill(template)
                rows.append({"id": f"v2-{genre}-{t_i:02d}-{k}", "genre": genre, "text": text, "spans": spans})
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
