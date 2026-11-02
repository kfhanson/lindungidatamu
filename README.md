# lindungidatamu

Find and mask Indonesian personal data before text reaches an AI model, a log or a dataset:
NIK, NPWP (old and new format), KK, BPJS, SIM, passports, bank accounts, cards, phone numbers,
emails, Indonesian addresses, secrets, and (with a name model) names and organisations.

```python
import lindungidatamu as ld

m = ld.mask("Kirim kontrak ke pelanggan, NIK 3174091203880004, rek. BCA 1234567890")
m.text
# 'Kirim kontrak ke pelanggan, NIK [NIK_1], rek. BCA [BANK_1]'

answer = call_your_llm(m.text)          # the model only sees placeholders
m.restore(answer)                       # real values back, on your side
```

> **Status: early development (0.1.0.dev0).** The API may change before 0.1.0.

**Bahasa Indonesia.** lindungidatamu menemukan dan menyamarkan data pribadi Indonesia (NIK, NPWP,
KK, BPJS, SIM, paspor, rekening, alamat, nama) sebelum teks dikirim ke model AI, log, atau dataset,
lalu mengembalikan nilai aslinya di jawaban. Berjalan di server Anda sendiri; tidak ada data yang
dikirim ke mana pun.

## Why

Under UU PDP (Law 27/2022), sending customers' personal data to an AI service is processing that
the company answers for. Generic tools are built around US and EU data: they don't know a NIK
from a phone number, or that an NPWP has had 16 digits since 2024. lindungidatamu does, checks what
it finds (a NIK's province and birth date, a card's check digit), and runs on your machine.

## Accuracy

See [the benchmark](bench/README.md). On real Indonesian text (IndoNLU's NERGrit and NERP test
sets), names of people are protected **90-93%** of the time with the GLiNER name model, and the
Indonesian layer removes a third to a half of the model's false alarms. On our synthetic sets of
chats, forms and documents, patterns protect **100%** of identifiers, numbers and addresses; those
sets are written by us and partly tuned on, so read them as "it handles these formats". A real-data
number for identifiers needs consented data and isn't there yet.

## Install

```bash
pip install lindungidatamu                 # patterns and masking, no dependencies
pip install "lindungidatamu[names]"        # + names, organisations, addresses (GLiNER, runs locally)
pip install "lindungidatamu[presidio]"     # + Indonesian recognisers for Microsoft Presidio
```

## What it finds

| Type | Example | Found when |
|---|---|---|
| `NIK` | `3174091203880004` | 16 digits with a valid province and birth date, or any number labelled NIK/KTP |
| `NPWP` | `01.234.567.8-901.000`, `NPWP 0012345678901000` | the dotted 15-digit format, or a number after the label NPWP (15 or 16 digits) |
| `KK`, `BPJS`, `SIM`, `PASSPORT` | `KK 3174…`, `BPJS Kesehatan 0001234567890` | after their label |
| `BANK` | `no. rek. BCA 1234567890` | after a label (rek., rekening, account no., a/c) |
| `CARD` | `4111 …` | issuer prefix and Luhn check; public test cards excluded |
| `PHONE`, `EMAIL` | `0812-3456-7890`, `budi@contoh.co.id` | Indonesian mobile formats; emails |
| `ADDRESS` | `Jl. Kenanga No. 7 RT 03/RW 05` | Indonesian street address cues |
| `IP`, `HOST` | `10.0.0.5`, `db01.corp` | private addresses, internal server names |
| `SECRET` | API keys, private keys, passwords | known key formats and `password = …` |
| `PERSON`, `ORG`, `ADDRESS` | `Budi Santoso` | with a name model (`lindungidatamu[names]`) |

Every pattern has test cases, including near misses it must not match, in `tests/fixtures.json`.

## More

```python
# A conversation: one vault, consistent placeholders across messages
vault = ld.Vault()
ld.mask("Budi Santoso minta refund", vault=vault, names=finder)   # [PERSON_1]
ld.mask("Tolong balas ke Budi", vault=vault, names=finder)        # [PERSON_1] again

# Your own words and number formats
ld.mask(text, terms=[ld.Term("CLIENT", "PT Sinar Abadi", aliases=("Sinar Abadi",))],
              formats=[ld.Format("POLICY", "POL-########")])      # # digit, @ letter, * either

# Names, organisations and addresses (runs locally; the first use downloads the model)
from lindungidatamu.names import GlinerNames
finder = GlinerNames()
ld.mask("Kontrak Budi Santoso dengan PT Maju Jaya", names=finder, keep=["Bank Indonesia"])

# Just find, without masking
ld.find("NIK 3174091203880004")   # [Entity(type='NIK', start=4, end=20, ...)]
```

### Microsoft Presidio

```python
from presidio_analyzer import AnalyzerEngine
from lindungidatamu.presidio import add_to

analyzer = add_to(AnalyzerEngine())
analyzer.analyze("NIK 3174091203880004", language="en")    # ID_NIK
```

## License

Apache-2.0.
