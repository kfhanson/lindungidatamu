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

Since 2026-11-03 health facilities stay visible (see below), so on the synthetic sets with the name
model protected is 99.1% (v1 and v2): the difference is "Klinik Sehat Sentosa", which those sets
label as an organisation to mask. Patterns alone and the real-text name numbers are unchanged.

## Install

Python 3.10 or newer (macOS ships 3.9: use `uv` or Homebrew Python).

```bash
pip install lindungidatamu                 # patterns and masking, no dependencies
pip install "lindungidatamu[names]"        # + names, organisations, addresses (GLiNER, runs locally)
pip install "lindungidatamu[presidio]"     # + Indonesian recognisers for Microsoft Presidio
```

## AI clients, logs and pandas

```python
from openai import OpenAI                     # also AsyncOpenAI, Anthropic, AsyncAnthropic
client = ld.wrap(OpenAI())                    # one wrapper = one conversation
r = client.chat.completions.create(model="gpt-4.1", messages=[{"role": "user", "content": text}])
r.choices[0].message.content                  # the model saw [NIK_1]; you get the real value back
```

`wrap` masks every request (messages, system prompts, tool results) and restores every answer, tool
call arguments and streams included (`stream=True`; a placeholder cut across two chunks is held back
until it is whole). The same value keeps its placeholder for the whole conversation, and a value the
model has seen once is masked wherever it appears again, also in earlier answers you send back.
Covered: OpenAI chat completions and responses, Anthropic messages, embeddings. Images, audio and
uploaded files are not masked; the SDKs' `.stream()` helpers and `with_raw_response` raise instead
of sending text unmasked. Add `names=GlinerNames()` for people and organisations.

```python
handler.addFilter(ld.LogFilter())             # log lines, arguments and tracebacks, patterns only
done = ld.mask_frame(df)                      # table mode on a DataFrame: done.frame, done.report
```

## Tables and files

```bash
pip install "lindungidatamu[files,names]"
lindungidatamu scan claims.parquet --names     # claims.policy.json: one action per column, and why
lindungidatamu mask claims.parquet --names     # claims.masked.parquet + a report (counts only)
lindungidatamu review claims.parquet --names   # claims.review.html: check it by eye, offline
```

A table's personal data is mostly in identifier columns, labelled by the column name rather than a
word before each number. So `scan` decides per column, from its name, a sample of its values and their
shape, and writes the decision down for you to check:

| Action | For | Example |
|---|---|---|
| `tokenize` | identifier and person columns: every value becomes a placeholder, no scanning | `member_no` → `[MEMBER_12]` |
| `scan_text` | mixed text: each distinct value is masked by patterns and the name model | `provider_name`, `notes` |
| `scan_patterns` | categories, codes, amounts as text: patterns only | `diagnosis`, `plan` |
| `keep` | numbers, dates, true/false | `amount_paid` |
| `drop` | remove the column (set it yourself) | |

Columns of one type share placeholders (`patient_member_no` and `emp_member_no` are both `MEMBER`),
so joins survive. Every text column is pattern-scanned whatever its action. After masking, a residual
check looks for identifiers and pattern matches still visible; if it finds any, nothing is written.

**One-way by default.** Without `--vault`, the values behind the placeholders are not kept anywhere.
With `--vault claims.vault.json` the masking is reversible: analyse the masked data with an AI model,
then `lindungidatamu restore findings.md --vault claims.vault.json` puts the real values back (exact
placeholders only). The vault holds every real value: keep it private, never send it with the data.

The same in Python, without pandas: `ld.table.profile(columns)` and `ld.table.mask_table(columns, policy)`,
where `columns` is a dict of column name → list of values.

## Re-identification risk

Masking takes out names and numbers, but the columns that stay can still single someone out together:
a woman born on 14 March 1987, living in one kelurahan, admitted on 3 February, is probably the only
one in the table. `risk` measures that and finds the smallest change that fixes it:

```bash
lindungidatamu risk claims.parquet                          # measure, and write the least fix into the policy
lindungidatamu risk claims.parquet --keep "birth_date=5 years"   # what the analysis needs; others give way
lindungidatamu mask claims.parquet                          # applies it, with the risk before and after in the report
```

```
Now:   k = 1; 99.9% of rows unique; 100.0% in groups of fewer than 5 people; average risk 0.999
       unique rows without each column: birth_date 11.4%, admitted 82.8%, kelurahan 98.9%, sex 99.8%

| Column     | Hierarchy | Publish as | Information kept |
| birth_date | date      | 5 years    | 26.0%            |
| sex        | sex       | value      | 100.0%           |
| kelurahan  | wilayah   | kabupaten  | 58.1%            |
| admitted   | date      | month      | 42.1%            |

After: k = 11; 0.0% of rows unique; 0.0% in groups of fewer than 5 people; average risk 0.016
```

(A made-up table of 196,793 claims; real tables vary.)

- **Quasi-identifiers** are found by `scan` and marked in the policy (`"role": "quasi"`): birth dates
  and ages, sex, area (kelurahan to provinsi, by Kemendagri code or name), postal codes, event dates. A
  diagnosis is marked `sensitive`.
- **k** is the smallest group of people sharing those columns (k-anonymity): an attacker who knows
  them about someone can't narrow it below 1 in k. Groups count people, not rows, when a member or
  patient column says whose claims are whose.
- **The fix** steps columns up their hierarchy (date → month → year → 5 years; kelurahan →
  kecamatan → kabupaten → provinsi; ICD-10 → category → chapter), searching every combination for
  the one that reaches k (default 5) losing the least information. At most 1% of rows
  (`--max-suppress`) may instead have those columns replaced by `*`. With many columns (more than
  3,000 combinations) it searches a beam of the most promising plans and then improves them by
  local moves; on test tables small enough to check every combination it finds the best plan or
  one within about 1%.
- **Dates become intervals.** With two or more dates, one is kept as the anchor (the event the others
  follow: admission, not discharge or policy start) and the others are published as whole years to
  it (birth date → age at admission) or days from it (discharge → length of stay). That is what most
  analyses need, and it identifies far less than calendar dates, so less has to give way.
  `--no-intervals` keeps them as dates; a column named in `--keep` stays a date.
- **l** reports, for each sensitive column, the fewest different values in any group: l = 1 means a
  group all shares one diagnosis, which you learn without knowing the row. `--l 2` makes the fix
  reach that too (generalising further, or suppressing those groups within the budget).

- **Large amounts** (claim paid, gaji, saldo, total belanja) single people out at the top: one Rp 2
  billion claim, one director's salary. The highest 0.5% of an amount column (`--top-code`, at least
  k people) are replaced by their mean, so totals and averages still add up.
- **Dates and ages inside text** ("laki-laki 45 tahun, dirawat 3 Feb 2024") are generalised when
  masking: a date to its year, an age to a 10-year band ("40-49 tahun"). Durations ("bekerja 5
  tahun", "3 tahun lalu") stay.
- **Area and postal codes** are treated as quasi-identifiers, not ID numbers to tokenize.
- **Area names** generalise like codes: "Kec. Beji" becomes "Kota Depok", then "Jawa Barat".
  Spellings ("Kab.", "Kota Adm.", "Jabar", "DKI") are matched against the Kemendagri 2025 list of
  provinsi, kabupaten/kota and kecamatan. A name used in more than one place (Bogor, Cipayung) is
  settled by "name, parent", by the column's level, or by where most of the column is. One that
  can't be settled is published no finer than what its places share, and never guessed.
- `review` shows all of this: generalised cells in the Rows tab, and a Re-identification risk tab with
  the measures over the whole table (not the sample) and examples per column.
- `--keep` accepts the C-numbers `--hide-columns` prints (`--keep "C13=month"`).

It works on any table with these columns, not only claims: employees, customers, patients, survey
answers.

In Python: `r = ld.risk(df, keep={"birth_date": "5 years"}, l=2)`, then `ld.mask_frame(df, r.policy, k=r.k, l=r.l)`.

## Safe AI analysis

Let an AI model analyse a table without seeing it. The model asks for counts, sums and means,
grouped and filtered; only answers that passed a privacy gate go back. Rows, identifier columns and
free text never leave your machine.

```bash
lindungidatamu serve claims.parquet       # an MCP server over stdio, for Claude Code, Claude Desktop, any MCP client
claude mcp add claims -- lindungidatamu serve /data/claims.parquet
lindungidatamu ask claims.parquet '{"group_by": [{"column": "admitted", "by": "month"}], "measures": ["people", {"mean": "paid"}]}'
```

```python
a = ld.Analysis(df, policy="claims.policy.json", state="claims.analysis")
a.describe()      # the columns that can be asked about, and how
a.aggregate(group_by=["sex", {"column": "kelurahan", "by": "kabupaten"}],
            measures=["people", {"mean": "amount_paid"}],
            where=[{"column": "diagnosis", "in": ["E11.9", "I10"]}])
```

The gate, on every answer:

- **Small groups** of fewer than k people (default 5, or the policy's risk k) are left out.
- **Dominated values**: a sum or mean that one person makes more than half of, or two people more
  than 85%, is withheld.
- **Noise**: counts move by up to 2, sums by a fraction of the largest person's share, and both are
  rounded to 3 significant figures. The noise is fixed by who is in the group, so asking again, or
  asking the same thing in other words, gives the same answer: it can't be averaged away.
- **Differencing**: two groups that differ by fewer than k people get the same answer, so "everyone
  in Cinere" minus "everyone in Cinere but her one claim" gives nothing. This holds whichever is
  asked first, and also when one of the two would be too small to show.
- **Identifiers** (tokenized columns, the person column) and free text can't be asked about.
  Numbers are grouped only in bins, dates by day to year, and area and ICD-10 codes by their levels.
  One answer has at most 500 groups.
- **State**: the folder (`FILE.analysis/`) keeps the noise secret, the groups already answered
  and a log of every question, refusals included. Keep it private: whoever has the secret can
  remove the noise.

This is statistical disclosure control, the kind national statistics offices use for table
builders. It is not differential privacy. An attacker who knows a lot about someone and asks many
cleverly combined questions can still learn something. The gate makes that slow, noisy and visible
in the log.

## What it finds

| Type | Example | Found when |
|---|---|---|
| `NIK` | `3174091203880004` | 16 digits with a valid province and birth date, or any number labelled NIK/KTP |
| `NPWP` | `01.234.567.8-901.000`, `NPWP 0012345678901000` | the dotted 15-digit format, or a number after the label NPWP (15 or 16 digits) |
| `KK`, `BPJS`, `SIM`, `PASSPORT` | `KK 3174…`, `BPJS Kesehatan 0001234567890`, `passport number (MH6653645)` | after their label; any format of code after a clear label (nomor KTP, SIM, paspor, social security, tax ID) |
| `BANK` | `no. rek. BCA 1234567890`, `reknya 26055704` | 8-17 digits after a label (rek., rekening, reknya, account no., a/c) |
| `CARD` | `4111 …`, `CC 5243699577738726` | issuer prefix and Luhn check (public test cards excluded), or any 13-19 digits after CC / kartu kredit / card number |
| `PHONE`, `EMAIL` | `0812-3456-7890`, `+62 21 5550 1234`, `budi@contoh.co.id` | Indonesian mobile formats; any number after hp / telepon / hubungi di, or starting with +; emails |
| `DOB` | `Tanggal Lahir: Semarang, 21/11/1996`, `lahir 5 Maret 2004` | a date after tanggal lahir, TTL, lahir, date of birth, DOB |
| `ADDRESS` | `Jl. Kenanga No. 7 RT 03/RW 05`, `Gang Mawar VII/41, Kel. Sukapada` | street cues (Jl., Jalan, Gang, Perumahan), kelurahan/desa, the line after Alamat:, warga / beralamat …, kode pos |
| `IP`, `HOST` | `10.0.0.5`, `db01.corp` | private addresses, internal server names |
| `SECRET` | API keys, private keys, passwords | known key formats and `password = …` |
| `ORG` | `KAP Budi Santoso, Hamid & Rekan` | a firm named after its partners, whole (no model needed) |
| `PERSON`, `ORG`, `ADDRESS` | `Budi Santoso` | with a name model (`lindungidatamu[names]`) |

Numbers are read through common OCR and typing slips (`3l75Ol29l2Ol8756`: O for 0, l for 1), and a masked tail
(`0812345678XXXX`) doesn't stop a number being found. Every pattern has test cases, including near misses it
must not match, in `tests/fixtures.json`.

With the name model, a few things are deliberately **not** masked as people:
- **Health facilities and their namesakes:** `RS Hermina Bekasi`, `RSUD Dr. Soetomo`, `Siloam Hospitals`.
  Indonesian hospitals are often named after people. A doctor *next to* a hospital (`RS Hermina, dr. Budi`)
  and a doctor's own practice (`Klinik dr. Budi`) are still masked. Turn the facility rule off with `public=False`.
- **Diseases named after people:** `G20-PARKINSON`, `Crohn disease`, `sindrom Down`.

All-caps text (`BAMBANG SUTRISNO`, common in Indonesian records) goes to the name model in title case, which
it reads far better. The model loads from your local cache without contacting the internet once it is
downloaded, and its version is pinned.

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

# A dataset: keep the vault for later, in a file only you can read (mode 0600)
vault = ld.Vault(merge_partial=False)   # in a table, "Budi" and "Budi Santoso" may be two people
...                                     # mask every value with vault=vault
vault.save("claims.vault.json")
ld.Vault.load("claims.vault.json").restore(report, strict=True)
# strict: only exact placeholders like [PERSON_12]. Without it, rewritten forms ("Person 12")
# are restored too: right for a chat answer, risky in a report ("member 812" is just words).

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

## Free and Pro

This library is free and stays complete on its own: everything needed to find, mask and restore
personal data safely is here, and nothing that keeps your data safe will move behind a paywall.
lindungidatamu Pro adds judgement, compliance evidence and convenience on top.

| | Free (this repo, Apache-2.0) | Pro |
|---|---|---|
| Finding | Patterns and validators, GLiNER name model (runs locally) | Fine-tuned Indonesian name model (all-caps text, institutions named after people) |
| Masking | `mask` / `restore`, Vault, your own terms and formats | |
| Tables and files | Per-column decisions (identifier, person, text, category), whole-column tokens, residual check, policy file, command line, review page | Decision model per column: UU PDP specific data (health, children, finance, criminal records, biometric, genetic) |
| Messages | | Decision model per message: UU PDP specific data, still identifiable after masking, confidential business information |
| Reports | Masking report: counts per column and type, versions | DPIA-ready report mapped to UU PDP articles, over-masking score |
| Integrations | `wrap` for OpenAI and Anthropic clients, `LogFilter`, `mask_frame`, Presidio add-on; *planned:* LangChain and LiteLLM hooks, self-hosted OpenAI-compatible proxy | *Planned:* proxy policy, audit log, admin view and SSO for teams; hosted API |

The free version uses no decision model: it finds identifiers and names, not facts. A diagnosis
next to a masked name stays visible; telling you that it is UU PDP health data is Pro's job.

## License

Apache-2.0. The area list (`wilayah.tsv.gz`, Kepmendagri No 300.2.2-2138 Tahun 2025) comes from
[cahyadsn/wilayah](https://github.com/cahyadsn/wilayah), MIT: see `src/lindungidatamu/wilayah-LICENSE.txt`.
`scripts/build_wilayah.py` rebuilds it.
