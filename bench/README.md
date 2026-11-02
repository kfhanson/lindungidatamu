# Benchmark

How well lindungidatamu hides Indonesian personal data, measured on 371 texts with 812 labelled
items: chats, customer service transcripts, contracts, HR messages, forms, IT logs, AI prompts,
and clean texts with hard negatives (invoice and order numbers, waybills, amounts, dates, public
institutions).

```bash
uv run python bench/generate.py                         # rebuilds data/id-pii-v1.jsonl (seeded: identical)
uv run python bench/run.py                              # patterns only
uv run --extra names python bench/run.py --names        # + the GLiNER name model
```

## What the numbers mean

- **Protected:** the share of labelled items at least 90% hidden by anything lindungidatamu found.
  For masking this is the number that matters: whatever isn't protected reaches the AI model.
- **Strict:** the right type and the exact span.
- **Over-masked:** things hidden that aren't personal data (an invoice number). No privacy cost, but
  the model loses context.

## Results

Two sets: **v1** (371 texts, 812 items) and **v2** (280 texts, 640 items, new genres and templates,
written before any fix). "Before" is the first version of the patterns; "after" is 2026-10-31.

| Set · system | Protected (all) | Identifiers and numbers | Addresses | Over-masked items (texts) | ms per text |
|---|---:|---:|---:|---:|---:|
| v1 · patterns, before | 51.4% | 81.6% | 4% | 1 (1) | 0.2 |
| v1 · patterns, after | 62.9% | **100%** | 100% | 1 (1) | 0.2 |
| v1 · patterns + GLiNER, before | 87.9% | 81.6% | 4% | 72 (60) | 127 |
| v1 · patterns + GLiNER, after | **99.5%** | **100%** | 100% | 1 (1) | 201 |
| v2 · patterns, before | 39.8% | 66.4% | 1% | 0 (0) | 0.1 |
| v2 · patterns, after | 60.0% | **100%** | 100% | 0 (0) | 0.3 |
| v2 · patterns + GLiNER, before | 79.4% | 66.4% | 1% | 129 (97) | 156 |
| v2 · patterns + GLiNER, after | **99.5%** | **100%** | 100% | 15 (15) | 108 |

Over-masking on v2 rose from 7 to 15 after the real-text work: in an all-lowercase chat the layer
no longer drops the model's lowercase "organisations", because on real lowercase news that rule
was unmasking actual people. Protecting comes first.

"Protected (all)" includes names and organisations, which patterns alone can't find (about 40% of
the items). With GLiNER, people are 98-99% protected (the misses are single Javanese names such as
Mulyono or Paijo) and organisations 100%. Times are a laptop CPU.

What changed between before and after:
- **Addresses** read on past the street: RT/RW, Kel., Desa, Dusun, Kec., Kab., Kota, postcode, and a
  city (a known one wherever it sits, any other only between commas).
- **Phones** in dotted and `+62 878 7675 911` forms; **bank accounts, BPJS and NPWP** with words
  between the label and the number (`rek gaji Mandiri …`, `bpjs aku …`, `Tax ID (NPWP): …`), and an
  account number straight after a bank's name (`ditransfer ke BCA 3376…`).
- **The name model's mistakes**, through an Indonesian list (`src/lindungidatamu/public.json`): forms
  of address and pronouns are never people (kak, bro, sy, Agent, Ketua RT); titles come off names
  (`Bapak [PERSON_1]`); public banks, payment apps, couriers, platforms and government bodies are
  never masked as organisations; lowercase phrases aren't organisations.

## Real Indonesian text: names (IndoNLU NERGrit and NERP)

The synthetic sets can't say how names fare in writing we didn't produce, so `bench/real_ner.py`
scores people and organisations on two public, labelled Indonesian test sets from
[IndoNLU](https://github.com/IndoNLP/indonlu) (Apache-2.0): **NERGrit** (209 encyclopedic sentences)
and **NERP** (840 news sentences, all lowercase). The files are fetched on first use and not kept
in this repo. Rules were tuned on the IndoNLU *validation* sets only; the *test* sets were scored
once, afterwards (2026-10-31):

| Test set | System | People protected | Organisations protected | Over-masked |
|---|---|---:|---:|---:|
| NERGrit | GLiNER alone | 93% (193/207) | 85% (106/124) | 102 |
| NERGrit | lindungidatamu | 93% (193/207) | 83% (103/124) | **69** |
| NERP | GLiNER alone | 90% (562/624) | – | 615 |
| NERP | lindungidatamu | 90% (561/624) | – | **340** |

- **People: about 90-93% on real text**, against 98-99% on the synthetic sets. The real number is
  the one to quote. All of it comes from the name model; the Indonesian layer keeps it and removes
  a third to a half of the false alarms (pronouns such as dia, ia, saya; roles such as presiden,
  korban; public banks, parties and ministries).
- **Organisations 83%** with lindungidatamu: the gap to the model alone is public bodies left
  unmasked on purpose (ministries, courts, parties are not personal data).
- NERP labels no organisations, so every organisation masked there counts as over-masked even when
  it's a real one (FIFA, Uni Eropa): that column overstates the problem for NERP.
- What's still over-masked is mostly ordinary nouns the model calls organisations in lowercase text
  (negara, gereja, cafe, angkot). Masking them costs the AI some context, not privacy.

## Limits, honestly

- **Synthetic and written by us.** Real messages are messier (typos, OCR noise, numbers split
  across lines). Treat these numbers as an upper bound until there's a set from real (consented,
  anonymised) data.
- **Both sets are now partly tuned on.** v1's misses shaped the fixes, and v2 was scored before
  them, but two of the fixes (known cities, a bank name before an account number) and the
  organisation rules came from v2's misses. 100% here doesn't mean 100% on real messages.
- **Names now have a real-text number** (above). Identifiers still don't: real messages with NIKs,
  account numbers and addresses can't be published, so that number needs a consented, anonymised
  set from real users.
- Values follow real formats but are random: NIKs pass the province and date checks, cards pass
  Luhn, nothing belongs to a real person.
