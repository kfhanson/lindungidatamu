"""ld.wrap against the real OpenAI and Anthropic SDKs, with the HTTP layer mocked: no network, no keys."""

import asyncio
import json

import pytest

import lindungidatamu as ld
from lindungidatamu.llm import Session

openai = pytest.importorskip("openai")
try:
    import httpx2 as httpx         # the SDKs' HTTP layer since 2026
except ImportError:
    httpx = pytest.importorskip("httpx")
anthropic = pytest.importorskip("anthropic")

NIK, PHONE = "3174091203880004", "081234567890"
TEXT = f"Cek klaim NIK {NIK}, hp {PHONE}"


def _sse(events, named=False):
    body = ""
    for e in events:
        if named:
            body += f"event: {e['type']}\n"
        body += f"data: {json.dumps(e)}\n\n"
    return body + ("" if named else "data: [DONE]\n\n")


def _pieces(text, n=3):
    return [text[i:i + n] for i in range(0, len(text), n)]


class Server:
    """Records what reached the 'API' and answers with the placeholders it was sent."""

    def __init__(self, answer):
        self.seen, self.answer = [], answer

    def __call__(self, request):
        body = json.loads(request.content)
        self.seen.append(body)
        return self.answer(body)


def _openai(answer, cls=None):
    server = Server(answer)
    if cls is openai.AsyncOpenAI:
        client = cls(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)))
    else:
        client = openai.OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(server)))
    return client, server


def _chat_reply(content, tool_calls=None):
    return httpx.Response(200, json={
        "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content,
                                                                       "tool_calls": tool_calls}}],
    })


def _placeholders(body):
    return [w for w in json.dumps(body).replace('"', " ").split() if w.startswith("[")]


def test_openai_chat_masks_the_request_and_restores_the_answer():
    client, server = _openai(lambda b: _chat_reply("Sudah dicek: " + b["messages"][-1]["content"].split("NIK ")[1]))
    chat = ld.wrap(client)
    r = chat.chat.completions.create(model="m", messages=[{"role": "user", "content": TEXT}])
    sent = json.dumps(server.seen[0])
    assert NIK not in sent and PHONE not in sent and "[NIK_1]" in sent
    assert r.choices[0].message.content == f"Sudah dicek: {NIK}, hp {PHONE}"
    assert chat.vault.value("NIK", 1) == NIK


def test_an_answer_sent_back_is_masked_again():
    client, server = _openai(lambda b: _chat_reply("NIK [NIK_1] valid"))
    chat = ld.wrap(client)
    messages = [{"role": "user", "content": TEXT}]
    r = chat.chat.completions.create(model="m", messages=messages)
    messages += [r.choices[0].message, {"role": "user", "content": "Lalu?"}]     # the SDK object, as people do
    chat.chat.completions.create(model="m", messages=messages)
    second = json.dumps(server.seen[1])
    assert NIK not in second and second.count("[NIK_1]") == 2


def test_tool_call_arguments_come_back_as_valid_json():
    calls = [{"id": "t1", "type": "function", "function": {"name": "lookup", "arguments": '{"nik": "[NIK_1]"}'}}]
    client, _ = _openai(lambda b: _chat_reply(None, calls))
    r = ld.wrap(client).chat.completions.create(model="m", messages=[{"role": "user", "content": TEXT}])
    assert json.loads(r.choices[0].message.tool_calls[0].function.arguments) == {"nik": NIK}
    assert r.choices[0].message.tool_calls[0].function.name == "lookup"


def _chat_stream(text):
    chunks = [{"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
               "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]} for p in _pieces(text)]
    chunks.append({"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    return httpx.Response(200, text=_sse(chunks), headers={"content-type": "text/event-stream"})


def test_openai_stream_restores_placeholders_cut_across_chunks():
    client, _ = _openai(lambda b: _chat_stream("NIK [NIK_1] dan hp [PHONE_1] sudah dicek."))
    chat = ld.wrap(client)
    with chat.chat.completions.create(model="m", messages=[{"role": "user", "content": TEXT}], stream=True) as s:
        text = "".join(c.choices[0].delta.content or "" for c in s)
    assert text == f"NIK {NIK} dan hp {PHONE} sudah dicek."


def test_async_openai_with_and_without_stream():
    async def run():
        client, server = _openai(lambda b: _chat_stream("[NIK_1] ok") if b.get("stream") else _chat_reply("[NIK_1] ok"),
                                 openai.AsyncOpenAI)
        chat = ld.wrap(client)
        r = await chat.chat.completions.create(model="m", messages=[{"role": "user", "content": TEXT}])
        s = await chat.chat.completions.create(model="m", messages=[{"role": "user", "content": TEXT}], stream=True)
        streamed = "".join([c.choices[0].delta.content or "" async for c in s])
        return r.choices[0].message.content, streamed, server.seen
    plain, streamed, seen = asyncio.run(run())
    assert plain == streamed == f"{NIK} ok"
    assert NIK not in json.dumps(seen)


def test_openai_responses():
    def answer(b):
        return httpx.Response(200, json={
            "id": "r1", "object": "response", "created_at": 0, "model": "m", "status": "completed",
            "output": [{"type": "message", "id": "o1", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": "Untuk [NIK_1].", "annotations": []}]}],
        })
    client, server = _openai(answer)
    r = ld.wrap(client).responses.create(model="m", instructions=f"Pelanggan hp {PHONE}", input=TEXT)
    assert PHONE not in json.dumps(server.seen[0]) and NIK not in json.dumps(server.seen[0])
    assert r.output_text == f"Untuk {NIK}."


def test_openai_responses_stream():
    text = "Untuk [NIK_1], hubungi [PHONE_1]."

    def answer(b):
        events = [{"type": "response.output_text.delta", "item_id": "o1", "output_index": 0, "content_index": 0,
                   "delta": p, "sequence_number": i, "logprobs": []} for i, p in enumerate(_pieces(text, 4))]
        events.append({"type": "response.output_text.done", "item_id": "o1", "output_index": 0, "content_index": 0,
                       "text": text, "sequence_number": 99, "logprobs": []})
        return httpx.Response(200, text=_sse(events, named=True), headers={"content-type": "text/event-stream"})
    client, _ = _openai(answer)
    events = list(ld.wrap(client).responses.create(model="m", input=TEXT, stream=True))
    streamed = "".join(e.delta for e in events if e.type == "response.output_text.delta")
    done = [e.text for e in events if e.type == "response.output_text.done"]
    assert streamed == done[0] == f"Untuk {NIK}, hubungi {PHONE}."


def _anthropic(answer):
    server = Server(answer)
    return anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(server))), server


def test_anthropic_messages_with_system_and_tool_use():
    def answer(b):
        return httpx.Response(200, json={
            "id": "m1", "type": "message", "role": "assistant", "model": "m", "stop_reason": "tool_use",
            "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
            "content": [{"type": "text", "text": "Saya cek [NIK_1]."},
                        {"type": "tool_use", "id": "t1", "name": "lookup", "input": {"nik": "[NIK_1]"}}],
        })
    client, server = _anthropic(answer)
    r = ld.wrap(client).messages.create(model="m", max_tokens=10, system=f"hp pelanggan {PHONE}",
                                        messages=[{"role": "user", "content": [{"type": "text", "text": TEXT}]}])
    assert PHONE not in json.dumps(server.seen[0]) and NIK not in json.dumps(server.seen[0])
    assert r.content[0].text == f"Saya cek {NIK}." and r.content[1].input == {"nik": NIK}
    assert r.content[1].name == "lookup" and r.id == "m1"


def test_anthropic_stream():
    text = "NIK [NIK_1] terdaftar."

    def answer(b):
        events = [
            {"type": "message_start", "message": {"id": "m1", "type": "message", "role": "assistant", "model": "m",
                                                  "content": [], "stop_reason": None, "stop_sequence": None,
                                                  "usage": {"input_tokens": 1, "output_tokens": 0}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            *[{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": p}}
              for p in _pieces(text, 2)],
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
             "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
        return httpx.Response(200, text=_sse(events, named=True), headers={"content-type": "text/event-stream"})
    client, _ = _anthropic(answer)
    events = list(ld.wrap(client).messages.create(model="m", max_tokens=10, stream=True,
                                                  messages=[{"role": "user", "content": TEXT}]))
    deltas = [e for e in events if e.type == "content_block_delta"]
    assert "".join(e.delta.text for e in deltas) == f"NIK {NIK} terdaftar."
    assert events.index(deltas[-1]) < [e.type for e in events].index("content_block_stop")


def test_helpers_that_would_bypass_masking_refuse():
    client, _ = _openai(lambda b: _chat_reply("x"))
    with pytest.raises(NotImplementedError):
        ld.wrap(client).chat.completions.with_raw_response
    a, _ = _anthropic(lambda b: None)
    with pytest.raises(NotImplementedError):
        ld.wrap(a).messages.stream


def test_with_options_stays_wrapped():
    client, server = _openai(lambda b: _chat_reply("ok"))
    ld.wrap(client).with_options(timeout=5).chat.completions.create(model="m", messages=[{"role": "user", "content": TEXT}])
    assert NIK not in json.dumps(server.seen[0])


def test_images_and_names_fields_are_left_alone():
    s = Session()
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,MDgxMjM0NTY3ODkw"}}
    msg = {"role": "user", "name": "budi_81234567890", "content": [{"type": "text", "text": TEXT}, image]}
    out = s.mask_request({"messages": [msg], "model": "m"})["messages"][0]
    assert out["content"][1] == image and out["name"] == msg["name"] and NIK not in json.dumps(out)


class Once:
    """A name model that finds Budi Santoso the first time only."""
    name = "once"

    def __init__(self):
        self.calls = 0

    def find(self, text):
        self.calls += 1
        i = text.find("Budi Santoso")
        return [ld.Entity("PERSON", i, i + 12, "Budi Santoso", "model", "once")] if self.calls == 1 and i >= 0 else []


def test_a_value_seen_once_is_masked_wherever_it_appears_again():
    s = Session(names=Once())
    assert s.mask_text("Budi Santoso minta refund") == "[PERSON_1] minta refund"
    assert s.mask_text("Jawaban untuk BUDI SANTOSO: oke") == "Jawaban untuk [PERSON_1]: oke"
