"""Mask what goes to an AI model, restore what comes back: wrap the client once.

    from openai import OpenAI
    client = ld.wrap(OpenAI())                  # also AsyncOpenAI, Anthropic, AsyncAnthropic
    r = client.chat.completions.create(model="...", messages=[{"role": "user", "content": text}])
    r.choices[0].message.content                # real values back
    client.vault                                # what was masked, for this conversation

One wrapper is one conversation: the same value gets the same placeholder in every request, and a
value the model has seen once is masked wherever it appears again, also in earlier answers you send
back. For a new conversation, wrap again (cheap) or pass vault=.

Covered: OpenAI chat completions (create, parse) and responses (create, parse), Anthropic messages
(create, count_tokens), embeddings, with or without stream=True. The SDKs' stream() helpers and
with_raw_response raise, because they would send text unmasked or return it unrestored; use
create(stream=True). Only text is masked: images, audio and uploaded files go as they are. Model
reasoning (thinking blocks) is left alone in both directions, because it is signed.
"""

import copy
import inspect
import json
import re

from .mask import _BACK, _EXACT, Vault, mask

# Request arguments that carry text to mask.
_REQUEST_KEYS = {"messages", "input", "instructions", "system", "prompt", "requests"}
# Keys that are never text: structure, ids, tool names, media, signed reasoning.
_SKIP = {
    "role", "type", "id", "model", "name", "object", "status", "call_id", "tool_call_id", "tool_use_id",
    "custom_id", "cache_control", "image_url", "input_audio", "url", "file_data", "file_id", "detail",
    "media_type", "format", "thinking", "signature", "encrypted_content", "system_fingerprint",
    "finish_reason", "stop_reason", "stop_sequence",
}
_CALLS = {"create", "parse", "count_tokens"}
_UNSUPPORTED = {"stream", "with_raw_response", "with_streaming_response"}
_CLIENT_COPIES = {"with_options", "copy"}


class Session:
    """One conversation's masking: the vault, a cache, and the known-value pass."""

    def __init__(self, vault: Vault | None = None, strict: bool = False, **find_options):
        self.vault = vault if vault is not None else Vault()
        self.strict = strict
        self.options = find_options
        self._cache: dict[str, str] = {}
        self._known_n = -1
        self._known_rx = None
        self._known: dict[str, str] = {}

    # --- requests ---------------------------------------------------------------------------
    def mask_text(self, text: str) -> str:
        if not text or not text.strip():
            return text
        out = self._cache.get(text)
        if out is None:
            out = self._cache[text] = mask(text, vault=self.vault, **self.options).text
        return self._known_values(out)

    def _known_values(self, text: str) -> str:
        """Values already in the vault, wherever they appear (e.g. in an answer you send back)."""
        if self._known_n != len(self.vault.items):
            self._known = {}
            for i in self.vault.items:
                if len(i.value) >= 3:
                    self._known.setdefault(i.value.lower(), i.placeholder)
            alts = sorted(self._known, key=len, reverse=True)
            self._known_rx = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, alts)) + r")(?!\w)",
                                        re.IGNORECASE) if alts else None
            self._known_n = len(self.vault.items)
        if self._known_rx is None:
            return text
        return self._known_rx.sub(lambda m: self._known[m.group(0).lower()], text)

    def mask_value(self, v, key: str | None = None):
        if key in _SKIP:
            return v
        if isinstance(v, str):
            return self.mask_text(v)
        if isinstance(v, (list, tuple)):
            return type(v)(self.mask_value(x) for x in v)
        if isinstance(v, dict):
            if v.get("type") == "base64":           # an image or PDF given inline
                return v
            return {k: self.mask_value(x, k) for k, x in v.items()}
        if hasattr(v, "model_dump"):                 # a message object from an earlier response
            return self.mask_value(v.model_dump(exclude_none=True), key)
        return v

    def mask_request(self, kwargs: dict) -> dict:
        return {k: self.mask_value(v) if k in _REQUEST_KEYS else v for k, v in kwargs.items()}

    # --- answers ----------------------------------------------------------------------------
    def restore_text(self, text: str, as_json: bool = False) -> str:
        rx = _EXACT if self.strict else _BACK

        def put(m):
            v = self.vault.value(m.group(1), m.group(2))
            if v is None:
                return m.group(0)
            return json.dumps(v, ensure_ascii=False)[1:-1] if as_json else v
        return rx.sub(put, text)

    def restore_value(self, v, key: str | None = None):
        """Real values back into an answer, in place (SDK objects are updated, not copied)."""
        if key in _SKIP:
            return v
        if isinstance(v, str):
            if key == "arguments":                   # tool call arguments: JSON text
                return self._restore_json(v)
            return self.restore_text(v)
        if isinstance(v, list):
            for i, x in enumerate(v):
                v[i] = self.restore_value(x)
            return v
        if isinstance(v, dict):
            for k in list(v):
                v[k] = self.restore_value(v[k], k)
            return v
        fields = getattr(v, "__dict__", None)
        if fields is not None and hasattr(v, "model_dump"):
            for k, x in list(fields.items()):
                new = self.restore_value(x, k)
                if new is not x:
                    fields[k] = new
        return v

    def _restore_json(self, text: str) -> str:
        try:
            return json.dumps(self.restore_value(json.loads(text)), ensure_ascii=False)
        except ValueError:
            return self.restore_text(text, as_json=True)


class _Buffer:
    """Streamed text: holds back the tail where a placeholder may be cut in two."""
    HOLD = 40

    def __init__(self, session: Session, as_json: bool = False):
        self.session, self.as_json, self.text = session, as_json, ""

    def feed(self, s: str) -> str:
        self.text += s
        cut = max(0, len(self.text) - self.HOLD)
        rx = _EXACT if self.session.strict else _BACK
        for m in rx.finditer(self.text):
            if m.start() < cut < m.end():
                cut = m.start()
        while 0 < cut < len(self.text) and self.text[cut - 1].isalnum() and self.text[cut].isalnum():
            cut -= 1
        out, self.text = self.text[:cut], self.text[cut:]
        return self.session.restore_text(out, self.as_json) if out else ""

    def flush(self) -> str:
        out, self.text = self.text, ""
        return self.session.restore_text(out, self.as_json) if out else ""


def _copy(obj, **changes):
    new = obj.model_copy() if hasattr(obj, "model_copy") else copy.copy(obj)
    for k, v in changes.items():
        new.__dict__[k] = v
    return new


class _Restorer:
    """Restores a stream of chunks or events: OpenAI chat chunks, Responses events, Anthropic events."""

    def __init__(self, session: Session):
        self.s = session
        self.buffers: dict = {}
        self.last: dict = {}           # key -> the last delta object, copied to carry the held-back tail
        self.chunk = None

    def _buf(self, key, as_json=False) -> _Buffer:
        if key not in self.buffers:
            self.buffers[key] = _Buffer(self.s, as_json)
        return self.buffers[key]

    def __call__(self, item) -> list:
        kind = getattr(item, "type", None)
        if getattr(item, "object", None) == "chat.completion.chunk" or (kind is None and hasattr(item, "choices")):
            return [self._chat(item)]
        if kind == "content_block_delta":
            return self._anthropic_delta(item)
        if kind == "content_block_stop":
            return self._flush_anthropic(item.index) + [item]
        if kind in ("response.output_text.delta", "response.refusal.delta", "response.function_call_arguments.delta"):
            key = (kind, getattr(item, "output_index", 0), getattr(item, "content_index", 0))
            self.last[key] = item
            item.delta = self._buf(key, kind.startswith("response.function")).feed(item.delta)
            return [item]
        if isinstance(kind, str) and kind.startswith("response.") and kind.endswith(".done"):
            out = []
            delta = kind[:-len(".done")] + ".delta"
            for key in [k for k in self.buffers if k[0] == delta]:
                rest = self.buffers.pop(key).flush()
                if rest:
                    out.append(_copy(self.last[key], delta=rest))
            return out + [self.s.restore_value(item)]
        if isinstance(kind, str) and ("delta" in kind):
            return [item]                            # reasoning and other deltas: left alone
        return [self.s.restore_value(item)]

    def _chat(self, chunk):
        self.chunk = chunk
        for c in chunk.choices or []:
            d = c.delta
            if d is None:
                continue
            if d.content:
                d.content = self._buf((c.index, "content")).feed(d.content)
            for tc in d.tool_calls or []:
                if tc.function is not None and tc.function.arguments:
                    key = (c.index, "tool", tc.index)
                    self.last[key] = tc
                    tc.function.arguments = self._buf(key, True).feed(tc.function.arguments)
            if c.finish_reason:
                self._flush_choice(c)
        return chunk

    def _flush_choice(self, c):
        d = c.delta
        rest = self.buffers.pop((c.index, "content"), None)
        if rest is not None and rest.text:
            d.content = (d.content or "") + rest.flush()
        for key in [k for k in self.buffers if k[0] == c.index and k[1] == "tool"]:
            tail = self.buffers.pop(key).flush()
            if tail:
                tc = self.last[key]
                more = _copy(tc, id=None, type=None, function=_copy(tc.function, name=None, arguments=tail))
                d.tool_calls = list(d.tool_calls or []) + [more]

    def _anthropic_delta(self, event):
        d = event.delta
        if d.type == "text_delta":
            self.last[event.index] = event
            d.text = self._buf(event.index).feed(d.text)
        elif d.type == "input_json_delta":
            self.last[event.index] = event
            d.partial_json = self._buf(event.index, True).feed(d.partial_json)
        return [event]

    def _flush_anthropic(self, index) -> list:
        buf = self.buffers.pop(index, None)
        rest = buf.flush() if buf else ""
        if not rest:
            return []
        ev = self.last[index]
        field = "text" if ev.delta.type == "text_delta" else "partial_json"
        return [_copy(ev, delta=_copy(ev.delta, **{field: rest}))]

    def leftover(self) -> list:
        """At the end of a stream: whatever is still held back, as one more chunk."""
        if not self.buffers:
            return []
        if self.chunk is not None:
            chunk = self.chunk.model_copy(deep=True)
            for c in chunk.choices:
                c.delta.content, c.delta.tool_calls, c.finish_reason = None, None, None
            for c in chunk.choices:
                if (c.index, "content") in self.buffers:
                    c.delta.content = self.buffers.pop((c.index, "content")).flush()
            self.buffers.clear()
            return [chunk]
        out = []
        for key in list(self.buffers):
            if isinstance(key, int):
                out += self._flush_anthropic(key)
            elif key in self.last:
                rest = self.buffers[key].flush()
                if rest:
                    out.append(_copy(self.last[key], delta=rest))
        self.buffers.clear()
        return out


class _Stream:
    """A restored stream that still behaves like the SDK's (with, close, response)."""

    def __init__(self, stream, session: Session):
        self._stream, self._r = stream, _Restorer(session)
        self._gen = self._run()

    def _run(self):
        for item in self._stream:
            yield from self._r(item)
        yield from self._r.leftover()

    def __iter__(self):
        return self._gen

    def __next__(self):
        return next(self._gen)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        close = getattr(self._stream, "close", None)
        if close:
            close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _AsyncStream:
    def __init__(self, stream, session: Session):
        self._stream, self._r = stream, _Restorer(session)
        self._gen = self._run()

    async def _run(self):
        async for item in self._stream:
            for out in self._r(item):
                yield out
        for out in self._r.leftover():
            yield out

    def __aiter__(self):
        return self._gen

    async def __anext__(self):
        return await self._gen.__anext__()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        close = getattr(self._stream, "close", None)
        if close:
            r = close()
            if inspect.isawaitable(r):
                await r

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _finish(result, session: Session):
    if hasattr(result, "__aiter__"):
        return _AsyncStream(result, session)
    if hasattr(result, "__iter__") and not hasattr(result, "model_dump") and not isinstance(result, (str, dict, list)):
        return _Stream(result, session)
    return session.restore_value(result)


class Wrapped:
    """An SDK client (or one of its resources) whose requests are masked and answers restored."""

    def __init__(self, target, session: Session, root: str):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_root", root)

    @property
    def vault(self) -> Vault:
        return self._session.vault

    def __getattr__(self, name):
        if name in _UNSUPPORTED:
            raise NotImplementedError(f"lindungidatamu.wrap: .{name} is not covered (text would go out unmasked "
                                      "or come back unrestored); use create(stream=True)")
        attr = getattr(self._target, name)
        if name in _CALLS and callable(attr):
            return self._call(attr)
        if name in _CLIENT_COPIES and callable(attr):
            return lambda *a, **k: Wrapped(attr(*a, **k), self._session, self._root)
        if not callable(attr) and type(attr).__module__.split(".")[0] == self._root and not hasattr(attr, "model_dump"):
            return Wrapped(attr, self._session, self._root)      # a resource: client.chat, .completions
        return attr

    def __setattr__(self, name, value):
        setattr(self._target, name, value)

    def __repr__(self):
        return f"lindungidatamu.wrap({self._target!r})"

    def _call(self, fn):
        session = self._session

        def call(*args, **kwargs):
            result = fn(*args, **session.mask_request(kwargs))
            if inspect.isawaitable(result):
                async def later():
                    return _finish(await result, session)
                return later()
            return _finish(result, session)
        return call


def wrap(client, *, vault: Vault | None = None, names=None, strict: bool = False, **find_options) -> Wrapped:
    """`client` with every request masked and every answer restored.

    vault     one per conversation (a new one by default); pass your own to keep it, or to share it
    names     a name finder (lindungidatamu.names.GlinerNames()) to mask people and organisations too
    strict    restore only exact placeholders ([PERSON_1]); by default rewritten ones ("Person 1") too
    Other options are those of lindungidatamu.find (types, terms, formats, keep).
    """
    if names is not None:
        find_options["names"] = names
    root = type(client).__module__.split(".")[0]
    return Wrapped(client, Session(vault, strict=strict, **find_options), root)
