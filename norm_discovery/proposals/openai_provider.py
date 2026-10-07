"""OpenAI-backed completion function for LLMProposalProvider (OpenAI Python SDK >= 1.x, Responses API).

* The API key is read by the SDK from the OPENAI_API_KEY environment variable; it is never stored or logged.
* Every call verifies that the model reported by the API matches the requested model; a mismatch raises
  ModelMismatchError (no silent substitution).
* Transient errors (rate limit, connection, timeout, 5xx) are retried up to `max_retries` times with
  exponential backoff; every retry is recorded.  Any other error is raised -- there is no fallback.
* Responses are cached on disk keyed by sha256(prompt, model, generation settings, replicate).  The
  replicate id is part of the key so that independent replications never share a response, while the same
  replicate re-issuing the identical prompt (e.g. the same run evaluated at a different rho, where the
  demonstration history and hypothesis set are identical up to that point) reuses it.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import tempfile
import time
from typing import Dict, List, Optional


class ModelMismatchError(RuntimeError):
    pass


class LLMCallError(RuntimeError):
    pass


def model_matches(requested: str, returned: Optional[str]) -> bool:
    """'gpt-5.5' matches 'gpt-5.5' and dated snapshots 'gpt-5.5-YYYY-MM-DD'; nothing else."""
    if not returned:
        return False
    if returned == requested:
        return True
    rest = returned[len(requested):] if returned.startswith(requested) else None
    return bool(rest) and rest[0] == "-" and rest[1:].replace("-", "").isdigit()


class OpenAICompletion:
    def __init__(self, model: str = "gpt-5.5", temperature: Optional[float] = None,
                 reasoning_effort: Optional[str] = None, max_output_tokens: Optional[int] = None,
                 cache_dir: Optional[str] = None, replicate: object = 0, max_retries: int = 3,
                 timeout: float = 600.0):
        from openai import OpenAI  # imported lazily so the deterministic code path never needs the SDK
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMCallError("OPENAI_API_KEY is not set (run `source ~/.bashrc` first)")
        self.client = OpenAI(timeout=timeout, max_retries=0)   # retries are handled (and logged) here
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.cache_dir = cache_dir
        self.replicate = replicate
        self.max_retries = max_retries
        self.calls: List[Dict] = []          # metadata of every call made through this object
        self.last_call: Optional[Dict] = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ settings / cache
    @property
    def settings(self) -> Dict:
        return {"model": self.model, "temperature": self.temperature, "reasoning_effort": self.reasoning_effort,
                "max_output_tokens": self.max_output_tokens, "api": "responses.create"}

    def cache_key(self, prompt: str) -> str:
        payload = json.dumps({"prompt": prompt, "settings": self.settings, "replicate": self.replicate},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Optional[str]:
        return os.path.join(self.cache_dir, key + ".json") if self.cache_dir else None

    # ------------------------------------------------------------------ call
    def __call__(self, prompt: str) -> str:
        key = self.cache_key(prompt)
        path = self._cache_path(key)
        lock = None
        if path:
            if not os.path.exists(path):
                try:
                    lock = os.open(path + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:      # another process is making this exact call: wait for it
                    t0 = time.time()
                    while time.time() - t0 < 1800 and os.path.exists(path + ".lock") and not os.path.exists(path):
                        time.sleep(0.5)
            if os.path.exists(path):
                if lock is not None:
                    os.close(lock)
                    os.remove(path + ".lock")
                with open(path) as f:
                    rec = json.load(f)
                rec = dict(rec, cache_hit=True)
                self._check_model(rec)
                self.last_call = rec
                self.calls.append(rec)
                return rec["output_text"]
        try:
            rec = self._call_api(prompt, key)
            if path:
                fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
                with os.fdopen(fd, "w") as f:
                    json.dump(rec, f, indent=1)
                os.replace(tmp, path)
        finally:
            if lock is not None:
                os.close(lock)
                try:
                    os.remove(path + ".lock")
                except FileNotFoundError:
                    pass
        rec = dict(rec, cache_hit=False)
        self.last_call = rec
        self.calls.append(rec)
        self._check_model(rec)
        return rec["output_text"]

    def _check_model(self, rec: Dict) -> None:
        if not model_matches(self.model, rec.get("returned_model")):
            raise ModelMismatchError(f"requested model {self.model!r} but the API reported "
                                     f"{rec.get('returned_model')!r}")

    def _call_api(self, prompt: str, key: str) -> Dict:
        import openai
        kwargs = {"model": self.model, "input": prompt}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        retries = []
        transient = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                     openai.InternalServerError)
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self.client.responses.create(**kwargs)
                break
            except transient as e:
                retries.append({"attempt": attempt, "error": f"{type(e).__name__}: {e}",
                                "at": _dt.datetime.now().isoformat(timespec="seconds")})
                if attempt == self.max_retries:
                    raise LLMCallError(f"OpenAI call failed after {attempt + 1} attempts: {retries}") from e
                time.sleep(min(60, 5 * 2 ** attempt))
            except openai.APIStatusError as e:   # 4xx etc.: not retried (e.g. unknown model, bad parameter)
                raise LLMCallError(f"OpenAI API error {getattr(e, 'status_code', '?')}: {e}") from e
        usage = None
        if getattr(resp, "usage", None) is not None:
            try:
                usage = resp.usage.model_dump()
            except Exception:
                usage = str(resp.usage)
        return {"cache_key": key, "requested_model": self.model, "returned_model": getattr(resp, "model", None),
                "response_id": getattr(resp, "id", None), "status": getattr(resp, "status", None),
                "incomplete_details": (resp.incomplete_details.model_dump()
                                       if getattr(resp, "incomplete_details", None) else None),
                "usage": usage, "settings": self.settings, "replicate": self.replicate,
                "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                "latency_s": round(time.time() - t0, 2), "retries": retries,
                "output_text": resp.output_text}
