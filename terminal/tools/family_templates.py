"""Writes serve/fixtures/family_templates.json.gz for serve/test_family_parsers.py: for each strata-poly family of setup.py, the chat
template and the pre-tokenizer the server is given, as setup's step 6 makes the pack.  tokenizer.ggml.pre and tokenizer.chat_template
come from the header of the first file the family pins (HTTP range reads, parsed in memory: nothing is saved but the result); a family
whose file has no template gets setup's pinned chat_template.jinja ("template"), checked against its size and SHA-256.  A family
already in the file is kept; delete its entry to read it again.

    cd terminal && python tools/family_templates.py [ENDPOINT]     # ENDPOINT: https://huggingface.co (default) or a mirror

A tool of the repository, not of the Terminal: nothing ships or runs it.

curl does the reads: on lab-pro6000 Python's TLS handshake with the mirror timed out where curl's did not.
"""
import gzip
import hashlib
import json
import os
import struct
import subprocess
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import setup  # noqa: E402
import strata_tokenizer as ST  # noqa: E402

END = (sys.argv[1] if len(sys.argv) > 1 else "https://huggingface.co").rstrip("/")
OUT = os.path.join("serve", "fixtures", "family_templates.json.gz")
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def get(url, first=None, last=None):
    rng = ["-r", f"{first}-{last}"] if first is not None else []
    return subprocess.run(["curl", "-sfL", "--retry", "8", "--retry-all-errors", "--retry-delay", "5", "-m", "900", *rng, url],
                          capture_output=True, check=True).stdout


class Short(Exception):
    """The header goes on past the bytes read."""


def parse(buf, want):
    if buf[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    at, found = 24, {}

    def need(end):
        if end > len(buf):
            raise Short

    def string(at):
        need(at + 8)
        n = struct.unpack_from("<Q", buf, at)[0]
        need(at + 8 + n)
        return buf[at + 8:at + 8 + n].decode("utf-8"), at + 8 + n

    for _ in range(struct.unpack_from("<Q", buf, 16)[0]):
        key, at = string(at)
        need(at + 4)
        kind = struct.unpack_from("<I", buf, at)[0]
        at += 4
        if kind == 8:
            value, at = string(at)
            if key in want:
                found[key] = value
        elif kind == 9:
            need(at + 12)
            item, n = struct.unpack_from("<IQ", buf, at)
            at += 12
            if item == 8:
                for _ in range(n):
                    at = string(at)[1]
            else:
                at += n * SIZES[item]
        else:
            at += SIZES[kind]
        if set(want) <= set(found):
            break
    return found


def header(url, want):
    n = 1 << 20
    while True:
        try:
            return parse(get(url, 0, n - 1), want)
        except Short:
            if n >= 256 << 20:
                raise
            n *= 4


def main():
    out = {}
    if os.path.exists(OUT):
        with gzip.open(OUT, "rt", encoding="utf-8") as f:
            out = json.load(f)
    for fam, d in setup.FAMILIES.items():
        if d.get("engine") != "poly" or fam in out:
            continue
        first = next(iter(d["sha256"]))
        owner, rest = d["hf"].split("huggingface.co/")[1].split("/resolve/")
        rev = rest.split("/")[0]
        files = json.loads(get(f"{END}/api/models/{owner}/revision/{rev}?expand%5B%5D=siblings"))["siblings"]
        # a family whose files are in a folder of the repository, or under another name there ("remote_suffix"),
        # is read from that very path: Inkling 2's repository has a file of the same name in another layout
        there = rest.split("/", 1)[1] + first + d.get("remote_suffix", "")
        path = next(s["rfilename"] for s in files
                    if s["rfilename"] == there or (there == first and s["rfilename"].split("/")[-1] == first))
        kv = header(f"{END}/{owner}/resolve/{rev}/{path}", {"tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.chat_template"})
        # the pack's pre-tokenizer, as tools/strata_tokenizer.py names it (Gemma 4's file has a tokenizer model and no pre)
        pre = ST.MODELS.get(kv.get("tokenizer.ggml.model")) or kv.get("tokenizer.ggml.pre", "qwen35")
        template, source = kv.get("tokenizer.chat_template"), f"{owner}@{rev} {path}"
        if template is None and d.get("template"):
            name, size, sha = d["template"]
            raw = get(END + d["template_hf"].split("huggingface.co", 1)[1] + name)
            if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha:
                sys.exit(f"{fam}: {name} is not the file setup pins")
            template, source = raw.decode("utf-8"), d["template_hf"].split("huggingface.co/")[1].replace("/resolve/", "@") + name
        out[fam] = {"pre": pre, "chat_template": template, "from": source}
        with open(OUT, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:   # the same bytes for the same data
            f.write(json.dumps(dict(sorted(out.items())), ensure_ascii=False, indent=1).encode("utf-8"))
        print(fam, pre, len(template or ""), source, flush=True)


if __name__ == "__main__":
    main()
