"""Re-run the VLM recognition engine for the reviewer ablations.

Same logic, models and prompts as VLM_Recognition_Engine.ipynb, but every
classifier, evaluator and feedback response from both rounds is saved, so
single-classifier, round-1 and round-2 results can all be scored from one run.

Money safety:
  * every successful API response is cached in api_cache.jsonl as it arrives, keyed by model, prompt, user text,
    image hash and --tag; a crashed or re-run command never pays twice (same command, add --resume)
  * --budget USD: hard stop once the spend of THIS command reaches it (cached answers cost 0)
  * errors that cannot succeed on retry (400/401/402/403/404) abort the whole run at once
  * every model ID is checked against OpenRouter's model list before the first paid call
  * --estimate prints the worst-case cost and exits without calling anything

Usage:
  python run_engine.py --plots ../plots_candidates --prompt full --out runs/axes_full_1 --estimate
  python run_engine.py --plots ../plots_candidates --prompt full --out runs/axes_full_1 --budget 3
  python run_engine.py --plots ../plots_candidates --prompt full --out runs/axes_full_2 --tag rep2 --budget 3
  python run_engine.py --plots ../plots_candidates_clean_512 --prompt full --out runs/noaxes_full_1 --budget 3
  python run_engine.py --plots ../plots_candidates --prompt minimal --out runs/axes_minimal_1 --budget 3
"""
import argparse
import base64
import hashlib
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

MAX_WORKERS = 3
MAX_RETRIES = 3
RETRY_DELAY = 5

# Classifier A was x-ai/grok-4.1-fast (highest binary flare recall on the synthetic benchmark, 67.5%); it is retired
# on OpenRouter, so A is the next-highest-recall benchmarked model, GPT-5 mini (62.7%), by the paper's own rule.
CLASSIFIER_A_MODEL = "openai/gpt-5-mini"
CLASSIFIER_B_MODEL = "qwen/qwen3.5-plus-02-15"
EVALUATOR_MODEL = "openai/gpt-5"
# The notebook used max_tokens=512; current Qwen-3.5-plus and GPT-5 spend all 512 on hidden reasoning and return
# nothing (smoke test 2026-10-29), so every call gets 4096 with provider-default reasoning (user's choice).
ENGINE_MAX_TOKENS = 4096
MAX_EMPTY = 3  # abort after this many empty answers that hit the token limit
# Qwen-3.5-plus sometimes reasons past 4096 tokens and never answers (1 of 3 calls in the smoke test), so its
# reasoning is capped below the total; the GPT models stop on their own (300-830 reasoning tokens).
REASONING_CAP = {"qwen/qwen3.5-plus-02-15": 3072}
FATAL_STATUS = {400, 401, 402, 403, 404}
CACHE = Path(__file__).with_name("api_cache.jsonl")

# Copied verbatim from the notebook (cell 4), including its typos.
DOMAIN_TEXT = """Flaring activity means there is decrease in
magnitude and you will be able to figure out a wave like structure against the noise/damped random walk nature of the
quasar light curves. Usual examples of flare types are gamma, gaussian, fast rise exponential decay, but these are not the only types of flare. The botttomline is that there will a rise and then decay.
Single point spikes are artifacts and donot classify them as flares. The magnitude usually decreases
by around ~0.5 (0.4 or 0.35 also works). This decrease in magnitude is from the start of the flare to the peak. It may happen
that the peak is towards the end or towards the beginning of the light curves. You will thus see only the rising or the falling part of the flaring activity.
Such cases are also flares, and the pattern may strech from start to end or vice versa meaning there can be maximum intensity at the beginning or end and the magnitude
keeps on increasing from there. Again remember that the y axis is inverted, since in astronomy less magnitude means higher brightness.
"""

PROMPTS = {
    "full": {
        "classifier": f"""You are an expert astronomer classifying quasar light curves for transient flares.

You will receive an r-band light-curve image (magnitude vs MJD).
IMPORTANT: The magnitude axis is inverted — lower magnitude = higher brightness.
A brightening event appears as points moving UPWARD on the plot.

Your task is to identify if the light curve has flaring activity in it. {DOMAIN_TEXT}

Respond ONLY in this format:
classification: <flare or non_flare>
reasoning: <Short reason for your classification>
""",
        "evaluator": f"""You are a senior astrophysicist evaluating a quasar flare detection pipeline.
Two classifiers have independently examined each light curve and DISAGREE.

For the light curve under review:
- The magnitude axis is inverted (lower mag = brighter).
- Single-point spikes are NOT flares.

Your task is to review the light curve and then the classifiers responses. {DOMAIN_TEXT}

Review the image and both classifier outputs. Respond in this format:
final_classification: <flare or non_flare>
reasoning: <brief explanation>
""",
    },
    # Minimal: the task, the inverted axis (a fact about the image) and the output
    # format only. No flare shapes, no amplitude, no spike rule, no edge-flare rule.
    "minimal": {
        "classifier": """You are classifying quasar light curves for transient flares.

You will receive an r-band light-curve image (magnitude vs time). The magnitude axis is inverted: points higher on the plot are brighter.

Respond ONLY in this format:
classification: <flare or non_flare>
reasoning: <Short reason for your classification>
""",
        "evaluator": """You are reviewing a quasar light curve for transient flares.
Two classifiers have independently examined the light curve and DISAGREE.
The magnitude axis is inverted: points higher on the plot are brighter.

Review the image and both classifier outputs. Respond in this format:
final_classification: <flare or non_flare>
reasoning: <brief explanation>
""",
    },
}

CLASSIFIER_USER_PROMPT = "Classify this quasar light curve: flare or non_flare?"
FEEDBACK_REQUEST_PROMPT = (
    "Based on the evaluations above, provide concise actionable feedback "
    "for the classifiers to reduce errors in the next round. Focus on mistake patterns."
)


# The "full" prompts are read from the notebook itself so they match the published
# engine byte for byte (the hand copy above lost trailing spaces).
_nb = json.loads((Path(__file__).parent.parent / "VLM_Recognition_Engine.ipynb").read_text(encoding="utf-8"))
_ns = {}
exec("".join(_nb["cells"][4]["source"]), _ns)
PROMPTS["full"] = {"classifier": _ns["CLASSIFIER_SYSTEM_PROMPT"], "evaluator": _ns["EVALUATOR_SYSTEM_PROMPT"]}
assert _ns["CLASSIFIER_USER_PROMPT"] == CLASSIFIER_USER_PROMPT
assert _ns["FEEDBACK_REQUEST_PROMPT"] == FEEDBACK_REQUEST_PROMPT


class Abort(Exception):
    pass


class Ledger:
    """Response cache plus spend tracking, shared by all threads."""

    def __init__(self, budget, tag, prices):
        self.budget, self.tag, self.prices = budget, tag, prices
        self.spent = 0.0
        self.calls = self.cached = self.empty = 0
        self.lock = threading.Lock()
        self.store = {}
        if CACHE.exists():
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.store[rec["key"]] = rec

    def key(self, model, system_prompt, user_text, image_b64, max_tokens):
        img = hashlib.sha256(image_b64.encode()).hexdigest() if image_b64 else ""
        raw = json.dumps([model, system_prompt, user_text, img, max_tokens, self.tag]
                         + ([REASONING_CAP[model]] if model in REASONING_CAP else []))
        return hashlib.sha256(raw.encode()).hexdigest()

    def cost(self, model, usage):
        if usage.get("cost") is not None:
            return float(usage["cost"])
        p = self.prices[model]
        return usage.get("prompt_tokens", 0) * p[0] + usage.get("completion_tokens", 0) * p[1]

    def check(self):
        with self.lock:
            if self.spent >= self.budget:
                raise Abort(f"budget ${self.budget:.2f} reached (spent ${self.spent:.4f})")

    def save(self, rec):
        with self.lock:
            self.store[rec["key"]] = rec
            self.spent += rec["cost"]
            self.calls += 1
            with open(CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")


LEDGER = None


def fetch_prices():
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as r:
        data = json.load(r)["data"]
    return {m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])) for m in data}


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def api_call(client, model, system_prompt, user_text, image_b64, max_tokens):
    """One cached, budgeted call. Returns (text, served_model, usage)."""
    k = LEDGER.key(model, system_prompt, user_text, image_b64, max_tokens)
    if k in LEDGER.store:
        with LEDGER.lock:
            LEDGER.cached += 1
        rec = LEDGER.store[k]
        return rec["text"], rec["served_model"], rec["usage"]
    if image_b64 is not None:
        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
    else:
        user_content = user_text
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        LEDGER.check()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
                extra_body={"usage": {"include": True},
                            **({"reasoning": {"max_tokens": REASONING_CAP[model]}} if model in REASONING_CAP else {})},
            )
            if not resp.choices:
                raise RuntimeError(f"no choices in response: {resp}")
            usage = resp.usage.model_dump() if resp.usage else {}
            text = (resp.choices[0].message.content or "").strip()
            rec = {"key": k, "model": model, "served_model": resp.model, "text": text, "usage": usage,
                   "cost": LEDGER.cost(model, usage), "utc": datetime.now(timezone.utc).isoformat()}
            if not text and usage.get("completion_tokens", 0) >= max_tokens:
                # paid for, but not cached: a resume should ask again
                with LEDGER.lock:
                    LEDGER.spent += rec["cost"]
                    LEDGER.empty += 1
                    n = LEDGER.empty
                if n > MAX_EMPTY:
                    raise Abort(f"{n} empty answers that hit max_tokens={max_tokens} (last: {model})")
                return text, resp.model, usage
            LEDGER.save(rec)
            return text, resp.model, usage
        except Abort:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in FATAL_STATUS:
                raise Abort(f"{model}: HTTP {status}: {exc}")
            last = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    raise Abort(f"{model}: {MAX_RETRIES} attempts failed, last error: {last}")


def call_vlm(client, model, system_prompt, user_text, image_b64=None):
    text, served, _ = api_call(client, model, system_prompt, user_text, image_b64, ENGINE_MAX_TOKENS)
    return text, served


def parse_classification(text):
    t = text.lower()
    if "non_flare" in t or "non-flare" in t or "no flare" in t:
        return "non_flare"
    if "flare" in t:
        return "flare"
    return "uncertain"


def run_classifiers(client, paths, system_prompt, log):
    def one(p):
        b64 = encode_image(p)
        out = {}
        for key, model in (("a", CLASSIFIER_A_MODEL), ("b", CLASSIFIER_B_MODEL)):
            text, served = call_vlm(client, model, system_prompt, CLASSIFIER_USER_PROMPT, b64)
            out[key] = {"label": parse_classification(text), "response": text, "served_model": served}
        log(p.name)
        return p.name, out

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        return dict(ex.map(one, paths))


def run_evaluator(client, items, img_map, eval_prompt):
    def one(item):
        fn = item["filename"]
        msg = (
            f"Light curve: {fn}\n"
            f"Classifier A (high recall): {item['a']['label']}\n  {item['a']['response']}\n\n"
            f"Classifier B (high precision): {item['b']['label']}\n  {item['b']['response']}\n\n"
            f"These classifiers disagree. Review the image and give your final classification."
        )
        text, served = call_vlm(client, EVALUATOR_MODEL, eval_prompt, msg, encode_image(img_map[fn]))
        return fn, {"label": parse_classification(text), "response": text, "served_model": served}

    if not items:
        return {}, "No disagreements to review."
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        evaluated = dict(ex.map(one, items))
    summary = "\n".join(f"- {fn}: {e['label']}. {e['response'][:200]}" for fn, e in evaluated.items())
    feedback, _ = call_vlm(
        client, EVALUATOR_MODEL, eval_prompt,
        f"Evaluator verdicts on disputed cases:\n\n{summary}\n\n{FEEDBACK_REQUEST_PROMPT}",
    )
    return evaluated, feedback


def split(results):
    agreed, disagreed = {}, []
    for fn, r in sorted(results.items()):
        la, lb = r["a"]["label"], r["b"]["label"]
        if la == lb and la not in ("error", "uncertain"):
            agreed[fn] = la
        else:
            disagreed.append({"filename": fn, **r})
    return agreed, disagreed


def run_single(client, model, paths, system_prompt, max_tokens, out, meta):
    """One model, one pass, no evaluator: the state-of-the-art comparison."""
    def one(p):
        text, served, usage = api_call(client, model, system_prompt, CLASSIFIER_USER_PROMPT, encode_image(p), max_tokens)
        print(f"  {p.name} {parse_classification(text)}  (spent ${LEDGER.spent:.3f})", flush=True)
        return p.name, {"label": parse_classification(text), "response": text, "served_model": served, "usage": usage}

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        res = dict(ex.map(one, paths))
    with open(out / "labels.csv", "w", encoding="utf-8") as f:
        f.write("dbID,single\n")
        for p in paths:
            f.write(f"{int(p.stem)},{res[p.name]['label']}\n")
    (out / "responses.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    meta.update({"mode": "single", "model": model, "max_tokens": max_tokens,
                 "finished_utc": datetime.now(timezone.utc).isoformat()})
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Done -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", required=True)
    ap.add_argument("--prompt", choices=sorted(PROMPTS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
    ap.add_argument("--single", help="run one model alone, e.g. openai/gpt-6.1-sol")
    ap.add_argument("--max-tokens", type=int, default=4096, help="single mode only")
    ap.add_argument("--budget", type=float, help="USD; hard stop when this command's spend reaches it (required)")
    ap.add_argument("--tag", default="", help="part of the cache key: give a replicate run its own answers (e.g. rep2)")
    ap.add_argument("--resume", action="store_true", help="allow an existing --out (cached calls are free)")
    ap.add_argument("--estimate", action="store_true", help="print the worst-case cost and exit; no API calls")
    args = ap.parse_args()
    global LEDGER

    paths = sorted(Path(args.plots).glob("*.png"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no PNG files in {args.plots}")

    prices = fetch_prices()
    models = [args.single] if args.single else [CLASSIFIER_A_MODEL, CLASSIFIER_B_MODEL, EVALUATOR_MODEL]
    missing = [m for m in models if m not in prices]
    if missing:
        raise SystemExit(f"not on OpenRouter: {missing}")
    # worst case: every call fills max_tokens; ~3,000 input tokens per image call (prompt + image)
    if args.single:
        worst = len(paths) * (3000 * prices[args.single][0] + args.max_tokens * prices[args.single][1])
    else:
        per_img = sum(3000 * prices[m][0] + ENGINE_MAX_TOKENS * prices[m][1] for m in (CLASSIFIER_A_MODEL, CLASSIFIER_B_MODEL))
        ev = 4000 * prices[EVALUATOR_MODEL][0] + ENGINE_MAX_TOKENS * prices[EVALUATOR_MODEL][1]
        worst = len(paths) * (2 * per_img + 2 * ev) + 2 * ev  # both rounds on every image, plus feedback
    print(f"{len(paths)} images, models {models}: worst-case ${worst:.2f}", flush=True)
    if args.estimate:
        return
    if args.budget is None:
        raise SystemExit("--budget is required for a paid run")

    out = Path(args.out)
    if out.exists() and not args.resume:
        raise SystemExit(f"{out} exists; pick a new --out (or --resume to re-run it from the cache)")
    out.mkdir(parents=True, exist_ok=True)

    load_dotenv(Path(__file__).with_name(".env"))
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not found (environment or ablation/.env)")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    LEDGER = Ledger(args.budget, args.tag, prices)
    try:
        run(args, client, paths, out)
    except Abort as exc:
        print(f"ABORTED: {exc}", flush=True)
    finally:
        print(f"spent ${LEDGER.spent:.4f} on {LEDGER.calls} paid calls; {LEDGER.cached} answers from the cache; "
              f"{LEDGER.empty} empty answers at the token limit", flush=True)


def run(args, client, paths, out):
    img_map = {p.name: p for p in paths}
    prompts = PROMPTS[args.prompt]
    done = [0]
    lock = threading.Lock()

    def log(name):
        with lock:
            done[0] += 1
            print(f"  {done[0]}/{len(paths)} {name}", flush=True)

    started = datetime.now(timezone.utc).isoformat()
    if args.single:
        meta = {"started_utc": started, "plots": str(Path(args.plots).resolve()),
                "prompt": args.prompt, "n_images": len(paths), "temperature": "API default (not set)",
                "tag": args.tag, "system_prompt": prompts["classifier"]}
        run_single(client, args.single, paths, prompts["classifier"], args.max_tokens, out, meta)
        return
    print(f"Round 1: {len(paths)} images from {args.plots}, prompt={args.prompt}", flush=True)
    r1 = run_classifiers(client, paths, prompts["classifier"], log)
    r1_agreed, r1_disagreed = split(r1)
    print(f"Round 1 agreed {len(r1_agreed)}, disagreed {len(r1_disagreed)}", flush=True)
    r1_eval, feedback = run_evaluator(client, r1_disagreed, img_map, prompts["evaluator"])

    r2, r2_eval = {}, {}
    if r1_disagreed:
        r2_prompt = (
            f"{prompts['classifier']}\n\nEVALUATOR FEEDBACK FROM PREVIOUS ROUND:\n{feedback}\n\n"
            f"Use this feedback to correct your classifications."
        )
        done[0] = 0
        disputed = [img_map[d["filename"]] for d in r1_disagreed]
        print(f"Round 2: {len(disputed)} disputed", flush=True)
        r2 = run_classifiers(client, disputed, r2_prompt, lambda n: None)
        _, r2_disagreed = split(r2)
        r2_eval, _ = run_evaluator(client, r2_disagreed, img_map, prompts["evaluator"])

    def final_label(fn, agreed, ev):
        if fn in agreed:
            return agreed[fn]
        lab = ev.get(fn, {}).get("label", "uncertain")
        return lab if lab in ("flare", "non_flare") else "uncertain"

    r2_agreed, _ = split(r2) if r2 else ({}, [])
    rows = []
    for p in paths:
        fn = p.name
        row = {
            "dbID": int(p.stem),
            "a_r1": r1[fn]["a"]["label"], "b_r1": r1[fn]["b"]["label"],
            "eval_r1": r1_eval.get(fn, {}).get("label", ""),
            "final_r1": final_label(fn, r1_agreed, r1_eval),
            "a_r2": r2.get(fn, {}).get("a", {}).get("label", ""),
            "b_r2": r2.get(fn, {}).get("b", {}).get("label", ""),
            "eval_r2": r2_eval.get(fn, {}).get("label", ""),
        }
        if fn in r1_agreed:
            row["final_r2"] = r1_agreed[fn]
        else:
            row["final_r2"] = final_label(fn, r2_agreed, r2_eval)
        rows.append(row)

    with open(out / "labels.csv", "w", encoding="utf-8") as f:
        cols = list(rows[0])
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    raw = {"round1": r1, "round1_evaluator": r1_eval, "feedback": feedback,
           "round2": r2, "round2_evaluator": r2_eval}
    (out / "responses.json").write_text(json.dumps(raw, indent=1), encoding="utf-8")
    meta = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "plots": str(Path(args.plots).resolve()), "prompt": args.prompt, "n_images": len(paths),
            "models": {"a": CLASSIFIER_A_MODEL, "b": CLASSIFIER_B_MODEL, "evaluator": EVALUATOR_MODEL},
            "max_tokens": ENGINE_MAX_TOKENS, "temperature": "API default (not set)", "tag": args.tag,
            "spent_usd_this_command": LEDGER.spent, "prompts": prompts}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Done -> {out}", flush=True)


if __name__ == "__main__":
    main()
