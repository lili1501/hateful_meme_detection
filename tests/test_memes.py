import json
import math
import subprocess
import sys
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from PIL import Image
from sklearn.metrics import roc_auc_score

from memes import report, serve_client
from memes.data import load_memes, read_jsonl, stratified_split, write_jsonl
from memes.labels import IGNORE, mask_prompt_labels
from memes.metrics import accuracy, auroc, best_threshold, bootstrap_ci, confusion
from memes.predict_vllm import first_token_logprobs, scores_from_outputs
from memes.prompt import QUESTION, build_messages
from memes.scoring import norm_token, score_from_logits, score_from_logprobs
from memes.synthetic import make_fake_split


# ---- metrics -------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_auroc_matches_sklearn_including_ties(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300) * 0.5 + y * 0.2, 1)        # rounding creates many ties
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auroc_known_values_and_errors():
    assert auroc(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) == 1.0
    assert auroc(np.array([0, 0, 1, 1]), np.array([.9, .8, .2, .1])) == 0.0
    assert auroc(np.array([0, 1, 0, 1]), np.array([.5, .5, .5, .5])) == 0.5
    with pytest.raises(ValueError):
        auroc(np.array([1, 1]), np.array([.2, .3]))


def test_threshold_and_confusion():
    y = np.array([0, 0, 0, 1, 1, 1]); s = np.array([.1, .2, .6, .4, .7, .9])
    assert accuracy(y, s, 0.5) == pytest.approx(4 / 6)
    t = best_threshold(y, s)
    assert accuracy(y, s, t) >= accuracy(y, s, 0.5)
    assert confusion(y, s, 0.5) == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}
    assert best_threshold(np.array([0, 1]), np.array([.5, .5])) == pytest.approx(0.5, abs=1e-6)


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0); y = rng.integers(0, 2, 400); s = rng.random(400) + 0.5 * y
    lo, hi = bootstrap_ci(y, s, n=300)
    assert lo < auroc(y, s) < hi


# ---- scoring from log-probs / logits ---------------------------------------------------------
def test_norm_token_variants():
    for t in ("Yes", " Yes", "▁Yes", "ĠYES", "yes\n"):
        assert norm_token(t) == "yes"


def test_score_from_logprobs_pools_variants_and_is_symmetric():
    lp = {"Yes": math.log(0.3), " Yes": math.log(0.3), "No": math.log(0.2), "The": math.log(0.2)}
    s, ok = score_from_logprobs(lp)
    assert ok and s == pytest.approx(0.6 / 0.8)
    s2, _ = score_from_logprobs({"No": math.log(0.3), " No": math.log(0.3), "Yes": math.log(0.2)})
    assert s2 == pytest.approx(1 - s)


def test_score_from_logprobs_edge_cases():
    assert score_from_logprobs({"hello": -0.1, "world": -3}) == (0.5, False)
    s, ok = score_from_logprobs({"Yes": -0.5})                     # "No" outside the top-k
    assert ok and s > 0.999999
    s, ok = score_from_logprobs({"No": -0.5})
    assert ok and s < 1e-6
    s, _ = score_from_logprobs({"Yes": -1000.0, "No": -0.0})      # no overflow
    assert 0 <= s < 1e-100


def test_score_from_logits():
    logits = np.zeros((2, 10)); logits[0, [1, 2]] = [2.0, 1.0]; logits[0, 5] = 0.0; logits[1, 5] = 3.0
    s = score_from_logits(logits, yes_ids=[1, 2], no_ids=[5])
    expect0 = (math.exp(2) + math.exp(1)) / (math.exp(2) + math.exp(1) + 1)
    assert s[0] == pytest.approx(expect0) and s[1] < 0.1
    big = np.zeros((1, 4)); big[0, 0] = 1e4
    assert np.isfinite(score_from_logits(big, [0], [1])).all()


# ---- vLLM output parsing (stand-in objects shaped like vllm.RequestOutput) --------------------
def fake_output(token_logprobs: dict[str, float]):
    lps = {i: SimpleNamespace(logprob=lp, decoded_token=tok) for i, (tok, lp) in enumerate(token_logprobs.items())}
    return SimpleNamespace(outputs=[SimpleNamespace(logprobs=[lps])])


def test_vllm_parsing():
    outs = [fake_output({" Yes": -0.1, " No": -2.4}), fake_output({" No": -0.05, "Yes": -3.0}), fake_output({"The": -0.1})]
    assert first_token_logprobs(outs[0]) == {" Yes": -0.1, " No": -2.4}
    scores, unparsed = scores_from_outputs(outs)
    assert scores[0] > 0.9 and scores[1] < 0.1 and scores[2] == 0.5 and unparsed == 1


# ---- label masking ----------------------------------------------------------------------------
def test_mask_prompt_labels():
    ids = np.array([[5, 6, 7, 8, 9, 0], [5, 6, 7, 8, 9, 10]]); att = np.array([[1, 1, 1, 1, 1, 0], [1] * 6])
    lab = mask_prompt_labels(ids, att, [3, 4])
    assert lab.tolist() == [[IGNORE] * 3 + [8, 9, IGNORE], [IGNORE] * 4 + [9, 10]]
    assert ids[0, 0] == 5                                    # input not modified
    with pytest.raises(ValueError):
        mask_prompt_labels(ids, att, [5, 4])                 # prompt swallows the whole row
    with pytest.raises(ValueError):
        mask_prompt_labels(np.array([[1, 2, 3]]), np.array([[0, 1, 1]]), [1])   # left padded


# ---- prompt -------------------------------------------------------------------------------------
def test_prompt_messages():
    m = build_messages('hi "there"')
    assert m[0]["content"][0] == {"type": "image"} and 'hi "there"' in m[0]["content"][1]["text"] and len(m) == 1
    assert build_messages("x", 1)[1]["content"][0]["text"] == "Yes" and build_messages("x", 0)[1]["content"][0]["text"] == "No"
    assert QUESTION.endswith("Answer Yes or No.")


# ---- data ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("hm")
    return str(d), make_fake_split(str(d), "train", 200, seed=1), make_fake_split(str(d), "dev_seen", 80, seed=2, start_id=5000)


def test_load_memes_and_split(fake):
    d, train, dev = fake
    rows = load_memes(train)
    assert len(rows) == 200 and rows[0]["path"].startswith(d) and set(rows[0]) == {"id", "path", "text", "label"}
    tr, va = stratified_split(rows, 0.1, seed=3)
    assert len(tr) + len(va) == 200 and not ({r["id"] for r in tr} & {r["id"] for r in va})
    assert abs(np.mean([r["label"] for r in va]) - np.mean([r["label"] for r in rows])) < 0.08
    assert [r["id"] for r in stratified_split(rows, 0.1, seed=3)[1]] == [r["id"] for r in va]      # deterministic
    assert [r["id"] for r in stratified_split(rows, 0.1, seed=4)[1]] != [r["id"] for r in va]


def test_load_memes_rejects_bad_input(tmp_path):
    img = tmp_path / "img"; img.mkdir(); Image.new("RGB", (4, 4)).save(img / "1.png")
    def check(rows, exc):
        write_jsonl(rows, str(tmp_path / "x.jsonl"))
        with pytest.raises(exc):
            load_memes(str(tmp_path / "x.jsonl"))
    good = {"id": 1, "img": "img/1.png", "label": 0, "text": "a"}
    check([{**good, "label": 2}], ValueError)
    check([{k: v for k, v in good.items() if k != "text"}], ValueError)
    check([good, good], ValueError)                                         # duplicate id
    check([{**good, "id": 2, "img": "img/missing.png"}], FileNotFoundError)
    unlabeled = {k: v for k, v in good.items() if k != "label"}
    check([unlabeled], ValueError)


# ---- report -------------------------------------------------------------------------------------
def _preds(path, ids, y, s):
    write_jsonl([{"id": int(i), "label": int(l), "score": float(x)} for i, l, x in zip(ids, y, s)], path)


def test_report_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    ids = np.arange(400); y = rng.integers(0, 2, 400)
    weak = np.clip(0.5 + 0.05 * (y - 0.5) + rng.normal(0, 0.1, 400), 0, 1)
    strong = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.1, 400), 0, 1)
    vid = np.arange(400, 600); vy = rng.integers(0, 2, 200)
    _preds(tmp_path / "w_dev.jsonl", ids, y, weak); _preds(tmp_path / "s_dev.jsonl", ids[::-1], y[::-1], strong[::-1])  # shuffled order
    _preds(tmp_path / "s_val.jsonl", vid, vy, np.clip(0.4 + 0.25 * (vy - 0.5) + rng.normal(0, 0.1, 200), 0, 1))
    rep = report.build_report({"zero_shot": str(tmp_path / "w_dev.jsonl"), "lora": str(tmp_path / "s_dev.jsonl")},
                              {"lora": str(tmp_path / "s_val.jsonl")})
    assert rep["runs"]["lora"]["auroc"] > rep["runs"]["zero_shot"]["auroc"]
    assert rep["runs"]["lora"]["auroc"] == pytest.approx(roc_auc_score(y, strong), abs=1e-12)
    d = rep["paired_auroc_diff"]; assert d["ci95"][0] > 0 and d["diff"] == pytest.approx(
        roc_auc_score(y, strong) - roc_auc_score(y, weak), abs=1e-12)
    assert rep["runs"]["zero_shot"]["at_val_threshold"]["threshold"] == 0.5          # no val file -> 0.5
    assert "| lora |" in report.to_markdown(rep)


def test_report_rejects_mismatched_runs_and_bad_scores(tmp_path):
    _preds(tmp_path / "a.jsonl", [1, 2, 3, 4], [0, 1, 0, 1], [.1, .9, .2, .8])
    _preds(tmp_path / "b.jsonl", [1, 2, 3, 5], [0, 1, 0, 1], [.1, .9, .2, .8])
    with pytest.raises(ValueError):
        report.build_report({"a": str(tmp_path / "a.jsonl"), "b": str(tmp_path / "b.jsonl")}, {})
    _preds(tmp_path / "n.jsonl", [1, 2], [0, 1], [float("nan"), .5])
    with pytest.raises(ValueError):
        report.load_preds(str(tmp_path / "n.jsonl"))


# ---- serve client (mock HTTP) -----------------------------------------------------------------------
def test_serve_client_roundtrip(fake):
    _, train, _ = fake
    row = load_memes(train)[0]
    seen = {}

    def handler(request: httpx.Request):
        body = json.loads(request.content); seen.update(body)
        top = [{"token": "Yes", "logprob": math.log(0.8)}, {"token": "No", "logprob": math.log(0.1)}]
        return httpx.Response(200, json={"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        s, ok, ms = serve_client.score_one(c, "http://x", "m", row)
    assert ok and s == pytest.approx(0.8 / 0.9) and ms >= 0
    assert seen["max_tokens"] == 1 and seen["logprobs"] is True
    assert seen["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert row["text"] in seen["messages"][0]["content"][1]["text"]


# ---- the GPU-side modules must at least import and show --help without torch ------------------------
@pytest.mark.parametrize("mod", ["train_lora", "predict_hf", "predict_vllm", "merge_lora", "serve_client", "report"])
def test_modules_import_without_gpu_stack(mod):
    r = subprocess.run([sys.executable, "-m", f"memes.{mod}", "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_filter_ids(fake):
    _, train, _ = fake
    rows = load_memes(train)
    _, va = stratified_split(rows, 0.1, seed=0)
    path = str(__import__("pathlib").Path(train).parent / "val_ids.jsonl")
    write_jsonl([{"id": r["id"]} for r in va], path)
    from memes.data import filter_ids
    assert [r["id"] for r in filter_ids(rows, path)] == sorted([r["id"] for r in va])
    write_jsonl([{"id": 999999}], path)
    with pytest.raises(ValueError):
        filter_ids(rows, path)
