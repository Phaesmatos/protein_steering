"""Activation steering for ESM-2. Research demonstration; no wet-lab validation."""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
XY = "APRKQESL"


def make_collagen(n, repeats=12, seed=42):
    """Synthetic G-X-Y motifs; negatives are composition-preserving phase shifts."""
    r = random.Random(seed)
    pairs = [(r.choice(XY), r.choice(XY)) for _ in range(n)]
    positives = ["".join("G" + x + y for _ in range(repeats)) for x, y in pairs]
    negatives = [s[1:] + s[:1] for s in positives]
    return list(zip(positives, negatives))


def load_pairs(path, seq_col, label_col, high, low, n, seed):
    pos, neg = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                seq = str(row[seq_col]).strip().upper().replace(" ", "")
                val = float(row[label_col])
            except (KeyError, ValueError, TypeError):
                continue
            if not 12 <= len(seq) <= 350 or any(a not in AA for a in seq):
                continue
            if val >= high:
                pos.append(seq)
            elif val <= low:
                neg.append(seq)
    if len(pos) < n or len(neg) < n:
        raise ValueError(f"Insufficient CSV examples: positive={len(pos)} negative={len(neg)}; need {n} each")
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    return list(zip(pos[:n], neg[:n]))


def model_setup(model_id, device):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).to(dev).eval()
    return tok, model, dev


def representation(model, tok, seq, layer, dev, position=None):
    ids = tok(seq, return_tensors="pt", truncation=False).to(dev)
    with torch.inference_mode():
        out = model.esm(**ids, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[layer + 1][0]
        # ESM prepends CLS, adds EOS; amino-acid token positions are 1..len(seq).
        if position is None:
            return h[1: len(seq) + 1].float().mean(0).cpu()
        return h[position + 1].float().cpu()


def direction(model, tok, pairs, layer, dev, motif=False):
    vectors = []
    for pos, neg in pairs:
        hp = representation(model, tok, pos, layer, dev, 0 if motif else None)
        hn = representation(model, tok, neg, layer, dev, 0 if motif else None)
        vectors.append(hp - hn)
    vec = torch.stack(vectors).mean(0)
    norm = vec.norm().item()
    if norm < 1e-9:
        raise RuntimeError("Steering vector is nearly zero")
    return vec / norm, norm


@contextmanager
def add_steering(model, layer, vector, alpha, mask_idx, dev):
    """Edit ESM encoder output at masked-token position, with forward hook."""
    if alpha == 0:
        yield
        return
    block = model.esm.encoder.layer[layer]
    vec = vector.to(dev)
    def hook(_module, _inputs, outputs):
        # Hugging Face EsmLayer returns tuple(hidden_states, attentions?)
        if isinstance(outputs, tuple):
            h = outputs[0].clone()
            h[:, mask_idx, :] += alpha * vec.to(h.dtype)
            return (h,) + outputs[1:]
        h = outputs.clone()
        h[:, mask_idx, :] += alpha * vec.to(h.dtype)
        return h
    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def masked_probability(model, tok, seq, position, target, layer, vec, alpha, dev):
    chars = list(seq)
    chars[position] = tok.mask_token
    input_seq = "".join(chars)
    encoded = tok(input_seq, return_tensors="pt", add_special_tokens=True).to(dev)
    mask_pos = torch.nonzero(encoded["input_ids"][0] == tok.mask_token_id).flatten()
    if mask_pos.numel() != 1:
        raise ValueError("Expected precisely one mask")
    mi = int(mask_pos.item())
    with add_steering(model, layer, vec, alpha, mi, dev):
        with torch.inference_mode():
            logits = model(**encoded).logits[0, mi].float()
    target_ids = tok(target, add_special_tokens=False)["input_ids"]
    if len(target_ids) != 1:
        raise ValueError(f"Unexpected token mapping for {target}")
    return float(logits.softmax(-1)[target_ids[0]].item())


def evaluate_motif(model, tok, heldout, layer, vec, alphas, dev):
    rows = []
    controls = {"steering": vec, "reverse": -vec}
    g = torch.Generator().manual_seed(912)
    controls["random"] = torch.randn(vec.shape, generator=g)
    controls["random"] /= controls["random"].norm()
    for alpha in alphas:
        for condition, v in controls.items():
            for phase, idx in [("G_phase", 6), ("non_G_phase", 7)]:
                values = [masked_probability(model, tok, positive, idx, "G", layer, v, alpha, dev) for positive, _ in heldout]
                rows.append({"alpha": alpha, "condition": condition, "phase": phase,
                             "mean_p_g": sum(values) / len(values), "n": len(values)})
    return rows


def evaluate_csv(model, tok, heldout, layer, vec, alphas, dev, seed):
    # Diagnostic only: probability of originally observed aa at a fixed held-out site.
    # It is NOT a measure of thermostability/solubility or biological success.
    rng = random.Random(seed + 53)
    rows = []
    for label, col in [("high", 0), ("low", 1)]:
        samples = [(pair[col], rng.randrange(2, len(pair[col])-2)) for pair in heldout]
        for alpha in alphas:
            for cond, v in [("steering", vec), ("reverse", -vec)]:
                values = [masked_probability(model, tok, seq, i, seq[i], layer, v, alpha, dev) for seq, i in samples]
                rows.append({"alpha": alpha, "condition": cond, "group": label,
                             "mean_p_native_aa": sum(values) / len(values), "n": len(values)})
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["collagen", "csv"], default="collagen")
    ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--layer", type=int, default=None, help="zero-based ESM encoder layer")
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-test", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 1, 3, 6, 12])
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--output", default="results/run")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--high", type=float, default=0.8)
    ap.add_argument("--low", type=float, default=0.2)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    tok, model, dev = model_setup(args.model, args.device)
    n_layers = len(model.esm.encoder.layer)
    layer = args.layer if args.layer is not None else n_layers // 2
    if not 0 <= layer < n_layers:
        ap.error(f"--layer must be 0..{n_layers-1}")
    if args.experiment == "collagen":
        pairs = make_collagen(args.n_train + args.n_test, seed=args.seed)
    else:
        if not args.csv:
            ap.error("--csv required with --experiment csv")
        pairs = load_pairs(args.csv, args.seq_col, args.label_col, args.high,
                           args.low, args.n_train + args.n_test, args.seed)
    train, test = pairs[:args.n_train], pairs[args.n_train:]
    print(f"Model={args.model} device={dev} layer={layer} train={len(train)} test={len(test)}")
    v, raw_norm = direction(model, tok, train, layer, dev, motif=args.experiment == "collagen")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": v, "model": args.model, "layer": layer, "train_examples": len(train)}, out / "direction.pt")
    if args.experiment == "collagen":
        rows = evaluate_motif(model, tok, test, layer, v, args.alphas, dev)
    else:
        rows = evaluate_csv(model, tok, test, layer, v, args.alphas, dev, args.seed)
    write_csv(out / "metrics.csv", rows)
    metadata = vars(args) | {"device_actual": str(dev), "layer_actual": layer,
                              "direction_raw_norm": raw_norm,
                              "warning": "in silico diagnostic only; no function/structure validation"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for r in rows:
        print(r)
    print(f"Saved: {out.resolve()}")

if __name__ == "__main__":
    main()
