#!/usr/bin/env python3
"""Is the regime (control vs demurrage) linearly decodable from the state the
policy actually reads?  — the representation-learnability probe.

Motivation (docs/runs/run-25): A1 gave a POWERED-NO — even a grounded teacher did
not transmit "deposit-less under demurrage". The pre-registered follow-up for a
POWERED-NO is *representation learnability*: before concluding the policy can't
learn the contingency, check whether the contingency is even PRESENT and linearly
recoverable in the encoded observation. This run also introduced a stabilizing
`F.layer_norm` on the state (fixing the NaN divergence that crashed runs
#21/#22); LayerNorm discards magnitude, so we must also check it did not strip a
scale-encoded regime signal.

Method (encoder-only, no trained checkpoint needed — this asks what the FIXED
tokenizer+encoder transform preserves, which is what bounds any policy built on
top):
  1. Collect observations from the sandbox under control (cf off) and demurrage
     (cf on) across several held-in seeds, driven by the heuristic so the
     trajectory visits realistic deposit/demurrage states.
  2. Encode each obs with the real build_brain backbone: `raw` = encode_state,
     `normed` = the LayerNorm'd state the policy now consumes.
  3. Fit a linear probe (logistic regression) regime<-representation with a
     held-out split; report balanced accuracy for raw vs normed vs a
     shuffled-label control.

Reading:
  - normed ≈ raw ≫ chance  -> regime IS in the representation and LayerNorm keeps
    it; A1's failure is about the POLICY/credit, not the encoding. Caveat closed.
  - normed ≪ raw           -> LayerNorm strips a scale-encoded regime signal; the
    stabilization confounds A1. Flag for a norm redesign.
  - raw ≈ chance           -> the tokenizer/encoder does not surface regime at
    all; representation-learnability is the real wall (independent of the norm).

Requires the [neural] extra (torch + llm_model_agi). Engine-side is stdlib.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergence.brains.heuristic import HeuristicBrain          # noqa: E402
from emergence.grounding import make_grounding_sandbox         # noqa: E402


def _collect_observations(seeds, *, days, n_savers, sole_banker, demurrage_per_day):
    """Run the sandbox under both regimes; return (obs_list, regime_labels).

    regime label: 0 = control (cf off), 1 = demurrage (cf on). A recording brain
    captures every observation the engine hands to a decide() call, tagged by the
    regime the episode was built with."""
    records: list = []           # (observation, regime_label)

    def make_factory(regime_label):
        def factory(agent, persona, rng):
            base = HeuristicBrain(persona)

            class _Recorder(HeuristicBrain):
                def decide(self, ag, observation):
                    records.append((observation, regime_label))
                    return base.decide(ag, observation)

            return _Recorder(persona)
        return factory

    for seed in seeds:
        for regime_label, cf in ((0, False), (1, True)):
            sim = make_grounding_sandbox(
                "guardian", rule="demurrage", n_savers=n_savers, seed=seed,
                days=days, cf_enabled=cf, brain_factory=make_factory(regime_label),
                sole_banker=sole_banker, demurrage_per_day=demurrage_per_day)
            sim.run()

    obs = [r[0] for r in records]
    y = [r[1] for r in records]
    return obs, y


def _encode_all(dev, obs_list):
    """Return (raw, normed) numpy arrays (N, D) for the state representation."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    raws, normeds = [], []
    with torch.no_grad():
        for obs in obs_list:
            tokens = dev.tokenize(obs).to(dev.cfg.device)
            if hasattr(dev.backbone, "encode_state"):
                h = dev.backbone.encode_state(tokens)          # (1, D), pre-norm
            else:
                h = dev.backbone.encode(tokens)[:, -1, :]
            raws.append(h.squeeze(0).cpu().numpy())
            normeds.append(F.layer_norm(h, (h.shape[-1],)).squeeze(0).cpu().numpy())
    return np.asarray(raws), np.asarray(normeds)


def _linear_probe(X, y, *, seed=0):
    """Balanced accuracy of a logistic-regression regime probe, held-out split.
    Falls back to a torch linear probe if scikit-learn is unavailable."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    cut = int(n * 0.7)
    tr, te = idx[:cut], idx[cut:]
    y = np.asarray(y)

    def _bal_acc(y_true, y_pred):
        accs = []
        for c in (0, 1):
            m = y_true == c
            if m.any():
                accs.append((y_pred[m] == c).mean())
        return float(np.mean(accs)) if accs else float("nan")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        return _bal_acc(y[te], pred)
    except Exception:
        import torch
        Xt = torch.tensor(X, dtype=torch.float32)
        Xt = (Xt - Xt[tr].mean(0)) / (Xt[tr].std(0) + 1e-6)
        yt = torch.tensor(y, dtype=torch.float32)
        w = torch.zeros(Xt.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            logit = Xt[tr] @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt[tr])
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = ((Xt[te] @ w + b) > 0).float().numpy()
        return _bal_acc(y[te], pred)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003])
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--n-savers", type=int, default=5)
    ap.add_argument("--sole-banker", action="store_true", default=True)
    ap.add_argument("--demurrage-per-day", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0, help="probe/torch seed")
    args = ap.parse_args(argv)

    import random
    random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except Exception:
        sys.exit("[fatal] torch is required for this probe (install the [neural] extra).")

    print(f"[collect] sandbox obs under control+demurrage, seeds={args.seeds}, "
          f"days={args.days}, n_savers={args.n_savers}, "
          f"demurrage_per_day={args.demurrage_per_day}", flush=True)
    obs, y = _collect_observations(
        args.seeds, days=args.days, n_savers=args.n_savers,
        sole_banker=args.sole_banker, demurrage_per_day=args.demurrage_per_day)
    n0, n1 = y.count(0), y.count(1)
    print(f"[collect] {len(y)} observations  (control={n0}, demurrage={n1})", flush=True)

    from agent.adapters.emergence import build_brain
    dev = build_brain("guardian", None, None)
    print(f"[encode] backbone d_model={dev.backbone.encode_state(dev.tokenize(obs[0])).shape[-1]}",
          flush=True)
    raw, normed = _encode_all(dev, obs)

    import numpy as np
    acc_raw = _linear_probe(raw, y, seed=args.seed)
    acc_normed = _linear_probe(normed, y, seed=args.seed)
    y_shuf = list(y); random.shuffle(y_shuf)
    acc_shuffle = _linear_probe(normed, np.asarray(y_shuf), seed=args.seed)

    print("\n=== regime decodability (balanced accuracy, held-out 30%) ===")
    print(f"  raw encode_state (pre-norm) : {acc_raw:.3f}")
    print(f"  normed (LayerNorm'd; what the policy reads) : {acc_normed:.3f}")
    print(f"  shuffled-label control : {acc_shuffle:.3f}  (chance ~0.5)")
    drop = acc_raw - acc_normed
    print("\n[reading]")
    if acc_raw < 0.6:
        print("  raw ~ chance: the tokenizer/encoder does not surface regime at all "
              "-> representation-learnability is the real wall (norm-independent).")
    elif drop > 0.1:
        print(f"  normed << raw (drop {drop:+.3f}): LayerNorm strips a scale-encoded "
              "regime signal -> the stabilization confounds A1; redesign the norm "
              "(e.g. keep a scale channel) or probe the trained checkpoint.")
    else:
        print(f"  normed ~ raw (drop {drop:+.3f}), both > chance: regime IS present and "
              "LayerNorm keeps it -> A1's POWERED-NO is about the POLICY/credit, not "
              "the encoding. LayerNorm caveat closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
