"""Extract the Gram sub-blocks of the three Gate-1 architectures whose evidence the C3-start optimiser
under-maximised, as a small regression fixture (statistics of the 512 training-pool evidence rows only)."""
import json, sys
from pathlib import Path
import numpy as np
REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
p4 = json.loads(Path(sys.argv[3]).read_text())
z = np.load(REPO / "artifacts/probes/qnli_s42/features.npz")
Phi, t = z["Phi"].astype(np.float64), z["t"].astype(np.float64)
K, b = Phi.T @ Phi, Phi.T @ t
mods = [f"roberta.encoder.layer.{l}.attention.self.{p}" for l in range(12) for p in ("query", "value")]
defs = {"L11_qv_r2": {22: 2, 23: 2}, "L03_qv_r4": {6: 4, 7: 4}, "L10+L11_qv_r2": {20: 2, 21: 2, 22: 2, 23: 2}}
dense = {r["arch_id"]: r for r in p4["gate1_arch_evidence"]}
out = {"source": "artifacts/probes/qnli_s42/features.npz (512 QNLI evidence rows, training pool; no audit data)",
       "features_file_sha256": json.loads((REPO / "artifacts/probes/qnli_s42/probe_result.json").read_text())["hashes"]["features_file_sha256"],
       "problems": {}}
for name, d in defs.items():
    cols, colmod = [], []
    for local, (m, r) in enumerate(sorted(d.items())):
        cols += [m * 8 + j for j in range(r)]
        colmod += [local] * r
    out["problems"][name] = {"K": K[np.ix_(cols, cols)].tolist(), "b": b[cols].tolist(), "col_module": colmod,
                             "n_modules": len(d), "modules": [mods[m] for m in sorted(d)],
                             "v1_recorded_evidence": dense[name]["E_recorded"],
                             "dense_search_maximum": dense[name]["E_dense_multistart"]}
OUT.write_text(json.dumps(out, indent=1))
print({k: (v["v1_recorded_evidence"], v["dense_search_maximum"]) for k, v in out["problems"].items()})
