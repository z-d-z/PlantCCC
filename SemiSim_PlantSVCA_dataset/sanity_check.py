"""Sanity-check the semi-synthetic dataset:
   1) True L-R pairs should show stronger bivariate Moran's I than negative controls
   2) Overwritten genes should retain real-data global statistics (mean, dropout)
"""
import numpy as np, pandas as pd, anndata as ad
import scipy.sparse as sp
from scipy.stats import pearsonr

DATA = "/home/claude/semisim/data"
adata = ad.read_h5ad(f"{DATA}/semisim_plant_svca.h5ad")
gt    = pd.read_csv(f"{DATA}/ground_truth.csv")
A     = sp.load_npz(f"{DATA}/neighbor_graph.npz")
genes = adata.var_names.values
gene_idx = {g: i for i, g in enumerate(genes)}

X_sim  = adata.X.toarray()
X_real = adata.layers["real_counts"].toarray()

def lag(v, A):
    d = np.asarray(A.sum(axis=1)).ravel(); d[d==0]=1.0
    return (A @ v) / d

print("=" * 100)
print("PART 1: Detectability of benchmark pairs (bivariate Moran's I of L vs lagged-R)")
print("=" * 100)
print(f"{'ligand':12s}{'receptor':12s}{'role':12s}{'label':32s}{'γ²':>7s}{'I_sim':>9s}{'I_real':>9s}")
print("-" * 100)

I_sim_by_role = {"true_lr": [], "confounder": [], "random": []}
for _, r in gt.iterrows():
    li, ri = gene_idx[r.ligand_agi], gene_idx[r.receptor_agi]
    L_sim, R_sim = X_sim[:, li], X_sim[:, ri]
    L_real, R_real = X_real[:, li], X_real[:, ri]
    r_sim = pearsonr(lag(L_sim, A), R_sim)[0] if R_sim.std() > 0 else 0
    r_real = pearsonr(lag(L_real, A), R_real)[0] if R_real.std() > 0 else 0
    I_sim_by_role[r.role].append(r_sim)
    g = "-" if (pd.isna(r.gamma2_true) or r.gamma2_true == 0) else f"{r.gamma2_true:.2f}"
    print(f"{r.ligand:12s}{r.receptor:12s}{r.role:12s}{r.label:32s}{g:>7s}"
          f"{r_sim:>9.3f}{r_real:>9.3f}")

print("\n--- group summary (sim) ---")
for k, v in I_sim_by_role.items():
    print(f"  {k:12s} n={len(v)}  mean I = {np.mean(v):+.3f}   range [{np.min(v):+.3f}, {np.max(v):+.3f}]")

print("\n" + "=" * 100)
print("PART 2: Statistical preservation — overwritten genes should retain real-data feel")
print("=" * 100)
print(f"\n{'gene':25s}{'role':12s}{'real_mean':>11s}{'sim_mean':>11s}{'real_drop':>11s}{'sim_drop':>11s}")
print("-" * 90)
for _, r in gt.iterrows():
    for tag, agi, sym in [("L", r.ligand_agi, r.ligand), ("R", r.receptor_agi, r.receptor)]:
        i = gene_idx[agi]
        rm, sm = X_real[:, i].mean(), X_sim[:, i].mean()
        rd, sd = (X_real[:, i] == 0).mean(), (X_sim[:, i] == 0).mean()
        label = f"{sym}|{agi}"
        print(f"{label:25s}{r.role:12s}{rm:>11.3f}{sm:>11.3f}{rd:>11.3f}{sd:>11.3f}")

print("\n" + "=" * 100)
print("PART 3: Library-size preservation — total counts per spot should stay similar")
print("=" * 100)
lib_real = X_real.sum(axis=1)
lib_sim  = X_sim.sum(axis=1)
print(f"  real lib size: median={np.median(lib_real):.0f}  mean={lib_real.mean():.0f}  "
      f"sd={lib_real.std():.0f}")
print(f"  sim  lib size: median={np.median(lib_sim):.0f}  mean={lib_sim.mean():.0f}  "
      f"sd={lib_sim.std():.0f}")
print(f"  per-spot lib-size correlation: r = {pearsonr(lib_real, lib_sim)[0]:.4f}")

print("\n" + "=" * 100)
print("PART 4: Non-overwritten genes are byte-identical to real data")
print("=" * 100)
mask_kept = ~adata.var["overwritten_by_sim"].values
diff = (X_sim[:, mask_kept] - X_real[:, mask_kept])
n_diff = (diff != 0).sum()
print(f"  {mask_kept.sum()} non-overwritten genes; differences vs real_counts layer: {n_diff}")
print(f"  -> {'PASS' if n_diff == 0 else 'FAIL'}: real background fully preserved")
