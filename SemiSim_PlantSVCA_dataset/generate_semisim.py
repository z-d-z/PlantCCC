"""
===============================================================================
Semi-synthetic Plant-SVCA Benchmark Generator
===============================================================================
Backbone : real Stereo-seq Arabidopsis leaf cross-section
           665 spots × 18,266 genes, 6 cell types
Strategy : preserve all real coordinates, cell-type labels, library sizes, and
           gene-wise expression statistics; OVERWRITE only the genes that
           participate in the benchmark pairs with new expression generated
           by a plant-adapted SVCA model whose ground truth is fully known.

Outputs  :
  - simulated.h5ad         : AnnData with .X = semi-synthetic counts
                             (real for background, regenerated for benchmark genes)
  - simulated_layers.h5ad  : same but also stores 'real' counts in .layers
  - ground_truth.csv       : per-pair labels (communication, sigma, gamma², …)
  - neighbor_graph.npz     : plant-aware adjacency from real coordinates
  - simulation_parameters.json
===============================================================================
"""

import os, json, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
import anndata as ad

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------------
SEED = 20260513
rng  = np.random.default_rng(SEED)

SRC   = "/mnt/user-data/uploads/1778609833877_S2-6_stereoseq.h5ad"
OUTDIR= "/home/claude/semisim/data"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. LOAD real anndata
# -----------------------------------------------------------------------------
print("[1] Loading real Stereo-seq anndata ...")
adata = ad.read_h5ad(SRC)
print(f"    {adata}")

coords    = adata.obsm["spatial"].astype(float)
cell_type = adata.obs["cell_type"].astype(str).values
N         = adata.n_obs
G         = adata.n_vars
genes     = adata.var_names.values
X_real    = adata.X.tocsr()
libsize   = np.asarray(X_real.sum(axis=1)).ravel()
print(f"    cells: {N}, genes: {G}")
print(f"    library size: median={np.median(libsize):.0f}, mean={libsize.mean():.0f}")
print(f"    cell-type counts: {pd.Series(cell_type).value_counts().to_dict()}")

# -----------------------------------------------------------------------------
# 2. BUILD plant-aware neighbour graph from REAL coordinates
#    Same protocol as before: Delaunay -> length cutoff -> boundary penalty.
#    Length cutoff is chosen relative to the actual spot pitch on this slide.
# -----------------------------------------------------------------------------
print("\n[2] Building plant-aware neighbour graph from real coordinates ...")

tri = Delaunay(coords)
edges = set()
for s in tri.simplices:
    for i in range(3):
        a, b = s[i], s[(i+1) % 3]
        edges.add((min(a, b), max(a, b)))

edge_lengths = np.array([np.linalg.norm(coords[a]-coords[b]) for a,b in edges])
# data-driven cutoff: 2× the median nearest-neighbour distance
from scipy.spatial import cKDTree
tree = cKDTree(coords)
dnn, _ = tree.query(coords, k=2)
nn_dist = dnn[:, 1]
EDGE_CUTOFF = 2.5 * np.median(nn_dist)
print(f"    median NN distance = {np.median(nn_dist):.1f}  -> edge cutoff = {EDGE_CUTOFF:.1f}")

BOUNDARY_PENALTY = 0.6
rows, cols, vals = [], [], []
n_kept = 0; n_dropped = 0
for a, b in edges:
    d = np.linalg.norm(coords[a]-coords[b])
    if d > EDGE_CUTOFF:
        n_dropped += 1
        continue
    w = 1.0
    if cell_type[a] != cell_type[b]:
        w *= BOUNDARY_PENALTY
    rows.extend([a, b]); cols.extend([b, a]); vals.extend([w, w])
    n_kept += 1

A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
deg = np.asarray(A.sum(axis=1)).ravel()
print(f"    kept {n_kept} edges, dropped {n_dropped}, mean degree {deg.mean():.2f}")

# -----------------------------------------------------------------------------
# 3. HEAT KERNEL builder (same as Section 1 pipeline)
# -----------------------------------------------------------------------------
def build_kernel(A, sigma_hops, decay=0.55):
    N = A.shape[0]
    d = np.asarray(A.sum(axis=1)).ravel(); d[d==0]=1.0
    D_inv = sp.csr_matrix((1.0/d, (np.arange(N), np.arange(N))), shape=(N,N))
    P = D_inv @ A
    K = sp.csr_matrix((N,N), dtype=float)
    P_pow = P.copy()
    for h in range(1, sigma_hops+1):
        K = K + (decay**h) * P_pow
        if h < sigma_hops:
            P_pow = P_pow @ P
    K = K.toarray()
    rs = K.sum(axis=1, keepdims=True); rs[rs==0] = 1.0
    return K / rs

# -----------------------------------------------------------------------------
# 4. SEMI-SYNTHETIC EXPRESSION GENERATOR for a single gene
#
#    Steps for one gene to be overwritten:
#      a) generate z-scaled SVCA signal s (mean 0, var 1)
#      b) compute a per-spot Poisson rate that:
#           - matches the gene's global mean μ_g in the real data
#           - is modulated by exp(0.7 * s) so spatial structure shows up
#           - is scaled by the spot's real library size relative to median
#               -> this preserves real per-spot capture efficiency
#      c) sample counts ~ Poisson(rate) and optionally inflate zeros to
#         match the gene's real dropout fraction (so a model that uses
#         dropout patterns as a feature cannot "cheat" by spotting that
#         our genes are too dense).
# -----------------------------------------------------------------------------
def realise_counts(z_signal, target_gene_idx=None, target_mean=None,
                   target_dropout=None, libsize_scale=None, modulation=0.7):
    """Convert z-scaled signal into raw counts that match real-data stats."""
    if libsize_scale is None:
        libsize_scale = libsize / np.median(libsize)
    # base mean such that geometric mean across spots ≈ target_mean
    z_signal = (z_signal - z_signal.mean()) / (z_signal.std() + 1e-9)
    rate = np.exp(modulation * z_signal)
    rate = rate / rate.mean()                       # mean(rate) = 1
    rate = rate * float(target_mean) * libsize_scale
    counts = rng.poisson(rate)
    if target_dropout is not None:
        # impose extra dropout if real-data dropout is higher than realised
        cur_drop = (counts == 0).mean()
        if target_dropout > cur_drop + 0.05:
            extra = (target_dropout - cur_drop) / (1.0 - cur_drop + 1e-9)
            mask = rng.uniform(0, 1, len(counts)) < extra
            counts = counts * (~mask)
    return counts.astype(np.int32)

# helper: per-cell-type intrinsic offset (z-scaled)
def tissue_offset(origin_type, intensity=1.0, noise=0.30):
    """Generate z-scaled per-spot signal that is high in origin_type."""
    sigma = 0.0
    z = np.zeros(N)
    for t in np.unique(cell_type):
        m = cell_type == t
        z[m] = intensity if t == origin_type else -intensity/3
    z = z + rng.normal(0, noise, N)
    return (z - z.mean()) / (z.std() + 1e-9)

def env_field(length_scale_frac=0.15):
    """Gaussian random field on the bounding box, sampled at real coords."""
    g = 80
    xmin, xmax = coords[:,0].min(), coords[:,0].max()
    ymin, ymax = coords[:,1].min(), coords[:,1].max()
    xs = np.linspace(xmin, xmax, g); ys = np.linspace(ymin, ymax, g)
    grid = rng.normal(0, 1, (g, g))
    # use length scale as a fraction of the diagonal
    diag = np.hypot(xmax-xmin, ymax-ymin)
    sigma_px = (length_scale_frac * diag) / max(xs[1]-xs[0], ys[1]-ys[0])
    smooth = gaussian_filter(grid, sigma=sigma_px, mode="reflect")
    ix = np.clip(((coords[:,0]-xmin)/(xmax-xmin) * (g-1)).astype(int), 0, g-1)
    iy = np.clip(((coords[:,1]-ymin)/(ymax-ymin) * (g-1)).astype(int), 0, g-1)
    v  = smooth[iy, ix]
    return (v - v.mean())/(v.std()+1e-9)

# -----------------------------------------------------------------------------
# 5. DEFINE benchmark pairs and overwrite their expression
# -----------------------------------------------------------------------------
print("\n[5] Defining benchmark pairs ...")

# Both genes in each pair MUST exist in the data (already verified)
# Tissue of origin matches the published biology of each peptide system.
TRUE_PAIRS = [
    # ligand_sym, ligand_agi,  receptor_sym, receptor_agi,  L_tissue,        R_tissue,        sigma, gamma2, label
    ("EPFL9",  "AT4G12970",  "ER",     "AT2G26330",
        "Palisade_mesophyll_cell", "Upper_epidermal_cell",  3, 0.40, "mesophyll_to_epidermis_medium"),
    ("CLE41",  "AT3G24770",  "PXY",    "AT5G61480",
        "Vascular_cell",           "Vascular_cell",         1, 0.45, "vascular_intra_short"),
    ("CLE9",   "AT1G26600",  "BAM1",   "AT5G65700",
        "Guard_cell",              "Upper_epidermal_cell",  2, 0.35, "guard_to_epidermis_short"),
    ("CIF2",   "AT4G34600",  "SGN3",   "AT4G20140",
        "Vascular_cell",           "Vascular_cell",         2, 0.40, "vascular_intra_medium"),
    ("PSK4",   "AT3G49780",  "PSKR1",  "AT2G02220",
        "Spongy_mesophyll_cell",   "Lower_epidermal_cell",  5, 0.30, "long_range_systemic"),
]

# Co-localization confounders: pick REAL tissue-specific non-LR genes,
# expressed in same tissue as a true ligand and as a true receptor, but
# WITHOUT any kernel coupling between them.
CONF_PAIRS = [
    # uses real high-specificity epidermis/mesophyll/vasc/guard genes found in find_confounder.py
    ("CONF_UE1_lig", "AT1G66100", "CONF_UE1_rec", "AT2G38540",
        "Upper_epidermal_cell", "Upper_epidermal_cell"),
    ("CONF_SM1_lig", "AT1G20620", "CONF_SM1_rec", "AT4G19420",
        "Spongy_mesophyll_cell","Spongy_mesophyll_cell"),
    ("CONF_LE1_lig", "AT1G68530", "CONF_LE1_rec", "AT2G28630",
        "Lower_epidermal_cell", "Lower_epidermal_cell"),
    ("CONF_GC1_lig", "AT5G25220", "CONF_GC1_rec", "AT3G61690",
        "Guard_cell",           "Guard_cell"),
    ("CONF_X1_lig",  "AT1G09310", "CONF_X1_rec",  "AT3G16370",
        "Upper_epidermal_cell", "Upper_epidermal_cell"),
]

# Spatially-shuffled random pairs: use real moderately-expressed genes but
# resample expression i.i.d. from negative binomial with the gene's real mean
# and a moderate dispersion. These should give ~zero spatial signal.
RAND_PAIRS = [
    ("RAND1_lig", "AT3G46970", "RAND1_rec", "AT4G33920"),
    ("RAND2_lig", "AT3G15210", "RAND2_rec", "AT1G29920"),
    ("RAND3_lig", "AT1G51500", "RAND3_rec", "AT1G65490"),
    ("RAND4_lig", "ATCG00020", "RAND4_rec", "AT4G27870"),
    ("RAND5_lig", "ATCG00490", "RAND5_rec", "AT3G41768"),
]

# precompute per-gene real-data stats
gene_to_idx = {g: i for i, g in enumerate(genes)}
def gene_stats(agi):
    i = gene_to_idx[agi]
    col = np.asarray(X_real[:, i].todense()).ravel()
    return {"mean": col.mean(), "dropout": (col == 0).mean(), "max": col.max(),
            "nnz": int((col > 0).sum())}

# We need to OVERWRITE columns of X_real with new counts for these genes
X_new = X_real.tolil()           # easy column overwrite
ground_truth = []
gene_was_overwritten = np.zeros(G, dtype=bool)

env_a = env_field(0.12)
env_b = env_field(0.08)

# ---------- 5A. TRUE L-R pairs ----------------------------------------------
print("\n  Overwriting TRUE L-R pair genes ...")
ALPHA2 = 0.30; BETA2 = 0.20
for (lig, lagi, rec, ragi, lt, rt, sigma, gamma2, label) in TRUE_PAIRS:
    # ligand: tissue-specific intrinsic + env, no interaction term
    z_lig = (np.sqrt(0.50) * tissue_offset(lt, 1.0, 0.25)
           + np.sqrt(0.20) * env_a
           + rng.normal(0, 0.30, N))
    # receptor: intrinsic in receptor tissue + env + interaction(K @ lig_signal) + noise
    K = build_kernel(A, sigma)
    interact = K @ z_lig
    interact = (interact - interact.mean()) / (interact.std()+1e-9)
    z_rec = (np.sqrt(ALPHA2) * tissue_offset(rt, 1.0, 0.25)
           + np.sqrt(BETA2)  * env_b
           + np.sqrt(gamma2) * interact
           + rng.normal(0, 0.25, N))

    # match real-data stats for each gene (overwrite mean to a floor of 0.5 so
    # there is meaningful signal even when original mean is near zero — we are
    # purposely making these "what the gene WOULD look like if well-captured")
    s_lig = gene_stats(lagi); s_rec = gene_stats(ragi)
    target_mean_lig = max(s_lig["mean"], 0.7)         # ensure detectable signal
    target_mean_rec = max(s_rec["mean"], 0.7)
    cnt_lig = realise_counts(z_lig, target_mean=target_mean_lig,
                             target_dropout=min(s_lig["dropout"], 0.5))
    cnt_rec = realise_counts(z_rec, target_mean=target_mean_rec,
                             target_dropout=min(s_rec["dropout"], 0.5))

    li, ri = gene_to_idx[lagi], gene_to_idx[ragi]
    X_new[:, li] = cnt_lig.reshape(-1,1)
    X_new[:, ri] = cnt_rec.reshape(-1,1)
    gene_was_overwritten[li] = True
    gene_was_overwritten[ri] = True

    ground_truth.append(dict(
        ligand=lig, ligand_agi=lagi, receptor=rec, receptor_agi=ragi,
        ligand_tissue=lt, receptor_tissue=rt,
        sigma_hops=sigma, gamma2_true=gamma2, label=label,
        communication=True, role="true_lr",
        alpha2=ALPHA2, beta2=BETA2,
        real_mean_ligand=s_lig["mean"], real_dropout_ligand=s_lig["dropout"],
        real_mean_receptor=s_rec["mean"], real_dropout_receptor=s_rec["dropout"],
        target_mean_ligand=target_mean_lig, target_mean_receptor=target_mean_rec,
    ))
    print(f"    [TRUE] {lig:7s}({lagi})->{rec:6s}({ragi})  σ={sigma}  γ²={gamma2:.2f}  "
          f"label={label}")

# ---------- 5B. Co-localization confounders ---------------------------------
print("\n  Overwriting CONFOUNDER pair genes ...")
for (lig, lagi, rec, ragi, lt, rt) in CONF_PAIRS:
    # BOTH genes have tissue-specific intrinsic + env, NO interaction
    z_lig = (np.sqrt(0.55) * tissue_offset(lt, 1.0, 0.30)
           + np.sqrt(0.20) * env_a
           + rng.normal(0, 0.30, N))
    z_rec = (np.sqrt(0.55) * tissue_offset(rt, 1.0, 0.30)
           + np.sqrt(0.20) * env_b
           + rng.normal(0, 0.30, N))
    s_lig = gene_stats(lagi); s_rec = gene_stats(ragi)
    target_mean_lig = max(s_lig["mean"], 0.7)
    target_mean_rec = max(s_rec["mean"], 0.7)
    cnt_lig = realise_counts(z_lig, target_mean=target_mean_lig,
                             target_dropout=min(s_lig["dropout"], 0.5))
    cnt_rec = realise_counts(z_rec, target_mean=target_mean_rec,
                             target_dropout=min(s_rec["dropout"], 0.5))
    li, ri = gene_to_idx[lagi], gene_to_idx[ragi]
    X_new[:, li] = cnt_lig.reshape(-1,1)
    X_new[:, ri] = cnt_rec.reshape(-1,1)
    gene_was_overwritten[li] = True
    gene_was_overwritten[ri] = True
    ground_truth.append(dict(
        ligand=lig, ligand_agi=lagi, receptor=rec, receptor_agi=ragi,
        ligand_tissue=lt, receptor_tissue=rt,
        sigma_hops=None, gamma2_true=0.0, label="colocalization_confounder",
        communication=False, role="confounder",
        alpha2=0.55, beta2=0.20,
        real_mean_ligand=s_lig["mean"], real_dropout_ligand=s_lig["dropout"],
        real_mean_receptor=s_rec["mean"], real_dropout_receptor=s_rec["dropout"],
        target_mean_ligand=target_mean_lig, target_mean_receptor=target_mean_rec,
    ))
    print(f"    [CONF] {lig:14s}({lagi})<->{rec:14s}({ragi})  shared tissue={lt}")

# ---------- 5C. Spatially-random pairs --------------------------------------
print("\n  Overwriting RANDOM pair genes ...")
for (lig, lagi, rec, ragi) in RAND_PAIRS:
    z_lig = rng.normal(0, 1, N)
    z_rec = rng.normal(0, 1, N)
    s_lig = gene_stats(lagi); s_rec = gene_stats(ragi)
    target_mean_lig = max(s_lig["mean"], 0.7)
    target_mean_rec = max(s_rec["mean"], 0.7)
    cnt_lig = realise_counts(z_lig, target_mean=target_mean_lig,
                             target_dropout=min(s_lig["dropout"], 0.5))
    cnt_rec = realise_counts(z_rec, target_mean=target_mean_rec,
                             target_dropout=min(s_rec["dropout"], 0.5))
    li, ri = gene_to_idx[lagi], gene_to_idx[ragi]
    X_new[:, li] = cnt_lig.reshape(-1,1)
    X_new[:, ri] = cnt_rec.reshape(-1,1)
    gene_was_overwritten[li] = True
    gene_was_overwritten[ri] = True
    ground_truth.append(dict(
        ligand=lig, ligand_agi=lagi, receptor=rec, receptor_agi=ragi,
        ligand_tissue="random", receptor_tissue="random",
        sigma_hops=None, gamma2_true=0.0, label="uniform_random",
        communication=False, role="random",
        alpha2=None, beta2=None,
        real_mean_ligand=s_lig["mean"], real_dropout_ligand=s_lig["dropout"],
        real_mean_receptor=s_rec["mean"], real_dropout_receptor=s_rec["dropout"],
        target_mean_ligand=target_mean_lig, target_mean_receptor=target_mean_rec,
    ))
    print(f"    [RAND] {lig:10s}({lagi})  {rec:10s}({ragi})")

print(f"\n[5] Done. Overwrote {gene_was_overwritten.sum()} genes "
      f"({gene_was_overwritten.sum()/G*100:.2f}% of transcriptome). "
      f"Remaining {G - gene_was_overwritten.sum()} genes keep their REAL counts.")

# -----------------------------------------------------------------------------
# 6. Build the simulated AnnData
# -----------------------------------------------------------------------------
print("\n[6] Building output AnnData ...")
X_sim = X_new.tocsr().astype(np.int32)
adata_sim = ad.AnnData(
    X=X_sim,
    obs=adata.obs.copy(),
    var=adata.var.copy(),
    obsm={"spatial": coords},
)
# enrich var with overwritten flags
adata_sim.var["overwritten_by_sim"] = gene_was_overwritten
# enrich obs
adata_sim.obs["library_size_real"] = libsize.astype(int)
adata_sim.obs["library_size_sim"]  = np.asarray(X_sim.sum(axis=1)).ravel().astype(int)
# attach uns
adata_sim.uns["simulation"] = {
    "seed": SEED,
    "edge_cutoff": float(EDGE_CUTOFF),
    "boundary_penalty": float(BOUNDARY_PENALTY),
    "n_true_pairs": len(TRUE_PAIRS),
    "n_confounder_pairs": len(CONF_PAIRS),
    "n_random_pairs": len(RAND_PAIRS),
    "alpha2": ALPHA2, "beta2": BETA2,
}

# Also store the real counts in a layer for transparency
adata_sim.layers["real_counts"] = X_real.astype(np.int32)
adata_sim.layers["sim_counts"]  = X_sim

# -----------------------------------------------------------------------------
# 7. Save everything
# -----------------------------------------------------------------------------
print("\n[7] Saving outputs ...")
adata_sim.write_h5ad(os.path.join(OUTDIR, "semisim_plant_svca.h5ad"), compression="gzip")

gt_df = pd.DataFrame(ground_truth)
gt_df.to_csv(os.path.join(OUTDIR, "ground_truth.csv"), index=False)

sp.save_npz(os.path.join(OUTDIR, "neighbor_graph.npz"), A)

# edge list
ri, ci = A.nonzero(); m = ri < ci
edge_df = pd.DataFrame({"cell_a": ri[m], "cell_b": ci[m],
                        "weight": np.asarray(A[ri[m], ci[m]]).ravel(),
                        "len": np.linalg.norm(coords[ri[m]] - coords[ci[m]], axis=1)})
edge_df.to_csv(os.path.join(OUTDIR, "neighbor_edges.csv"), index=False)

params = {
    "seed": SEED,
    "source_dataset": os.path.basename(SRC),
    "n_cells": int(N), "n_genes_total": int(G),
    "n_genes_overwritten": int(gene_was_overwritten.sum()),
    "cell_types": sorted(set(cell_type)),
    "neighbor_graph": {
        "method": "Delaunay + length cutoff + tissue-boundary penalty",
        "edge_cutoff_units": float(EDGE_CUTOFF),
        "boundary_penalty": float(BOUNDARY_PENALTY),
        "median_nn_distance": float(np.median(nn_dist)),
        "n_edges": int(A.nnz // 2), "mean_degree": float(deg.mean()),
    },
    "svca_decomposition": {
        "formula": "y = alpha*X_intrinsic + beta*X_env + gamma*(K_sigma L) + epsilon",
        "alpha2_target": ALPHA2, "beta2_target": BETA2,
        "noise_sd_on_z_scale": 0.25,
        "kernel": "rownorm(sum_{h=1..sigma} 0.55^h * P^h) on plant graph",
    },
    "count_model": "Poisson(target_mean * exp(0.7*z)/mean * libsize/median_libsize)",
    "pairs": {
        "true_lr_pairs":       [dict(zip(["ligand","ligand_agi","receptor","receptor_agi",
                                          "ligand_tissue","receptor_tissue","sigma_hops",
                                          "gamma2","label"], p)) for p in TRUE_PAIRS],
        "confounder_pairs":    [dict(zip(["ligand","ligand_agi","receptor","receptor_agi",
                                          "ligand_tissue","receptor_tissue"], p)) for p in CONF_PAIRS],
        "random_pairs":        [dict(zip(["ligand","ligand_agi","receptor","receptor_agi"], p))
                                 for p in RAND_PAIRS],
    },
}
with open(os.path.join(OUTDIR, "simulation_parameters.json"), "w") as f:
    json.dump(params, f, indent=2, default=str)

print(f"\n[OK] All outputs in {OUTDIR}")
for fn in sorted(os.listdir(OUTDIR)):
    sz = os.path.getsize(os.path.join(OUTDIR, fn)) / 1024
    print(f"     {fn:35s}  {sz:8.1f} KB")
