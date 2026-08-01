"""Generate figures for the supplementary materials report."""
import os
import numpy as np, pandas as pd, anndata as ad
import matplotlib.pyplot as plt
import scipy.sparse as sp
from matplotlib.collections import LineCollection
from scipy.stats import pearsonr

DATA = "/home/claude/semisim/data"
FIGS = "/home/claude/semisim/figures"
os.makedirs(FIGS, exist_ok=True)

adata = ad.read_h5ad(f"{DATA}/semisim_plant_svca.h5ad")
gt    = pd.read_csv(f"{DATA}/ground_truth.csv")
A     = sp.load_npz(f"{DATA}/neighbor_graph.npz")
coords = adata.obsm["spatial"]
ct = adata.obs["cell_type"].astype(str).values
genes = adata.var_names.values
gene_idx = {g:i for i,g in enumerate(genes)}

# colors for cell types
CT_COL = {
    "Upper_epidermal_cell":   "#3B6FB2",
    "Palisade_mesophyll_cell":"#2CA02C",
    "Spongy_mesophyll_cell":  "#8C6D31",
    "Lower_epidermal_cell":   "#9467BD",
    "Vascular_cell":          "#C44545",
    "Guard_cell":             "#FF7F0E",
}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "figure.dpi": 130, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})

# =============================================================================
# FIG S1: Real leaf cross-section + plant-aware neighbour graph
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(13, 5.4))

ax = axes[0]
for t, c in CT_COL.items():
    m = ct == t
    ax.scatter(coords[m,0], coords[m,1], s=22, c=c, label=f"{t.replace('_',' ')} (n={m.sum()})",
               edgecolors="white", linewidths=0.4)
ax.set_aspect("equal")
ax.set_title("A. Real Stereo-seq A. thaliana leaf cross-section (665 spots, 6 cell types)")
ax.legend(loc="upper center", fontsize=7, frameon=False, ncol=6, bbox_to_anchor=(0.5, -0.05))
ax.set_xticks([]); ax.set_yticks([])
for s in ("top","right","bottom","left"): ax.spines[s].set_visible(False)

ax = axes[1]
rows_, cols_ = A.nonzero(); mask = rows_ < cols_
segs = np.stack([coords[rows_[mask]], coords[cols_[mask]]], axis=1)
ax.add_collection(LineCollection(segs, colors="#888888", linewidths=0.3, alpha=0.7))
for t, c in CT_COL.items():
    m = ct == t
    ax.scatter(coords[m,0], coords[m,1], s=14, c=c, edgecolors="none", zorder=3)
ax.set_aspect("equal")
ax.set_title(f"B. Plant-aware neighbour graph built from real coordinates "
             f"({A.nnz//2} edges, mean degree {A.nnz/A.shape[0]:.2f})")
ax.set_xticks([]); ax.set_yticks([])
for s in ("top","right","bottom","left"): ax.spines[s].set_visible(False)

plt.suptitle("Figure S1.  Real spatial backbone of the semi-synthetic benchmark",
             y=1.02, fontsize=11, fontweight="bold")
plt.savefig(f"{FIGS}/FigS1_real_backbone.png"); plt.close()

# =============================================================================
# FIG S2: Real vs sim — preservation of background, overwriting of LR genes
# =============================================================================
fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))

# show one background gene that should look identical between real and sim
bg_gene_idx = np.where(~adata.var["overwritten_by_sim"].values)[0]
nnz_bg = np.asarray((adata.layers["real_counts"][:, bg_gene_idx] > 0).sum(axis=0)).ravel()
gi = bg_gene_idx[np.argmax(nnz_bg)]                # background gene with most expression
bg_name = genes[gi]

# show one true LR ligand (CIF2) before & after
ovr_lig_idx = gene_idx["AT4G34600"]                # CIF2
ovr_rec_idx = gene_idx["AT4G20140"]                # SGN3

for col, (idx, title, kind) in enumerate([
    (gi,           f"Background gene\n{bg_name}",       "bg"),
    (ovr_lig_idx,  "CIF2 (TRUE ligand)\nAT4G34600",     "lig"),
    (ovr_rec_idx,  "SGN3 (TRUE receptor)\nAT4G20140",   "rec"),
    (gene_idx["AT1G66100"], "CONF_UE1_lig\n(real epidermis gene)", "conf"),
]):
    # row 0 = real, row 1 = sim
    for row, mat_name in enumerate(["real_counts", "sim_counts"]):
        ax = axes[row, col]
        v = np.asarray(adata.layers[mat_name][:, idx].todense()).ravel()
        sc = ax.scatter(coords[:,0], coords[:,1], c=v, s=8, cmap="Reds",
                        vmin=0, vmax=max(v.max(), 1), edgecolors="none")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ("top","right","bottom","left"): ax.spines[s].set_visible(False)
        ttl = (f"REAL — {title}" if row==0 else f"SIM — {title}")
        ax.set_title(ttl, fontsize=8.5)
        plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02).ax.tick_params(labelsize=7)
plt.suptitle("Figure S2.  Real vs semi-synthetic expression for representative genes\n"
             "(left column: background gene preserved byte-identical; right columns: regenerated)",
             y=1.01, fontsize=11, fontweight="bold")
plt.savefig(f"{FIGS}/FigS2_real_vs_sim.png"); plt.close()

# =============================================================================
# FIG S3: All TRUE L-R pairs — ligand vs receptor spatial maps
# =============================================================================
true_pairs = gt[gt["role"] == "true_lr"].reset_index(drop=True)
fig, axes = plt.subplots(len(true_pairs), 2, figsize=(13, 2.0*len(true_pairs)))
for i, r in true_pairs.iterrows():
    for j, (agi, sym, role) in enumerate([(r.ligand_agi, r.ligand, "Ligand"),
                                            (r.receptor_agi, r.receptor, "Receptor")]):
        ax = axes[i, j]
        v = np.asarray(adata.X[:, gene_idx[agi]].todense()).ravel()
        sc = ax.scatter(coords[:,0], coords[:,1], c=v, s=10, cmap="Reds",
                        vmin=0, vmax=max(v.max(), 1), edgecolors="none")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ("top","right","bottom","left"): ax.spines[s].set_visible(False)
        ax.set_title(f"{role}: {sym} ({agi})  [{r.label}, σ={r.sigma_hops}, γ²={r.gamma2_true}]",
                     fontsize=8)
        plt.colorbar(sc, ax=ax, fraction=0.022, pad=0.02).ax.tick_params(labelsize=7)
plt.suptitle("Figure S3.  Spatial expression of all five TRUE L-R pairs after semi-synthetic injection",
             y=1.005, fontsize=11, fontweight="bold")
plt.savefig(f"{FIGS}/FigS3_true_lr_maps.png"); plt.close()

# =============================================================================
# FIG S4: Detectability landscape — bivariate Moran's I per pair class
# =============================================================================
def lag(v, A):
    d = np.asarray(A.sum(axis=1)).ravel(); d[d==0]=1
    return (A @ v) / d

records = []
for _, r in gt.iterrows():
    L = np.asarray(adata.X[:, gene_idx[r.ligand_agi]].todense()).ravel()
    R = np.asarray(adata.X[:, gene_idx[r.receptor_agi]].todense()).ravel()
    if R.std() > 0 and L.std() > 0:
        I = pearsonr(lag(L, A), R)[0]
    else:
        I = 0.0
    records.append({"role": r.role, "label": r.label, "I": I,
                    "pair": f"{r.ligand}→{r.receptor}",
                    "gamma2": r.gamma2_true if pd.notna(r.gamma2_true) else 0})
mi = pd.DataFrame(records)
mi.to_csv(f"{FIGS}/FigS4_moran_table.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 4.6))
order_role = ["true_lr", "confounder", "random"]
palette = {"true_lr":"#C44545", "confounder":"#A45BC4", "random":"#777777"}
for i, role in enumerate(order_role):
    d = mi[mi["role"] == role].reset_index(drop=True)
    xs = i + np.random.uniform(-0.10, 0.10, len(d))
    ax.scatter(xs, d["I"], s=120, c=palette[role], edgecolors="black",
               linewidths=0.6, zorder=3, label=f"{role} (n={len(d)})")
    for _, row in d.iterrows():
        ax.text(i + 0.15, row["I"], row["pair"], fontsize=6.5, va="center")
    # group mean
    mean = d["I"].mean()
    ax.hlines(mean, i-0.25, i+0.25, colors=palette[role], linestyles="--", lw=1.5)

ax.axhline(0, color="grey", lw=0.7, ls=":")
ax.set_xticks(range(len(order_role)))
ax.set_xticklabels(["TRUE L-R\n(n=5)", "Co-localization\nconfounders (n=5)",
                    "Spatially-random\npairs (n=5)"], fontsize=9)
ax.set_ylabel("Bivariate Moran's I\n(Pearson r of lagged ligand vs receptor)")
ax.set_title("Figure S4.  Detectability landscape — naive co-expression statistics CANNOT\n"
             "separate TRUE pairs from co-localization confounders. Dashed lines = group means.",
             fontweight="bold", fontsize=10)
ax.spines[["top","right"]].set_visible(False)
plt.savefig(f"{FIGS}/FigS4_detectability.png"); plt.close()

# =============================================================================
# FIG S5: Statistical preservation — sim vs real mean/dropout for ALL non-overwritten genes
# =============================================================================
n_genes = adata.n_vars
mu_real = np.asarray(adata.layers["real_counts"].mean(axis=0)).ravel()
mu_sim  = np.asarray(adata.layers["sim_counts"].mean(axis=0)).ravel()
drop_real = np.asarray((adata.layers["real_counts"] == 0).sum(axis=0)).ravel() / adata.n_obs
drop_sim  = np.asarray((adata.layers["sim_counts"]  == 0).sum(axis=0)).ravel() / adata.n_obs
overwritten = adata.var["overwritten_by_sim"].values

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
ax.scatter(np.log10(mu_real[~overwritten] + 1e-3), np.log10(mu_sim[~overwritten] + 1e-3),
           s=2, alpha=0.25, c="#555555", label=f"background ({(~overwritten).sum()})", rasterized=True)
ax.scatter(np.log10(mu_real[overwritten] + 1e-3),  np.log10(mu_sim[overwritten] + 1e-3),
           s=70, c="#C44545", edgecolors="black", linewidths=0.4, zorder=5,
           label=f"benchmark genes ({overwritten.sum()})")
lo, hi = -3, 3
ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("log10(real mean + 0.001)")
ax.set_ylabel("log10(sim mean + 0.001)")
ax.set_title("A. Mean expression preservation")
ax.legend(fontsize=8, loc="lower right", frameon=False)
ax.spines[["top","right"]].set_visible(False)

ax = axes[1]
ax.scatter(drop_real[~overwritten], drop_sim[~overwritten],
           s=2, alpha=0.25, c="#555555", label=f"background ({(~overwritten).sum()})", rasterized=True)
ax.scatter(drop_real[overwritten], drop_sim[overwritten],
           s=70, c="#C44545", edgecolors="black", linewidths=0.4, zorder=5,
           label=f"benchmark genes ({overwritten.sum()})")
ax.plot([0,1],[0,1],"k--",lw=0.7)
ax.set_xlabel("real dropout fraction")
ax.set_ylabel("sim dropout fraction")
ax.set_title("B. Dropout preservation")
ax.legend(fontsize=8, loc="upper left", frameon=False)
ax.spines[["top","right"]].set_visible(False)

plt.suptitle("Figure S5.  Statistical preservation across the transcriptome\n"
             "Background genes lie exactly on y=x; benchmark genes are deliberately raised\n"
             "to a detectable mean (≥0.7) so injected SVCA signals can be evaluated.",
             y=1.04, fontsize=10, fontweight="bold")
plt.savefig(f"{FIGS}/FigS5_stat_preservation.png", dpi=180); plt.close()

print("Figures generated:")
for f in sorted(os.listdir(FIGS)):
    sz = os.path.getsize(f"{FIGS}/{f}") / 1024
    print(f"  {f:40s}  {sz:8.1f} KB")
