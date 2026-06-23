"""Phase B visualisation — render Objective B attention into figures.

Produces:
  * ``heatmaps/latent{li}_overlay.png`` — latent->image attention (mean over layers+heads)
    folded into the 2D patch grid and overlaid on the input image.
  * ``heatmaps/latent{li}_by_layer.png`` — the same latent's image attention per layer
    (small multiples), to see at which depth the latent looks at the image.
  * ``attn_text2latent.png`` — heatmap of generated tokens (cols) x latents (rows),
    mean over layers+heads.

All plotting is best-effort: if matplotlib / PIL are unavailable the caller still keeps the
``.npz`` tensors. Heavy imports are kept inside the functions.
"""
import os
import numpy as np


def _grid_hw(grid_thw, merge, n_image):
    """Merged patch-grid (gh, gw) for a single image; None if it can't be inferred."""
    if grid_thw is None:
        return None
    g = np.asarray(grid_thw).reshape(-1, 3)
    if g.shape[0] != 1:
        return None  # multi-image: skip clean 2D folding
    t, h, w = int(g[0, 0]), int(g[0, 1]), int(g[0, 2])
    gh, gw = h // merge, w // merge
    if t * gh * gw != n_image:
        return None
    return gh, gw


def _suppress_sinks(B):
    """Zero each (layer,head,latent)'s single argmax patch — those are attention sinks
    (degenerate patches that soak up mass without carrying content), so they dominate the
    overlays. Returns a copy."""
    flat = B.reshape(-1, B.shape[-1]).copy()
    flat[np.arange(flat.shape[0]), flat.argmax(axis=1)] = 0.0
    return flat.reshape(B.shape)


def render_all(out_dir, attn_summary, latent_positions, tokenizer, trace, image_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hdir = os.path.join(out_dir, "heatmaps")
    os.makedirs(hdir, exist_ok=True)

    l2i = _suppress_sinks(attn_summary["latent2image"].astype(np.float32))  # [L,H,N_lat,N_img]
    t2l = attn_summary["text2latent"].astype(np.float32)    # [L, H, Q_text, N_lat]
    n_lat = l2i.shape[2]
    n_img = l2i.shape[3]
    gh_gw = _grid_hw(attn_summary["grid_thw"], attn_summary["merge"], n_img)

    # ---- text -> latent matrix ----
    m = t2l.mean(axis=(0, 1))                               # [Q_text, N_lat]
    fig, ax = plt.subplots(figsize=(max(4, n_lat * 0.6), 6))
    im = ax.imshow(m, aspect="auto", cmap="viridis")
    ax.set_xlabel("latent index")
    ax.set_ylabel("generated token (query, top→bottom)")
    ax.set_title("text → latent attention (mean over layers/heads)")
    ax.set_xticks(range(n_lat))
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "attn_text2latent.png"), dpi=110)
    plt.close(fig)

    if gh_gw is None:
        print("[viz] could not fold image attention to a 2D grid "
              "(multi-image or size mismatch); skipped overlays.")
        return
    gh, gw = gh_gw

    img = None
    if image_path and os.path.exists(image_path):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")

    l2i_mean = l2i.mean(axis=(0, 1))                        # [N_lat, N_img]
    for li in range(n_lat):
        heat = l2i_mean[li].reshape(gh, gw)
        _overlay(plt, img, heat, os.path.join(hdir, f"latent{li}_overlay.png"),
                 title=f"latent {li} → image (sink-suppressed, mean L/H)")

    # per-layer small multiples for each latent
    n_layers = l2i.shape[0]
    for li in range(n_lat):
        per_layer = l2i[:, :, li, :].mean(axis=1)          # [L, N_img]
        ncol = 7
        nrow = int(np.ceil(n_layers / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.6, nrow * 1.6))
        axes = np.atleast_1d(axes).ravel()
        for l in range(n_layers):
            axes[l].imshow(per_layer[l].reshape(gh, gw), cmap="magma")
            axes[l].set_title(f"L{l}", fontsize=6)
            axes[l].axis("off")
        for l in range(n_layers, len(axes)):
            axes[l].axis("off")
        fig.suptitle(f"latent {li} → image attention by layer", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(hdir, f"latent{li}_by_layer.png"), dpi=100)
        plt.close(fig)


def _overlay(plt, img, heat, path, title):
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()
    fig, ax = plt.subplots(figsize=(5, 5))
    if img is not None:
        ax.imshow(img)
        from matplotlib.transforms import Bbox  # noqa: F401 (kept for clarity)
        ax.imshow(_resize(heat, img.size[1], img.size[0]), cmap="jet", alpha=0.45)
    else:
        ax.imshow(heat, cmap="jet")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _resize(arr, out_h, out_w):
    """Nearest-neighbour upsample a small [gh, gw] map to [out_h, out_w] without scipy."""
    gh, gw = arr.shape
    ys = (np.linspace(0, gh - 1, out_h)).round().astype(int)
    xs = (np.linspace(0, gw - 1, out_w)).round().astype(int)
    return arr[ys][:, xs]


def render_nearest(out_dir, nearest, image_path):
    """Overlay each latent's cosine-to-image-patch map on the image (token-free localiser)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cos = nearest["cos_map"].astype(np.float32)         # [N_lat, N_img]
    grid = nearest["grid_thw"]; merge = nearest["merge"]
    n_img = cos.shape[1]
    gh_gw = _grid_hw(grid, merge, n_img)
    if gh_gw is None:
        print("[viz] nearest-patch: cannot fold to 2D grid; skipped.")
        return
    gh, gw = gh_gw
    hdir = os.path.join(out_dir, "heatmaps")
    os.makedirs(hdir, exist_ok=True)
    img = None
    if image_path and os.path.exists(image_path):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
    for li in range(cos.shape[0]):
        heat = cos[li].reshape(gh, gw)
        _overlay(plt, img, heat, os.path.join(hdir, f"latent{li}_nearest.png"),
                 title=f"latent {li}: cosine to image patches")
