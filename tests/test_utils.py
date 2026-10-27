import numpy as np
import pytest

from src.utils import plot_dimensionality_reduction

pytest.importorskip("umap")
pytest.importorskip("seaborn")


def test_embedding_plots(tmp_path):
    rng = np.random.default_rng(0)
    embeddings = np.concatenate([rng.normal(0, 1, (40, 16)), rng.normal(3, 1, (40, 16))])
    labels = np.array([0] * 40 + [1] * 40)

    info = plot_dimensionality_reduction(embeddings, labels, tmp_path)
    for name in ["PCA", "tSNE", "UMAP"]:
        assert (tmp_path / f"BinaryMapTransformer_{name}.png").stat().st_size > 0
    assert len(info["pca_explained_variance_ratio"]) == 2
