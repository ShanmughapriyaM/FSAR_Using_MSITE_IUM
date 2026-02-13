import numpy as np


def segmented_multiscale_encoding(X, scales=(1, 2, 4)):
    """
    Segmented multi-scale temporal encoding (training-free)

    Parameters
    ----------
    X : np.ndarray
        Shape [T, D] temporal feature sequence
    scales : tuple
        Number of temporal segments at each scale

    Returns
    -------
    np.ndarray
        Shape [sum(scales) * D]
    """

    assert X.ndim == 2, "Input must be [T, D]"
    T, D = X.shape

    encodings = []

    for s in scales:
        # Compute segment boundaries
        boundaries = np.linspace(0, T, s + 1).astype(int)

        for i in range(s):
            start, end = boundaries[i], boundaries[i + 1]

            # Safety: if segment is empty
            if start >= end:
                seg_feat = np.zeros(D, dtype=X.dtype)
            else:
                seg_feat = X[start:end].mean(axis=0)

            encodings.append(seg_feat)

    return np.concatenate(encodings, axis=0)
