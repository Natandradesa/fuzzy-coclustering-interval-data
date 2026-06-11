import numpy as np
from sklearn.metrics import confusion_matrix


def confusion_matrix_coclustering(z_true, z_pred, w_true, w_pred):
    """
    Compute the co-clustering contingency matrix via the Kronecker product
    of row and column contingency matrices.

    Parameters
    ----------
    z_true : ndarray of shape (n_samples,)
        True row cluster labels.

    z_pred : ndarray of shape (n_samples,)
        Predicted row cluster labels.

    w_true : ndarray of shape (n_features,)
        True column cluster labels.

    w_pred : ndarray of shape (n_features,)
        Predicted column cluster labels.

    Returns
    -------
    score : ndarray
        Confusion matrix

    """
    n_zz = confusion_matrix(z_true, z_pred)
    n_ww = confusion_matrix(w_true, w_pred)
    return np.kron(n_zz, n_ww)


def _comb_2(n):
    """Compute the number of combinations of 2 elements from n (i.e., n choose 2)."""
    return (n * (n - 1)) // 2


def adjusted_rand_index_coclustering(z_true, z_pred, w_true, w_pred):
    """
    Compute the Adjusted Rand Index (ARI) for co-clustering, known as CARI.

    This metric compares two co-clustering partitions of a data matrix (i.e.,
    one set of labels for rows and another for columns), adjusting for chance
    similarly to the classic ARI.

    Parameters
    ----------
    z_true : ndarray of shape (n_samples,)
        True row cluster labels.

    z_pred : ndarray of shape (n_samples,)
        Predicted row cluster labels.

    w_true : ndarray of shape (n_features,)
        True column cluster labels.

    w_pred : ndarray of shape (n_features,)
        Predicted column cluster labels.

    Returns
    -------
    score : float
        The co-clustering Adjusted Rand Index (CARI), ranging from -1.0 to 1.0.

    References
    ----------
    Robert, V., Vasseur, Y., & Brault, V. (2021).
    Comparing high-dimensional partitions with the co-clustering adjusted rand index.
    Journal of Classification, 38(1), 158–186.
    """

    cm = confusion_matrix_coclustering(
        z_true=z_true, z_pred=z_pred, w_true=w_true, w_pred=w_pred
    )

    N = np.sum(cm)
    index = np.sum(_comb_2(cm))
    a = np.sum(_comb_2(cm.sum(axis=1)))
    b = np.sum(_comb_2(cm.sum(axis=0)))
    expected_index = (a * b) / _comb_2(N) if N >= 2 else 0
    max_index = (a + b) / 2

    if max_index == expected_index:
        return 1.0 if index == expected_index else 0.0
    else:
        return (index - expected_index) / (max_index - expected_index)
