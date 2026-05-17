import numpy as np

from deconfounding_interp.directions import (
    average_directions,
    cosine_similarity,
    difference_in_means,
    orthonormal_basis,
    remove_subspace,
    subspace_overlap_fraction,
)


def test_difference_in_means():
    pos = np.array([[2.0, 1.0], [4.0, 1.0]])
    neg = np.array([[1.0, 1.0], [1.0, 3.0]])
    np.testing.assert_allclose(difference_in_means(pos, neg), np.array([2.0, -1.0]))


def test_average_directions_normalizes():
    averaged = average_directions(np.array([[1.0, 0.0], [1.0, 0.0]]))
    np.testing.assert_allclose(averaged, np.array([1.0, 0.0]))


def test_subspace_removal():
    basis = orthonormal_basis(np.array([[1.0, 0.0, 0.0]]))
    cleaned = remove_subspace(np.array([1.0, 1.0, 0.0]), basis)
    assert abs(cosine_similarity(cleaned, np.array([0.0, 1.0, 0.0])) - 1.0) < 1e-12
    assert subspace_overlap_fraction(np.array([1.0, 1.0, 0.0]), basis) == 0.5
