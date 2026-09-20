import numpy as np
import pytest

from postprocess import remove_small_components, separate_touching_nuclei


def test_remove_small_components():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1, 1] = True
    mask[4:7, 4:7] = True
    cleaned = remove_small_components(mask, 4)
    assert not cleaned[1, 1]
    assert cleaned[5, 5]


def test_postprocess_arguments_are_validated():
    mask = np.zeros((8, 8), dtype=bool)
    with pytest.raises(ValueError):
        remove_small_components(mask, 0)
    with pytest.raises(ValueError):
        separate_touching_nuclei(mask, 0, 0.1)
    with pytest.raises(ValueError):
        separate_touching_nuclei(mask, 2, 1.1)


def test_each_disconnected_component_receives_an_instance():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:6, 2:6] = True
    mask[13:18, 13:18] = True
    labels = separate_touching_nuclei(mask, 3, 0.2)
    assert labels.max() == 2
