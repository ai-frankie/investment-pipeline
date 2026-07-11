import numpy as np
from learn_weights import weights_from_coefs

def test_negative_coef_gets_zero_weight():
    coefs = np.array([0.5, -0.5, 0.25])
    w = weights_from_coefs(coefs, ["a", "b", "c"])
    assert w["b"] == 0.0
    assert abs(w["a"] + w["c"] - 1.0) < 1e-9
    assert w["a"] > w["c"] > 0
