"""Portable, data-only scalar scikit-learn forest payloads."""
import struct
import math


def encode(model, width: int) -> bytes:
    if model.__class__.__name__ not in {"ExtraTreesRegressor", "RandomForestRegressor"}:
        raise TypeError("forest backend requires ExtraTreesRegressor or RandomForestRegressor")
    trees = tuple(model.estimators_)
    if not trees or model.n_features_in_ != width or model.n_outputs_ != 1:
        raise ValueError("forest input width or scalar output contract mismatch")
    nodes = []
    roots = []
    for estimator in trees:
        tree = estimator.tree_
        base = len(nodes)
        roots.append(base)
        missing = getattr(tree, "missing_go_to_left", [0] * tree.node_count)
        for i in range(tree.node_count):
            left, right = int(tree.children_left[i]), int(tree.children_right[i])
            threshold, value = float(tree.threshold[i]), float(tree.value[i, 0, 0])
            if not math.isfinite(threshold) or not math.isfinite(value):
                raise ValueError("forest parameters must be finite")
            nodes.append(struct.pack("<qqqddQ", base + left if left >= 0 else -1,
                base + right if right >= 0 else -1, int(tree.feature[i]),
                threshold, value, int(missing[i])))
    return (b"FNOFOR1\0" + struct.pack("<QQQ", width, len(trees), len(nodes))
            + struct.pack("<" + "Q" * len(roots), *roots) + b"".join(nodes))
