"""Deterministic lowering of selected estimators to portable C++ source."""

from __future__ import annotations

import math
from typing import Sequence


_SUPPORTED = {
    "ElasticNet",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "KNeighborsRegressor",
    "Lasso",
    "LGBMRegressor",
    "LinearRegression",
    "RandomForestRegressor",
    "Ridge",
    "XGBRegressor",
    "XGBRFRegressor",
}


def supports(model: object) -> bool:
    """Return whether one fitted estimator graph has a compiled v1 lowering."""

    if model.__class__.__name__ in _SUPPORTED:
        return True
    if model.__class__.__name__ != "VotingRegressor":
        return False
    estimators = tuple(getattr(model, "estimators_", ()))
    return bool(estimators) and all(supports(item) for item in estimators)


def _number(value: object) -> str:
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError("compiled models require finite parameters")
    return selected.hex()


def _array(kind: str, name: str, values: Sequence[object]) -> str:
    if kind == "double":
        body = ",".join(_number(value) for value in values)
    else:
        body = ",".join(str(int(value)) for value in values)
    return f"static constexpr {kind} {name}[] = {{{body}}};"


class _Generator:
    def __init__(self, input_width: int) -> None:
        self.input_width = input_width
        self.definitions: list[str] = []
        self.batches: dict[str, str] = {}
        self.count = 0

    def model(self, estimator: object) -> str:
        name = f"model_{self.count}"
        self.count += 1
        class_name = estimator.__class__.__name__
        if class_name in {"Ridge", "Lasso", "ElasticNet", "LinearRegression"}:
            self._linear(name, estimator)
        elif class_name in {"ExtraTreesRegressor", "RandomForestRegressor"}:
            self._forest(name, estimator)
        elif class_name == "KNeighborsRegressor":
            self._neighbors(name, estimator)
        elif class_name == "GradientBoostingRegressor":
            self._gradient_boosting(name, estimator)
        elif class_name in {"XGBRegressor", "XGBRFRegressor"}:
            self._xgboost(name, estimator)
        elif class_name == "LGBMRegressor":
            self._lightgbm(name, estimator)
        elif class_name == "VotingRegressor":
            self._voting(name, estimator)
        else:
            raise TypeError(
                f"compiled v1 does not support {class_name}; use "
                "fno.Export.sklearn(..., backend='joblib')"
            )
        return name

    def _linear(self, name: str, estimator: object) -> None:
        import numpy as np

        coefficients = np.asarray(getattr(estimator, "coef_", None), dtype=np.float64)
        if coefficients.ndim != 1 or len(coefficients) != self.input_width:
            raise ValueError("compiled v1 linear models require one scalar output")
        intercept = np.asarray(getattr(estimator, "intercept_", 0.0), dtype=np.float64)
        if intercept.size != 1:
            raise ValueError("compiled v1 linear models require one scalar output")
        weights = f"{name}_weights"
        self.definitions.extend(
            (
                _array("double", weights, coefficients),
                f"static inline double {name}(const double* x) {{\n"
                f"  double y = {_number(intercept.reshape(-1)[0])};\n"
                f"  for (std::size_t j = 0; j < {self.input_width}; ++j) y += {weights}[j] * x[j];\n"
                "  return y;\n}",
                f"static inline void {name}_batch(const double* input, std::size_t rows, std::size_t columns, double* output, std::size_t) {{\n"
                f"  for (std::size_t row = 0; row < rows; ++row) output[row] = {name}(input + row * columns);\n"
                "}",
            )
        )
        self.batches[name] = f"{name}_batch"

    def _neighbors(self, name: str, estimator: object) -> None:
        import numpy as np

        features = np.asarray(getattr(estimator, "_fit_X", None), dtype=np.float64)
        targets = np.asarray(getattr(estimator, "_y", None), dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.input_width:
            raise ValueError("compiled KNN input width does not match the FNOM contract")
        if targets.ndim != 1 or len(targets) != len(features):
            raise ValueError("compiled v1 KNN requires one scalar output")
        neighbors = int(getattr(estimator, "n_neighbors", 0))
        if not 1 <= neighbors <= len(features):
            raise ValueError("compiled KNN has an invalid neighbor count")
        weights = getattr(estimator, "weights", "uniform")
        if weights not in {"uniform", "distance"}:
            raise TypeError("compiled v1 KNN supports uniform or distance weights")
        metric = getattr(estimator, "effective_metric_", None)
        parameters = getattr(estimator, "effective_metric_params_", {})
        if metric == "euclidean":
            power = 2
        elif metric == "manhattan":
            power = 1
        elif metric == "minkowski" and int(parameters.get("p", 2)) in {1, 2}:
            power = int(parameters.get("p", 2))
        else:
            raise TypeError("compiled v1 KNN supports Euclidean and Manhattan distance")
        prefix = f"{name}_knn"
        distance_term = (
            "const double difference = x[column] - "
            f"{prefix}_features[sample * {self.input_width} + column];\n"
            + ("      distance += difference * difference;" if power == 2 else "      distance += std::abs(difference);")
        )
        distance_finish = "distance = std::sqrt(distance);" if power == 2 else ""
        if weights == "uniform":
            reduction = (
                "  double prediction = 0.0;\n"
                f"  for (std::size_t item = 0; item < {neighbors}; ++item) prediction += {prefix}_targets[candidates[item].second];\n"
                f"  return prediction / {_number(neighbors)};"
            )
        else:
            reduction = (
                "  double zero_sum = 0.0;\n"
                "  std::size_t zero_count = 0;\n"
                f"  for (std::size_t item = 0; item < {neighbors}; ++item) {{\n"
                "    if (candidates[item].first == 0.0) {\n"
                f"      zero_sum += {prefix}_targets[candidates[item].second];\n"
                "      ++zero_count;\n"
                "    }\n"
                "  }\n"
                "  if (zero_count != 0) return zero_sum / static_cast<double>(zero_count);\n"
                "  double weighted_sum = 0.0;\n"
                "  double weight_sum = 0.0;\n"
                f"  for (std::size_t item = 0; item < {neighbors}; ++item) {{\n"
                "    const double weight = 1.0 / candidates[item].first;\n"
                f"    weighted_sum += weight * {prefix}_targets[candidates[item].second];\n"
                "    weight_sum += weight;\n"
                "  }\n"
                "  return weighted_sum / weight_sum;"
            )
        self.definitions.extend(
            (
                _array("double", f"{prefix}_features", features.reshape(-1)),
                _array("double", f"{prefix}_targets", targets),
                f"static inline double {name}(const double* x) {{\n"
                "  std::vector<std::pair<double, std::size_t>> candidates;\n"
                f"  candidates.reserve({neighbors});\n"
                f"  for (std::size_t sample = 0; sample < {len(features)}; ++sample) {{\n"
                "    double distance = 0.0;\n"
                f"    for (std::size_t column = 0; column < {self.input_width}; ++column) {{\n"
                f"      {distance_term}\n"
                "    }\n"
                f"    {distance_finish}\n"
                "    const std::pair<double, std::size_t> candidate{distance, sample};\n"
                f"    if (candidates.size() < {neighbors}) {{\n"
                "      candidates.push_back(candidate);\n"
                "      std::push_heap(candidates.begin(), candidates.end());\n"
                "    } else if (candidate < candidates.front()) {\n"
                "      std::pop_heap(candidates.begin(), candidates.end());\n"
                "      candidates.back() = candidate;\n"
                "      std::push_heap(candidates.begin(), candidates.end());\n"
                "    }\n"
                "  }\n"
                f"{reduction}\n"
                "}",
                self._row_batch(name),
            )
        )
        self.batches[name] = f"{name}_batch"

    def _gradient_boosting(self, name: str, estimator: object) -> None:
        import numpy as np

        estimators = np.asarray(getattr(estimator, "estimators_", None), dtype=object)
        if estimators.ndim != 2 or estimators.shape[1] != 1 or len(estimators) == 0:
            raise ValueError("compiled v1 gradient boosting requires one scalar output")
        initial = getattr(estimator, "init_", None)
        constant = np.asarray(getattr(initial, "constant_", None), dtype=np.float64)
        if constant.size != 1:
            raise TypeError("compiled v1 requires a constant GradientBoosting initializer")
        children = [self._tree(name, index, tree) for index, tree in enumerate(estimators[:, 0])]
        learning_rate = _number(getattr(estimator, "learning_rate", 0.0))
        expression = " + ".join(f"{child}(x)" for child in children)
        self.definitions.extend(
            (
                f"static inline double {name}(const double* x) {{ return {_number(constant.reshape(-1)[0])} + {learning_rate} * ({expression}); }}",
                self._row_batch(name),
            )
        )
        self.batches[name] = f"{name}_batch"

    def _xgboost(self, name: str, estimator: object) -> None:
        import json

        booster = estimator.get_booster()
        configuration = json.loads(booster.save_config())
        learner = configuration["learner"]
        parameters = learner["learner_model_param"]
        if int(parameters["num_target"]) != 1 or int(parameters["num_class"]) != 0:
            raise ValueError("compiled v1 XGBoost requires one scalar regression output")
        objective = learner["objective"]["name"]
        if objective not in {"reg:squarederror", "reg:absoluteerror"}:
            raise TypeError(f"compiled v1 does not support XGBoost objective {objective!r}")
        base_score = float(str(parameters["base_score"]).strip("[]"))
        trees = [json.loads(value) for value in booster.get_dump(dump_format="json")]
        children = []
        for index, tree in enumerate(trees):
            child = f"{name}_xgb_tree_{index}"
            self.definitions.append(
                f"static inline double {child}(const double* x) {{\n"
                + self._xgboost_node(tree, "  ")
                + "\n}"
            )
            children.append(child)
        expression = " + ".join(f"{child}(x)" for child in children)
        self.definitions.extend(
            (
                f"static inline double {name}(const double* x) {{ return {_number(base_score)} + {expression}; }}",
                self._row_batch(name),
            )
        )
        self.batches[name] = f"{name}_batch"

    def _xgboost_node(self, node: dict[str, object], indent: str) -> str:
        if "leaf" in node:
            return f"{indent}return {_number(node['leaf'])};"
        split = str(node["split"])
        if not split.startswith("f") or not split[1:].isdigit():
            raise TypeError("compiled v1 XGBoost requires positional feature names")
        column = int(split[1:])
        if column >= self.input_width:
            raise ValueError("XGBoost feature index exceeds the FNOM input width")
        children = {int(child["nodeid"]): child for child in node["children"]}
        yes = children[int(node["yes"])]
        no = children[int(node["no"])]
        missing_yes = int(node["missing"]) == int(node["yes"])
        condition = (
            f"std::isnan(x[{column}]) ? {'true' if missing_yes else 'false'} : "
            f"x[{column}] < {_number(node['split_condition'])}"
        )
        return (
            f"{indent}if ({condition}) {{\n"
            + self._xgboost_node(yes, indent + "  ")
            + f"\n{indent}}} else {{\n"
            + self._xgboost_node(no, indent + "  ")
            + f"\n{indent}}}"
        )

    def _lightgbm(self, name: str, estimator: object) -> None:
        dump = estimator.booster_.dump_model()
        objective = str(dump.get("objective", ""))
        if not objective.startswith(("regression", "l1", "huber", "fair", "quantile")):
            raise TypeError(f"compiled v1 does not support LightGBM objective {objective!r}")
        children = []
        for item in dump["tree_info"]:
            child = f"{name}_lgb_tree_{int(item['tree_index'])}"
            self.definitions.append(
                f"static inline double {child}(const double* x) {{\n"
                + self._lightgbm_node(item["tree_structure"], "  ")
                + "\n}"
            )
            children.append(child)
        expression = " + ".join(f"{child}(x)" for child in children)
        self.definitions.extend(
            (
                f"static inline double {name}(const double* x) {{ return {expression}; }}",
                self._row_batch(name),
            )
        )
        self.batches[name] = f"{name}_batch"

    def _lightgbm_node(self, node: dict[str, object], indent: str) -> str:
        if "leaf_value" in node:
            return f"{indent}return {_number(node['leaf_value'])};"
        if str(node.get("decision_type", "<=")) != "<=":
            raise TypeError("compiled v1 LightGBM does not support categorical splits")
        column = int(node["split_feature"])
        if column >= self.input_width:
            raise ValueError("LightGBM feature index exceeds the FNOM input width")
        default_left = bool(node.get("default_left", True))
        missing_type = str(node.get("missing_type", "None"))
        comparison = f"x[{column}] <= {_number(node['threshold'])}"
        if missing_type == "None":
            condition = (
                f"(std::isnan(x[{column}]) ? 0.0 : x[{column}]) <= "
                f"{_number(node['threshold'])}"
            )
        elif missing_type == "NaN":
            condition = (
                f"std::isnan(x[{column}]) ? {'true' if default_left else 'false'} : "
                + comparison
            )
        elif missing_type == "Zero":
            condition = (
                f"(std::isnan(x[{column}]) || x[{column}] == 0.0) ? "
                f"{'true' if default_left else 'false'} : {comparison}"
            )
        else:
            raise TypeError(f"compiled v1 does not support LightGBM missing type {missing_type!r}")
        return (
            f"{indent}if ({condition}) {{\n"
            + self._lightgbm_node(node["left_child"], indent + "  ")
            + f"\n{indent}}} else {{\n"
            + self._lightgbm_node(node["right_child"], indent + "  ")
            + f"\n{indent}}}"
        )

    def _tree(self, owner: str, index: int, estimator: object) -> str:
        import numpy as np

        tree = estimator.tree_
        values = np.asarray(tree.value)
        if values.shape[1:] != (1, 1):
            raise ValueError("compiled v1 trees require one scalar output")
        name = f"{owner}_boost_tree_{index}"
        self.definitions.extend(
            (
                _array("int", f"{name}_left", tree.children_left),
                _array("int", f"{name}_right", tree.children_right),
                _array("int", f"{name}_feature", tree.feature),
                _array("double", f"{name}_threshold", tree.threshold),
                _array("double", f"{name}_value", values[:, 0, 0]),
                f"static inline double {name}(const double* x) {{\n"
                "  int node = 0;\n"
                f"  while ({name}_left[node] >= 0) {{\n"
                f"    const int column = {name}_feature[node];\n"
                f"    node = static_cast<float>(x[column]) <= {name}_threshold[node] ? {name}_left[node] : {name}_right[node];\n"
                "  }\n"
                f"  return {name}_value[node];\n"
                "}",
            )
        )
        return name

    @staticmethod
    def _row_batch(name: str) -> str:
        return (
            f"static inline void {name}_batch(const double* input, std::size_t rows, std::size_t columns, double* output, std::size_t requested_threads) {{\n"
            "  const std::size_t workers = std::min(rows, std::max<std::size_t>(1, requested_threads));\n"
            "  auto evaluate = [&](std::size_t begin, std::size_t end) {\n"
            f"    for (std::size_t row = begin; row < end; ++row) output[row] = {name}(input + row * columns);\n"
            "  };\n"
            "  if (workers == 1 || rows < 256) { evaluate(0, rows); return; }\n"
            "  std::vector<std::thread> pool;\n"
            "  pool.reserve(workers - 1);\n"
            "  const std::size_t quotient = rows / workers;\n"
            "  const std::size_t remainder = rows % workers;\n"
            "  std::size_t begin = 0;\n"
            "  for (std::size_t worker = 0; worker + 1 < workers; ++worker) {\n"
            "    const std::size_t end = begin + quotient + (worker < remainder ? 1 : 0);\n"
            "    pool.emplace_back(evaluate, begin, end);\n"
            "    begin = end;\n"
            "  }\n"
            "  evaluate(begin, rows);\n"
            "  for (auto& worker : pool) worker.join();\n"
            "}"
        )

    def _forest(self, name: str, estimator: object) -> None:
        import numpy as np

        trees = tuple(getattr(estimator, "estimators_", ()))
        if not trees:
            raise ValueError("compiled forests must be fitted before export")
        offsets = [0]
        left: list[int] = []
        right: list[int] = []
        feature: list[int] = []
        threshold: list[float] = []
        value: list[float] = []
        for tree_estimator in trees:
            tree = tree_estimator.tree_
            values = np.asarray(tree.value)
            if values.shape[1:] != (1, 1):
                raise ValueError("compiled v1 forests require one scalar output")
            base = offsets[-1]
            left.extend(-1 if item < 0 else base + int(item) for item in tree.children_left)
            right.extend(-1 if item < 0 else base + int(item) for item in tree.children_right)
            feature.extend(int(item) for item in tree.feature)
            threshold.extend(float(item) for item in tree.threshold)
            value.extend(float(item) for item in values[:, 0, 0])
            offsets.append(base + int(tree.node_count))
        prefix = f"{name}_tree"
        self.definitions.extend(
            (
                _array("std::size_t", f"{prefix}_offset", offsets),
                _array("int", f"{prefix}_left", left),
                _array("int", f"{prefix}_right", right),
                _array("int", f"{prefix}_feature", feature),
                _array("double", f"{prefix}_threshold", threshold),
                _array("double", f"{prefix}_value", value),
                f"static inline double {name}(const double* x) {{\n"
                "  double sum = 0.0;\n"
                f"  for (std::size_t tree = 0; tree < {len(trees)}; ++tree) {{\n"
                f"    int node = static_cast<int>({prefix}_offset[tree]);\n"
                f"    while ({prefix}_left[node] >= 0) {{\n"
                f"      const int column = {prefix}_feature[node];\n"
                # scikit-learn's tree predictor narrows input values to its
                # float32 DTYPE before comparing them with float64 thresholds.
                f"      node = static_cast<float>(x[column]) <= {prefix}_threshold[node] ? {prefix}_left[node] : {prefix}_right[node];\n"
                "    }\n"
                f"    sum += {prefix}_value[node];\n"
                "  }\n"
                f"  return sum / {_number(len(trees))};\n"
                "}",
                f"static inline void {name}_batch(const double* input, std::size_t rows, std::size_t columns, double* output, std::size_t requested_threads) {{\n"
                f"  const std::size_t tree_count = {len(trees)};\n"
                "  const std::size_t workers = std::min(tree_count, std::max<std::size_t>(1, requested_threads));\n"
                "  std::vector<double> partial(workers * rows, 0.0);\n"
                "  auto evaluate_trees = [&](std::size_t worker, std::size_t first, std::size_t last) {\n"
                "    double* local = partial.data() + worker * rows;\n"
                "    for (std::size_t tree = first; tree < last; ++tree) {\n"
                f"      const int root = static_cast<int>({prefix}_offset[tree]);\n"
                "      for (std::size_t row = 0; row < rows; ++row) {\n"
                "        const double* x = input + row * columns;\n"
                "        int node = root;\n"
                f"        while ({prefix}_left[node] >= 0) {{\n"
                f"          const int column = {prefix}_feature[node];\n"
                f"          node = static_cast<float>(x[column]) <= {prefix}_threshold[node] ? {prefix}_left[node] : {prefix}_right[node];\n"
                "        }\n"
                f"        local[row] += {prefix}_value[node];\n"
                "      }\n"
                "    }\n"
                "  };\n"
                "  std::vector<std::thread> pool;\n"
                "  pool.reserve(workers - 1);\n"
                "  const std::size_t quotient = tree_count / workers;\n"
                "  const std::size_t remainder = tree_count % workers;\n"
                "  std::size_t first = 0;\n"
                "  for (std::size_t worker = 0; worker + 1 < workers; ++worker) {\n"
                "    const std::size_t last = first + quotient + (worker < remainder ? 1 : 0);\n"
                "    pool.emplace_back(evaluate_trees, worker, first, last);\n"
                "    first = last;\n"
                "  }\n"
                "  evaluate_trees(workers - 1, first, tree_count);\n"
                "  for (auto& worker : pool) worker.join();\n"
                "  for (std::size_t row = 0; row < rows; ++row) {\n"
                "    double sum = 0.0;\n"
                "    for (std::size_t worker = 0; worker < workers; ++worker) sum += partial[worker * rows + row];\n"
                "    output[row] = sum / static_cast<double>(tree_count);\n"
                "  }\n"
                "}",
            )
        )
        self.batches[name] = f"{name}_batch"

    def _voting(self, name: str, estimator: object) -> None:
        import numpy as np

        estimators = tuple(getattr(estimator, "estimators_", ()))
        if not estimators:
            raise ValueError("compiled voting regressors must be fitted before export")
        raw_weights = getattr(estimator, "weights", None)
        weights = np.ones(len(estimators)) if raw_weights is None else np.asarray(raw_weights)
        if weights.shape != (len(estimators),) or not np.all(np.isfinite(weights)):
            raise ValueError("VotingRegressor weights are invalid")
        total = float(np.sum(weights))
        if total == 0.0:
            raise ValueError("VotingRegressor weights must not sum to zero")
        children = [self.model(item) for item in estimators]
        expression = " + ".join(
            f"({_number(weight / total)} * {child}(x))"
            for child, weight in zip(children, weights, strict=True)
        )
        self.definitions.append(
            f"static inline double {name}(const double* x) {{ return {expression}; }}"
        )
        statements = ["  std::fill(output, output + rows, 0.0);", "  std::vector<double> child(rows);"]
        for child, weight in zip(children, weights, strict=True):
            statements.extend(
                (
                    f"  {self.batches[child]}(input, rows, columns, child.data(), requested_threads);",
                    f"  for (std::size_t row = 0; row < rows; ++row) output[row] += {_number(weight / total)} * child[row];",
                )
            )
        self.definitions.append(
            f"static inline void {name}_batch(const double* input, std::size_t rows, std::size_t columns, double* output, std::size_t requested_threads) {{\n"
            + "\n".join(statements)
            + "\n}"
        )
        self.batches[name] = f"{name}_batch"


def generate_source(model: object, *, input_width: int, output_width: int) -> str:
    """Lower one fitted estimator graph into a stable C ABI translation unit."""

    if input_width < 1:
        raise ValueError("compiled models require at least one input feature")
    if output_width != 1:
        raise ValueError("compiled v1 currently requires one scalar output")
    generator = _Generator(input_width)
    root = generator.model(model)
    root_batch = generator.batches[root]
    definitions = "\n\n".join(generator.definitions)
    return f"""// Generated by FoamNordic compiled runtime v1. Do not edit.
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <thread>
#include <utility>
#include <vector>

namespace {{
{definitions}
}}

extern "C" std::size_t foamnordic_input_width() {{ return {input_width}; }}
extern "C" std::size_t foamnordic_output_width() {{ return 1; }}
static inline void foamnordic_predict_rows(
    const double* input,
    std::size_t begin,
    std::size_t end,
    std::size_t columns,
    double* output) {{
  for (std::size_t row = begin; row < end; ++row) output[row] = {root}(input + row * columns);
}}
extern "C" int foamnordic_predict(
    const double* input,
    std::size_t rows,
    std::size_t columns,
    double* output,
    std::size_t output_columns) {{
  if (!input || !output || columns != {input_width} || output_columns != 1) return 1;
  foamnordic_predict_rows(input, 0, rows, columns, output);
  return 0;
}}
extern "C" int foamnordic_predict_parallel(
    const double* input,
    std::size_t rows,
    std::size_t columns,
    double* output,
    std::size_t output_columns,
    std::size_t requested_threads) {{
  if (!input || !output || columns != {input_width} || output_columns != 1) return 1;
  const std::size_t workers = std::min(rows, std::max<std::size_t>(1, requested_threads));
  if (rows >= 32768) {{
    {root_batch}(input, rows, columns, output, workers);
    return 0;
  }}
  if (workers == 1 || rows < 256) {{
    foamnordic_predict_rows(input, 0, rows, columns, output);
    return 0;
  }}
  std::vector<std::thread> pool;
  pool.reserve(workers - 1);
  const std::size_t quotient = rows / workers;
  const std::size_t remainder = rows % workers;
  std::size_t begin = 0;
  for (std::size_t worker = 0; worker + 1 < workers; ++worker) {{
    const std::size_t end = begin + quotient + (worker < remainder ? 1 : 0);
    pool.emplace_back([=] {{ foamnordic_predict_rows(input, begin, end, columns, output); }});
    begin = end;
  }}
  foamnordic_predict_rows(input, begin, rows, columns, output);
  for (auto& worker : pool) worker.join();
  return 0;
}}
"""


__all__ = ["generate_source", "supports"]
