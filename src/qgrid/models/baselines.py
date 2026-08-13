"""Non-neural baselines (torch-free module)."""
from __future__ import annotations


def build_sklearn_baseline(model_type: str, seed: int):
    if model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                          random_state=seed)
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=seed)
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, max_depth=3,
                             random_state=seed, eval_metric="logloss")
    if model_type == "svm":
        from sklearn.svm import SVC
        return SVC(kernel="rbf", probability=True, random_state=seed)
    raise ValueError(f"Unknown baseline: {model_type}")
