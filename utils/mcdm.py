import numpy as np, pandas as pd

def topsis(matrix: np.ndarray, weights: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    X = matrix.astype(float)
    Xn = X / np.sqrt((X**2).sum(axis=0))
    w  = weights / (weights.sum() if weights.sum() != 0 else 1.0)
    V  = Xn * w
    ideal = np.where(benefit, V.max(axis=0), V.min(axis=0))
    anti  = np.where(benefit, V.min(axis=0), V.max(axis=0))
    d_pos = np.linalg.norm(V - ideal, axis=1)
    d_neg = np.linalg.norm(V - anti , axis=1)
    return d_neg / (d_pos + d_neg + 1e-12)

def topsis_rank(df: pd.DataFrame, criteria_cols, weights, benefit_flags, index_col=None):
    C = df[criteria_cols].apply(pd.to_numeric, errors="coerce").dropna()
    scores = topsis(C.values, np.asarray(weights), np.asarray(benefit_flags, dtype=bool))
    res = C.copy()
    res["score_topsis"] = scores
    if index_col and index_col in df.columns:
        res[index_col] = df.loc[C.index, index_col]
    return res.sort_values("score_topsis", ascending=False)
