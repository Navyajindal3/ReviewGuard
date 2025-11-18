# bundle_multi_vec_pipeline.py
import argparse, itertools, joblib
from pathlib import Path

from custom_vec import CombinedPrefitVectorizer
from sklearn.pipeline import Pipeline

def vec_dim(v):
    """
    Return the feature dimension a fitted vectorizer produces.
    Works for CountVectorizer, TfidfVectorizer, and HashingVectorizer.
    """
    # Tfidf/Count after fit
    if hasattr(v, "get_feature_names_out"):
        try:
            return len(v.get_feature_names_out())
        except Exception:
            pass
    if hasattr(v, "vocabulary_"):
        try:
            return len(v.vocabulary_)
        except Exception:
            pass
    # HashingVectorizer has fixed n_features (no vocabulary)
    n = getattr(v, "n_features", None)
    if n is not None:
        return int(n)
    # As a last resort, do a tiny transform to infer width (slow but safe)
    try:
        import numpy as np
        from scipy.sparse import issparse
        X = v.transform(["probe text"])
        if hasattr(X, "shape"):
            return int(X.shape[1])
    except Exception:
        pass
    return None

def model_expected_dim(model):
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    if hasattr(model, "coef_"):
        return int(model.coef_.shape[1])
    return None

def load_vec(path):
    obj = joblib.load(path)
    d = vec_dim(obj)
    if d is None:
        raise SystemExit(f"{path} does not look like a fitted text vectorizer (can't determine dimension).")
    return obj, d

def main():
    ap = argparse.ArgumentParser(description="Bundle multiple pre-fitted vectorizers + a trained classifier into one pipeline.pkl")
    ap.add_argument("--model", required=True, help="Path to trained classifier .pkl (e.g., LogisticRegression)")
    ap.add_argument("--vectors", nargs="+", required=True, help="Paths to fitted vectorizers .pkl (4 files)")
    ap.add_argument("--order", nargs="+", help="Explicit order (filenames) matching training order")
    ap.add_argument("--val_csv", help="Optional CSV with columns: text,label to auto-pick best order")
    ap.add_argument("--out", default="pipeline.pkl", help="Output pipeline path")
    args = ap.parse_args()

    clf = joblib.load(args.model)
    m_dim = model_expected_dim(clf)
    print(f"Model expects: {m_dim} features")

    # Load vectorizers and compute their dims
    loaded = []
    name_to_vec = {}
    for vp in args.vectors:
        name = Path(vp).name
        v, d = load_vec(vp)
        loaded.append((name, v, d))
        name_to_vec[name] = (v, d)
        print(f"  {name} -> dim {d}")

    # Decide order
    if args.order:
        # Validate provided order
        missing = [n for n in args.order if n not in name_to_vec]
        if missing:
            raise SystemExit(f"--order contains names not in --vectors: {missing}")
        ordered = [(n, *name_to_vec[n]) for n in args.order]
    else:
        # Keep the given listing order (you can refine with --val_csv)
        ordered = loaded

    sum_dim = sum(d for _, _, d in ordered)
    print(f"Sum of vectorizer dims: {sum_dim}")
    if m_dim is not None and sum_dim != m_dim:
        raise SystemExit(f"Mismatch: sum(vectorizers)={sum_dim} but model expects {m_dim}. "
                         f"Use the exact vectorizer set/order from training.")

    # If a validation CSV is provided, try all permutations and pick the best accuracy
    if args.val_csv:
        import pandas as pd
        from sklearn.metrics import accuracy_score
        df = pd.read_csv(args.val_csv)
        xs = df["text"].astype(str).tolist()
        ys = df["label"].tolist()

        best_acc, best_perm = -1.0, None
        for perm in itertools.permutations(ordered):
            names = [p[0] for p in perm]
            V = CombinedPrefitVectorizer([p[1] for p in perm])
            pipe = Pipeline([("vect", V), ("clf", clf)])
            try:
                yhat = pipe.predict(xs)
                acc = accuracy_score(ys, yhat)
                print(f"Order {names} -> acc={acc:.4f}")
                if acc > best_acc:
                    best_acc, best_perm = acc, perm
            except Exception as e:
                print(f"Order {names} failed: {e}")

        if best_perm is None:
            raise SystemExit("No order worked on the validation set.")
        ordered = list(best_perm)
        print("Chosen order:", [p[0] for p in ordered], "acc:", best_acc)

    # Build and save the final pipeline
    V = CombinedPrefitVectorizer([p[1] for p in ordered])
    pipe = Pipeline([("vect", V), ("clf", clf)])
    joblib.dump(pipe, args.out)

    print("Saved pipeline to:", Path(args.out).resolve())
    print("Final order:", [p[0] for p in ordered])

if __name__ == "__main__":
    main()
