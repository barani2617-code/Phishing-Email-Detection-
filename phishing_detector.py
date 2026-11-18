#!/usr/bin/env python3
"""
Phishing Email Detection Model  -  Thiranex Cyber Security Internship, Task 3
=============================================================================
Trains a scikit-learn model that classifies emails as "Phishing" or "Safe".

Task requirements covered
  * Train on a dataset of phishing and legitimate emails      -> `train`
  * Extract features (URLs, keywords, etc.)                   -> phishing_features.py + TF-IDF
  * Classify emails as "Phishing" or "Safe"                   -> `predict`
  * Display accuracy and confusion matrix                     -> printed + PNG saved

Usage
  python phishing_detector.py train --data Phishing_Email.csv
  python phishing_detector.py predict --text "URGENT verify your account http://1.2.3.4/login"
  python phishing_detector.py predict --file suspicious_email.txt
"""
import argparse
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from phishing_features import HandcraftedFeatures, explain

TEXT_COLS = ["text", "Email Text", "email", "body", "message", "content", "v2", "Message"]
LABEL_COLS = ["label", "Email Type", "class", "type", "target", "v1", "Category", "Label"]
PHISH_WORDS = ("phish", "spam", "malicious", "fraud")
SAFE_WORDS = ("safe", "ham", "legit", "benign", "normal")
CLASS_NAMES = ["Safe", "Phishing"]            # index 0 / 1


# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
def pick_column(df, requested, candidates, what):
    if requested:
        if requested not in df.columns:
            sys.exit(f"Column '{requested}' not found. Columns are: {list(df.columns)}")
        return requested
    for c in candidates:
        if c in df.columns:
            return c
    sys.exit(f"Could not guess the {what} column. Columns are {list(df.columns)}. "
             f"Pass it with --{'text' if what == 'text' else 'label'}-col.")


def map_label(value, phish_label=None):
    """Convert any label spelling to 1 (phishing) or 0 (safe)."""
    v = str(value).strip().lower()
    if phish_label is not None:
        return int(v == str(phish_label).strip().lower())
    if v in ("1", "true", "yes") or any(w in v for w in PHISH_WORDS):
        return 1
    if v in ("0", "false", "no") or any(w in v for w in SAFE_WORDS):
        return 0
    raise ValueError(f"Unrecognised label '{value}'. Use --phish-label.")


def load_dataset(path, text_col=None, label_col=None, phish_label=None):
    df = pd.read_csv(path)
    tcol = pick_column(df, text_col, TEXT_COLS, "text")
    lcol = pick_column(df, label_col, LABEL_COLS, "label")
    df = df[[tcol, lcol]].dropna()
    df.columns = ["text", "label"]
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    before = len(df)
    # Duplicates are dangerous: the same email in train AND test inflates accuracy.
    df = df.drop_duplicates(subset="text")
    df["y"] = [map_label(v, phish_label) for v in df["label"]]
    info = {"rows_after_cleaning": len(df), "duplicates_removed": before - len(df),
            "phishing": int(df["y"].sum()), "safe": int((df["y"] == 0).sum())}
    return df["text"].reset_index(drop=True), df["y"].reset_index(drop=True), info


# ---------------------------------------------------------------------------
# 2. MODEL
# ---------------------------------------------------------------------------
def build_pipeline(kind="logreg"):
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2,
                            max_features=20000, sublinear_tf=True)
    hand = Pipeline([("features", HandcraftedFeatures()),
                     ("scale", StandardScaler(with_mean=False))])
    features = FeatureUnion([("tfidf", tfidf), ("hand", hand)])
    clfs = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     n_jobs=-1, random_state=42),
        "nb": MultinomialNB(),
    }
    return Pipeline([("features", features), ("clf", clfs[kind])])


def top_features(pipe, n=12):
    """For logistic regression: words/features pushing towards Phishing / Safe."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return None
    names = pipe.named_steps["features"].get_feature_names_out()
    order = clf.coef_[0].argsort()
    return {"Phishing": [names[i] for i in order[::-1][:n]],
            "Safe": [names[i] for i in order[:n]]}


# ---------------------------------------------------------------------------
# 3. TRAIN + EVALUATE
# ---------------------------------------------------------------------------
def train(args):
    X, y, info = load_dataset(args.data, args.text_col, args.label_col, args.phish_label)
    print(f"Dataset: {info}")
    if min(info["phishing"], info["safe"]) < 10:
        sys.exit("Need at least 10 emails of each class.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42)
    print(f"Train: {len(X_train)} emails | Test: {len(X_test)} emails (never used for tuning)")

    # --- choose the model using cross-validation on the TRAINING set only ----
    kinds = ["logreg", "rf", "nb"] if args.model == "auto" else [args.model]
    cv_scores = {}
    if len(kinds) > 1:
        print(f"\nComparing models with {args.cv}-fold cross-validation (F1 on training data):")
        for k in kinds:
            s = cross_val_score(build_pipeline(k), X_train, y_train, cv=args.cv, scoring="f1")
            cv_scores[k] = round(float(s.mean()), 4)
            print(f"   {k:7s} F1 = {s.mean():.4f}  (+/- {s.std():.4f})")
        best = max(cv_scores, key=cv_scores.get)
    else:
        best = kinds[0]
    print(f"\nSelected model: {best}")

    pipe = build_pipeline(best).fit(X_train, y_train)

    # --- final, one-time evaluation on the held-out test set -----------------
    pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "model": best, "cv_f1": cv_scores,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "dataset": info, "test_size": len(y_test),
    }
    report = classification_report(y_test, pred, target_names=CLASS_NAMES, digits=4)

    print("\n" + "=" * 60)
    print(f" Accuracy : {metrics['accuracy']:.2%}")
    print(f" Precision: {metrics['precision']:.2%}  (of emails flagged Phishing, how many really were)")
    print(f" Recall   : {metrics['recall']:.2%}  (of real phishing emails, how many we caught)")
    print(f" F1-score : {metrics['f1']:.2%}")
    print("-" * 60)
    print(" Confusion matrix (rows = actual, columns = predicted)")
    print(f"                  Pred Safe   Pred Phishing")
    print(f"   Actual Safe    {tn:9d}   {fp:13d}   <- FP: safe mail wrongly flagged")
    print(f"   Actual Phish   {fn:9d}   {tp:13d}   <- FN: phishing that got through")
    print("=" * 60 + "\n" + report)

    # --- save evidence --------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(pipe, os.path.join(args.outdir, "phishing_model.joblib"))
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.outdir, "classification_report.txt"), "w") as f:
        f.write(report)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix - {best} (accuracy {metrics['accuracy']:.2%})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    tf = top_features(pipe)
    if tf:
        print("Most influential features (logistic regression):")
        print("   Push towards PHISHING:", ", ".join(tf["Phishing"]))
        print("   Push towards SAFE    :", ", ".join(tf["Safe"]))
    print(f"\nSaved to '{args.outdir}/': phishing_model.joblib, metrics.json, "
          "classification_report.txt, confusion_matrix.png")
    return metrics


# ---------------------------------------------------------------------------
# 4. PREDICT
# ---------------------------------------------------------------------------
def predict(args):
    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        sys.exit("Give --text or --file.")
    pipe = joblib.load(os.path.join(args.outdir, "phishing_model.joblib"))
    proba = pipe.predict_proba([text])[0]
    label = CLASS_NAMES[int(proba.argmax())]
    print(f"Verdict   : {label.upper()}")
    print(f"Confidence: {proba.max():.1%}  (P(phishing) = {proba[1]:.1%})")
    flags = explain(text)
    print("Signals found in this email:", flags if flags else "none of the hand-crafted signals")
    print("Note: a model is an aid, not proof - always verify suspicious mail manually.")
    return label, float(proba[1])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phishing email detector (educational)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train and evaluate the model")
    t.add_argument("--data", required=True, help="CSV file with email text + label")
    t.add_argument("--text-col"); t.add_argument("--label-col")
    t.add_argument("--phish-label", help="label value meaning phishing (if auto-detect fails)")
    t.add_argument("--model", choices=["auto", "logreg", "rf", "nb"], default="auto")
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--cv", type=int, default=3)
    t.add_argument("--outdir", default="results")
    p = sub.add_parser("predict", help="classify one email")
    p.add_argument("--text"); p.add_argument("--file")
    p.add_argument("--outdir", default="results")
    args = ap.parse_args(argv)
    return train(args) if args.cmd == "train" else predict(args)


if __name__ == "__main__":
    main()
