
from pathlib import Path
import numpy as np

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score

# Caricamento
BASE_DIR = Path(__file__).resolve().parent.parent

with np.load(BASE_DIR / "risultati" / "dataset_ml.npz") as dati:
    X = dati["X"]
    y = dati["y"]
    gruppi = dati["groups"]
    nomi = dati["feature_names"].astype(str)

# Solo feature del modulo
mask = np.array(["_magnitude_" in n for n in nomi])
X = X[:, mask]

# Modello con selezione automatica
pipeline = make_pipeline(
    MinMaxScaler(),
    SelectKBest(score_func=f_classif),
    GaussianNB()
)

parametri = {
    "selectkbest__k": [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 56]
}

# Cinque suddivisioni casuali per soggetto
cv = GroupShuffleSplit(
    n_splits=5,
    test_size=1/3,
    random_state=42
)

accuratezze = []
recall_p = []

for numero, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups=gruppi), start=1
):

    assert set(gruppi[train_idx]).isdisjoint(
        set(gruppi[test_idx])
    )

    ricerca = GridSearchCV(
        pipeline,
        parametri,
        cv=GroupKFold(n_splits=4),
        scoring="balanced_accuracy",
        n_jobs=1
    )

    ricerca.fit(
        X[train_idx],
        y[train_idx],
        groups=gruppi[train_idx]
    )

    pred = ricerca.predict(X[test_idx])

    acc = accuracy_score(y[test_idx], pred)
    recall = recall_score(
        y[test_idx], pred,
        pos_label="P"
    )

    accuratezze.append(acc)
    recall_p.append(recall)

    print(
        f"Test {numero}: "
        f"accuracy={acc:.2%}, "
        f"recall P={recall:.2%}, "
        f"k={ricerca.best_params_['selectkbest__k']}"
    )

print("\nRISULTATI COMPLESSIVI")
print(f"Accuracy media: {np.mean(accuratezze):.2%}")
print(f"Deviazione standard: {np.std(accuratezze):.2%}")
print(f"Recall P medio: {np.mean(recall_p):.2%}")
