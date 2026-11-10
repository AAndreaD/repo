
from pathlib import Path
from collections import Counter
import numpy as np

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)

# ==========================================
# CARICAMENTO DEL DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "risultati" / "dataset_ml.npz"

with np.load(DATA_PATH) as dati:
    X = dati["X"]
    y = dati["y"]
    gruppi = dati["groups"]
    nomi = dati["feature_names"].astype(str)

# Utilizziamo soltanto le feature del modulo
mask = np.array([
    "_magnitude_" in nome for nome in nomi
])

X = X[:, mask]
nomi = nomi[mask]

print("Matrice:", X.shape)

# Numeri di feature da confrontare
VALORI_K = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 56]

# ==========================================
# MODELLO E SELEZIONE
# ==========================================

pipeline = make_pipeline(
    MinMaxScaler(),
    SelectKBest(score_func=f_classif),
    GaussianNB()
)

parametri = {
    "selectkbest__k": VALORI_K
}

cv_esterna = GroupKFold(n_splits=5)

predizioni = np.empty_like(y)
conteggio_feature = Counter()

# ==========================================
# NESTED CROSS-VALIDATION
# ==========================================

for fold, (train_idx, test_idx) in enumerate(
    cv_esterna.split(X, y, groups=gruppi),
    start=1
):

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    gruppi_train = gruppi[train_idx]

    assert set(gruppi_train).isdisjoint(
        set(gruppi[test_idx])
    )

    # Cross-validation interna:
    # sceglie il numero di feature
    ricerca = GridSearchCV(
        estimator=pipeline,
        param_grid=parametri,
        cv=GroupKFold(n_splits=4),
        scoring="balanced_accuracy",
        refit=True,
        n_jobs=1
    )

    ricerca.fit(
        X_train,
        y_train,
        groups=gruppi_train
    )

    modello = ricerca.best_estimator_
    k_migliore = ricerca.best_params_["selectkbest__k"]

    # Valutazione sui soggetti del test esterno
    y_pred = modello.predict(X_test)
    predizioni[test_idx] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    balanced = balanced_accuracy_score(y_test, y_pred)

    print(
        f"\nFold {fold}: "
        f"k={k_migliore}, "
        f"accuracy={accuracy:.2%}, "
        f"balanced accuracy={balanced:.2%}"
    )

    # Registra le feature selezionate
    selettore = modello.named_steps["selectkbest"]
    indici = selettore.get_support(indices=True)

    selezionate = nomi[indici]

    conteggio_feature.update(selezionate)

# ==========================================
# RISULTATI COMPLESSIVI
# ==========================================

cm = confusion_matrix(
    y, predizioni, labels=["N", "P"]
)

print("\nRISULTATI FINALI")
print("Accuracy:", f"{accuracy_score(y, predizioni):.2%}")
print(
    "Balanced accuracy:",
    f"{balanced_accuracy_score(y, predizioni):.2%}"
)

print("\nMatrice di confusione:")
print(cm)

print("\nFEATURE SELEZIONATE PIÙ SPESSO")

for nome, volte in conteggio_feature.most_common(15):
    print(f"{volte}/5 fold - {nome}")
