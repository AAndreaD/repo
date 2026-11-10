
from pathlib import Path
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# CARICAMENTO DATI
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "risultati" / "dataset_ml.npz"

with np.load(DATA_PATH) as dati:
    X = dati["X"]
    y = dati["y"]
    gruppi = dati["groups"]
    nomi = dati["feature_names"].astype(str)

# Seleziona le feature del modulo
mask_modulo = np.array([
    "_magnitude_" in nome for nome in nomi
])

X_modulo = X[:, mask_modulo]

print("Feature totali:", X.shape[1])
print("Feature del modulo:", X_modulo.shape[1])

# ==========================================
# CONFIGURAZIONI DA CONFRONTARE
# ==========================================

esperimenti = {
    "A - Tutte le feature": (
        X,
        make_pipeline(
            MinMaxScaler(),
            GaussianNB()
        )
    ),

    "B - Solo modulo": (
        X_modulo,
        make_pipeline(
            MinMaxScaler(),
            GaussianNB()
        )
    ),

    "C - Migliori 10 del modulo": (
        X_modulo,
        make_pipeline(
            MinMaxScaler(),
            SelectKBest(score_func=f_classif, k=10),
            GaussianNB()
        )
    )
}

# ==========================================
# CONFRONTO CON GROUP K-FOLD
# ==========================================

cv = GroupKFold(n_splits=5)

for nome, (X_esperimento, modello) in esperimenti.items():

    predizioni = np.empty_like(y)
    accuratezze = []

    for train_idx, test_idx in cv.split(
        X_esperimento, y, groups=gruppi
    ):

        assert set(gruppi[train_idx]).isdisjoint(
            set(gruppi[test_idx])
        )

        modello.fit(
            X_esperimento[train_idx],
            y[train_idx]
        )

        y_pred = modello.predict(
            X_esperimento[test_idx]
        )

        predizioni[test_idx] = y_pred

        accuracy = accuracy_score(
            y[test_idx], y_pred
        )

        accuratezze.append(accuracy)

    accuracy_totale = accuracy_score(y, predizioni)
    cm = confusion_matrix(
        y, predizioni, labels=["N", "P"]
    )

    recall_p = cm[1, 1] / cm[1].sum()

    print("\n" + "=" * 50)
    print(nome)
    print("=" * 50)

    print(f"Accuracy: {accuracy_totale:.2%}")
    print(f"Recall patologica: {recall_p:.2%}")

    print("Accuracy per fold:")
    print([round(a, 3) for a in accuratezze])

    print("Matrice di confusione:")
    print(cm)
