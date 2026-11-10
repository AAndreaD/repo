
from pathlib import Path
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# CARICAMENTO DEL DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "risultati" / "dataset_ml.npz"

with np.load(DATA_PATH) as dati:
    X = dati["X"]
    y = dati["y"]
    gruppi = dati["groups"]
    feature_names = dati["feature_names"]

print("Dimensioni X:", X.shape)
print("Dimensioni y:", y.shape)
print("Numero soggetti:", len(np.unique(gruppi)))

if not np.isfinite(X).all():
    raise ValueError("La matrice contiene valori non finiti.")

# ==========================================
# CROSS-VALIDATION PER SOGGETTO
# ==========================================

cv = GroupKFold(n_splits=5)

predizioni = np.empty_like(y)
accuratezze = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups=gruppi), start=1
):

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # Controllo: nessuno studente compare
    # contemporaneamente in training e test
    assert set(gruppi[train_idx]).isdisjoint(
        set(gruppi[test_idx])
    )

    # Normalizzazione e classificatore
    modello = make_pipeline(
        MinMaxScaler(),
        GaussianNB()
    )

    # Fit soltanto sul training set
    modello.fit(X_train, y_train)

    # Predizione sul test set
    y_pred = modello.predict(X_test)

    predizioni[test_idx] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    accuratezze.append(accuracy)

    print(
        f"Fold {fold}: "
        f"{len(train_idx)} training, "
        f"{len(test_idx)} test, "
        f"accuracy = {accuracy:.2%}"
    )

# ==========================================
# VALUTAZIONE COMPLESSIVA
# ==========================================

accuracy_totale = accuracy_score(y, predizioni)

cm = confusion_matrix(
    y, predizioni, labels=["N", "P"]
)

print("\nRISULTATI FINALI")
print(f"Accuracy complessiva: {accuracy_totale:.2%}")

print("\nMatrice di confusione")
print("Righe = classe reale; colonne = predizione")
print("Ordine classi: N, P")
print(cm)
