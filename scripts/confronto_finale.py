
from pathlib import Path
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix
)

# ==========================================
# 1. CARICAMENTO
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "risultati" / "dataset_ml.npz"

with np.load(DATA_PATH) as dati:
    X = dati["X"]
    y = dati["y"]
    gruppi = dati["groups"]
    nomi = dati["feature_names"].astype(str)

# ==========================================
# 2. FEATURE FINALI
# ==========================================

feature_finali = [
    "acc_magnitude_centro_freq_dom",
    "acc_magnitude_fine_freq_dom",
    "acc_magnitude_intero_freq_dom"
]

indici = []

for nome in feature_finali:
    trovati = np.flatnonzero(nomi == nome)

    if len(trovati) != 1:
        raise ValueError(
            f"Feature non trovata o duplicata: {nome}"
        )

    indici.append(int(trovati[0]))

X_ridotto = X[:, indici]

print("Dataset originale:", X.shape)
print("Dataset ridotto:", X_ridotto.shape)

# ==========================================
# 3. CINQUE TEST CON GLI STESSI SOGGETTI
# ==========================================

cv = GroupShuffleSplit(
    n_splits=5,
    test_size=1/3,
    random_state=42
)

risultati = {
    "224 feature": [],
    "3 feature": []
}

recall_p = {
    "224 feature": [],
    "3 feature": []
}

for test, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups=gruppi),
    start=1
):

    assert set(gruppi[train_idx]).isdisjoint(
        set(gruppi[test_idx])
    )

    print(f"\n========== TEST {test} ==========")

    configurazioni = {
        "224 feature": X,
        "3 feature": X_ridotto
    }

    for nome, matrice in configurazioni.items():

        modello = make_pipeline(
            MinMaxScaler(),
            GaussianNB()
        )

        modello.fit(
            matrice[train_idx],
            y[train_idx]
        )

        predizioni = modello.predict(
            matrice[test_idx]
        )

        accuracy = accuracy_score(
            y[test_idx], predizioni
        )

        recall = recall_score(
            y[test_idx],
            predizioni,
            pos_label="P"
        )

        risultati[nome].append(accuracy)
        recall_p[nome].append(recall)

        print(
            f"\n{nome}: "
            f"accuracy={accuracy:.2%}, "
            f"recall P={recall:.2%}"
        )

        print(
            confusion_matrix(
                y[test_idx],
                predizioni,
                labels=["N", "P"]
            )
        )

# ==========================================
# 4. CONFRONTO FINALE
# ==========================================

print("\n" + "=" * 50)
print("CONFRONTO FINALE")
print("=" * 50)

for nome in risultati:

    valori = np.array(risultati[nome])

    print(f"\n{nome}")
    print(f"Accuracy media: {np.mean(valori):.2%}")
    print(f"Deviazione standard: {np.std(valori):.2%}")
    print(
        f"Recall P medio: "
        f"{np.mean(recall_p[nome]):.2%}"
    )

print("\nConfronto completato.")
