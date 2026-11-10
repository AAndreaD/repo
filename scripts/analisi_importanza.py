
from pathlib import Path
from collections import Counter

import numpy as np

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# ==============================================
# 1. CARICAMENTO DEL DATASET
# ==============================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "risultati" / "dataset_ml.npz"

with np.load(DATA_PATH) as dati:
    X = dati["X"]
    y = dati["y"]
    gruppi = dati["groups"]
    nomi = dati["feature_names"].astype(str)

# Manteniamo le 56 feature del modulo
mask = np.array(["_magnitude_" in n for n in nomi])

X = X[:, mask]
nomi = nomi[mask]

print(f"Dataset: {X.shape[0]} registrazioni")
print(f"Feature disponibili: {X.shape[1]}")

# ==============================================
# 2. MODELLO
# ==============================================

pipeline = make_pipeline(
    MinMaxScaler(),
    SelectKBest(score_func=f_classif),
    GaussianNB()
)

parametri = {
    "selectkbest__k": [5, 8, 10, 15, 20, 30, 40, 56]
}

cv_esterna = GroupKFold(n_splits=5)

importanza_totale = np.zeros(len(nomi))
frequenza_selezione = Counter()
coppie_correlate = Counter()

predizioni = np.empty_like(y)

# ==============================================
# 3. CROSS-VALIDATION
# ==============================================

for fold, (train_idx, test_idx) in enumerate(
    cv_esterna.split(X, y, groups=gruppi),
    start=1
):

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

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
        X_train,
        y_train,
        groups=gruppi[train_idx]
    )

    modello = ricerca.best_estimator_
    predizioni[test_idx] = modello.predict(X_test)

    selettore = modello.named_steps["selectkbest"]
    indici = selettore.get_support(indices=True)

    nomi_selezionati = nomi[indici]
    frequenza_selezione.update(nomi_selezionati)

    print(f"\nFOLD {fold}")
    print(f"Numero feature: {len(indici)}")

    # ==========================================
    # 4. CORRELAZIONE SUL TRAINING
    # ==========================================

    X_sel = X_train[:, indici]

    with np.errstate(divide="ignore", invalid="ignore"):
        correlazioni = np.corrcoef(X_sel, rowvar=False)

    correlate_fold = 0

    for i in range(len(indici)):
        for j in range(i + 1, len(indici)):

            r = correlazioni[i, j]

            if np.isfinite(r) and abs(r) >= 0.90:
                coppia = tuple(sorted([
                    str(nomi_selezionati[i]),
                    str(nomi_selezionati[j])
                ]))

                coppie_correlate[coppia] += 1
                correlate_fold += 1

    print(f"Coppie molto correlate: {correlate_fold}")

    # ==========================================
    # 5. PERMUTATION IMPORTANCE SUL TEST
    # ==========================================

    risultato = permutation_importance(
        modello,
        X_test,
        y_test,
        scoring="balanced_accuracy",
        n_repeats=30,
        random_state=42,
        n_jobs=1
    )

    importanza_totale += risultato.importances_mean

    accuracy = balanced_accuracy_score(
        y_test, predizioni[test_idx]
    )

    print(f"Balanced accuracy: {accuracy:.2%}")

# ==============================================
# 6. RISULTATI FINALI
# ==============================================

importanza_media = importanza_totale / 5

ordinamento = np.argsort(importanza_media)[::-1]

print("\n" + "=" * 55)
print("PERMUTATION IMPORTANCE - PRIME 15 FEATURE")
print("=" * 55)

for indice in ordinamento[:15]:
    print(
        f"{nomi[indice]}\n"
        f"  Importanza media: {importanza_media[indice]:+.4f} | "
        f"Selezionata: {frequenza_selezione[nomi[indice]]}/5"
    )

print("\nCOPPIE CON CORRELAZIONE FORTE (|r| >= 0.90)")

if coppie_correlate:
    for coppia, volte in coppie_correlate.most_common(10):
        print(f"{volte}/5 fold: {coppia[0]} <-> {coppia[1]}")
else:
    print("Nessuna coppia fortemente correlata trovata.")

print("\nMATRICE DI CONFUSIONE")
print(confusion_matrix(y, predizioni, labels=["N", "P"]))

print("\nBALANCED ACCURACY COMPLESSIVA")
print(f"{balanced_accuracy_score(y, predizioni):.2%}")

print("\nAnalisi completata.")
