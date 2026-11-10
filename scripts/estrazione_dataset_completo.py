"""Parte 5 - Gait Analysis Lab (FIA).

Estrae 224 feature da accelerometro e giroscopio per ogni registrazione N/P
valida nel dataset completo e genera X, y e groups per la classificazione.

Dipendenze: numpy (gia' presente in requirements.txt).
Posizionare questo file in repo/scripts/estrazione_dataset_completo.py
Eseguire da repo/: python3 scripts/estrazione_dataset_completo.py
"""

from collections import defaultdict
from pathlib import Path
import csv
import re

import numpy as np

# --------------------------------------------------
# CONFIGURAZIONE
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset_completo"
OUTPUT_DIR = ROOT / "risultati"

# Stessa finestra usata nel confronto grafico: 2-10 secondi.
# In questo script "intero" significa tutta questa finestra.
# Per analizzare ogni registrazione intera impostare FINESTRA = None.
FINESTRA = (2.0, 10.0)

SENSORI = {
    "acc": "Accelerometer.csv",
    "gyro": "Gyroscope.csv",
}
CANALI = ("x", "y", "z", "magnitude")
STATISTICHE = ("mean", "var", "std", "range", "iqr", "energy", "freq_dom")
PORZIONI = ("intero", "inizio", "centro", "fine")

REGEX_CARTELLA = re.compile(
    r"^Gruppo_?(\d+)_stud_?(\d+)_(N|P)(?:$|[-_].*)",
    re.IGNORECASE,
)


def cerca_registrazioni():
    """Restituisce (gruppo, studente, tipo) -> possibili cartelle."""
    trovate = defaultdict(list)
    for path in sorted(DATA_DIR.rglob("Accelerometer.csv")):
        match = REGEX_CARTELLA.match(path.parent.name)
        if match:
            gruppo, studente, tipo = match.groups()
            trovate[(int(gruppo), int(studente), tipo.upper())].append(path.parent)
    return trovate


def leggi_sensore(percorso):
    """Legge tempo e assi XYZ; scarta righe non numeriche e timestamp ripetuti."""
    if not percorso.is_file() or percorso.stat().st_size == 0:
        raise ValueError(f"file assente o vuoto: {percorso.name}")

    try:
        dati = np.genfromtxt(
            percorso, delimiter=",", names=True,
            encoding="utf-8-sig", dtype=float,
        )
    except (OSError, IndexError, ValueError, UnicodeError) as exc:
        raise ValueError(f"CSV non leggibile ({percorso.name}): {exc}") from exc

    dati = np.atleast_1d(dati)
    colonne = ("seconds_elapsed", "x", "y", "z")
    if dati.dtype.names is None or not set(colonne).issubset(dati.dtype.names):
        raise ValueError(f"colonne obbligatorie mancanti: {percorso.name}")

    t = np.asarray(dati["seconds_elapsed"], dtype=float)
    assi = {asse: np.asarray(dati[asse], dtype=float) for asse in ("x", "y", "z")}

    validi = np.isfinite(t)
    for valori in assi.values():
        validi &= np.isfinite(valori)
    t = t[validi]
    assi = {asse: valori[validi] for asse, valori in assi.items()}
    if len(t) < 30:
        raise ValueError(f"campioni validi insufficienti: {percorso.name}")

    ordine = np.argsort(t)
    t = t[ordine]
    assi = {asse: valori[ordine] for asse, valori in assi.items()}
    unici = np.r_[True, np.diff(t) > 0]
    t = t[unici]
    assi = {asse: valori[unici] for asse, valori in assi.items()}
    t = t - t[0]

    if FINESTRA is not None:
        inizio, fine = FINESTRA
        if t[-1] < fine - 0.01:
            raise ValueError(f"durata inferiore a {fine:g} s: {percorso.name}")
        mask = (t >= inizio) & (t <= fine)
        t = t[mask]
        assi = {asse: valori[mask] for asse, valori in assi.items()}

    if len(t) < 30 or not np.all(np.diff(t) > 0):
        raise ValueError(f"campioni/timestamp insufficienti: {percorso.name}")
    return t, assi


def frequenza_dominante(t, valori):
    """Stima la frequenza dominante via FFT (segnale ricampionato uniformemente)."""
    n = len(valori)
    if n < 3 or t[-1] <= t[0]:
        return 0.0
    t_uniforme = np.linspace(t[0], t[-1], n)
    segnale = np.interp(t_uniforme, t, valori)
    segnale = segnale - segnale.mean()  # elimina la componente continua
    ampiezze = np.abs(np.fft.rfft(segnale))
    frequenze = np.fft.rfftfreq(n, d=t_uniforme[1] - t_uniforme[0])
    if len(ampiezze) < 2 or np.max(ampiezze[1:]) == 0:
        return 0.0
    return float(frequenze[1 + np.argmax(ampiezze[1:])])


def statistiche(t, valori):
    q25, q75 = np.percentile(valori, [25, 75])
    return {
        "mean": float(np.mean(valori)),
        "var": float(np.var(valori)),
        "std": float(np.std(valori)),
        "range": float(np.ptp(valori)),
        "iqr": float(q75 - q25),
        "energy": float(np.sum(valori ** 2)),
        "freq_dom": frequenza_dominante(t, valori),
    }


def feature_sensore(t, assi, prefisso):
    """Restituisce 4 canali x 4 porzioni x 7 statistiche = 112 feature."""
    segnali = dict(assi)
    segnali["magnitude"] = np.sqrt(
        assi["x"] ** 2 + assi["y"] ** 2 + assi["z"] ** 2
    )
    features = {}
    for canale in CANALI:
        valori = segnali[canale]
        indici = np.arange(len(valori))
        primo, centro, ultimo = np.array_split(indici, 3)
        segmenti = {
            "intero": indici, "inizio": primo,
            "centro": centro, "fine": ultimo,
        }
        for porzione in PORZIONI:
            idx = segmenti[porzione]
            if len(idx) < 3:
                raise ValueError("una porzione ha meno di tre campioni")
            misure = statistiche(t[idx], valori[idx])
            for statistica in STATISTICHE:
                features[f"{prefisso}_{canale}_{porzione}_{statistica}"] = misure[statistica]
    return features


def estrai_registrazione(cartella):
    features = {}
    for prefisso, filename in SENSORI.items():
        t, assi = leggi_sensore(cartella / filename)
        features.update(feature_sensore(t, assi, prefisso))
    if len(features) != 224 or not all(np.isfinite(x) for x in features.values()):
        raise ValueError("feature non valide o numero di colonne errato")
    return features


def main():
    if not DATA_DIR.is_dir():
        raise SystemExit(f"Dataset completo non trovato: {DATA_DIR}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    registrazioni = cerca_registrazioni()
    soggetti = sorted({(g, s) for g, s, _ in registrazioni})
    righe = []
    registro = []
    escluse = 0

    for gruppo, studente in soggetti:
        coppia = {}
        for tipo in ("N", "P"):
            candidate = registrazioni.get((gruppo, studente, tipo), [])
            if not candidate:
                registro.append([gruppo, studente, tipo, "NO", "registrazione assente", ""])
                continue
            motivi = []
            for cartella in candidate:
                try:
                    feat = estrai_registrazione(cartella)
                    coppia[tipo] = feat
                    registro.append([gruppo, studente, tipo, "OK", "", str(cartella)])
                    break
                except (ValueError, OSError) as exc:
                    motivi.append(f"{cartella.name}: {exc}")
            if tipo not in coppia:
                registro.append([gruppo, studente, tipo, "NO", "; ".join(motivi), ""])

        # Manteniamo solo coppie N/P utilizzabili, evitando lo sbilanciamento.
        if "N" in coppia and "P" in coppia:
            for tipo in ("N", "P"):
                righe.append({"gruppo": gruppo, "studente": studente,
                             "tipo": tipo, **coppia[tipo]})
        else:
            escluse += 1
            print(f"Escluso gruppo {gruppo}, studente {studente}: coppia N/P incompleta")

    with (OUTPUT_DIR / "controllo_registrazioni.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gruppo", "studente", "tipo", "stato", "motivo", "percorso"])
        writer.writerows(registro)

    if not righe:
        raise SystemExit("Nessuna coppia N/P valida: consulta controllo_registrazioni.csv")

    nomi_feature = [k for k in righe[0] if k not in ("gruppo", "studente", "tipo")]
    with (OUTPUT_DIR / "features_dataset_completo.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gruppo", "studente", "tipo"] + nomi_feature)
        writer.writeheader()
        writer.writerows(righe)

    X = np.array([[r[k] for k in nomi_feature] for r in righe], dtype=float)
    y = np.array([r["tipo"] for r in righe])
    gruppi = np.array([f'{r["gruppo"]}_{r["studente"]}' for r in righe])
    np.savez_compressed(OUTPUT_DIR / "dataset_ml.npz", X=X, y=y,
                        groups=gruppi, feature_names=np.array(nomi_feature))

    print("\nESTRAZIONE COMPLETATA")
    print(f"Soggetti individuati: {len(soggetti)}")
    print(f"Coppie N/P valide: {len(righe) // 2}")
    print(f"Coppie escluse: {escluse}")
    print(f"Matrice X: {X.shape}; etichette y: {y.shape}")
    print(f"Numero feature: {len(nomi_feature)}")
    print("File generati in risultati/:")
    print(" - features_dataset_completo.csv")
    print(" - dataset_ml.npz")
    print(" - controllo_registrazioni.csv")


if __name__ == "__main__":
    main()
