
import csv
import numpy as np

from confronto_sensori import (
    BASE_DIR,
    STUDENTI,
    SENSORI,
    trova_cartelle,
    carica_sensore
)

# ==============================================
# CALCOLO DELLE FEATURE
# ==============================================

def calcola_statistiche(tempo, valori):

    valori = np.asarray(valori, dtype=float)

    # Caratteristiche nel dominio temporale
    statistiche = {
        "mean": float(np.mean(valori)),
        "var": float(np.var(valori)),
        "std": float(np.std(valori)),
        "range": float(np.ptp(valori)),
        "iqr": float(
            np.percentile(valori, 75) -
            np.percentile(valori, 25)
        ),
        "energy": float(np.sum(valori ** 2))
    }

    # Frequenza dominante tramite FFT
    dt = np.median(np.diff(tempo))

    frequenza = 0.0

    if dt > 0 and len(valori) >= 3:
        centrato = valori - np.mean(valori)

        spettro = np.abs(np.fft.rfft(centrato))
        frequenze = np.fft.rfftfreq(
            len(centrato), d=dt
        )

        # Esclude la componente a frequenza zero
        if len(spettro) > 1 and np.max(spettro[1:]) > 0:
            indice = np.argmax(spettro[1:]) + 1
            frequenza = float(frequenze[indice])

    statistiche["freq_dom"] = frequenza

    return statistiche


# ==============================================
# FEATURE DI UN SENSORE
# ==============================================

def compute_features(tempo, segnali, prefisso):

    risultato = {}

    # Aggiungiamo il modulo del vettore
    x = segnali["x"]
    y = segnali["y"]
    z = segnali["z"]

    canali = dict(segnali)

    canali["magnitude"] = np.sqrt(
        x**2 + y**2 + z**2
    )

    for asse, valori in canali.items():

        indici = np.arange(len(valori))

        parti = np.array_split(indici, 3)

        porzioni = {
            "intero": indici,
            "inizio": parti[0],
            "centro": parti[1],
            "fine": parti[2]
        }

        for nome_porzione, idx in porzioni.items():

            statistiche = calcola_statistiche(
                tempo[idx], valori[idx]
            )

            for nome_feature, valore in statistiche.items():

                chiave = (
                    f"{prefisso}_{asse}_"
                    f"{nome_porzione}_{nome_feature}"
                )

                risultato[chiave] = valore

    return risultato


# ==============================================
# ESTRAZIONE DAL DATASET
# ==============================================

if __name__ == "__main__":

    archivio = trova_cartelle()
    righe = []

    for gruppo, studente in STUDENTI:

        for tipo in ("N", "P"):

            riga = {
                "gruppo": gruppo,
                "studente": studente,
                "tipo": tipo
            }

            valido = True

            for sensore, (_, _, prefisso) in SENSORI.items():

                risultato = carica_sensore(
                    archivio,
                    gruppo,
                    studente,
                    tipo,
                    sensore
                )

                if risultato is None:
                    valido = False
                    print(
                        f"Dati mancanti: G{gruppo} "
                        f"S{studente} {tipo} {sensore}"
                    )
                    break

                tempo, segnali = risultato

                features = compute_features(
                    tempo, segnali, prefisso
                )

                riga.update(features)

            if valido:
                righe.append(riga)

    if not righe:
        raise SystemExit("Nessuna registrazione valida.")

    # Controlla che le righe abbiano le stesse feature
    nomi_feature = list(righe[0].keys())[3:]

    if any(set(r.keys()) != set(righe[0].keys()) for r in righe):
        raise SystemExit("Feature non coerenti tra registrazioni.")

    output_dir = BASE_DIR / "risultati"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "features_esplorazione.csv"

    with output_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(righe[0].keys())
        )

        writer.writeheader()
        writer.writerows(righe)

    print("\nESTRAZIONE COMPLETATA")
    print(f"Registrazioni: {len(righe)}")
    print(f"Feature per registrazione: {len(nomi_feature)}")
    print(f"Matrice X: ({len(righe)}, {len(nomi_feature)})")
    print(f"File salvato: {output_file}")
