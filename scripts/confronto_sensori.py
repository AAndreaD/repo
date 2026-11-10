
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

# ==============================================
# CONFIGURAZIONE
# ==============================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset_completo"
OUTPUT_DIR = BASE_DIR / "risultati"

STUDENTI = [(10, 1), (10, 2), (4, 1), (4, 3)]

SENSORI = {
    "Accelerometer": ("Accelerometro", "m/s²", "acc"),
    "Gyroscope": ("Giroscopio", "rad/s", "gyro")
}

FINESTRA = (2.0, 10.0)

PATTERN = re.compile(
    r"^Gruppo_?(\d+)_stud_?(\d+)_(N|P)(?:$|[-_].*)",
    re.IGNORECASE
)


# ==============================================
# RICERCA REGISTRAZIONI
# ==============================================

def trova_cartelle():
    archivio = {}

    for file in sorted(DATASET_DIR.rglob("Accelerometer.csv")):
        match = PATTERN.match(file.parent.name)

        if match is None:
            continue

        gruppo = int(match.group(1))
        studente = int(match.group(2))
        tipo = match.group(3).upper()

        chiave = (gruppo, studente, tipo)

        archivio.setdefault(chiave, []).append(file.parent)

    return archivio


# ==============================================
# CARICAMENTO DEL SEGNALE
# ==============================================

def carica_sensore(archivio, gruppo, studente, tipo, sensore):

    chiave = (gruppo, studente, tipo)
    colonne = ("seconds_elapsed", "x", "y", "z")

    for cartella in archivio.get(chiave, []):
        percorso = cartella / f"{sensore}.csv"

        if not percorso.is_file() or percorso.stat().st_size == 0:
            continue

        try:
            dati = np.genfromtxt(
                percorso,
                delimiter=",",
                names=True,
                encoding="utf-8-sig"
            )

            dati = np.atleast_1d(dati)

            if dati.dtype.names is None:
                continue

            if not all(c in dati.dtype.names for c in colonne):
                continue

            tempo = np.asarray(
                dati["seconds_elapsed"], dtype=float
            )

            segnali = {
                asse: np.asarray(dati[asse], dtype=float)
                for asse in ("x", "y", "z")
            }

            validi = np.isfinite(tempo)

            for valori in segnali.values():
                validi &= np.isfinite(valori)

            tempo = tempo[validi]

            for asse in segnali:
                segnali[asse] = segnali[asse][validi]

            if len(tempo) < 3:
                continue

            tempo = tempo - tempo[0]

            mask = (
                (tempo >= FINESTRA[0]) &
                (tempo <= FINESTRA[1])
            )

            tempo = tempo[mask]

            for asse in segnali:
                segnali[asse] = segnali[asse][mask]

            if len(tempo) < 3:
                continue

            return tempo, segnali

        except (ValueError, IndexError, OSError, TypeError):
            continue

    return None


# ==============================================
# CONFRONTO N / P
# ==============================================

def confronta(archivio, gruppo, studente, sensore, nome, unita):

    normale = carica_sensore(
        archivio, gruppo, studente, "N", sensore
    )

    patologica = carica_sensore(
        archivio, gruppo, studente, "P", sensore
    )

    if normale is None or patologica is None:
        print(
            f"DATI MANCANTI: gruppo {gruppo}, "
            f"studente {studente}, {sensore}"
        )
        return False

    fig, axes = plt.subplots(
        3, 2, figsize=(14, 9),
        sharex=True, sharey="row"
    )

    colori = {
        "x": "tab:red",
        "y": "tab:green",
        "z": "tab:blue"
    }

    for colonna, (tipo, risultato) in enumerate(
        (("N", normale), ("P", patologica))
    ):
        tempo, segnali = risultato

        for riga, asse in enumerate(("x", "y", "z")):
            ax = axes[riga, colonna]

            ax.plot(
                tempo,
                segnali[asse],
                color=colori[asse],
                linewidth=0.8
            )

            ax.set_xlim(*FINESTRA)
            ax.set_ylabel(f"{asse.upper()} ({unita})")
            ax.grid(True, alpha=0.3)

        titolo = (
            "Camminata normale" if tipo == "N"
            else "Patologica simulata"
        )

        axes[0, colonna].set_title(titolo)
        axes[-1, colonna].set_xlabel("Tempo (s)")

    fig.suptitle(
        f"{nome} - Gruppo {gruppo} - Studente {studente}",
        fontsize=14
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUTPUT_DIR.mkdir(exist_ok=True)

    fig.savefig(
        OUTPUT_DIR / f"{sensore}_G{gruppo}_S{studente}.png",
        dpi=150
    )

    print(f"\n{nome} - Gruppo {gruppo}, Studente {studente}")

    for asse in ("x", "y", "z"):
        std_n = np.std(normale[1][asse])
        std_p = np.std(patologica[1][asse])

        print(
            f"Asse {asse.upper()}: "
            f"std N = {std_n:.3f}, "
            f"std P = {std_p:.3f}"
        )

    return True


# ==============================================
# PROGRAMMA PRINCIPALE
# ==============================================

if __name__ == "__main__":

    if not DATASET_DIR.exists():
        raise SystemExit("Dataset completo non trovato.")

    archivio = trova_cartelle()

    conteggio = 0

    for sensore, (nome, unita, sigla) in SENSORI.items():

        for gruppo, studente in STUDENTI:

            if confronta(
                archivio, gruppo, studente,
                sensore, nome, unita
            ):
                conteggio += 1

    print(f"\nConfronti generati: {conteggio}")
    print(f"Grafici salvati in: {OUTPUT_DIR}")

    plt.show()
