
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import csv

# ==================================================
# CONFIGURAZIONE
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset_completo"

GRUPPI_DA_ANALIZZARE = [10, 4]

# Riconosce i nomi delle cartelle delle registrazioni
PATTERN = re.compile(
    r"^Gruppo_?(\d+)_stud_?(\d+)_(N|P)(?:$|[-_].*)",
    re.IGNORECASE
)


# ==================================================
# RICERCA AUTOMATICA DEI DATI
# ==================================================

def trova_registrazioni():

    registrazioni = {}

    # Cerca tutti i file dell'accelerometro
    for csv_path in DATASET_DIR.rglob("Accelerometer.csv"):

        nome_cartella = csv_path.parent.name
        match = PATTERN.match(nome_cartella)

        if not match:
            continue

        gruppo = int(match.group(1))
        studente = int(match.group(2))
        tipo = match.group(3).upper()

        identificativo = (gruppo, studente)

        if identificativo not in registrazioni:
            registrazioni[identificativo] = {}

        if tipo in registrazioni[identificativo]:
            print(f"Attenzione: registrazione duplicata per {identificativo}, tipo {tipo}")
            continue

        registrazioni[identificativo][tipo] = csv_path

    return registrazioni


# ==================================================
# ANALISI DI UNA REGISTRAZIONE
# ==================================================

def analizza_camminata(csv_path, gruppo, studente, tipo):

    # Carica il file CSV
    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        encoding="utf-8"
    )

    # Controlla che le colonne richieste esistano
    colonne = ("seconds_elapsed", "x", "y", "z")

    if not all(col in data.dtype.names for col in colonne):
        print(f"Colonne mancanti in {csv_path}")
        return

    seconds = np.atleast_1d(data["seconds_elapsed"])
    x = np.atleast_1d(data["x"])
    y = np.atleast_1d(data["y"])
    z = np.atleast_1d(data["z"])

    if len(seconds) < 2:
        print(f"Dati insufficienti: {csv_path}")
        return

    tipo_nome = (
        "Normale" if tipo == "N"
        else "Patologica simulata"
    )

    # Informazioni nel terminale
    print(f"\nGruppo {gruppo} - Studente {studente}")
    print(f"Camminata: {tipo_nome}")
    print(f"Campioni: {len(seconds)}")
    print(f"Durata: {seconds[-1] - seconds[0]:.2f} s")

    # Creazione dei grafici
    fig, axes = plt.subplots(
        3, 1,
        sharex=True,
        figsize=(12, 8)
    )

    for ax, signal, label, color in zip(
        axes,
        (x, y, z),
        ("X", "Y", "Z"),
        ("tab:red", "tab:green", "tab:blue")
    ):

        ax.plot(
            seconds,
            signal,
            color=color,
            linewidth=0.8
        )

        ax.set_ylabel(f"Acc {label} (m/s²)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Tempo (s)")

    fig.suptitle(
        f"Gruppo {gruppo} - Studente {studente} - {tipo_nome}"
    )

    fig.tight_layout()
    plt.show()


# ==================================================
# VERIFICA VALIDITÀ DEI FILE CSV
# ==================================================

def csv_valido(percorso):
    """Verifica che il CSV contenga almeno due campioni validi."""

    if not percorso.is_file() or percorso.stat().st_size == 0:
        return False

    try:
        with percorso.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)

            colonne = {"seconds_elapsed", "x", "y", "z"}

            if not reader.fieldnames or not colonne.issubset(reader.fieldnames):
                return False

            for _ in range(2):
                riga = next(reader, None)

                if riga is None:
                    return False

                for colonna in colonne:
                    if not np.isfinite(float(riga[colonna])):
                        return False

            return True

    except (OSError, ValueError, TypeError, KeyError, UnicodeError, csv.Error):
        return False


# ==================================================
# PROGRAMMA PRINCIPALE
# ==================================================

if __name__ == "__main__":

    # Verifica che il dataset esista
    if not DATASET_DIR.exists():
        print(f"Dataset non trovato: {DATASET_DIR}")
        raise SystemExit(1)

    # Cerca tutte le registrazioni disponibili
    registrazioni = trova_registrazioni()

    print(f"Studenti individuati: {len(registrazioni)}")

    # Seleziona gli studenti che hanno sia N che P
    studenti_completi = [
        studente
        for studente, dati in registrazioni.items()
        if "N" in dati and "P" in dati
    ]

    print(
        f"Studenti con entrambe le registrazioni: "
        f"{len(studenti_completi)}"
    )

    # Analizza esclusivamente i gruppi 10 e 3
    for gruppo in GRUPPI_DA_ANALIZZARE:

        studenti_gruppo = sorted(
            [
                (g, s)
                for g, s in studenti_completi
                if g == gruppo
                and csv_valido(registrazioni[(g, s)]["N"])
                and csv_valido(registrazioni[(g, s)]["P"])
            ],
            key=lambda elemento: elemento[1]
        )

        print(
            f"\nGruppo {gruppo}: "
            f"{len(studenti_gruppo)} studenti completi"
        )

        # Analizza due studenti per gruppo
        for g, studente in studenti_gruppo[:2]:

            dati = registrazioni[(g, studente)]

            for tipo in ("N", "P"):
                analizza_camminata(
                    dati[tipo],
                    g,
                    studente,
                    tipo
                )

    print("\nEsplorazione completata.")


