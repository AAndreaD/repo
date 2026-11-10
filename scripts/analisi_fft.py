
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from confronto_sensori import (
    BASE_DIR,
    STUDENTI,
    trova_cartelle,
    carica_sensore
)

# ==========================================
# CONFIGURAZIONE
# ==========================================

OUTPUT_DIR = BASE_DIR / "risultati" / "fft"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQ_MAX = 10  # Hz

# ==========================================
# CALCOLO FFT
# ==========================================

def calcola_fft(tempo, segnale):

    # Rimuoviamo il valore medio
    segnale = segnale - np.mean(segnale)

    # Periodo di campionamento stimato
    dt = np.median(np.diff(tempo))

    if dt <= 0:
        raise ValueError("Periodo di campionamento non valido")

    # Trasformata di Fourier
    spettro = np.abs(np.fft.rfft(segnale))

    # Frequenze associate
    frequenze = np.fft.rfftfreq(
        len(segnale),
        d=dt
    )

    # Escludiamo la componente a 0 Hz
    if len(spettro) <= 1:
        raise ValueError("Segnale troppo breve")

    indice = np.argmax(spettro[1:]) + 1

    freq_dominante = frequenze[indice]

    return frequenze, spettro, freq_dominante


# ==========================================
# CONFRONTO NORMALE / PATOLOGICA
# ==========================================

def confronta_fft(archivio, gruppo, studente):

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 8)
    )

    risultati = {}

    for colonna, tipo in enumerate(("N", "P")):

        dati = carica_sensore(
            archivio,
            gruppo,
            studente,
            tipo,
            "Accelerometer"
        )

        if dati is None:
            print(
                f"Dati mancanti: G{gruppo} "
                f"S{studente} {tipo}"
            )
            plt.close(fig)
            return

        tempo, segnali = dati

        # Modulo tridimensionale dell'accelerazione
        modulo = np.sqrt(
            segnali["x"]**2 +
            segnali["y"]**2 +
            segnali["z"]**2
        )

        frequenze, spettro, freq_dom = calcola_fft(
            tempo, modulo
        )

        risultati[tipo] = freq_dom

        # Grafico temporale
        axes[0, colonna].plot(
            tempo, modulo,
            color="tab:blue",
            linewidth=0.8
        )

        axes[0, colonna].set_ylabel(
            "Modulo accelerazione (m/s²)"
        )

        # Spettro FFT
        axes[1, colonna].plot(
            frequenze, spettro,
            color="tab:orange",
            linewidth=1
        )

        axes[1, colonna].set_xlim(0, FREQ_MAX)

        # Evidenzia il picco dominante
        if freq_dom <= FREQ_MAX:

            indice = np.argmin(
                np.abs(frequenze - freq_dom)
            )

            axes[1, colonna].scatter(
                freq_dom,
                spettro[indice],
                color="red",
                zorder=5
            )

        nome = (
            "Normale" if tipo == "N"
            else "Patologica simulata"
        )

        axes[0, colonna].set_title(
            f"{nome} - f dominante: {freq_dom:.2f} Hz"
        )

        axes[1, colonna].set_xlabel(
            "Frequenza (Hz)"
        )

        for riga in range(2):
            axes[riga, colonna].grid(
                True, alpha=0.3
            )

    fig.suptitle(
        f"FFT - Gruppo {gruppo}, Studente {studente}"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"FFT_G{gruppo}_S{studente}.png",
        dpi=150
    )

    print(f"\nGruppo {gruppo} - Studente {studente}")
    print(f"Frequenza N: {risultati['N']:.3f} Hz")
    print(f"Frequenza P: {risultati['P']:.3f} Hz")

    differenza = risultati["P"] - risultati["N"]
    print(f"Differenza P-N: {differenza:+.3f} Hz")


# ==========================================
# PROGRAMMA PRINCIPALE
# ==========================================

if __name__ == "__main__":

    archivio = trova_cartelle()

    for gruppo, studente in STUDENTI:
        confronta_fft(
            archivio,
            gruppo,
            studente
        )

    print("\nAnalisi FFT completata.")
    plt.show()
