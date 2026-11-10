
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from confronto_sensori import (
    BASE_DIR,
    STUDENTI,
    trova_cartelle,
    carica_sensore
)

# ==========================================
# CONFIGURAZIONE
# ==========================================

FREQ_MIN = 0.5
FREQ_MAX = 5.0

OUTPUT_DIR = BASE_DIR / "risultati" / "fft"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# ANALISI SPETTRALE
# ==========================================

def analizza_frequenze(tempo, segnali):

    modulo = np.sqrt(
        segnali["x"]**2 +
        segnali["y"]**2 +
        segnali["z"]**2
    )

    dt = np.median(np.diff(tempo))
    fs = 1.0 / dt

    nperseg = min(512, len(modulo))

    frequenze, potenza = welch(
        modulo,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="linear",
        scaling="density"
    )

    maschera = (
        (frequenze >= FREQ_MIN) &
        (frequenze <= FREQ_MAX)
    )

    f = frequenze[maschera]
    p = potenza[maschera]

    if len(f) == 0:
        return None

    indice = np.argmax(p)
    dominante = float(f[indice])

    return f, p, dominante


# ==========================================
# CONFRONTO PER STUDENTE
# ==========================================

if __name__ == "__main__":

    archivio = trova_cartelle()

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9)
    )

    for ax, (gruppo, studente) in zip(
        axes.flat, STUDENTI
    ):

        print(f"\nGruppo {gruppo} - Studente {studente}")

        for tipo, colore in (
            ("N", "tab:blue"),
            ("P", "tab:orange")
        ):

            dati = carica_sensore(
                archivio,
                gruppo,
                studente,
                tipo,
                "Accelerometer"
            )

            if dati is None:
                print(f"Registrazione {tipo} mancante")
                continue

            risultato = analizza_frequenze(*dati)

            if risultato is None:
                continue

            f, p, dominante = risultato

            ax.plot(
                f, p,
                label=f"{tipo}: {dominante:.2f} Hz",
                color=colore
            )

            ax.axvline(
                dominante,
                color=colore,
                linestyle="--",
                alpha=0.5
            )

            print(
                f"{tipo}: frequenza dominante "
                f"{dominante:.3f} Hz"
            )

        ax.set_title(
            f"Gruppo {gruppo} - Studente {studente}"
        )

        ax.set_xlabel("Frequenza (Hz)")
        ax.set_ylabel("Densità spettrale di potenza")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Confronto N/P - Metodo di Welch",
        fontsize=14
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "confronto_welch.png",
        dpi=150
    )

    plt.show()
