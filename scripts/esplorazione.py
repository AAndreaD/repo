
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Percorso del dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

# Individua automaticamente le cartelle delle registrazioni
cartelle = sorted(
    cartella
    for cartella in DATASET_DIR.iterdir()
    if cartella.is_dir() and cartella.name.endswith(("_N", "_P"))
)


def analizza_camminata(cartella):
    """Carica e visualizza i dati dell'accelerometro."""

    csv_path = cartella / "Accelerometer.csv"

    if not csv_path.exists():
        print(f"File non trovato: {csv_path}")
        return

    # Caricamento dei dati
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    seconds = np.atleast_1d(data["seconds_elapsed"])
    x = np.atleast_1d(data["x"])
    y = np.atleast_1d(data["y"])
    z = np.atleast_1d(data["z"])

    # Informazioni sulla registrazione
    print(f"\nRegistrazione: {cartella.name}")
    print(f"Campioni caricati: {len(seconds)}")
    print(f"Durata: {seconds[-1] - seconds[0]:.2f} secondi")

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
        ax.plot(seconds, signal, color=color, linewidth=0.8)
        ax.set_ylabel(f"Acc {label} (m/s²)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Tempo (s)")

    tipo = (
        "Normale"
        if cartella.name.endswith("_N")
        else "Patologica simulata"
    )

    fig.suptitle(f"{cartella.name} - Camminata {tipo}")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":

    if not cartelle:
        print("Nessuna cartella di registrazione trovata.")
    else:
        print(f"Registrazioni trovate: {len(cartelle)}")

        for cartella in cartelle:
            analizza_camminata(cartella)
