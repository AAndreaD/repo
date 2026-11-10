
from pathlib import Path
import csv
import math
from statistics import median

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "risultati" / "features_esplorazione.csv"
OUTPUT_FILE = BASE_DIR / "risultati" / "ranking_feature_modulo.csv"

# ==============================================
# CARICAMENTO DELLE FEATURE
# ==============================================

with INPUT_FILE.open("r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)
    righe = list(reader)
    colonne = reader.fieldnames

# Selezioniamo le feature del modulo
feature_modulo = [
    nome for nome in colonne
    if "_magnitude_" in nome
]

# Raggruppa N e P dello stesso studente
studenti = {}

for riga in righe:
    chiave = (int(riga["gruppo"]), int(riga["studente"]))
    tipo = riga["tipo"]

    if tipo in studenti.setdefault(chiave, {}):
        raise ValueError(f"Registrazione duplicata: {chiave}, {tipo}")

    studenti[chiave][tipo] = riga

coppie = {
    chiave: dati
    for chiave, dati in studenti.items()
    if "N" in dati and "P" in dati
}

print(f"Studenti analizzati: {len(coppie)}")
print(f"Feature del modulo: {len(feature_modulo)}")

# ==============================================
# CONFRONTO DELLE FEATURE
# ==============================================

risultati = []

for feature in feature_modulo:

    differenze = []
    variazioni = []

    for chiave, dati in sorted(coppie.items()):

        normale = float(dati["N"][feature])
        patologica = float(dati["P"][feature])

        if not (math.isfinite(normale) and math.isfinite(patologica)):
            raise ValueError(f"Valore non valido: {chiave}, {feature}")

        differenza = patologica - normale

        # Variazione relativa simmetrica, compresa tra -200 e 200
        denominatore = abs(patologica) + abs(normale) + 1e-12
        variazione = 200 * differenza / denominatore

        differenze.append(differenza)
        variazioni.append(variazione)

    aumenti = sum(d > 0 for d in differenze)
    diminuzioni = sum(d < 0 for d in differenze)

    consistenza = max(aumenti, diminuzioni)

    risultati.append({
        "feature": feature,
        "aumenti": aumenti,
        "diminuzioni": diminuzioni,
        "consistenza": consistenza,
        "variazione_mediana": median(variazioni),
        "intensita_mediana": median(
            abs(v) for v in variazioni
        )
    })

# Ordina prima per consistenza e poi per differenza relativa
risultati.sort(
    key=lambda r: (
        r["consistenza"],
        r["intensita_mediana"]
    ),
    reverse=True
)

# ==============================================
# RISULTATI
# ==============================================

coerenti = [
    r for r in risultati
    if r["consistenza"] == len(coppie)
]

print(f"\nFeature con andamento coerente: {len(coerenti)}")

print("\nPRIME 15 FEATURE:")
print("-" * 85)

for r in risultati[:15]:

    direzione = (
        "AUMENTA" if r["aumenti"] > r["diminuzioni"]
        else "DIMINUISCE"
        if r["diminuzioni"] > r["aumenti"]
        else "MISTO"
    )

    print(
        f"{r['feature']}\n"
        f"  Coerenza: {r['consistenza']}/{len(coppie)} | "
        f"{direzione} | "
        f"Variazione simmetrica mediana: "
        f"{r['variazione_mediana']:.1f}%"
    )

# Salva il ranking completo
OUTPUT_FILE.parent.mkdir(exist_ok=True)

with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(risultati[0].keys())
    )
    writer.writeheader()
    writer.writerows(risultati)

print(f"\nRanking salvato in: {OUTPUT_FILE}")
