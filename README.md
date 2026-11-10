# repo
esercizio lab FIA

# Gait Analysis Lab

Laboratorio di Fondamenti di Intelligenza Artificiale.

## Obiettivo

Sviluppare una pipeline di machine learning per distinguere
camminate normali (N) e camminate patologiche simulate (P),
utilizzando dati registrati tramite i sensori degli smartphone.

## Dataset

Il dataset analizzato comprende:

- 22 soggetti
- 44 registrazioni (22 N e 22 P)
- Accelerometro e giroscopio
- Segnali tridimensionali lungo gli assi x, y, z

Le registrazioni sono elaborate attraverso una finestra temporale
comune e suddivise in segmenti per l'estrazione delle feature.

I dataset e i risultati generati sono esclusi dal repository Git.

## Metodologia

1. Analisi esplorativa dei segnali.
2. Calcolo del modulo tridimensionale dei sensori.
3. Estrazione di caratteristiche statistiche e spettrali.
4. Costruzione della matrice delle feature X e delle etichette y.
5. Normalizzazione dei dati.
6. Classificazione tramite Gaussian Naive Bayes.
7. Valutazione con separazione dei soggetti tra training e test.
8. Selezione delle feature mediante SelectKBest.
9. Analisi delle correlazioni e permutation importance.
10. Confronto tra il modello iniziale e quello raffinato.

### Feature extraction

Per ogni registrazione vengono inizialmente calcolate 224 feature:

- Media
- Varianza
- Deviazione standard
- Range
- Intervallo interquartile
- Energia
- Frequenza dominante tramite FFT

Le feature sono calcolate sui diversi segmenti temporali e sui
segnali dell'accelerometro e del giroscopio.

### Feature finali

Il confronto finale utilizza tre caratteristiche spettrali:

- `acc_magnitude_intero_freq_dom`
- `acc_magnitude_centro_freq_dom`
- `acc_magnitude_fine_freq_dom`

Queste rappresentano la frequenza dominante del modulo
dell'accelerazione in differenti segmenti temporali.

## Risultati

Confronto su cinque suddivisioni casuali del dataset,
mantenendo separati i soggetti di training e test:

| Modello | Accuracy media | Recall P medio |
|---|---:|---:|
| GaussianNB - 224 feature | 61.25% | 62.50% |
| GaussianNB - 3 feature | 82.50% | 82.50% |

Una precedente nested cross-validation a 5 fold, con selezione
automatica di 2-4 feature, ha ottenuto un'accuracy dell'86.36%.

I risultati sono esplorativi: il numero di soggetti è limitato
e le feature finali sono state individuate analizzando lo stesso
dataset. Le camminate patologiche sono simulate, non cliniche.

## Installazione

Requisiti: Python e pip.

Creare e attivare un ambiente virtuale, se non già disponibile:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Installare le dipendenze:

```bash
pip install -r requirements.txt
```

Il dataset completo utilizzato nel laboratorio è disponibile
separatamente e può essere clonato nella cartella `dataset_completo`:

```bash
git clone https://github.com/alecang/ml_lab_dataset26-27.git dataset_completo
```

## Esecuzione

Dalla directory principale del repository:

Estrarre le feature dal dataset completo:

```bash
python3 scripts/estrazione_dataset_completo.py
```

Eseguire la classificazione iniziale:

```bash
python3 scripts/classificazione.py
```

Eseguire il raffinamento con nested cross-validation:

```bash
python3 scripts/raffinamento_feature.py
```

Confrontare il modello a 224 feature con quello a 3 feature:

```bash
python3 scripts/confronto_finale.py
```

Gli ulteriori script nella cartella `scripts/` permettono
di effettuare analisi esplorative, confronti tra sensori,
visualizzazioni FFT, analisi delle correlazioni e
permutation importance.

## Tecnologie

- Python
- NumPy
- Matplotlib
- SciPy
- Scikit-learn
- Git e GitHub

## Conclusioni

Il laboratorio evidenzia come la selezione di caratteristiche
informative possa migliorare le prestazioni di un classificatore
rispetto all'impiego indiscriminato di numerose feature.

In particolare, le caratteristiche spettrali ottenute mediante
FFT si sono dimostrate utili per distinguere le due condizioni
di camminata nel dataset analizzato.
