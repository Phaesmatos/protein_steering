# Protein Activation Steering Lab — ESM-2 (Windows / RTX 4050 6 GB)

Questo progetto è una **replica didattica indipendente e ridotta**, NON il codice degli autori né una replica numerica degli esperimenti pubblicati. Nessun risultato sperimentale è pre-calcolato. Tutti i numeri vanno generati localmente.

## 1. Ambiente

- Windows 10/11, Python **3.11** (raccomandato), GPU NVIDIA 6 GB, driver NVIDIA aggiornati, ~16 GB RAM.
- Apri PowerShell nella cartella estratta.
- Esegui `powershell -ExecutionPolicy Bypass -File .\run_windows.ps1`. Questo crea la venv, installa PyTorch CUDA, le dipendenze, esegue i test e l'esperimento da 35M. **Richiede Internet per pacchetti e pesi**. La build `cu128` richiede driver compatibili; se non lo sono, scegli l'indice CUDA indicato dal selettore ufficiale [PyTorch](https://pytorch.org/get-started/locally/), oppure usa PyTorch CPU.
- Alternativa manuale: `py -3.11 -m venv .venv`, `\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128`, `\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- Verifica GPU: `\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"`.

## 2. Cosa succede

ESM-2 è un **masked protein language model**, non un modello autoregressivo. Si parte da sequenze `GXYGXY...` (motivo triplette del collagene; X,Y simulati) e da negativi ottenuti ruotando la stessa sequenza di un residuo. Gli esempi sono sintetici, **non sequenze sperimentalmente validate**. La composizione aminoacidica di ciascuna coppia è identica, ma le fasi differiscono.

Vettore `v = normalize(mean_i(h_l(x_i^+) - h_l(x_i^-)))` estratto al residuo di fase 0. Per ogni test indipendente, la posizione 6 (fase G) o 7 (fase non-G) è mascherata, si applica `h_l[:,mask,:] += alpha*v` tramite forward hook, poi si misura `P(G | sequenza mascherata)`. L'intervento modifica **solo il token mascherato nel layer scelto**. Pesi congelati, nessun fine-tuning.

Controlli: alpha=0 (baseline), segno invertito, direzione casuale di pari norma, posizione non-G, split train/test distinto. Questi controlli aiutano a distinguere effetti specifici da perturbazioni generiche. **Importante:** le probabilità in presenza della maschera possono essere già molto alte; nessun guadagno è garantito.

Dati: `results/collagen_35m/metrics.csv`, `direction.pt`, `metadata.json`, `metrics.png`.

## 3. Prova il modello da 150M sulla RTX 4050

```powershell
.\.venv\Scripts\python.exe steer.py --experiment collagen --model facebook/esm2_t30_150M_UR50D --layer 15 --n-train 30 --n-test 15 --alphas 0 1 3 6 12 --output results/collagen_150m
.\.venv\Scripts\python.exe plot.py results/collagen_150m/metrics.csv
```

150M è una scelta ragionevole per 6 GB con sequenze corte e batch 1, ma velocità e memoria dipendono dalle versioni e dai processi concorrenti. Se CUDA va in OOM, chiudi altri carichi GPU o usa 35M.

## 4. Esperimento su una proprietà reale (termostabilità o solubilità)

**Separato dal demo collagene.** Prepara un CSV con colonne `sequence,label`: sequenze con **etichette misurate e fonte tracciabile**, ad es. solubilità 0–1 oppure temperatura di melting in °C. Non mischiare proprietà differenti. Se puoi usa i dataset disponibili nel repository ufficiale [Steering-PLMs](https://github.com/Long-Kai/Steering-PLMs), verificandone le colonne, unità, licenze e provenienza. La CLI richiede *sequenze non ambigue* di 12–350 residui; quelle escluse vengono ignorate.

Esempio per solubilità con valori tra 0 e 1:

```powershell
.\.venv\Scripts\python.exe steer.py --experiment csv --csv data/solubility.csv --seq-col sequence --label-col label --high 0.8 --low 0.2 --n-train 24 --n-test 12 --model facebook/esm2_t30_150M_UR50D --layer 15 --output results/solubility
.\.venv\Scripts\python.exe plot.py results/solubility/metrics.csv
```

Esempio per Tm con valori in Celsius: usa `--high 70 --low 50`.

Per CSV, il vettore usa **mean pooling su tutti i residui** (non il solo sito del motivo collagene). Il test in CSV misura la probabilità di ricostruire **un amminoacido originario mascherato**, non la proprietà biologica. È un test di intervento, **non dimostra un aumento di Tm o solubilità**. Servono ulteriori verifiche con un modello esterno indipendente, un test set fuori distribuzione e infine misure sperimentali. Le righe non sono deduplicate per omologia: sequenze simili tra train e test causano leakage. Per un lavoro pubblicabile, creare prima cluster di omologia e split per cluster, con dataset di dimensione adeguata.

## 5. Come interpretare i risultati

- Un miglioramento nel demo è `P(G)` più alta alla posizione G, **rispetto alla baseline** e con un effetto più specifico dei controlli casuale/inverso. Un aumento indiscriminato anche alla fase non-G può indicare solo bias per glicina.
- Per scegliere alpha, usare un set di validazione separato: **non selezionare alpha sui test riportati**. La CLI per semplicità usa un singolo held-out diagnostico e non effettua tuning statistico.
- Per risultati affidabili: almeno 3–5 seed, intervalli bootstrap per campione, screening del layer e alpha solo sul validation set, controllo di perdita di plausibilità sequenziale e validazione biologica separata.
- È un proof-of-concept dell'**intervento nelle attivazioni**, non una riproduzione di proprietà di collagene funzionale, stabilità o strutture 3D.

## 6. Articoli e riproducibilità

1. Huang et al., **Steering Protein Language Models** (ICML 2025): https://proceedings.mlr.press/v267/huang25ba.html — repo ufficiale https://github.com/Long-Kai/Steering-PLMs (codice ESM2-650M e dataset termostabilità/solubilità). L'esperimento qui è una riduzione con ESM2 35M/150M.
2. **InterPLM** (preprint 2024): https://www.biorxiv.org/content/10.1101/2024.11.14.623630v1 — interventi su feature SAE per periodicità glicina nel collagene. Il nostro vettore differenza-media **non è** un intervento SAE; riproduce solo un concetto verificabile simile.
3. Qayyum et al., **PROSOUNDS** (Bioinformatics 2026): https://pmc.ncbi.nlm.nih.gov/articles/PMC13501300/ — steering pesato per incertezza. **Non implementato** in questa versione.
4. Modello: https://huggingface.co/facebook/esm2_t30_150M_UR50D — controllare sempre terms/licenza del checkpoint.

## 7. Diagnostica

Se il modello mostra `ImportError` o un problema con Transformers 5, usa l'intervallo di `requirements.txt` (4.45–4.x). Se viene installato `torch` CPU, reinstalla con il comando CUDA della sezione 1. Se `torch.cuda.is_available()` è False, il programma usa CPU (molto più lento). Se il modulo `esm` fallisce, controlla la versione Transformers e ambiente. Il modello non è compreso nello ZIP: viene scaricato automaticamente alla prima esecuzione.
