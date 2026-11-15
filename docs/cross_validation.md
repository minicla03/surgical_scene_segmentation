# Cross-validation per video e seed sullo split ufficiale

Come si stima la variabilità dei risultati nei tre dataset. Codice: [src/datasets/kfold.py](../src/datasets/kfold.py).
Metriche: [metrics_convention.md](metrics_convention.md).

| Dataset | Split | Variabilità |
|---|---|---|
| CholecSeg8k | K-fold per video (K = 4), nessuno split ufficiale | media ± dev. std. sui 4 fold |
| DSAD | split ufficiale (train / validation / test) | media ± dev. std. su 3 seed |
| SAR-RARP50 | split ufficiale (train / validation) | media ± dev. std. su 3 seed |

DSAD e SAR-RARP50 non usano la cross-validation: si tiene lo split degli autori, così i risultati sono confrontabili con quelli pubblicati, e la varianza si ricava da 3 seed di training (§6).

## 1. CholecSeg8k

I frame di uno stesso video sono quasi identici tra loro. Dividere per frame metterebbe frame quasi uguali sia in
training sia in validation (leakage) e gonfierebbe le metriche. Per questo l'**unità di split è il video**: tutti i frame di un video stanno sempre nello stesso insieme. Con pochi video (17) il risultato di un singolo split dipende molto da quali finiscono in validation; il K-fold dà una stima più robusta.

## 2. Regole dello schema

1. **K = 4**, seed **42**. Il seed dello split (`data.split.seed`) è diverso da quello del training: tutti i fold
   di una stessa cross-validation devono condividere la stessa partizione.
2. **Una classe presente in un solo video** non può essere allenata e validata insieme. Il video che la contiene è
   **bloccato in train** (non va mai in validation) e la classe è **N/A**: viene appresa ma non valutata.
3. **Una classe in più video** compare in validation nel maggior numero possibile di fold, cioè
   `min(K, video con la classe)`, restando sempre anche in train. Con 2 soli video, uno sta in train e l'altro in
   validation.
4. **Valutabile = in train e in validation.** Una classe presente in validation ma non in train in quel fold è N/A:
   il valore 0 non misurerebbe il modello, ma solo il fatto che non l'ha mai vista.
5. **Bilanciamento.** A parità di copertura (punto 3), i fold sono scelti in modo che la **quota di pixel delle classi
   popolose** (almeno l'1% dei pixel) in ogni validation resti vicina a quella del dataset, e che le validation
   abbiano dimensioni simili.
6. **Test (opzionale).** `test_ratio > 0` tiene fuori dei video come test **prima** del K-fold, con priorità
   train > validation > test: non garantisce di contenere tutte le classi. Oggi `test_ratio = 0`.
7. **Riproducibilità.** I video sono ordinati per id prima dell'assegnazione: i fold dipendono solo da id dei video e
   seed, non dall'ordine con cui il dataset li elenca.

Una classe è considerata presente in un video se ha **almeno un pixel** in almeno un frame (nessuna soglia minima).

## 3. Algoritmo

`plan_kfold(presence, group_ids, class_names, n_folds, ..., group_pixels)` lavora su matrici video × classe, quindi
non dipende dal dataset.

1. **Ordinamento** dei video per id.
2. **Blocco in train** dei video che contengono classi presenti in un solo video.
3. **Test** (se `test_ratio > 0`): selezione greedy con vincolo sulla copertura.
4. **Assegnazione ai fold**: stratificazione multilabel (`MultilabelStratifiedKFold`) sulla presenza delle classi.
5. **Scambi greedy** di video tra fold (mantengono il numero di video per fold), accettati solo se migliorano
   strettamente la coppia (criterio 1, criterio 2):
   1. *copertura*: numero di fold in cui ogni classe è valutabile (obiettivo `min(K, video liberi con la classe)`);
   2. *bilanciamento*: costo = scostamento relativo al quadrato tra la quota di pixel di ogni classe popolosa nella
      validation del fold e la quota sui video validabili, pesato sulla quota della classe (le più popolose contano
      di più), più un termine sulla dimensione dei fold. Entrano solo le classi valutabili con almeno
      `balance_min_pixel_pct` % dei pixel.
6. **Tentativi**: i passi 4-5 sono ripetuti `n_restarts` volte (30) con seed `seed`, `seed+1`, ... e si tiene il
   risultato migliore (copertura, poi costo). Tutto è deterministico.
7. **Controllo**: se una classe resta sotto il suo obiettivo di copertura, viene emesso un avviso.

La stratificazione sulla sola presenza delle classi non controlla le quantità: sui dati reali di Cholec la quota di
pixel in validation delle classi popolose si scostava da quella dei video validabili fino al 38% (Abdominal Wall), 57%
(Grasper, Gallbladder) e 101% (Gastrointestinal Tract). Il passo 5.2 porta lo scostamento relativo medio da 27% a 11%
e il massimo da 101% a 32%, senza cambiare la copertura.

## 4. Risultati (CholecSeg8k)

Calcolati dalle tabelle video × classe in `docs/reports/dataset_validation/` (K = 4, seed 42, con bilanciamento).
Il controllo sulle maschere reali non è ancora stato eseguito.

Video bloccati in train: video01 (Blood), video12 (Hepatic Vein), video09 (Liver Ligament).

| Fold | Video in validation | Frame in validation |
|---|---|---|
| 0 | 18, 20, 24, 27 | 1680 |
| 1 | 25, 26, 52 | 1440 |
| 2 | 35, 43, 48, 55 | 1440 |
| 3 | 17, 28, 37 | 1360 |

**9 classi su 13 sono valutabili** (con le regole di §2):

| Classe | Video | Fold validi | Nota |
|---|---|---|---|
| Blood, Hepatic Vein, Liver Ligament | 1 | 0 | video bloccato in train |
| Connective Tissue | 2 | 0 | entrambi i video (01 e 12) sono bloccati: contengono Blood e Hepatic Vein |
| Cystic Duct | 3 | 1 (fold 3) | solo 2 frame in video17: stima molto rumorosa |
| le altre 8 classi | 9-17 | 4 | |

Distribuzione delle classi popolose (quota di pixel in validation rispetto ai video validabili): scostamento
relativo medio 11%, massimo 32% (Grasper); Background ±7%, Abdominal Wall ±9%, Liver ±17%, Fat ±22%, Gallbladder
±22%, Gastrointestinal Tract ±25%. Con 14 video validabili lo scostamento non può azzerarsi.

La distribuzione completa per fold (frame, pixel e video per classe in train e validation) si genera con il notebook
`notebooks/kfold_validation.ipynb`, che produce anche un PDF in `docs/reports/kfold_validation/`.

## 5. Configurazione ed esecuzione

Nel YAML del dataset (`configs/dataset/baseline/<dataset>.yaml`), sezione `data.split`:

```yaml
split:
  strategy: kfold      # CholecSeg8k.  DSAD: official
  group_by: video
  seed: 42
  n_folds: 4
  fold_index: 0        # ogni job riceve il proprio fold
  test_ratio: 0.0
  balance: true
  n_restarts: 30
  balance_min_pixel_pct: 1.0
```

Gli script di lancio leggono la strategia dallo YAML:

```bash
cd /homes/cmininno/tesi/training_script/baseline
./submitt_all.sh                          # kfold -> un job per fold; official -> un job per seed
./submit_mask2former_pretrained.sh dsad   # idem per Mask2Former pretrainato
```

Override opzionali: `N_FOLDS`, `SEEDS`, `SPLIT_STRATEGY`, `EXTRA_ARGS`. Per ogni run si hanno nome W&B,
tag, group e cartella dei checkpoint distinti (§7).

## 6. Seed sullo split ufficiale (DSAD, SAR-RARP50)

- Lo split è quello ufficiale e non cambia. Si lanciano 3 training con `experiment.seed` = 42, 43, 44
  (`experiment.seed_runs=true`, impostato dagli script).
- **Cosa misura la varianza sui seed**: la casualità del training (inizializzazione, ordine dei batch, augmentation).
  **Non** misura la variabilità dovuta alla scelta dei dati, che è ciò che stima la cross-validation. È la
  differenza da dichiarare quando si confrontano le deviazioni standard dei diversi dataset.
- **DSAD**: spleen è presente solo in train (108 frame) e manca sia dalla validation sia dal test: è N/A in tutti i
  run. Oltre alla validation si valuta anche il **test** ufficiale (chiavi `final/test_last/*`, `final/test_best/*`).
- **SAR-RARP50**: ha solo train e validation, tutte le classi sono presenti in validation.

## 7. Nomi, metriche e aggregazione

| Run | W&B run | group | tag | checkpoint |
|---|---|---|---|---|
| fold | `{exp}_lr{lr}_fold{k}` | `{exp}_lr{lr}_cv` | `fold:k` | `.../fold{k}` |
| seed | `{exp}_lr{lr}_seed{s}` | `{exp}_lr{lr}_seeds` | `seed:s` | `.../seed{s}` |

Per ogni run, a fine training si valutano i pesi dell'**ultima epoca** (`last`, con early stopping è l'epoca di
arresto) e quelli del **miglior checkpoint** (`best`), con tutte le metriche, sulla validation e, dove c'è, sul test.
Le classi N/A sono NaN e restano fuori dalle medie. Per ogni classe si riportano anche i **pixel assegnati** dal
modello (`pred_pixels`): la regola per le classi N/A è che il loro Dice resta fuori dalla media anche se il modello le
predice, e si riporta quanti pixel di validation vengono assegnati a quella classe.

Dopo i run:

```bash
/homes/cmininno/tesi/tesi_venv/bin/python cv_summary.py --group dsad_U-Net_lr1e-4_seeds   # o ..._cv per i fold
```

(usare il Python del venv del progetto: con altri interpreti i pacchetti dell'utente possono essere incompatibili
con NumPy 2 e il log delle tabelle su W&B fallisce; i CSV e i valori scalari vengono comunque salvati.)

Per `best`, `last`, `test_best` e `test_last` produce: tabella macro, tabella per classe con valore per run, **media,
deviazione standard (ddof=1) e numero di run validi**, tabella dei pixel assegnati con il numero di run in cui la
classe è valutabile, e le epoche di best e di stop. CSV in `cv_summary/<group>/`, più la run `..._summary` su W&B.

## 8. Limiti e punti aperti

- **Cholec: classi con pochi video.** 4 classi su 13 non sono valutabili e Cystic Duct lo è in un fold con due frame.
  Le medie macro vanno dichiarate come calcolate sulle classi valutabili.
- **Connective Tissue** non è valutabile: è presente solo in due video, che sono gli stessi bloccati in train per
  Blood e Hepatic Vein. Valutarla richiederebbe di rinunciare ad allenare una di quelle classi in un fold.
- **Video non validati.** I video bloccati in train (3 su 17) non sono mai in validation, nemmeno per le altre classi.
- **Soglia di presenza.** Una classe con pochi pixel conta come presente; non c'è una soglia minima.
- **Selezione sulla validation.** La best epoch è scelta sulla validation: il valore è leggermente ottimistico.
  Anche `last` dipende dall'early stopping. Per DSAD il test ufficiale non entra nella scelta.
- **Val loss.** Le classi viste solo in train producono falsi positivi in validation che la cross-entropy penalizza
  sempre di più al crescere della sicurezza del modello: la val loss di Cholec resta alta. Per questo early stopping e
  scelta del checkpoint usano il **Dice accumulato (`dice_acc`) di validazione**, non la loss.

## 9. Verifica

- Test automatici (`test/test_kfold.py`, `test_kfold_integration.py`, `test_kfold_distribution.py`,
  `test_cv_summary.py`, `test_metric_manager.py`): partizione disgiunta e completa, video bloccati mai in validation,
  copertura, riproducibilità con ordine rimescolato, il bilanciamento non peggiora la copertura e non aumenta il
  costo, tabelle di distribuzione, aggregazione per fold e per seed, pixel assegnati alle classi N/A.
- Il notebook `kfold_validation.ipynb` è stato eseguito con un dataset sintetico costruito dai numeri dei report
  (stesso flusso, PDF compreso); non è stato ancora eseguito sulle maschere reali.
- Lo smoke test su cluster (un fold di DSAD con U-Net, 2 epoche) è precedente al bilanciamento e ai seed.
- Non ancora verificati: controllo su CholecSeg8k reale, `cv_summary.py` con W&B online dopo le ultime modifiche,
  valutazione sul test di DSAD su cluster.
