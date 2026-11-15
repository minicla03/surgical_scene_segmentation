# Convenzione delle metriche di segmentazione

Vale per **tutti i modelli** (U-Net, Mask2Former, U-Mamba) e tutti i dataset: la valutazione passa sempre da
`MetricManager` ([src/metrics/metric_manager.py](../src/metrics/metric_manager.py)), con gli stessi parametri
nei config di training (`evaluation.metrics`). Nessun modello calcola metriche proprie.

## 1. Pipeline di valutazione

1. Il modello produce punteggi per classe alla **risoluzione dell'input** (Mask2Former li riporta all'input con
   interpolazione bilineare; U-Net e U-Mamba producono già quella risoluzione).
2. `argmax` sulle classi, poi one-hot. Il target è la maschera di classe alla stessa risoluzione.
3. Tutte le metriche sono calcolate **alla risoluzione di preprocessing** (es. 240×427 per CholecSeg8k, con padding a 240×432; 384×480
   per DSAD; 384×672 per SAR-RARP50, 16:9 come i frame originali), non a quella originale dei frame. Le distanze (HD95) sono in **pixel**.

## 2. Metriche

| Chiave W&B | Cosa misura | Implementazione |
|---|---|---|
| `dice`, `iou` | Dice / IoU **per immagine**, poi media sulle immagini in cui la classe è nel ground truth | `monai.metrics.DiceMetric`, `monai.metrics.MeanIoU` |
| `dice_acc`, `iou_acc` | Dice / IoU **accumulati** su tutto l'insieme valutato (TP/FP/FN sommati su tutti i pixel di tutte le immagini) | matrice di confusione C×C |
| `hd95` | Hausdorff distance al 95° percentile, per immagine | `monai.metrics.HausdorffDistanceMetric` |

Per le chiavi per classe: `val_per_class/<metrica>/<classe>`, ad es. `val_per_class/dice_acc/liver`.

### 2.1 Dice e IoU per immagine (MONAI 1.3.0)

Parametri: `include_background=True`, `reduction="mean"`, `ignore_empty=True` (default), `get_not_nans=False`.

Per immagine e per classe: `Dice = 2·TP / (|GT| + |Pred|)`, `IoU = TP / (|GT| + |Pred| − TP)`.
Se `|GT| = 0` il valore è **NaN**, non 0 né 1. Se `|GT| > 0` e la classe non è mai predetta, il valore è 0.

Valore per classe = media (ignorando i NaN) sulle immagini. Macro = media sulle classi di ogni immagine, poi sulle immagini.

**Conseguenza:** i falsi positivi su immagini senza la classe nel GT **non vengono contati**.

### 2.2 Dice e IoU accumulati

Con `TP_c`, `FP_c`, `FN_c` sommati sull'intero insieme valutato:

- `Dice_acc_c = 2·TP_c / (2·TP_c + FP_c + FN_c)`
- `IoU_acc_c = TP_c / (TP_c + FP_c + FN_c)`

La classe è N/A se non ha pixel di GT in tutto l'insieme valutato (`TP_c + FN_c = 0`). I falsi positivi su
qualunque immagine sono contati. Il macro è la media sulle classi valutabili del valore accumulato per classe.

I due modi non sono alternativi: si riportano entrambi (vedi §5).

### 2.3 HD95

Parametri: `include_background=True`, `percentile=95`, `reduction="mean"`, distanza euclidea, simmetrica
(`directed=False`: massimo tra le due direzioni), in pixel.

Casi particolari (applicati in `MetricManager._distance_buffer`):

- classe assente dal GT dell'immagine → **NaN** (N/A, esclusa dalle medie);
- classe presente nel GT ma mai predetta (MONAI restituisce `inf`) → **diagonale dell'immagine** (errore massimo
  possibile), così non predire nulla non è mai premiato e la media resta finita.

HD95 è costosa: in training non viene calcolata; in validazione ogni `evaluation.hd95_freq` epoche (default 10);
nella valutazione finale sempre.

## 3. Regola N/A (identica per ogni metrica)

- Una classe **senza pixel di GT** nell'insieme valutato è **N/A (NaN)**, mai 0, ed è esclusa da ogni media.
- Nel K-fold una classe è valutabile in un fold solo se compare **sia in train sia in validation**. Altrimenti è N/A
  in quel fold (anche se è nel GT di validation: senza averla vista in training il valore 0 non misurerebbe il modello).
  Vedi [src/datasets/kfold.py](../src/datasets/kfold.py).
- Le classi presenti in un solo video sono bloccate in train e non sono mai valutate.
- Classe assente dalla validation per come è costruito lo split (spleen in DSAD, che manca anche dal test; le classi
  N/A di CholecSeg8k): è N/A **anche se il modello la predice**, ed è esclusa dalla media. Al suo posto si riporta
  quanti pixel di validation il modello assegna a quella classe (§3.1).
- Lo **0 vale solo per le classi che potrebbero comparire nella validation**: presenti nel GT e non predette.

### 3.1 Pixel assegnati a ogni classe

Per **ogni** classe (non mascherata) si registrano `pred_pixels` (pixel assegnati dal modello nell'insieme valutato) e
`pred_pixel_percentage` (% sui pixel valutati), dalle somme per colonna della matrice di confusione. Chiavi per
classe: `<split>_per_class/pred_pixels/<classe>` e `.../pred_pixel_percentage/<classe>`; nei riepiloghi finali
`final/<tag>_per_class/pred_pixels/<classe>`. `cv_summary.py` li raccoglie in una tabella dedicata
(`*_pred_pixels.csv`) con il numero di run in cui la classe è valutabile: per le classi N/A (0 run) è il numero da
riportare.

## 4. Valori finali per run

Al termine del training di ogni run (un fold, oppure un seed sullo split ufficiale) si valutano il modello con i
**pesi dell'ultima epoca** (`last`; con early stopping è l'epoca di arresto) e quello del **miglior checkpoint**
(`best`, scelto sul `dice_acc`), calcolando **tutte** le metriche (HD95 compresa) sulla validation. Se lo split
ha un **test** (DSAD ufficiale) gli stessi pesi sono valutati anche sul test. Chiavi: `final/last/*`, `final/best/*`,
`final/test_last/*`, `final/test_best/*`, `final/*_per_class/*`, `final/best_epoch`, `final/last_epoch`.

Aggregazione (`cv_summary.py`): media e deviazione standard (ddof=1) dei valori dei soli run in cui la classe è
valutabile, con il numero di run validi. Una classe mai valutabile ha 0 run validi e media N/A. Sui fold la deviazione
standard include la variabilità dei dati; sui seed misura solo la casualità del training (vedi
[cross_validation.md](cross_validation.md)).

## 5. Cosa riportare nella tesi

- Metrica principale per i confronti tra modelli: da concordare con il relatore (proposta: `dice_acc` / `iou_acc`,
  stabili per le classi rare). `dice` / `iou` per immagine come controllo.
- HD95 in pixel, con la convenzione di §2.3.
- Il numero di classi valutate e, per ognuna, il numero di fold validi.

## 6. Limiti noti

- Le metriche per immagine hanno alta varianza per le classi rare presenti in pochi frame.
- Una classe presente in pochi frame (es. Cystic Duct in CholecSeg8k) dà valori poco affidabili in ogni convenzione.
- L'early stopping monitora il `dice_acc` (Dice accumulato, che conta i falsi positivi) con `min_delta = 1e-3`: un
  miglioramento minore non azzera la patience né sostituisce il best checkpoint. La scelta della best epoch dipende
  quindi da questa convenzione.
- `last` è scelto dall'early stopping (selezione indiretta sulla validation).
- **Val loss.** Le classi viste solo in train producono falsi positivi in validation che la cross-entropy penalizza
  sempre di più al crescere della sicurezza del modello, quindi la val loss resta alta (CholecSeg8k, con le classi
  sempre in train). Per questo early stopping e scelta del checkpoint usano il Dice accumulato di validazione, non la loss.
