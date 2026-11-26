# Week 7 Project: ML-Powered IDS (SIEM Component 2)

A supervised intrusion detection model trained on CICIDS2017 that names the attack type, maps it to MITRE ATT&CK, and writes alerts into the SIEM log collector.

## What it does

- Reads CICIDS2017 flows (15 labels: BENIGN plus 14 attack types)
- Classifies each flow with XGBoost and gives a confidence score
- Maps the predicted type to an ATT&CK tactic and technique, with a first analyst step
- Writes alerts into the shared `events` table with `source='ids'`

## Where the code is

- `week7/day1` to `day3`: rule IDS vs Random Forest vs XGBoost
- `week7/day4`: all 15 classes
- `week7/day5`: pipeline class, time-ordered test, saved model
- `week7/day6`: `mitre_map.py` (label to ATT&CK lookup)
- `week7/day7`: collector integration

The saved model (`data/ids_xgb.joblib`) and the data are not pushed.

## Results

### 1. A hand-written rule IDS vs machine learning (Wednesday, 60,000 flows)

The rule was "over 100 forward packets and no reply." It caught 0 of 21,872 attacks and raised 20 false alarms. A Random Forest on the same data scored attack precision 0.97 and recall 0.99 (same-day split).

### 2. Random Forest vs XGBoost (same-day split, 5 classes)

| Model | Macro F1 | Train time |
|---|---|---|
| Random Forest | 0.9930 | 39.7 s |
| XGBoost | 0.9944 | 45.8 s |

The gap is within run-to-run noise. XGBoost missed fewer Hulk flows (recall 0.9998 vs 0.9970).

### 3. All 15 classes (shuffled split, 21,571 test flows)

Accuracy 0.99, but macro recall 0.90 and macro precision 0.94. Most classes score 0.99 to 1.00. The weak ones:

| Class | Test flows | Precision | Recall |
|---|---|---|---|
| Web Attack XSS | 196 | 0.45 | 0.28 |
| Web Attack Brute Force | 441 | 0.73 | 0.86 |
| Web Attack Sql Injection | 6 | 1.00 | 0.50 |

Heartbleed (3 test flows) and Infiltration (11) are too small to judge.

### 4. Stricter test: time-ordered attacks + unseen-day benign

Each attack was trained on its first 70% of rows and tested on the last 30%. Benign traffic from Friday was held out entirely.

- Accuracy 0.97, macro precision 0.90, macro recall 0.82, macro F1 0.83
- Bot recall fell from 1.00 (shuffled) to **0.16**: about 491 of 585 missed
- slowloris 0.88, Hulk 0.97, GoldenEye 0.97, Slowhttptest 0.96
- False alarms on 24,000 unseen Friday benign flows: **5.4 per 10,000** at confidence 0.5, **1.7 per 10,000** at 0.9

### 5. Comparison with the Week 6 anomaly detector (20,000 Wednesday flows each)

| | Autoencoder | Supervised IDS |
|---|---|---|
| Hulk alerts | 51 | 5,518 |
| GoldenEye alerts | 0 | 324 |
| False alarms per 1,000 alerts | ~300 | ~1.6 |

The IDS wrote 6,192 alerts into the collector, 6,182 real and 10 false (precision 0.998). Those samples overlap with training data, so this figure checks the plumbing, not new accuracy. The 12 wrong-type alerts all stayed inside the DoS family, which maps to one ATT&CK technique.

## Limitations

- Each attack type appears on only one day in CICIDS2017, so a true cross-day test of attack types is not possible with this dataset.
- Class caps make the sample attack-heavy (about 70% attacks), unlike a real network (about 1%).
- Rare classes (Heartbleed, Infiltration, SQL Injection) have too few flows for reliable scores.
- XSS is poorly detected, and Bot recall drops sharply on later traffic.
- CICIDS flows contain no IP addresses, so `src_ip` is empty. Week 8 and 9 need IP-bearing events for correlation.
- The ATT&CK mapping is a judgement call. Some entries (Infiltration, Heartbleed) are approximate.

## Part of

The AI-powered mini SIEM (Phase 2): Week 5 log collector, Week 6 anomaly detector, Week 7 IDS (this project).