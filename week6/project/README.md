# Week 6: Anomaly Detector (SIEM Component 1)

An unsupervised autoencoder that learns normal network traffic from CICIDS2017 Monday (benign only), then flags flows it cannot rebuild well. Alarms are written into the SIEM log collector (`events` table, `source='anomaly'`).

## How it works
1. Six flow features, `log1p` + StandardScaler fitted on Monday only.
2. Autoencoder 6 -> 12 -> 3 -> 12 -> 6 (Keras), trained on 80k benign flows.
3. Alarm when reconstruction error is above the 99.5th percentile of normal error (0.2009).
4. `week6/day6/day6.py` trains and saves the model. `week6/day7/day7.py` scores a batch and writes alarms to the collector.

## Results (Wednesday DoS file, 150k-flow sample)

| Attack | Statistical z-score | Isolation Forest | Autoencoder (99th pct) |
|---|---|---|---|
| Slowhttptest | n/a | 4.1% | 64.0% |
| slowloris | n/a | 2.9% | 31.4% |
| Hulk | n/a | 0.2% | 2.6% |
| GoldenEye | n/a | 3.9% | 0.0% |
| PortScan (Friday file) | 0.0% | 0.0% | n/a |
| Benign false alarms | 1.8% | 1.9% | 0.9% |

## Alert fatigue (realistic world, about 1% attacks)

| Threshold | Alarms per 10k flows | Real | False | Precision |
|---|---|---|---|---|
| 99th pct | 96 | 41 | 880 | 0.04 |
| 99.5th pct (chosen) | 58 | 26 | 531 | 0.05 |
| 99.9th pct | 8 | 0 | 76 | 0.00 |

## SIEM integration (Day 7)
20,000 flows scored, 217 alarms written (96 Slowhttptest, 65 benign, 51 Hulk, 5 slowloris). Existing collector rows (9 pcap, 1000 cicids) were untouched.

## Honest limitations
- A **narrow detector**: good at slow-connection attacks (Slowhttptest), weak on Hulk, blind to GoldenEye and PortScan.
- Precision is about 0.05 when attacks are rare. Moving the threshold trades volume, it does not fix noise.
- CICIDS flows have no IP addresses, so alarms cannot be grouped by source yet. Week 9 correlation needs IP-bearing events.
- Isolation Forest and the autoencoder fail on different attacks, which is the argument for running several detectors and correlating them.