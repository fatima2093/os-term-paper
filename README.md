# CSE-307 Term Paper: Learned Page Replacement (Track 1)

## What this project does
Compares FIFO, LRU and Optimal (Belady's) page replacement with a simple
learned policy (Decision Tree) on a synthetic trace whose pattern shifts
halfway: first half locality-heavy, second half random.

## Setup
```
python -m pip install numpy scikit-learn matplotlib
```

## How to run
```
python main.py
```
Output is printed in the terminal and saved in `results/`
(`results.csv` and `chart.png`).

## Experiment setup
- 5 memory frames, 50 distinct pages
- Trace: 500 locality-heavy accesses (6 pages), then 500 random accesses
- Learned policy: Decision Tree (features: recency, frequency), trained
  to imitate Optimal's eviction choice on a separate locality-only trace

## Results
| Policy | Faults (before) | Faults (after) | Hit % (before) | Hit % (after) |
|---|---|---|---|---|
| FIFO | 92 | 454 | 81.6 | 9.2 |
| LRU | 92 | 452 | 81.6 | 9.6 |
| Optimal | 40 | 348 | 92.0 | 30.4 |
| Learned | 100 | 454 | 80.0 | 9.2 |

## Summary
All policies perform well before the shift and degrade sharply after it,
because random accesses have no locality to exploit. The learned model was
trained only on locality-heavy data, so it also fails to adapt to the new
pattern. Optimal is the upper bound but cannot be used in real systems.

## AI assistance disclosure
I used Claude (Anthropic) for implementation help with the code. The
experiment results and analysis in the report are my own.