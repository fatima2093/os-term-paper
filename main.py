# CSE-307 Term Paper - Track 1: Learned Page Replacement
import random
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

FRAMES = 5
HALF = 500
TOTAL_PAGES = 50

def make_trace(seed, shift=True):
    random.seed(seed)
    part1 = [random.choice([0, 1, 2, 3, 4, 5]) for _ in range(HALF)]
    part2 = [random.randrange(TOTAL_PAGES) for _ in range(HALF)]
    return part1 + part2 if shift else part1 + part1

def fifo(trace, frames):
    memory, faults = [], []
    for page in trace:
        if page in memory:
            faults.append(0)
        else:
            faults.append(1)
            if len(memory) == frames:
                memory.pop(0)
            memory.append(page)
    return faults

def lru(trace, frames):
    memory, last_used, faults = [], {}, []
    for t, page in enumerate(trace):
        if page in memory:
            faults.append(0)
        else:
            faults.append(1)
            if len(memory) == frames:
                victim = min(memory, key=lambda p: last_used[p])
                memory.remove(victim)
            memory.append(page)
        last_used[page] = t
    return faults

def next_use(trace, page, t):
    for i in range(t + 1, len(trace)):
        if trace[i] == page:
            return i
    return 10**9

def optimal(trace, frames):
    memory, faults = [], []
    for t, page in enumerate(trace):
        if page in memory:
            faults.append(0)
        else:
            faults.append(1)
            if len(memory) == frames:
                victim = max(memory, key=lambda p: next_use(trace, p, t))
                memory.remove(victim)
            memory.append(page)
    return faults

def features(page, t, last_used, count):
    return [t - last_used[page], count[page]]

def train_model(train_trace, frames):
    X, y = [], []
    memory, last_used, count = [], {}, {}
    for t, page in enumerate(train_trace):
        if page not in memory:
            if len(memory) == frames:
                victim = max(memory, key=lambda p: next_use(train_trace, p, t))
                for p in memory:
                    X.append(features(p, t, last_used, count))
                    y.append(1 if p == victim else 0)
                memory.remove(victim)
            memory.append(page)
        last_used[page] = t
        count[page] = count.get(page, 0) + 1
    model = DecisionTreeClassifier(max_depth=4, random_state=0)
    model.fit(X, y)
    return model

def learned(trace, frames, model):
    memory, last_used, count, faults = [], {}, {}, []
    for t, page in enumerate(trace):
        if page in memory:
            faults.append(0)
        else:
            faults.append(1)
            if len(memory) == frames:
                scores = [model.predict_proba([features(p, t, last_used, count)])[0][-1]
                          for p in memory]
                victim = memory[int(np.argmax(scores))]
                memory.remove(victim)
            memory.append(page)
        last_used[page] = t
        count[page] = count.get(page, 0) + 1
    return faults

test_trace = make_trace(seed=1)
train_trace = make_trace(seed=2, shift=False)[:HALF]
model = train_model(train_trace, FRAMES)

results = {
    "FIFO": fifo(test_trace, FRAMES),
    "LRU": lru(test_trace, FRAMES),
    "Optimal": optimal(test_trace, FRAMES),
    "Learned": learned(test_trace, FRAMES, model),
}

print(f"{'Policy':<9}{'Faults(before)':>15}{'Faults(after)':>15}{'Hit%(before)':>14}{'Hit%(after)':>13}")
rows = []
for name, f in results.items():
    fb, fa = sum(f[:HALF]), sum(f[HALF:])
    hb, ha = 100 * (1 - fb / HALF), 100 * (1 - fa / HALF)
    rows.append((name, fb, fa, hb, ha))
    print(f"{name:<9}{fb:>15}{fa:>15}{hb:>14.1f}{ha:>13.1f}")

os.makedirs("results", exist_ok=True)
with open("results/results.csv", "w") as fh:
    fh.write("policy,faults_before,faults_after,hit_before,hit_after\n")
    for r in rows:
        fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.1f},{r[4]:.1f}\n")

names = [r[0] for r in rows]
x = np.arange(len(names))
plt.figure(figsize=(7, 4))
plt.bar(x - 0.2, [r[3] for r in rows], 0.4, label="Before shift (locality)")
plt.bar(x + 0.2, [r[4] for r in rows], 0.4, label="After shift (random)")
plt.xticks(x, names)
plt.ylabel("Hit ratio (%)")
plt.title(f"Page replacement hit ratio ({FRAMES} frames)")
plt.legend()
plt.tight_layout()
plt.savefig("results/chart.png", dpi=150)
print("Saved results/results.csv and results/chart.png")