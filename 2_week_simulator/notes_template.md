# Exp N — [Architecture] ([Map]) · Notes

Fill this in after the experiment trains successfully.

---

## 1. Loss Curve

*What did the training and validation loss curves look like?*

- Did loss drop smoothly, oscillate, or flatline?
- Did train and val loss track each other, or did they diverge (sign of overfitting)?
- Roughly which epoch did val loss stop improving?

```
[paste your observations here, or attach curves.png]
```

---

## 2. Accuracy

*Final val accuracy:*  _____ %  (reference: check experiment_progression.md)

Per-action breakdown:

| Action   | Val Accuracy |
|----------|-------------|
| forward  |             |
| backward |             |
| left     |             |
| right    |             |

*Which actions did the model struggle with most? Why might that be?*

---

## 3. Delta from Previous Experiment

*What changed vs the previous experiment — and did the result match your expectation?*

| | Previous Exp | This Exp |
|---|---|---|
| Architecture change | — | [what you added] |
| Val accuracy | ___% | ___% |
| Direction (better/worse/same) | — | |

Did the result match your hypothesis? Why or why not?

```
[write here]
```

---

## 4. Deliberate Break

*Pick one thing to break intentionally, run training, observe what happens.*

What I broke:

```
[describe the change]
```

What happened:

```
[describe the result — loss curve, accuracy, error message]
```

Why it happened:

```
[your explanation]
```

---

## 5. System Trace

*Pick one sample and manually trace it through the model in a notebook.*

Print shape and value after each module. Note anything unexpected.

```
[write here]
```

---

## 6. Architecture Question

*One question this experiment raised that you want to answer in a future experiment.*

```
[write here]
```

---

## 7. Key Takeaway

*One sentence: what is the most important thing you learned from this experiment?*

```
[write here]
```
