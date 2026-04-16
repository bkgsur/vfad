# Wiki Schema — Rules for LLM Maintenance

This document defines the structure and maintenance rules for the VFAD wiki.
Read this before making any changes to the wiki.

---

## Purpose

This wiki is a **persistent, compounding knowledge base** for the VFAD project.
Every concept, experiment result, and architectural insight is written down once,
cross-linked, and updated as new experiments run. It grows alongside the project.

---

## Directory Structure

```
wiki/
├── SCHEMA.md           ← this file — rules for maintaining the wiki
├── index.md            ← catalog of every page, one-line summary per entry
├── log.md              ← append-only record of every wiki operation
├── glossary.md         ← one-line definitions for all technical terms
├── concepts/           ← one page per ML concept
├── experiments/        ← one page per experiment
├── architecture/       ← how modules compose into full models
└── sources/            ← summaries of papers and external references
```

---

## Page Formats

### concepts/ page
```
# [Concept Name]

## What it is
One paragraph, plain language. No assumed ML knowledge.

## Why it matters in this project
Concrete connection to the VFAD experiments. Which experiments use it. What it contributes.

## How it works
The mechanism. Include a shape walkthrough for any tensor operation.
Use concrete numbers (B=2, T=5, D=8) not abstract letters alone.

## Key intuition
One or two sentences. The thing you'd say if explaining to a friend.

## Where in the code
File paths and line references to the implementation.

## Experiments that use this
Links to experiment pages.

## See also
Links to related concept pages.
```

### experiments/ page
```
# Exp N — [Architecture] ([Map])

## What changed from previous experiment
One sentence. What single new component was added.

## Architecture
Bullet list: input → module → module → output, with shapes.

## Results
| Metric | Value |
Val accuracy, per-action breakdown, comparison to previous experiment.

## What the results tell us
Interpretation. Did it improve? Why or why not? What does this reveal about the architecture?

## Key takeaway
One sentence.

## Concepts used
Links to concept pages.

## Code
Links to model.py and __main__.py.
```

### architecture/ page
```
# [Architecture Name]

## Purpose
What problem this architecture solves.

## Components
How shared modules are assembled. Data flow with shapes.

## Used in experiments
Which experiments use this architecture.

## Tradeoffs
What it gains vs what it costs (parameters, complexity, training time).
```

### sources/ page
```
# [Paper/Resource Title]

## One-line summary
## Key claims relevant to this project
## Which experiments it informed
## Link / citation
```

---

## Operations

### INGEST (after each experiment trains)
1. Write or update the `experiments/expN.md` page with results
2. Write any new concept pages introduced by this experiment
3. Update existing concept pages with new cross-references if relevant
4. Update `glossary.md` with any new terms
5. Update `index.md` with new/changed pages
6. Append to `log.md`

### QUERY
When the user asks a question about the project:
1. Read `index.md` to locate relevant pages
2. Read those pages
3. Synthesise an answer with citations to wiki pages
4. If the answer is non-trivial and reusable, file it back as a new page or update

### LINT (periodically)
Check for:
- Orphan pages (not linked from index.md)
- Stale claims (experiment result referenced before it was run)
- Missing cross-references between related concept pages
- Concepts used in experiments but not yet documented

---

## Rules

1. **Never delete pages** — mark stale content with `> ⚠️ Outdated:` blockquotes instead
2. **Always update index.md** when adding or renaming a page
3. **Always append to log.md** after any operation
4. **Concepts belong in concepts/** — do not embed concept explanations inside experiment pages; link instead
5. **Use concrete numbers** for any tensor shape explanation
6. **One concept per page** — if a page covers two concepts, split it
