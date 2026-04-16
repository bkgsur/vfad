# vision_only_vla — CNN + MLP Architecture

## Purpose
The simplest VLA architecture: take an image, extract features with a CNN,
map features to action logits with an MLP. No language, no temporal reasoning.

## Components

```
Camera image (128×128 RGB)
  ↓
CNNBackbone — extracts spatial features, outputs flat vector
  ↓
MLPHead — maps features to 4 action logits
  ↓
[forward, backward, left, right] logits
```

## Data flow with shapes

**Exp 1 (3-conv, curved road):**
```
(B, 3, 128, 128) → CNNBackbone(3conv) → (B, 64) → MLPHead(64,64,4) → (B, 4)
```

**Exp 4 (4-conv, forked road):**
```
(B, 3, 128, 128) → CNNBackbone(4conv) → (B, 128) → MLPHead(128,64,4) → (B, 4)
```

## Used in experiments
- **Exp 1** — curved road baseline, 3-conv backbone
- **Exp 4** — forked road baseline, 4-conv backbone

## Tradeoffs

| | Value |
|--|-------|
| Parameters (Exp 1) | 28,004 |
| Parameters (Exp 4) | ~50,000 (est.) |
| Strengths | Simple, fast, interpretable |
| Weaknesses | No temporal consistency (one action per frame), no language input, limited feature capacity |
| When it works | Single-path tasks with clear visual cues |
| When it fails | Multi-path tasks requiring language disambiguation |

## model.json format field
`"vision_only_vla"`

## See also
- [CNN](../concepts/cnn.md)
- [MLP Head](../concepts/mlp_head.md)
- [act_vla](act_vla.md) — the next architecture, adds Transformer
