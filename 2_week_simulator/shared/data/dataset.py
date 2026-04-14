"""
VLADataset: loads training-data JSON files for any experiment.

Responsibilities:
- Parse JSON structure (metadata + samples)
- Decode base64 JPEG images → normalised float tensors
- Extract binary action labels [forward, backward, left, right]
- Perform random 80/20 train/val split
- Optionally return language_id for forked-road experiments
"""
