"""
Exports trained model weights to model.json format.

The simulator reads model.json directly — this file produces
the exact format the simulator expects (matching the existing spec).

Output keys: format, num_params, weights (flattened layer arrays)
"""
