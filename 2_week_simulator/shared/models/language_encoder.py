"""
Language encoder used in Exp 5, 8, 9.

Converts a language instruction ID (e.g. 0="turn left", 1="turn right")
into a fixed-size embedding vector that conditions the action prediction.

Input:  (B,) integer language IDs
Output: (B, lang_dim) language embedding vectors
"""
