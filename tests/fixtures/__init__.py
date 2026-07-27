"""Deterministic local fixtures for GLiNER2 tests.

These fixtures build tiny tokenizers, encoders, and checkpoints entirely
offline so tests never touch the network. They are shared by the span
regression suite and the boundary architecture suite.
"""
