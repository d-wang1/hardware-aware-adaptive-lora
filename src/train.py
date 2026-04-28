"""Unified training entry point. Dispatches by config 'method' field:
{uniform, adalora, gradient_adaptive, hardware_aware}. Phase 3+4 — to be implemented.

Usage (planned):
    python -m src.train --config configs/uniform_lora.yaml --seed 42
"""
