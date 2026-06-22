"""Monet latent-token inspection tooling (read-only analysis).

See ``__plans__/latent_inspection_plan.md`` for the full design.

Phase A — ``generate_latents``: greedy HF generation that captures the latent
hidden-state vectors and their positions into a Trace.
Phase B — (to be added) ``inspect``: single teacher-forced replay producing
logit-lens tables and attention maps.
"""
