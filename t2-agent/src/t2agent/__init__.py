"""T2-Agent unified reproduction package.

This package merges the single-label five-way verdict pipeline and the
multi-label eight-way distortion-taxonomy pipeline into a single agent that
produces both outputs in one MCTS run per sample.
"""

__all__ = [
    "agent",
    "config",
    "data",
    "labels",
    "llm",
    "tools",
    "logging_utils",
    "metrics_plot",
]
