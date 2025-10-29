# mcp_pipeline/__init__.py
"""
MCP Pipeline package initialization.

This package provides tools for:
- Molecular analysis (MolecularAnalyzer)
- Sequence analysis (SequenceAnalyzer)
- DeepDTA encoding and dataset utilities (DeepDTAProcessor)
"""

from .mcp_tools import MolecularAnalyzer, SequenceAnalyzer, DeepDTAProcessor

__all__ = [
    "MolecularAnalyzer",
    "SequenceAnalyzer",
    "DeepDTAProcessor",
]

__version__ = "0.2.0"
