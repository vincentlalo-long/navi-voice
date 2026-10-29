"""
Backward compatibility layer for modules.tagging.
Directs imports to modules.tagging.classifier.
"""

from modules.tagging.classifier import CategoryClassifier

__all__ = ["CategoryClassifier"]
