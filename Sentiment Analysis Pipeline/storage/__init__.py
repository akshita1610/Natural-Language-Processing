"""
Storage Module for Sentiment Analysis Pipeline

This module handles data storage and retrieval:
- CSV file storage
- SQLite database storage
- PostgreSQL database storage (optional)
"""

from .data_storage import DataStorage

__all__ = ['DataStorage']
