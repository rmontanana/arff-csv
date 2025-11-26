"""
ARFF-CSV Converter.

A Python library for converting between CSV and ARFF (Weka) file formats.

This module provides functionality to:
- Convert CSV files to ARFF format (Weka's Attribute-Relation File Format)
- Convert ARFF files to CSV format
- Automatically detect data types and handle nominal attributes
- Preserve data integrity during conversions
"""

from arff_csv.converter import ArffConverter, csv_to_arff, arff_to_csv
from arff_csv.parser import ArffParser, ArffData
from arff_csv.writer import ArffWriter
from arff_csv.exceptions import (
    ArffCsvError,
    ArffParseError,
    ArffWriteError,
    CsvParseError,
    InvalidAttributeError,
    MissingDataError,
)

__version__ = "1.0.0"
__author__ = "Ricardo Montañana"
__email__ = "ricardo.montanana@example.com"

__all__ = [
    # Main converter class and functions
    "ArffConverter",
    "csv_to_arff",
    "arff_to_csv",
    # Parser
    "ArffParser",
    "ArffData",
    # Writer
    "ArffWriter",
    # Exceptions
    "ArffCsvError",
    "ArffParseError",
    "ArffWriteError",
    "CsvParseError",
    "InvalidAttributeError",
    "MissingDataError",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]
