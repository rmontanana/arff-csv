# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2024-XX-XX

### Added

- Initial release of ARFF-CSV Converter
- `ArffConverter` class for bidirectional conversion between CSV and ARFF formats
- `ArffParser` class for reading and parsing ARFF files
- `ArffWriter` class for writing ARFF files
- `ArffData` dataclass for holding parsed ARFF data
- Support for all standard ARFF attribute types:
  - NUMERIC (floating-point numbers)
  - INTEGER (integer numbers)
  - REAL (alias for NUMERIC)
  - STRING (text values)
  - NOMINAL (categorical values)
  - DATE (date/time values with optional format)
- Sparse ARFF format support
- Missing value handling (standard `?` representation)
- Command-line interface with three commands:
  - `csv2arff` - Convert CSV files to ARFF format
  - `arff2csv` - Convert ARFF files to CSV format
  - `info` - Display information about ARFF files
- **CSV analysis mode** (`--analyze` / `-a`) for `csv2arff` command:
  - Analyzes CSV files and suggests column types (NOMINAL, STRING, NUMERIC, INTEGER)
  - Detects binary columns (0/1, yes/no, true/false, si/no)
  - Recognizes common target column names (class, target, label, y)
  - Identifies categorical integers based on unique value count
  - Configurable nominal threshold (`--nominal-threshold`)
  - Configurable preview rows (`--preview-rows`)
  - Generates ready-to-use conversion command
- Convenience functions `csv_to_arff()` and `arff_to_csv()`
- Direct pandas DataFrame integration
- Full type annotations (PEP 484 compliant)
- Comprehensive exception hierarchy:
  - `ArffCsvError` (base exception)
  - `ArffParseError`
  - `ArffWriteError`
  - `CsvParseError`
  - `InvalidAttributeError`
  - `MissingDataError`
- Comprehensive test suite with high coverage
- Support for Python 3.10, 3.11, 3.12, 3.13, and 3.14

### Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0

[Unreleased]: https://github.com/rmontanana/arff-csv-converter/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/rmontanana/arff-csv-converter/releases/tag/v1.0.0
