"""
Tests for the ARFF parser module.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arff_csv.parser import ArffParser, ArffData, Attribute, AttributeType
from arff_csv.exceptions import ArffParseError, MissingDataError


class TestArffParser:
    """Tests for ArffParser class."""

    def test_parse_basic_arff(self, sample_arff_content: str) -> None:
        """Test parsing a basic ARFF file."""
        parser = ArffParser()
        arff_data = parser.parse_string(sample_arff_content)

        assert arff_data.relation_name == "iris"
        assert len(arff_data.attributes) == 5
        assert len(arff_data.data) == 6

    def test_parse_relation_name(self, sample_arff_content: str) -> None:
        """Test that relation name is correctly parsed."""
        parser = ArffParser()
        arff_data = parser.parse_string(sample_arff_content)

        assert arff_data.relation_name == "iris"

    def test_parse_attributes(self, sample_arff_content: str) -> None:
        """Test that attributes are correctly parsed."""
        parser = ArffParser()
        arff_data = parser.parse_string(sample_arff_content)

        attr_names = arff_data.get_attribute_names()
        assert attr_names == ["sepallength", "sepalwidth", "petallength", 
                             "petalwidth", "class"]

        # Check numeric attributes
        numeric_attrs = arff_data.get_numeric_attributes()
        assert len(numeric_attrs) == 4

        # Check nominal attribute
        nominal_attrs = arff_data.get_nominal_attributes()
        assert nominal_attrs == ["class"]

    def test_parse_nominal_values(self, sample_arff_content: str) -> None:
        """Test that nominal values are correctly parsed."""
        parser = ArffParser()
        arff_data = parser.parse_string(sample_arff_content)

        class_attr = arff_data.attributes[-1]
        assert class_attr.type == AttributeType.NOMINAL
        assert class_attr.nominal_values == [
            "Iris-setosa", "Iris-versicolor", "Iris-virginica"
        ]

    def test_parse_data_values(self, sample_arff_content: str) -> None:
        """Test that data values are correctly parsed."""
        parser = ArffParser()
        arff_data = parser.parse_string(sample_arff_content)

        # Check first row
        first_row = arff_data.data.iloc[0]
        assert first_row["sepallength"] == 5.1
        assert first_row["sepalwidth"] == 3.5
        assert first_row["class"] == "Iris-setosa"

    def test_parse_comments(self, sample_arff_content: str) -> None:
        """Test that comments are correctly parsed."""
        parser = ArffParser()
        arff_data = parser.parse_string(sample_arff_content)

        assert len(arff_data.comments) == 2
        assert "Sample ARFF file" in arff_data.comments[0]

    def test_parse_missing_values(self, sample_arff_with_missing: str) -> None:
        """Test parsing ARFF with missing values."""
        parser = ArffParser()
        arff_data = parser.parse_string(sample_arff_with_missing)

        # Check for NaN values
        assert pd.isna(arff_data.data.iloc[1]["a"])
        assert pd.isna(arff_data.data.iloc[2]["b"])
        assert pd.isna(arff_data.data.iloc[3]["c"])

    def test_parse_string_attributes(self, sample_arff_with_strings: str) -> None:
        """Test parsing ARFF with string attributes."""
        parser = ArffParser()
        arff_data = parser.parse_string(sample_arff_with_strings)

        assert arff_data.data.iloc[0]["name"] == "John Doe"
        assert arff_data.data.iloc[0]["description"] == "A person with spaces"
        assert arff_data.data.iloc[1]["name"] == "Jane's name"
        assert arff_data.data.iloc[1]["description"] == "Description with, comma"

    def test_parse_sparse_format(self, sample_arff_sparse: str) -> None:
        """Test parsing ARFF with sparse data format."""
        parser = ArffParser()
        arff_data = parser.parse_string(sample_arff_sparse)

        # First row: {0 1.0, 2 3.0}
        assert arff_data.data.iloc[0]["a"] == 1.0
        assert pd.isna(arff_data.data.iloc[0]["b"])
        assert arff_data.data.iloc[0]["c"] == 3.0
        assert pd.isna(arff_data.data.iloc[0]["d"])

        # Third row: all values present
        assert arff_data.data.iloc[2]["a"] == 5.0
        assert arff_data.data.iloc[2]["b"] == 6.0
        assert arff_data.data.iloc[2]["c"] == 7.0
        assert arff_data.data.iloc[2]["d"] == 8.0

    def test_parse_date_attribute(self, sample_arff_date: str) -> None:
        """Test parsing ARFF with date attributes."""
        parser = ArffParser()
        arff_data = parser.parse_string(sample_arff_date)

        date_attr = arff_data.attributes[0]
        assert date_attr.type == AttributeType.DATE
        assert date_attr.date_format == "yyyy-MM-dd"

    def test_parse_file(self, sample_arff_file: Path) -> None:
        """Test parsing ARFF from file."""
        parser = ArffParser()
        arff_data = parser.parse_file(sample_arff_file)

        assert arff_data.relation_name == "iris"
        assert len(arff_data.data) == 6

    def test_parse_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        parser = ArffParser()

        with pytest.raises(FileNotFoundError):
            parser.parse_file("/nonexistent/file.arff")

    def test_parse_missing_relation(self) -> None:
        """Test error when @RELATION is missing."""
        parser = ArffParser()
        content = """
@ATTRIBUTE a NUMERIC
@DATA
1.0
"""
        with pytest.raises(MissingDataError, match="No @RELATION found"):
            parser.parse_string(content)

    def test_parse_missing_attributes(self) -> None:
        """Test error when no attributes are defined."""
        parser = ArffParser()
        content = """
@RELATION test
@DATA
"""
        with pytest.raises(MissingDataError, match="No @ATTRIBUTE definitions"):
            parser.parse_string(content)

    def test_parse_invalid_attribute_type(self) -> None:
        """Test error for invalid attribute type."""
        parser = ArffParser()
        content = """
@RELATION test
@ATTRIBUTE a INVALID_TYPE
@DATA
"""
        with pytest.raises(ArffParseError, match="Unknown attribute type"):
            parser.parse_string(content)

    def test_parse_wrong_number_of_values(self) -> None:
        """Test error when data row has wrong number of values."""
        parser = ArffParser()
        content = """
@RELATION test
@ATTRIBUTE a NUMERIC
@ATTRIBUTE b NUMERIC
@DATA
1.0,2.0,3.0
"""
        with pytest.raises(ArffParseError, match="Wrong number of values"):
            parser.parse_string(content)

    def test_parse_quoted_relation_name(self) -> None:
        """Test parsing quoted relation name."""
        parser = ArffParser()
        content = """
@RELATION 'My Dataset Name'
@ATTRIBUTE a NUMERIC
@DATA
1.0
"""
        arff_data = parser.parse_string(content)
        assert arff_data.relation_name == "My Dataset Name"

    def test_parse_quoted_attribute_name(self) -> None:
        """Test parsing quoted attribute name."""
        parser = ArffParser()
        content = """
@RELATION test
@ATTRIBUTE 'Column With Spaces' NUMERIC
@DATA
1.0
"""
        arff_data = parser.parse_string(content)
        assert arff_data.attributes[0].name == "Column With Spaces"

    def test_parse_integer_type(self) -> None:
        """Test parsing INTEGER type attribute."""
        parser = ArffParser()
        content = """
@RELATION test
@ATTRIBUTE id INTEGER
@DATA
1
2
3
"""
        arff_data = parser.parse_string(content)
        assert arff_data.attributes[0].type == AttributeType.INTEGER

    def test_parse_real_type(self) -> None:
        """Test parsing REAL type attribute."""
        parser = ArffParser()
        content = """
@RELATION test
@ATTRIBUTE value REAL
@DATA
1.5
2.5
"""
        arff_data = parser.parse_string(content)
        assert arff_data.attributes[0].type == AttributeType.NUMERIC

    def test_parse_empty_data(self) -> None:
        """Test parsing ARFF with empty data section."""
        parser = ArffParser()
        content = """
@RELATION test
@ATTRIBUTE a NUMERIC
@DATA
"""
        arff_data = parser.parse_string(content)
        assert len(arff_data.data) == 0
        assert list(arff_data.data.columns) == ["a"]

    def test_custom_missing_value(self) -> None:
        """Test custom missing value string."""
        parser = ArffParser(missing_value="NA")
        content = """
@RELATION test
@ATTRIBUTE a NUMERIC
@DATA
1.0
NA
3.0
"""
        arff_data = parser.parse_string(content)
        assert pd.isna(arff_data.data.iloc[1]["a"])


class TestAttribute:
    """Tests for Attribute class."""

    def test_is_numeric_true(self) -> None:
        """Test is_numeric returns True for numeric types."""
        attr = Attribute(name="test", type=AttributeType.NUMERIC)
        assert attr.is_numeric() is True

        attr = Attribute(name="test", type=AttributeType.REAL)
        assert attr.is_numeric() is True

        attr = Attribute(name="test", type=AttributeType.INTEGER)
        assert attr.is_numeric() is True

    def test_is_numeric_false(self) -> None:
        """Test is_numeric returns False for non-numeric types."""
        attr = Attribute(name="test", type=AttributeType.STRING)
        assert attr.is_numeric() is False

        attr = Attribute(name="test", type=AttributeType.NOMINAL)
        assert attr.is_numeric() is False

        attr = Attribute(name="test", type=AttributeType.DATE)
        assert attr.is_numeric() is False


class TestArffData:
    """Tests for ArffData class."""

    def test_get_attribute_names(self, sample_arff_content: str) -> None:
        """Test getting attribute names."""
        parser = ArffParser()
        arff_data = parser.parse_string(sample_arff_content)

        names = arff_data.get_attribute_names()
        assert len(names) == 5
        assert "sepallength" in names
        assert "class" in names

    def test_get_numeric_attributes(self, sample_arff_content: str) -> None:
        """Test getting numeric attribute names."""
        parser = ArffParser()
        arff_data = parser.parse_string(sample_arff_content)

        numeric = arff_data.get_numeric_attributes()
        assert len(numeric) == 4
        assert "class" not in numeric

    def test_get_nominal_attributes(self, sample_arff_content: str) -> None:
        """Test getting nominal attribute names."""
        parser = ArffParser()
        arff_data = parser.parse_string(sample_arff_content)

        nominal = arff_data.get_nominal_attributes()
        assert nominal == ["class"]
