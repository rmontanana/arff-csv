"""
Tests for the CLI module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arff_csv.cli import main, create_parser


class TestCLI:
    """Tests for CLI functionality."""

    def test_create_parser(self) -> None:
        """Test parser creation."""
        parser = create_parser()
        assert parser is not None

    def test_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test help output."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        
        # Help should exit with 0
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "csv2arff" in captured.out

    def test_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test version output."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        
        # Version should exit with 0
        assert exc_info.value.code == 0

    def test_no_command(self) -> None:
        """Test running without command."""
        result = main([])
        assert result == 0

    def test_csv2arff_command(
        self, sample_csv_file: Path, temp_dir: Path
    ) -> None:
        """Test csv2arff command."""
        output_path = temp_dir / "output.arff"

        result = main([
            "csv2arff",
            str(sample_csv_file),
            str(output_path),
            "-r", "test_relation",
        ])

        assert result == 0
        assert output_path.exists()

    def test_csv2arff_verbose(
        self, sample_csv_file: Path, temp_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test csv2arff with verbose output."""
        output_path = temp_dir / "output.arff"

        result = main([
            "csv2arff",
            str(sample_csv_file),
            str(output_path),
            "-v",
        ])

        assert result == 0
        captured = capsys.readouterr()
        assert "Successfully converted" in captured.out

    def test_csv2arff_with_nominal(
        self, sample_csv_file: Path, temp_dir: Path
    ) -> None:
        """Test csv2arff with nominal columns."""
        output_path = temp_dir / "output.arff"

        result = main([
            "csv2arff",
            str(sample_csv_file),
            str(output_path),
            "-n", "name", "passed",
        ])

        assert result == 0

    def test_csv2arff_with_comments(
        self, sample_csv_file: Path, temp_dir: Path
    ) -> None:
        """Test csv2arff with comments."""
        output_path = temp_dir / "output.arff"

        result = main([
            "csv2arff",
            str(sample_csv_file),
            str(output_path),
            "-c", "Test comment",
        ])

        assert result == 0
        content = output_path.read_text()
        assert "% Test comment" in content

    def test_csv2arff_file_not_found(self, temp_dir: Path) -> None:
        """Test csv2arff with missing input file."""
        output_path = temp_dir / "output.arff"

        result = main([
            "csv2arff",
            "/nonexistent/file.csv",
            str(output_path),
        ])

        assert result == 1

    def test_arff2csv_command(
        self, sample_arff_file: Path, temp_dir: Path
    ) -> None:
        """Test arff2csv command."""
        output_path = temp_dir / "output.csv"

        result = main([
            "arff2csv",
            str(sample_arff_file),
            str(output_path),
        ])

        assert result == 0
        assert output_path.exists()

    def test_arff2csv_verbose(
        self, sample_arff_file: Path, temp_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test arff2csv with verbose output."""
        output_path = temp_dir / "output.csv"

        result = main([
            "arff2csv",
            str(sample_arff_file),
            str(output_path),
            "-v",
        ])

        assert result == 0
        captured = capsys.readouterr()
        assert "Successfully converted" in captured.out

    def test_arff2csv_include_index(
        self, sample_arff_file: Path, temp_dir: Path
    ) -> None:
        """Test arff2csv with index included."""
        output_path = temp_dir / "output.csv"

        result = main([
            "arff2csv",
            str(sample_arff_file),
            str(output_path),
            "--include-index",
        ])

        assert result == 0

    def test_arff2csv_file_not_found(self, temp_dir: Path) -> None:
        """Test arff2csv with missing input file."""
        output_path = temp_dir / "output.csv"

        result = main([
            "arff2csv",
            "/nonexistent/file.arff",
            str(output_path),
        ])

        assert result == 1

    def test_info_command(
        self, sample_arff_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test info command."""
        result = main([
            "info",
            str(sample_arff_file),
        ])

        assert result == 0
        captured = capsys.readouterr()
        assert "Relation: iris" in captured.out
        assert "Attributes:" in captured.out

    def test_info_file_not_found(self) -> None:
        """Test info with missing file."""
        result = main([
            "info",
            "/nonexistent/file.arff",
        ])

        assert result == 1

    def test_csv2arff_custom_delimiter(self, temp_dir: Path) -> None:
        """Test csv2arff with custom delimiter."""
        # Create CSV with semicolon delimiter
        csv_path = temp_dir / "input.csv"
        csv_path.write_text("a;b;c\n1;2;3\n4;5;6")

        output_path = temp_dir / "output.arff"

        result = main([
            "csv2arff",
            str(csv_path),
            str(output_path),
            "--delimiter", ";",
        ])

        assert result == 0

    def test_csv2arff_custom_missing(self, temp_dir: Path) -> None:
        """Test csv2arff with custom missing value."""
        csv_path = temp_dir / "input.csv"
        csv_path.write_text("a,b\n1,2\nNA,3")

        output_path = temp_dir / "output.arff"

        result = main([
            "csv2arff",
            str(csv_path),
            str(output_path),
            "-m", "NA",
        ])

        assert result == 0


class TestCLIEdgeCases:
    """Tests for CLI edge cases."""

    def test_empty_args(self) -> None:
        """Test with empty arguments."""
        result = main([])
        assert result == 0

    def test_invalid_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test with invalid command."""
        # argparse will handle this with an error
        with pytest.raises(SystemExit):
            main(["invalid_command"])
