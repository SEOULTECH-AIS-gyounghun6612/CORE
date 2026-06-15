"""CSV file reader and writer."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Any

from ._base import File_Process, Handle_exp, Suffix_check, BASIC_FILE_ERROR


CSV_FILE_READ_ERROR = {
    **BASIC_FILE_ERROR,
    csv.Error: "CSV 파싱 오류 발생",
}


class Csv(File_Process):
    """Handles CSV file persistence as rows of dicts (header required)."""

    @classmethod
    @Handle_exp(extra_exp=CSV_FILE_READ_ERROR)
    def Read_from(
        cls, file: Path, enc: str = "UTF-8", delimiter: str = ","
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Reads a CSV file into a list of dicts via DictReader.

        Args:
            file: Input CSV file path.
            enc: Text encoding.
            delimiter: Field delimiter character.

        Returns:
            A tuple of ``(is_ok, rows)``.
        """
        _, _file = Suffix_check(file, ".csv")

        if not _file.exists():
            return False, []

        with _file.open(encoding=enc, newline="") as _f:
            return True, list(csv.DictReader(_f, delimiter=delimiter))

    @classmethod
    @Handle_exp()
    def Write_to(
        cls,
        file: Path,
        data: list[dict[str, Any]],
        enc: str = "UTF-8",
        delimiter: str = ",",
        fieldnames: list[str] | None = None,
    ) -> bool:
        """Writes a list of dicts to a CSV file via DictWriter.

        Args:
            file: Output CSV file path.
            data: Rows to write; each row must be a dict.
            enc: Text encoding.
            delimiter: Field delimiter character.
            fieldnames: Column order; inferred from first row when omitted.

        Returns:
            ``True`` when the write succeeds.
        """
        cls.Ensure_dir(file)
        _, _path = Suffix_check(file, ".csv", True)

        _fields = fieldnames or (list(data[0].keys()) if data else [])

        with _path.open(mode="w", encoding=enc, newline="") as _f:
            _writer = csv.DictWriter(_f, fieldnames=_fields, delimiter=delimiter)
            _writer.writeheader()
            _writer.writerows(data)
        return True
