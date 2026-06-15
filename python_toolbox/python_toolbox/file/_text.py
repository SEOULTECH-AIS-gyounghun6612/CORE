"""Plain-text file reader and writer."""
from __future__ import annotations
from pathlib import Path

from ._base import File_Process, Handle_exp, Suffix_check


class Text(File_Process):
    """Handles plain-text file persistence."""

    @classmethod
    @Handle_exp()
    def Read_from(
        cls, file: Path,
        enc: str = "UTF-8", start: int = 0, delim: str = "\n"
    ) -> tuple[bool, list[str]]:
        """Reads a text file and splits it by a delimiter.

        Args:
            file: Input text file path.
            enc: Text encoding.
            start: Start index applied after splitting.
            delim: Delimiter used to split the text.

        Returns:
            A tuple of ``(is_ok, parts)``.
        """
        _, _file = Suffix_check(file, ".txt")
        if _file.exists():
            return True, _file.read_text(enc).split(delim)[start:]
        return False, []

    @classmethod
    @Handle_exp()
    def Write_to(
        cls, file: Path, data: str | list[str], enc: str = "UTF-8",
        anno: list[str] | str | None = None
    ) -> bool:
        """Writes plain text to a file.

        Args:
            file: Output text file path.
            data: String or string list to write.
            enc: Text encoding.
            anno: Optional header lines written before ``data``.

        Returns:
            ``True`` when the write succeeds.
        """
        cls.Ensure_dir(file)
        _, _path = Suffix_check(file, ".txt", True)

        _data = (
            [anno] if isinstance(anno, str) else anno
        ) if anno else []
        _data += [data] if isinstance(data, str) else data

        _path.write_text("\n".join(_data), enc)
        return True
