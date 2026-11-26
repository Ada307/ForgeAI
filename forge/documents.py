"""Document ingestion module for ForgeAI.

Recursively discovers and ingests PDF, CSV, and TXT files, preserving
source metadata, page/row references, and document boundaries.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import csv
import os
import warnings
import pypdf


class DocumentIngestionError(Exception):
    """Raised when document ingestion fails or no usable documents are found."""
    pass


@dataclass
class Document:
    """Structured record representing an ingested document or document section."""
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.doc_id


@dataclass
class SkippedFile:
    """Record of a file that was skipped during ingestion."""
    path: str
    reason: str


class DocumentLoader:
    """Recursively discovers and loads documents from a directory.

    Supported formats: .txt, .csv, .pdf (case-insensitive).
    Preserves relative paths, PDF page numbers, and CSV row numbers with headers.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".csv", ".pdf"}

    def __init__(self) -> None:
        self.skipped_files: List[SkippedFile] = []

    def _record_skip(self, path: str, reason: str) -> None:
        self.skipped_files.append(SkippedFile(path=path, reason=reason))
        warnings.warn(f"Skipping '{path}': {reason}", UserWarning, stacklevel=2)

    def load(self, folder_path: Union[str, Path]) -> List[Document]:
        """Load all usable documents from folder_path recursively."""
        self.skipped_files = []
        root_path = Path(folder_path).resolve()

        if not root_path.exists():
            raise FileNotFoundError(f"Document directory does not exist: '{folder_path}'")
        if not root_path.is_dir():
            raise NotADirectoryError(f"Provided path is not a directory: '{folder_path}'")

        # Discover all files deterministically
        discovered_files: List[Path] = []
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                discovered_files.append(Path(dirpath) / filename)

        # Sort deterministically by relative path
        discovered_files.sort(key=lambda p: p.relative_to(root_path).as_posix())

        documents: List[Document] = []

        for file_path in discovered_files:
            rel_path = file_path.relative_to(root_path).as_posix()
            ext = file_path.suffix.lower()

            if ext not in self.SUPPORTED_EXTENSIONS:
                self._record_skip(rel_path, f"Unsupported file extension '{ext}'")
                continue

            try:
                if ext == ".txt":
                    docs = self._load_txt(file_path, rel_path)
                elif ext == ".csv":
                    docs = self._load_csv(file_path, rel_path)
                elif ext == ".pdf":
                    docs = self._load_pdf(file_path, rel_path)
                else:
                    continue

                documents.extend(docs)
            except Exception as exc:
                self._record_skip(rel_path, f"Failed to read file: {exc}")

        if not documents:
            raise DocumentIngestionError(
                f"No usable documents found in '{folder_path}'. "
                f"Checked {len(discovered_files)} files; skipped {len(self.skipped_files)}."
            )

        return documents

    def _load_txt(self, file_path: Path, rel_path: str) -> List[Document]:
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding="latin-1")
            except Exception as exc:
                self._record_skip(rel_path, f"Unreadable text encoding: {exc}")
                return []

        text = content.strip()
        if not text:
            self._record_skip(rel_path, "File is empty or contains only whitespace")
            return []

        metadata = {
            "source": rel_path,
            "file_type": "txt",
            "filename": file_path.name,
        }
        return [Document(doc_id=rel_path, text=text, metadata=metadata)]

    def _load_csv(self, file_path: Path, rel_path: str) -> List[Document]:
        content: Optional[str] = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                content = file_path.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            self._record_skip(rel_path, "Unable to decode CSV file with supported encodings")
            return []

        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            self._record_skip(rel_path, "CSV file is empty")
            return []

        reader = csv.reader(lines)
        try:
            headers = next(reader, None)
        except Exception as exc:
            self._record_skip(rel_path, f"Malformed CSV header: {exc}")
            return []

        if not headers or not any(h.strip() for h in headers):
            self._record_skip(rel_path, "CSV missing column headers")
            return []

        clean_headers = [h.strip() for h in headers]
        docs: List[Document] = []
        row_idx = 0

        for raw_row in reader:
            if not raw_row or not any(cell.strip() for cell in raw_row):
                continue
            row_idx += 1

            # Format row with column names and values to preserve meaning
            fields_formatted: List[str] = []
            for i, val in enumerate(raw_row):
                col_name = clean_headers[i] if i < len(clean_headers) and clean_headers[i] else f"column_{i+1}"
                val_clean = val.strip()
                if val_clean:
                    fields_formatted.append(f"{col_name}: {val_clean}")

            if not fields_formatted:
                continue

            row_text = f"Row {row_idx}: " + ", ".join(fields_formatted)
            doc_id = f"{rel_path}:r{row_idx}"
            metadata = {
                "source": rel_path,
                "file_type": "csv",
                "filename": file_path.name,
                "row": row_idx,
                "row_number": row_idx,
            }
            docs.append(Document(doc_id=doc_id, text=row_text, metadata=metadata))

        if not docs:
            self._record_skip(rel_path, "CSV contains headers but no valid data rows")
            return []

        return docs

    def _load_pdf(self, file_path: Path, rel_path: str) -> List[Document]:
        try:
            reader = pypdf.PdfReader(str(file_path))
        except Exception as exc:
            self._record_skip(rel_path, f"Unreadable or corrupt PDF: {exc}")
            return []

        total_pages = len(reader.pages)
        if total_pages == 0:
            self._record_skip(rel_path, "PDF contains no pages")
            return []

        docs: List[Document] = []
        for page_idx, page in enumerate(reader.pages, start=1):
            page_id = f"{rel_path}:p{page_idx}"
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                self._record_skip(page_id, f"Page {page_idx} extraction failed: {exc}")
                continue

            clean_text = page_text.strip()
            if not clean_text:
                self._record_skip(page_id, f"Page {page_idx} contains no extractable text")
                continue

            doc_id = page_id
            metadata = {
                "source": rel_path,
                "file_type": "pdf",
                "filename": file_path.name,
                "page": page_idx,
                "page_number": page_idx,
                "total_pages": total_pages,
            }
            docs.append(Document(doc_id=doc_id, text=clean_text, metadata=metadata))

        if not docs:
            self._record_skip(
                rel_path,
                "PDF contains no extractable text (scanned image or empty; OCR not supported)"
            )
            return []

        return docs


def load_documents(folder_path: Union[str, Path]) -> List[Document]:
    """Convenience function to load documents using DocumentLoader."""
    return DocumentLoader().load(folder_path)
