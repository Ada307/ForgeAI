"""Tests for document ingestion module (forge/documents.py)."""

import io
from pathlib import Path
import tempfile
import unittest
import warnings
import pypdf

from forge.documents import DocumentIngestionError, DocumentLoader, load_documents


def _create_minimal_pdf(file_path: Path, text: str = "Industrial furnace safety procedures.") -> None:
    """Helper to create a minimal valid text-containing PDF using exact xref offsets."""
    stream_data = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET\n".encode("latin-1")
    header = b"%PDF-1.4\n"
    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = (
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
    )
    obj4 = b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    obj5 = (
        b"5 0 obj\n<< /Length " + str(len(stream_data)).encode("ascii") + b" >>\nstream\n"
        + stream_data + b"endstream\nendobj\n"
    )

    off1 = len(header)
    off2 = off1 + len(obj1)
    off3 = off2 + len(obj2)
    off4 = off3 + len(obj3)
    off5 = off4 + len(obj4)
    xref_pos = off5 + len(obj5)

    xref = (
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        + f"{off1:010d} 00000 n \n".encode("ascii")
        + f"{off2:010d} 00000 n \n".encode("ascii")
        + f"{off3:010d} 00000 n \n".encode("ascii")
        + f"{off4:010d} 00000 n \n".encode("ascii")
        + f"{off5:010d} 00000 n \n".encode("ascii")
    )
    trailer = (
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
        + f"{xref_pos}\n".encode("ascii")
        + b"%%EOF\n"
    )

    pdf_bytes = header + obj1 + obj2 + obj3 + obj4 + obj5 + xref + trailer
    file_path.write_bytes(pdf_bytes)


def _create_blank_pdf(file_path: Path) -> None:
    """Helper to create a PDF page with no extractable text."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with open(file_path, "wb") as f:
        writer.write(f)


class TestDocumentIngestion(unittest.TestCase):
    """Test suite for DocumentLoader and load_documents."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_txt_loading(self) -> None:
        """Test loading a standard .txt document."""
        txt_file = self.root_path / "instructions.txt"
        txt_content = "Calibrate pressure relief valve to 150 PSI before initial startup."
        txt_file.write_text(txt_content, encoding="utf-8")

        docs = load_documents(self.root_path)
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc.doc_id, "instructions.txt")
        self.assertEqual(doc.text, txt_content)
        self.assertEqual(doc.metadata["source"], "instructions.txt")
        self.assertEqual(doc.metadata["file_type"], "txt")

    def test_csv_loading_and_row_preservation(self) -> None:
        """Test CSV loading with column names and row numbers preserved."""
        csv_file = self.root_path / "telemetry.csv"
        csv_content = (
            "device_id,temperature,status\n"
            "DEV-01,72.4,nominal\n"
            "DEV-02,98.1,warning\n"
        )
        csv_file.write_text(csv_content, encoding="utf-8")

        docs = load_documents(self.root_path)
        self.assertEqual(len(docs), 2)

        doc1, doc2 = docs[0], docs[1]
        self.assertEqual(doc1.metadata["source"], "telemetry.csv")
        self.assertEqual(doc1.metadata["row"], 1)
        self.assertIn("device_id: DEV-01", doc1.text)
        self.assertIn("temperature: 72.4", doc1.text)
        self.assertIn("status: nominal", doc1.text)

        self.assertEqual(doc2.metadata["source"], "telemetry.csv")
        self.assertEqual(doc2.metadata["row"], 2)
        self.assertIn("device_id: DEV-02", doc2.text)
        self.assertIn("temperature: 98.1", doc2.text)

    def test_pdf_loading(self) -> None:
        """Test PDF loading and page number preservation."""
        pdf_file = self.root_path / "safety.pdf"
        _create_minimal_pdf(pdf_file, "Industrial furnace safety procedures.")

        docs = load_documents(self.root_path)
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc.metadata["source"], "safety.pdf")
        self.assertEqual(doc.metadata["file_type"], "pdf")
        self.assertEqual(doc.metadata["page"], 1)
        self.assertIn("Industrial furnace safety procedures", doc.text)

    def test_nested_directories_and_duplicate_filenames(self) -> None:
        """Test relative path preservation for duplicate filenames in subfolders."""
        dir_a = self.root_path / "dept_a"
        dir_b = self.root_path / "dept_b"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)

        (dir_a / "log.txt").write_text("Department A log entry.", encoding="utf-8")
        (dir_b / "log.txt").write_text("Department B log entry.", encoding="utf-8")

        docs = load_documents(self.root_path)
        self.assertEqual(len(docs), 2)

        # Deterministic sorting ensures dept_a appears before dept_b
        self.assertEqual(docs[0].metadata["source"], "dept_a/log.txt")
        self.assertEqual(docs[0].doc_id, "dept_a/log.txt")
        self.assertEqual(docs[0].text, "Department A log entry.")

        self.assertEqual(docs[1].metadata["source"], "dept_b/log.txt")
        self.assertEqual(docs[1].doc_id, "dept_b/log.txt")
        self.assertEqual(docs[1].text, "Department B log entry.")

    def test_empty_and_invalid_documents_with_warnings(self) -> None:
        """Test that unreadable, empty, and unsupported files generate explicit warnings."""
        (self.root_path / "empty.txt").write_text("", encoding="utf-8")
        (self.root_path / "empty.csv").write_text("", encoding="utf-8")
        (self.root_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        _create_blank_pdf(self.root_path / "blank.pdf")
        (self.root_path / "valid.txt").write_text("Valid text content.", encoding="utf-8")

        loader = DocumentLoader()
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            docs = loader.load(self.root_path)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].text, "Valid text content.")

        # Check recorded skipped files and explicit warning messages
        skipped_paths = [s.path for s in loader.skipped_files]
        self.assertIn("empty.txt", skipped_paths)
        self.assertIn("empty.csv", skipped_paths)
        self.assertIn("image.png", skipped_paths)
        self.assertIn("blank.pdf", skipped_paths)
        self.assertGreaterEqual(len(recorded_warnings), 4)

    def test_missing_and_empty_folders(self) -> None:
        """Test error handling for missing and empty folder paths."""
        non_existent = self.root_path / "non_existent_dir"
        with self.assertRaises(FileNotFoundError):
            load_documents(non_existent)

        # Empty directory should raise DocumentIngestionError
        with self.assertRaises(DocumentIngestionError):
            load_documents(self.root_path)

    def test_pdf_partial_page_extraction_failure_recorded(self) -> None:
        """Test recording of individual blank or unextractable PDF pages when others succeed."""
        pdf_file = self.root_path / "mixed.pdf"
        _create_minimal_pdf(pdf_file, "Page 1 valid text.")
        reader = pypdf.PdfReader(str(pdf_file))
        writer = pypdf.PdfWriter()
        writer.append(reader)
        writer.add_blank_page(width=612, height=792)
        with open(pdf_file, "wb") as f:
            writer.write(f)

        loader = DocumentLoader()
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            docs = loader.load(self.root_path)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertIn("Page 1 valid text", docs[0].text)

        # Page 2 should be explicitly recorded in skipped_files
        skipped_paths = [s.path for s in loader.skipped_files]
        self.assertIn("mixed.pdf:p2", skipped_paths)


if __name__ == "__main__":
    unittest.main()
