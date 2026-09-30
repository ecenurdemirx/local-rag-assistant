import io
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import config
import database
import document_loader
import evaluate
import llm
import rag_pipeline
import retriever


def write_test_pdf(path, page_texts):
    """Writes a tiny deterministic text PDF using only pypdf."""
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_reference = writer._add_object(font)

    for page_text in page_texts:
        page = writer.add_blank_page(width=300, height=300)
        if page_text is None:
            continue
        escaped_text = (
            page_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        )
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 40 250 Td ({escaped_text}) Tj ET".encode("ascii"))
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})
        })
        page[NameObject("/Contents")] = writer._add_object(content)

    with path.open("wb") as pdf_file:
        writer.write(pdf_file)


class ChunkingTests(unittest.TestCase):
    def test_chunking_preserves_words_and_overlap(self):
        chunks = document_loader.chunk_text("one two three four five six", 14, 5)
        self.assertEqual(chunks, ["one two three", "three four", "four five six"])

    def test_invalid_chunk_settings(self):
        for size, overlap in ((0, 0), (10, -1), (10, 10), (10, 11)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                document_loader.chunk_text("text", size, overlap)

    def test_recursive_loader_skips_empty_and_invalid_utf8_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "nested").mkdir()
            (root / "nested" / "valid.md").write_text("useful text", encoding="utf-8")
            (root / "empty.txt").write_text("", encoding="utf-8")
            (root / "invalid.txt").write_bytes(b"\xff")
            documents = document_loader.load_documents(root)
        self.assertEqual(
            [(item["source"], item["text"]) for item in documents],
            [("empty.txt", ""), ("nested/valid.md", "useful text")],
        )


class PdfLoaderTests(unittest.TestCase):
    def test_text_pdf_is_discovered_and_preserves_relative_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "nested").mkdir()
            write_test_pdf(
                root / "nested" / "manual.pdf",
                ["First page PDF fact.", None, "Final page PDF detail."],
            )
            documents = document_loader.load_documents(root)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["source"], "nested/manual.pdf")
        self.assertEqual(
            documents[0]["text"], "First page PDF fact.\n\nFinal page PDF detail."
        )

    def test_extracted_pdf_text_uses_standard_chunking(self):
        with tempfile.TemporaryDirectory() as folder:
            pdf_path = Path(folder) / "long.pdf"
            write_test_pdf(pdf_path, ["one two three four five six"])
            document = document_loader.load_documents(folder)[0]

        self.assertEqual(
            document_loader.chunk_text(document["text"], 14, 5),
            ["one two three", "three four", "four five six"],
        )

    def test_empty_pdf_warns_and_does_not_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            pdf_path = Path(folder) / "scanned.pdf"
            write_test_pdf(pdf_path, [None])
            output = io.StringIO()
            with redirect_stdout(output):
                documents = document_loader.load_documents(folder)

        self.assertEqual(documents, [])
        self.assertIn("No extractable text found in scanned.pdf; skipping", output.getvalue())


class SimilarityTests(unittest.TestCase):
    def test_cosine_similarity(self):
        self.assertAlmostEqual(retriever.cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(retriever.cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertEqual(retriever.cosine_similarity([], []), 0.0)
        self.assertEqual(retriever.cosine_similarity([0, 0], [1, 0]), 0.0)
        self.assertEqual(retriever.cosine_similarity([1], [1, 0]), 0.0)

    @patch("embeddings.get_embedding", return_value=[1.0, 0.0])
    @patch("database.get_all_chunks")
    def test_retrieve_ranks_and_handles_fewer_than_k(self, get_chunks, _get_embedding):
        get_chunks.return_value = [
            {"source": "low.txt", "chunk_index": 0, "content": "low", "embedding": [0, 1]},
            {"source": "high.txt", "chunk_index": 0, "content": "high", "embedding": [1, 0]},
        ]
        results = retriever.retrieve("query", top_k=5)
        self.assertEqual([item["source"] for item in results], ["high.txt", "low.txt"])


class EvaluationSetupTests(unittest.TestCase):
    def test_load_cases_accepts_supported_and_unsupported_questions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cases.json"
            path.write_text(
                '[{"question": "Supported?", "expected_source": "guide.txt"}, '
                '{"question": "Unsupported?", "expected_source": null}]',
                encoding="utf-8",
            )
            cases = evaluate.load_cases(path)
        self.assertEqual(cases[0]["expected_source"], "guide.txt")
        self.assertIsNone(cases[1]["expected_source"])

    def test_load_cases_rejects_missing_expected_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cases.json"
            path.write_text('[{"question": "Supported?"}]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must include expected_source"):
                evaluate.load_cases(path)

    def test_load_cases_reports_missing_file(self):
        with self.assertRaisesRegex(ValueError, "Evaluation file not found"):
            evaluate.load_cases("does-not-exist.json")

    def test_load_cases_reports_invalid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cases.json"
            path.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid JSON"):
                evaluate.load_cases(path)

    def test_load_cases_rejects_invalid_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cases.json"
            path.write_text('[{"question": "", "expected_source": 42}]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty 'question'"):
                evaluate.load_cases(path)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.db_patch = patch.object(config, "DATABASE_PATH", self.db_path)
        self.db_patch.start()
        database.initialize_database()

    def tearDown(self):
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def test_empty_question(self):
        result = rag_pipeline.query_pipeline("   ")
        self.assertEqual(result["retrieved_chunks"], [])

    def test_empty_index_returns_help_without_models(self):
        with patch("retriever.retrieve") as retrieve, patch("llm.generate_answer") as generate:
            result = rag_pipeline.query_pipeline("question")
        retrieve.assert_not_called()
        generate.assert_not_called()
        self.assertIn("knowledge base is currently empty", result["answer"])

    def test_atomic_index_replacement(self):
        database.replace_chunks([
            {"source": "a.txt", "chunk_index": 0, "content": "hello", "embedding": [1.0, 0.0]}
        ])
        self.assertEqual(database.count_chunks(), 1)
        self.assertEqual(database.get_all_chunks()[0]["source"], "a.txt")
        self.assertTrue(database.index_uses_current_model())

        with self.assertRaises(sqlite3.IntegrityError):
            database.replace_chunks([
                {"source": None, "chunk_index": 0, "content": "broken", "embedding": [0.0]}
            ])
        self.assertEqual(database.get_all_chunks()[0]["source"], "a.txt")

    def test_empty_ingestion_preserves_previous_index(self):
        database.replace_chunks([
            {"source": "a.txt", "chunk_index": 0, "content": "hello", "embedding": [1.0]}
        ])
        for documents in ([], [{"source": "empty.txt", "text": "   "}]):
            with patch("document_loader.load_documents", return_value=documents):
                with self.assertRaisesRegex(ValueError, "existing index preserved"):
                    rag_pipeline.ingest_documents()
            self.assertEqual(database.count_chunks(), 1)

    def test_abstention_has_no_answer_sources(self):
        database.replace_chunks([
            {"source": "a.txt", "chunk_index": 0, "content": "hello", "embedding": [1.0]}
        ])
        with patch("retriever.retrieve", return_value=[
            {"source": "a.txt", "chunk_index": 0, "content": "hello", "score": 0.9}
        ]), patch("llm.generate_answer", return_value=llm.INSUFFICIENT_CONTEXT_RESPONSE):
            result = rag_pipeline.query_pipeline("unknown")
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["answer"], llm.INSUFFICIENT_CONTEXT_RESPONSE)

    def test_embedding_model_metadata_mismatch(self):
        database.replace_chunks([
            {"source": "a.txt", "chunk_index": 0, "content": "hello", "embedding": [1.0]}
        ])
        with patch.object(config, "EMBEDDING_MODEL", "different-model"):
            self.assertFalse(database.index_uses_current_model())

    @patch("retriever.retrieve", return_value=[])
    def test_missing_context_does_not_call_llm(self, _retrieve):
        database.replace_chunks([
            {"source": "a.txt", "chunk_index": 0, "content": "hello", "embedding": [1.0]}
        ])
        with patch("llm.generate_answer") as generate:
            result = rag_pipeline.query_pipeline("unknown")
        generate.assert_not_called()
        self.assertIn("enough information", result["answer"])

    @patch("retriever.retrieve")
    def test_below_threshold_context_does_not_call_llm(self, retrieve):
        database.replace_chunks([
            {"source": "a.txt", "chunk_index": 0, "content": "hello", "embedding": [1.0]}
        ])
        retrieve.return_value = [
            {"source": "a.txt", "chunk_index": 0, "content": "hello", "score": 0.1}
        ]
        with patch("llm.generate_answer") as generate:
            result = rag_pipeline.query_pipeline("unknown")
        generate.assert_not_called()
        self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main()
