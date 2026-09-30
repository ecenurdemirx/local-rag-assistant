"""Offline control-flow tests; enable real model cases with RAG_LIVE_TESTS=1."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import llm
import embeddings
import rag_pipeline

FALLBACK = llm.INSUFFICIENT_CONTEXT_RESPONSE
CHUNKS = [{"source": "classification.txt", "chunk_index": 0,
           "content": "Binary classification has two possible classes.", "score": 0.99}]
CASES = [
    ("What is binary classification?", True),
    ("How many possible classes does binary classification have?", True),
    ("Who invented binary classification?", False),
    ("Who won the 2022 FIFA World Cup?", False),
    ("Explain everything about machine learning.", False),
    ("How many classes does binary classification have and who invented it?", False),
    ("In which year was binary classification invented?", False),
    ("What is the capital of Japan?", False),
]


def event(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class AnswerabilityTests(unittest.TestCase):
    def test_supported_and_paraphrased_questions(self):
        for question, supported in CASES:
            if not supported:
                continue
            with self.subTest(question=question), patch(
                "llm._complete", side_effect=["ANSWERABLE", "1", "SUPPORTED"]
            ) as complete:
                self.assertEqual(llm.generate_answer(question, CHUNKS), CHUNKS[0]["content"])
                self.assertEqual(complete.call_count, 3)

    def test_negative_decisions_never_generate(self):
        for question, supported in CASES:
            if supported:
                continue
            with self.subTest(question=question), patch(
                "llm._complete", return_value="NOT_ANSWERABLE"
            ) as complete:
                self.assertEqual(llm.generate_answer(question, CHUNKS), FALLBACK)
                complete.assert_called_once()

    def test_ambiguous_classification_fails_closed(self):
        for decision in ("", "answerable", "ANSWERABLE.", "NOT_ANSWERABLE",
                         "ANSWERABLE because it mentions classification", "ANSWERABLE\nNOT_ANSWERABLE"):
            with self.subTest(decision=decision), patch("llm._complete", return_value=decision) as complete:
                self.assertEqual(llm.generate_answer(CASES[0][0], CHUNKS), FALLBACK)
                complete.assert_called_once()

    def test_invalid_or_invented_generation_is_rejected(self):
        for selection in ("", "0", "2", "1,1", "-1", "1.0", "1 and 2",
                          "Invented by Alan Turing.", FALLBACK):
            with self.subTest(selection=selection), patch(
                "llm._complete", side_effect=["ANSWERABLE", selection]
            ) as complete:
                self.assertEqual(llm.generate_answer(CASES[0][0], CHUNKS), FALLBACK)
                self.assertEqual(complete.call_count, 2)

    def test_verification_rejects_true_but_irrelevant_definition(self):
        with patch("llm._complete", side_effect=["ANSWERABLE", "1", "NOT_SUPPORTED"]):
            self.assertEqual(llm.generate_answer("Who invented binary classification?", CHUNKS), FALLBACK)

    def test_ambiguous_verification_fails_closed(self):
        for verdict in ("", "supported", "SUPPORTED.", "SUPPORTED but incomplete"):
            with self.subTest(verdict=verdict), patch(
                "llm._complete", side_effect=["ANSWERABLE", "1", verdict]
            ):
                self.assertEqual(llm.generate_answer(CASES[0][0], CHUNKS), FALLBACK)

    def test_multiple_passages_are_exact_evidence(self):
        chunks = [{**CHUNKS[0], "content": "Training learns from data. Prediction uses new data."}]
        with patch("llm._complete", side_effect=["ANSWERABLE", "1,2", "SUPPORTED"]):
            self.assertEqual(llm.generate_answer("What are both stages?", chunks), chunks[0]["content"])

    def test_no_context_does_not_call_model(self):
        with patch("llm._get_client") as client:
            self.assertEqual(llm.generate_answer("question", []), FALLBACK)
        client.assert_not_called()

    def test_stream_handles_empty_events(self):
        events = [SimpleNamespace(choices=[]), event(None), event("ANSWER"), event("ABLE")]
        client = SimpleNamespace(settings=SimpleNamespace(),
                                 complete_streaming_chat=lambda _messages: iter(events))
        with patch("llm._get_client", return_value=client):
            self.assertEqual(llm._complete("rules", {"question": "test"}, 16), "ANSWERABLE")
        self.assertEqual(client.settings.max_tokens, 16)

    def test_runtime_error_does_not_leak_unverified_answer(self):
        with patch("llm._complete", side_effect=["ANSWERABLE", "1", RuntimeError("failure")]):
            with self.assertRaisesRegex(RuntimeError, "Foundry Local answer generation failed"):
                llm.generate_answer(CASES[0][0], CHUNKS)

    def test_prompts_require_complete_explicit_support(self):
        with patch("llm._complete", side_effect=["ANSWERABLE", "1", "SUPPORTED"]) as complete:
            llm.generate_answer(CASES[0][0], CHUNKS)
        for call in complete.call_args_list:
            self.assertTrue("No guessing" in call.args[0] or "Do not guess" in call.args[0])
        self.assertIn("If any essential part of the answer is missing", complete.call_args_list[1].args[0])

    def test_high_score_does_not_bypass_answerability_or_expose_sources(self):
        with patch("database.initialize_database"), patch("database.count_chunks", return_value=1), \
             patch("database.index_uses_current_model", return_value=True), \
             patch("retriever.retrieve", return_value=CHUNKS), \
             patch("llm._complete", return_value="NOT_ANSWERABLE") as complete:
            result = rag_pipeline.query_pipeline("Who invented binary classification?")
        self.assertEqual(result["answer"], FALLBACK)
        self.assertEqual(result["sources"], [])
        complete.assert_called_once()


@unittest.skipUnless(os.getenv("RAG_LIVE_TESTS") == "1", "requires real Foundry Local models")
class LiveAnswerabilityTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        llm.unload_model()
        embeddings.unload_model()

    def test_indexed_documents_reject_missing_specific_facts(self):
        for question in (
            "Who invented binary classification?",
            "How many classes does binary classification have and who invented it?",
            "Explain everything about machine learning.",
            "Who won the 2022 FIFA World Cup?",
        ):
            with self.subTest(question=question):
                result = rag_pipeline.query_pipeline(question)
                self.assertEqual(result["answer"], FALLBACK)
                self.assertEqual(result["sources"], [])

    def test_real_model_decisions_and_answers(self):
        for question, supported in CASES:
            with self.subTest(question=question):
                answer = llm.generate_answer(question, CHUNKS)
                if supported:
                    self.assertEqual(answer, CHUNKS[0]["content"])
                else:
                    self.assertEqual(answer, FALLBACK)


if __name__ == "__main__":
    unittest.main()
