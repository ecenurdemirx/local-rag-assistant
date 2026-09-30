"""Small repeatable retrieval evaluation that does not invoke the chat model."""

import atexit
import database
import embeddings

import argparse
import json

import config
import retriever


def load_cases(file_path):
    """Load and validate evaluation cases supplied by the user."""
    try:
        with open(file_path, encoding="utf-8") as stream:
            cases = json.load(stream)
    except FileNotFoundError as error:
        raise ValueError(f"Evaluation file not found: {file_path}") from error
    except OSError as error:
        raise ValueError(f"Could not read evaluation file '{file_path}': {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in evaluation file '{file_path}' at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error

    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation file must contain a non-empty JSON array.")

    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Evaluation case {index} must be a JSON object.")
        question = case.get("question")
        if "expected_source" not in case:
            raise ValueError(f"Evaluation case {index} must include expected_source (string or null).")
        expected_source = case.get("expected_source")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Evaluation case {index} must have a non-empty 'question'.")
        if expected_source is not None and (
            not isinstance(expected_source, str) or not expected_source.strip()
        ):
            raise ValueError(
                f"Evaluation case {index} 'expected_source' must be a non-empty string or null."
            )
    return cases


def evaluate(cases):
    results = []
    for case in cases:
        matches = retriever.retrieve(case["question"], top_k=config.TOP_K)
        matches = [match for match in matches if match["score"] >= config.SIMILARITY_THRESHOLD]
        sources = [match["source"] for match in matches]
        expected_source = case.get("expected_source")
        passed = not sources if expected_source is None else expected_source in sources
        results.append({**case, "retrieved_sources": sources, "passed": passed})
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate local RAG retrieval")
    parser.add_argument("--cases", help="JSON file containing evaluation cases")
    args = parser.parse_args()
    if not args.cases:
        parser.error(
            "an evaluation file is required; for the bundled demo use "
            "--cases evaluation_cases.example.json"
        )
    try:
        cases = load_cases(args.cases)
    except ValueError as error:
        parser.error(str(error))
    database.initialize_database()
    if database.count_chunks() == 0 or not database.index_uses_current_model():
        parser.error("Missing or incompatible index; run python main.py --reindex first.")
    atexit.register(embeddings.unload_model)
    results = evaluate(cases)
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        sources = ", ".join(result["retrieved_sources"]) or "no relevant source"
        print(f"{status}: {result['question']} -> {sources}")
    passed = sum(result["passed"] for result in results)
    accuracy = (passed / len(results) * 100) if results else 0.0
    print(f"\nPassed: {passed}/{len(results)}")
    print(f"Retrieval accuracy: {accuracy:.1f}%")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
