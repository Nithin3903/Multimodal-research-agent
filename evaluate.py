import subprocess
import sys
import re
from dataclasses import dataclass
from typing import List, Optional


# ============================================================
# CONFIGURATION
# ============================================================

RAG_SCRIPT = "rag.py"


# ============================================================
# TEST CASE
# ============================================================

@dataclass
class TestCase:
    name: str
    question: str
    expected_terms: List[str]
    category: str


# ============================================================
# BENCHMARK
# ============================================================

TEST_CASES = [

    TestCase(
        name="Main objective",
        question=(
            "What is the main objective of the research paper?"
        ),
        expected_terms=[
            "HAISAS",
            "aquaponics",
            "IoT",
            "fish health"
        ],
        category="Text"
    ),

    TestCase(
        name="Table IV",
        question=(
            "Which model performs best in Table IV, "
            "and what does the paper say about its performance?"
        ),
        expected_terms=[
            "XGBoost",
            "98%",
            "0.94"
        ],
        category="Table"
    ),

    TestCase(
        name="Page 1 image",
        question=(
            "What is shown in the image on page 1?"
        ),
        expected_terms=[
            "Hindi",
            "educational"
        ],
        category="Visual"
    ),

    TestCase(
        name="Model selection",
        question=(
            "Why was XGBoost selected even though "
            "Random Forest had a higher cross-validation score?"
        ),
        expected_terms=[
            "Random Forest",
            "XGBoost",
            "inference",
            "regularized"
        ],
        category="Multimodal"
    ),

    TestCase(
        name="Document comparison",
        question=(
            "Compare the documents available in the system "
            "and explain what each document is about."
        ),
        expected_terms=[
            "New_research_paper.pdf",
            "DocScanner",
            "research",
            "educational"
        ],
        category="Document Comparison"
    ),

]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:

    text = text.lower()

    # Treat underscores/backslashes as spaces.
    text = text.replace("_", " ")
    text = text.replace("\\", " ")

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# RUN ONE TEST
# ============================================================

def run_single_question(
    question: str
) -> str:

    """
    Run rag.py in a fresh process for one question.

    This prevents output from one benchmark question
    from getting mixed with another benchmark question.
    """

    input_data = (
        question
        + "\n"
        + "exit\n"
    )

    try:

        result = subprocess.run(
            [
                sys.executable,
                RAG_SCRIPT
            ],
            input=input_data,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace"
        )

    except Exception as error:

        return (
            "EVALUATION_RUN_ERROR\n"
            f"{type(error).__name__}: {error}"
        )

    output = result.stdout

    if result.stderr:

        output += (
            "\n\nSTDERR:\n"
            + result.stderr
        )

    return output


# ============================================================
# EXTRACT ANSWER
# ============================================================

def extract_answer(
    output: str
) -> Optional[str]:

    """
    Extract the answer from ONE rag.py execution.

    Handles both:
      ANSWER
    and:
      DOCUMENT COMPARISON
    """

    # --------------------------------------------------------
    # Standard ANSWER block
    # --------------------------------------------------------

    standard_pattern = re.compile(
        r"={10,}\s*"
        r"ANSWER"
        r"\s*={10,}"
        r"(.*?)"
        r"(?=\
={10,}\s*SOURCES USED\s*={10,}"
        r"|\
={10,}\s*DOCUMENTS USED\s*={10,}"
        r"|Ask a question"
        r"|$)",
        flags=re.IGNORECASE | re.DOTALL
    )

    standard_matches = standard_pattern.findall(
        output
    )

    if standard_matches:

        answer = standard_matches[-1].strip()

        if answer:
            return answer

    # --------------------------------------------------------
    # Document comparison block
    # --------------------------------------------------------

    comparison_pattern = re.compile(
        r"={10,}\s*"
        r"DOCUMENT COMPARISON"
        r"\s*={10,}"
        r"(.*?)"
        r"(?=\
={10,}\s*DOCUMENTS USED\s*={10,}"
        r"|Ask a question"
        r"|$)",
        flags=re.IGNORECASE | re.DOTALL
    )

    comparison_matches = (
        comparison_pattern.findall(
            output
        )
    )

    if comparison_matches:

        answer = (
            comparison_matches[-1]
            .strip()
        )

        if answer:
            return answer

    return None


# ============================================================
# CHECK EXPECTED TERMS
# ============================================================

def evaluate_terms(
    answer: str,
    expected_terms: List[str]
):

    normalized_answer = normalize_text(
        answer
    )

    matched = []
    missing = []

    for term in expected_terms:

        normalized_term = normalize_text(
            term
        )

        if normalized_term in normalized_answer:

            matched.append(
                term
            )

        else:

            missing.append(
                term
            )

    if expected_terms:

        score = (
            len(matched)
            /
            len(expected_terms)
        )

    else:

        score = 0.0

    # 70% evidence threshold.
    passed = score >= 0.70

    return {
        "passed": passed,
        "score": score,
        "matched": matched,
        "missing": missing
    }


# ============================================================
# PRINT INDIVIDUAL RESULT
# ============================================================

def print_test_result(
    index: int,
    test: TestCase,
    answer: Optional[str],
    evaluation: dict
):

    status = (
        "PASS"
        if evaluation["passed"]
        else "FAIL"
    )

    print()
    print("=" * 70)
    print(
        f"TEST {index}: {test.name}"
    )
    print("=" * 70)

    print(
        f"Category : {test.category}"
    )

    print(
        f"Question : {test.question}"
    )

    print()

    print(
        f"Result   : {status}"
    )

    print(
        f"Score    : "
        f"{evaluation['score'] * 100:.1f}%"
    )

    print()

    print(
        "Matched:"
    )

    if evaluation["matched"]:

        print(
            ", ".join(
                evaluation["matched"]
            )
        )

    else:

        print(
            "None"
        )

    print()

    print(
        "Missing:"
    )

    if evaluation["missing"]:

        print(
            ", ".join(
                evaluation["missing"]
            )
        )

    else:

        print(
            "None"
        )

    print()

    print(
        "Answer:"
    )

    if answer:

        print(
            answer
        )

    else:

        print(
            "No answer detected."
        )


# ============================================================
# MEMORY TEST
# ============================================================

def run_memory_test():

    """
    Memory needs TWO questions in the SAME rag.py process.

    Q1 establishes the entity.
    Q2 uses a pronoun/reference.
    """

    first_question = (
        "Which model performs best in Table IV?"
    )

    second_question = (
        "Why was it selected?"
    )

    input_data = (
        first_question
        + "\n"
        + second_question
        + "\n"
        + "exit\n"
    )

    try:

        result = subprocess.run(
            [
                sys.executable,
                RAG_SCRIPT
            ],
            input=input_data,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace"
        )

    except Exception as error:

        return None, (
            f"{type(error).__name__}: {error}"
        )

    output = result.stdout

    # --------------------------------------------------------
    # Extract all normal answer blocks.
    # --------------------------------------------------------

    pattern = re.compile(
        r"={10,}\s*"
        r"ANSWER"
        r"\s*={10,}"
        r"(.*?)"
        r"(?=\
={10,}\s*SOURCES USED\s*={10,}"
        r"|\
Ask a question"
        r"|$)",
        flags=re.IGNORECASE | re.DOTALL
    )

    answers = [
        item.strip()
        for item in pattern.findall(
            output
        )
        if item.strip()
    ]

    if len(answers) < 2:

        return None, (
            "Could not extract both memory-test answers."
        )

    # The second answer is the one we evaluate.
    return answers[-1], None


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RAG EVALUATION")
    print("=" * 70)

    print(
        f"Agent: {RAG_SCRIPT}"
    )

    print(
        f"Tests: {len(TEST_CASES)}"
    )

    results = []

    # --------------------------------------------------------
    # Run normal tests independently.
    # --------------------------------------------------------

    for index, test in enumerate(
        TEST_CASES,
        start=1
    ):

        print()
        print(
            f"Running test {index}/{len(TEST_CASES)}: "
            f"{test.name}"
        )

        output = run_single_question(
            test.question
        )

        answer = extract_answer(
            output
        )

        if answer is None:

            evaluation = {
                "passed": False,
                "score": 0.0,
                "matched": [],
                "missing": test.expected_terms
            }

        else:

            evaluation = evaluate_terms(
                answer,
                test.expected_terms
            )

        results.append(
            {
                "test": test,
                "answer": answer,
                "evaluation": evaluation
            }
        )

        print_test_result(
            index,
            test,
            answer,
            evaluation
        )

    # --------------------------------------------------------
    # Memory test separately.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MEMORY TEST")
    print("=" * 70)

    memory_answer, memory_error = (
        run_memory_test()
    )

    memory_expected = [
        "XGBoost",
        "inference"
    ]

    if memory_error:

        memory_evaluation = {
            "passed": False,
            "score": 0.0,
            "matched": [],
            "missing": memory_expected
        }

        print(
            "Memory test error:",
            memory_error
        )

    else:

        memory_evaluation = evaluate_terms(
            memory_answer or "",
            memory_expected
        )

    memory_test = TestCase(
        name="Conversation memory",
        question="Why was it selected?",
        expected_terms=memory_expected,
        category="Memory"
    )

    results.append(
        {
            "test": memory_test,
            "answer": memory_answer,
            "evaluation": memory_evaluation
        }
    )

    print_test_result(
        len(results),
        memory_test,
        memory_answer,
        memory_evaluation
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total = len(results)

    passed = sum(
        1
        for result in results
        if result["evaluation"]["passed"]
    )

    failed = total - passed

    accuracy = (
        passed
        /
        total
        *
        100
        if total
        else 0
    )

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Total tests      : {total}"
    )

    print(
        f"Passed           : {passed}"
    )

    print(
        f"Failed           : {failed}"
    )

    print(
        f"Overall accuracy : {accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # Category results.
    # --------------------------------------------------------

    categories = {}

    for result in results:

        category = result[
            "test"
        ].category

        if category not in categories:

            categories[
                category
            ] = {
                "total": 0,
                "passed": 0
            }

        categories[
            category
        ]["total"] += 1

        if result[
            "evaluation"
        ]["passed"]:

            categories[
                category
            ]["passed"] += 1

    print()
    print(
        "CATEGORY PERFORMANCE"
    )

    print(
        "-" * 70
    )

    for category, data in categories.items():

        category_accuracy = (
            data["passed"]
            /
            data["total"]
            *
            100
        )

        print(
            f"{category:<24}"
            f"{data['passed']}/"
            f"{data['total']} "
            f"({category_accuracy:.1f}%)"
        )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()