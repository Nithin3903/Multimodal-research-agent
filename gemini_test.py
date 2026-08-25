import os

from dotenv import load_dotenv
from google import genai


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found in .env"
    )


# ============================================================
# CREATE CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "gemini-3.1-flash-lite"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("GEMINI API TEST")
print("=" * 70)

print(
    "\nAPI key detected successfully."
)

print(
    "API key is NOT being printed for security."
)

print(
    f"\nModel: {MODEL_NAME}"
)

print(
    "\nSending test request..."
)


# ============================================================
# GEMINI REQUEST
# ============================================================

try:

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=(
            "Explain Retrieval Augmented Generation "
            "(RAG) in two simple sentences."
        )
    )

    print("\nGemini response:")
    print("-" * 70)

    print(
        interaction.output_text
    )

    print("-" * 70)

    print(
        "\nGEMINI API TEST SUCCESSFUL"
    )


except Exception as exc:

    print("\n")
    print("=" * 70)
    print("GEMINI API TEST FAILED")
    print("=" * 70)

    print(
        f"\nError:\n{exc}"
    )