import asyncio
import importlib
import os
import shutil
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Local Multimodal Research Agent",
    description="Local AI research assistant powered by RAG and Ollama",
    version="0.3.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

DOCUMENTS_DIR = "data/documents"

PROCESSED_DIR = "data/processed"

VECTORSTORE_DIR = "data/vectorstore"


# ============================================================
# PROCESSING STATE
# ============================================================

processing_status = {
    "processing": False,
    "message": "Ready",
    "document": None,
    "documents": [],
    "step": 0,
    "total_steps": 5,
}


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):

    question: str


# ============================================================
# SAFE DEFAULT answer_question
# (used before any document is uploaded)
# ============================================================

def answer_question(question):

    return {
        "answer": (
            "No document has been uploaded yet. "
            "Please upload one or more PDF files "
            "using the Upload button, then ask "
            "your question."
        ),
        "sources": []
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "Local Multimodal Research Agent"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "processing": processing_status["processing"],
    }


# ============================================================
# STATUS ENDPOINT
# ============================================================

@app.get("/status")
def get_status():

    return {
        "processing": processing_status["processing"],
        "message": processing_status["message"],
        "document": processing_status["document"],
        "documents": processing_status["documents"],
        "step": processing_status["step"],
        "total_steps": processing_status["total_steps"],
    }


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

@app.get("/document")
def document_info():

    os.makedirs(
        DOCUMENTS_DIR,
        exist_ok=True
    )

    pdf_files = [
        file
        for file in os.listdir(DOCUMENTS_DIR)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:

        return {
            "document": None,
            "documents": [],
            "status": "No document uploaded"
        }

    return {
        "document": pdf_files[0],
        "documents": pdf_files,
        "status": (
            "Processing"
            if processing_status["processing"]
            else "Ready"
        )
    }


# ============================================================
# RUN PIPELINE SCRIPT
# ============================================================

def run_pipeline_script(script_name):

    print()
    print("=" * 60)
    print(f"RUNNING: {script_name}")
    print("=" * 60)

    # Force the child Python process to use UTF-8 for stdout/stderr.
    # This prevents Windows cp1252 encoding errors when the PDF
    # contains Unicode characters such as ∗, ±, ×, etc.

    environment = os.environ.copy()

    environment["PYTHONIOENCODING"] = "utf-8"

    process = __import__("subprocess").run(
        [
            sys.executable,
            script_name
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment
    )

    if process.stdout:
        try:
            print(process.stdout)
        except UnicodeEncodeError:
            print(process.stdout.encode("cp1252", errors="replace").decode("cp1252"))

    if process.returncode != 0:

        try:
            print(process.stderr)
        except UnicodeEncodeError:
            print(process.stderr.encode("cp1252", errors="replace").decode("cp1252"))

        raise RuntimeError(
            f"{script_name} failed.\n\n"
            f"{process.stderr}"
        )

    print(
        f"{script_name} completed successfully."
    )


# ============================================================
# RELOAD SEARCH PIPELINE
# ============================================================

def reload_search_pipeline():

    global answer_question

    print()
    print("=" * 60)
    print("RELOADING SEARCH PIPELINE")
    print("=" * 60)

    import hybrid_search
    import rag

    # Reload hybrid_search first so it reads
    # the newly generated semantic chunks
    # and newly generated FAISS index.

    importlib.reload(
        hybrid_search
    )

    # Reload rag.py so its
    # "from hybrid_search import hybrid_search"
    # receives the new function.

    importlib.reload(
        rag
    )

    # Update the function used by /ask.

    answer_question = (
        rag.process_question_api
    )

    print(
        "Search pipeline reloaded successfully."
    )


# ============================================================
# BACKGROUND PROCESSING TASK
# ============================================================

async def _process_pipeline(safe_filenames):

    global processing_status

    try:

        # ----------------------------------------------------
        # Step 1: Document ingestion
        # ----------------------------------------------------

        processing_status["message"] = (
            "Step 1/4 — Extracting and structuring document..."
        )
        processing_status["step"] = 1

        await asyncio.to_thread(
            run_pipeline_script,
            "document_ingestion.py"
        )


        # ----------------------------------------------------
        # Step 2: Semantic chunking
        # ----------------------------------------------------

        processing_status["message"] = (
            "Step 2/4 — Creating semantic chunks..."
        )
        processing_status["step"] = 2

        await asyncio.to_thread(
            run_pipeline_script,
            "semantic_chunker.py"
        )


        # ----------------------------------------------------
        # Step 3: Embeddings
        # ----------------------------------------------------

        processing_status["message"] = (
            "Step 3/4 — Generating embeddings..."
        )
        processing_status["step"] = 3

        await asyncio.to_thread(
            run_pipeline_script,
            "embedder.py"
        )


        # ----------------------------------------------------
        # Step 4: FAISS vector store
        # ----------------------------------------------------

        processing_status["message"] = (
            "Step 4/4 — Building vector store..."
        )
        processing_status["step"] = 4

        await asyncio.to_thread(
            run_pipeline_script,
            "rebuild_vectorstore.py"
        )


        # ----------------------------------------------------
        # Step 5: Reload in-memory search system
        # ----------------------------------------------------

        processing_status["message"] = (
            "Finalizing — Loading new document into search..."
        )
        processing_status["step"] = 5

        await asyncio.to_thread(
            reload_search_pipeline
        )


        # ----------------------------------------------------
        # Complete
        # ----------------------------------------------------

        processing_status = {
            "processing": False,
            "message": "Ready",
            "document": ", ".join(safe_filenames),
            "documents": safe_filenames,
            "step": 5,
            "total_steps": 5,
        }


        print()
        print("=" * 60)
        print("DOCUMENT PROCESSING COMPLETE")
        print("=" * 60)


    except Exception as error:

        processing_status = {
            "processing": False,
            "message": f"Processing failed: {str(error)[:200]}",
            "document": None,
            "documents": [],
            "step": 0,
            "total_steps": 5,
        }


        print()
        print("=" * 60)
        print("DOCUMENT PROCESSING FAILED")
        print("=" * 60)

        print(
            str(error)
        )


# ============================================================
# UPLOAD PDF
# ============================================================

@app.post("/upload")
async def upload_pdf(
    files: list[UploadFile] = File(...)
):

    global processing_status

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not files:

        raise HTTPException(
            status_code=400,
            detail="No files selected."
        )

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are supported. Got: {file.filename}"
            )

    if processing_status["processing"]:

        raise HTTPException(
            status_code=409,
            detail="Another document is currently being processed. Please wait."
        )


    # --------------------------------------------------------
    # Prepare directories
    # --------------------------------------------------------

    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)


    processing_status = {
        "processing": True,
        "message": "Uploading documents...",
        "document": ", ".join(f.filename for f in files if f.filename),
        "documents": [],
        "step": 0,
        "total_steps": 5,
    }


    try:

        # ----------------------------------------------------
        # Remove previous PDFs
        # ----------------------------------------------------

        for existing_file in os.listdir(DOCUMENTS_DIR):

            existing_path = os.path.join(
                DOCUMENTS_DIR,
                existing_file
            )

            if (
                os.path.isfile(existing_path)
                and
                existing_file.lower().endswith(".pdf")
            ):

                os.remove(existing_path)


        # ----------------------------------------------------
        # Save uploaded PDFs
        # ----------------------------------------------------

        safe_filenames = []

        for file in files:
            safe_filename = os.path.basename(
                file.filename
            )
            safe_filenames.append(safe_filename)

            pdf_path = os.path.join(
                DOCUMENTS_DIR,
                safe_filename
            )

            with open(pdf_path, "wb") as output_file:

                shutil.copyfileobj(
                    file.file,
                    output_file
                )


        print()
        print("=" * 60)
        print("PDFs UPLOADED")
        print("=" * 60)

        for name in safe_filenames:
            print(f"File: {name}")


        processing_status["message"] = "Uploaded. Starting pipeline..."
        processing_status["documents"] = safe_filenames

        # ----------------------------------------------------
        # Run pipeline in background so HTTP response returns
        # immediately and the client can poll /status.
        # ----------------------------------------------------

        asyncio.create_task(
            _process_pipeline(safe_filenames)
        )

        return {
            "success": True,
            "document": ", ".join(safe_filenames),
            "documents": safe_filenames,
            "message": (
                "Documents uploaded. Processing started in background. "
                "Poll /status to track progress."
            )
        }


    except Exception as error:

        processing_status = {
            "processing": False,
            "message": "Upload failed",
            "document": None,
            "documents": [],
            "step": 0,
            "total_steps": 5,
        }


        print()
        print("=" * 60)
        print("UPLOAD FAILED")
        print("=" * 60)

        print(str(error))


        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# QUESTION ENDPOINT
# ============================================================

@app.post("/ask")
def ask_question_api(
    request: QuestionRequest
):

    question = request.question.strip()


    if not question:

        return {
            "answer": "Please enter a question.",
            "sources": []
        }


    if processing_status["processing"]:

        return {
            "answer": (
                "The document is still being processed. "
                f"Current step: {processing_status['message']} "
                "Please wait until processing is complete."
            ),
            "sources": []
        }


    try:

        result = answer_question(
            question
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


    return {
        "question": question,

        "answer": result["answer"],

        "sources": [

            {
                "page": source.get("page"),

                "chunk_id": source.get("chunk_id"),

                "section": source.get("section"),

                "subsection": source.get("subsection"),

                "region": source.get("region"),

                "score": source.get("final_score"),

                "source": source.get("source"),

                "document": source.get("source"),
            }

            for source in result["sources"]

        ]
    }