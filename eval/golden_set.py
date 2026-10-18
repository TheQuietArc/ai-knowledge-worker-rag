"""
The golden test set: questions whose correct answers we know in advance.

Each question has:
  keywords          — short strings that MUST appear in the retrieved context for the
                      question to be answerable. Retrieval is scored by whether (and how
                      high) chunks containing these keywords are ranked. Keep them short
                      and lowercase; matching is case-insensitive substring search.
  reference_answer  — the correct answer in plain words, used by the LLM judge.
  category          — which kind of question it is, so we can see where the system is
                      weak (e.g. direct facts vs. multi-fact questions).

Pydantic validates every entry at import time, so a typo in this file fails loudly
instead of silently skewing the metrics.
"""

from pydantic import BaseModel, Field, field_validator


class TestQuestion(BaseModel):
    __test__ = False  # stop pytest from mistaking this model for a test class

    question: str = Field(description="The question to ask the system")
    keywords: list[str] = Field(min_length=1, description="Must appear in relevant chunks")
    reference_answer: str = Field(description="The correct answer, for the LLM judge")
    category: str = Field(description="direct_fact | multi_fact | descriptive")

    @field_validator("keywords")
    @classmethod
    def lowercase_keywords(cls, v: list[str]) -> list[str]:
        return [k.lower().strip() for k in v]


GOLDEN_SET: list[TestQuestion] = [
    # --- Store Intelligence System: direct facts ---
    TestQuestion(
        question="What object detection model does the Store Intelligence System use?",
        keywords=["yolov8n"],
        reference_answer="It uses YOLOv8n for object detection.",
        category="direct_fact",
    ),
    TestQuestion(
        question="Which algorithm tracks people across video frames in the Store Intelligence System?",
        keywords=["bytetrack"],
        reference_answer="ByteTrack is used for object tracking.",
        category="direct_fact",
    ),
    TestQuestion(
        question="How does the system re-identify the same person?",
        keywords=["histogram"],
        reference_answer="It uses colour histogram-based re-identification (Re-ID).",
        category="direct_fact",
    ),
    TestQuestion(
        question="What database does the Store Intelligence System use?",
        keywords=["sqlite"],
        reference_answer="It uses SQLite.",
        category="direct_fact",
    ),
    TestQuestion(
        question="How are real-time dashboard updates delivered?",
        keywords=["server-sent events"],
        reference_answer="Through a Server-Sent Events (SSE) dashboard.",
        category="direct_fact",
    ),
    TestQuestion(
        question="Which tech challenge was the Store Intelligence System originally built for?",
        keywords=["purplle"],
        reference_answer="It was originally built for the Purplle Tech Challenge 2026.",
        category="direct_fact",
    ),
    # --- Store Intelligence System: multi-fact ---
    TestQuestion(
        question="Which framework powers the backend API and how many endpoints does it expose?",
        keywords=["fastapi", "9 endpoints"],
        reference_answer="The backend API is built with FastAPI and has 9 endpoints.",
        category="multi_fact",
    ),
    TestQuestion(
        question="How is the Store Intelligence System deployed and how many tests does it have?",
        keywords=["docker compose", "33"],
        reference_answer="It is deployed with Docker Compose and has 33 pytest test cases.",
        category="multi_fact",
    ),
    # --- Store Intelligence System: descriptive ---
    TestQuestion(
        question="What are the known limitations of the dashboard?",
        keywords=["simulated events"],
        reference_answer=(
            "The live dashboard uses simulated events for demo purposes rather than "
            "live CCTV footage."
        ),
        category="descriptive",
    ),
    TestQuestion(
        question="What bug did Ashutosh catch where the dashboard showed outdated numbers?",
        keywords=["stale"],
        reference_answer=(
            "The video pipeline's data went stale but the dashboard kept showing old "
            "values, such as customers in a full queue when the store was empty. He "
            "noticed by manually comparing the live footage with the dashboard."
        ),
        category="descriptive",
    ),
    # --- Retail sales analysis ---
    TestQuestion(
        question="What kind of analysis does the retail sales project perform?",
        keywords=["rfm"],
        reference_answer="RFM (Recency, Frequency, Monetary) analysis of retail sales data.",
        category="direct_fact",
    ),
    TestQuestion(
        question="How many transactions did the retail analysis cover?",
        keywords=["1 million"],
        reference_answer="Over 1 million transactions.",
        category="direct_fact",
    ),
    # --- Profile ---
    TestQuestion(
        question="Which university does Ashutosh study at?",
        keywords=["vit bhopal"],
        reference_answer="He studies B.Tech Computer Science at VIT Bhopal.",
        category="direct_fact",
    ),
    TestQuestion(
        question="Which roles is Ashutosh targeting?",
        keywords=["data analyst", "software development engineer"],
        reference_answer="Software Development Engineer (SDE) and Data Analyst roles.",
        category="multi_fact",
    ),
    TestQuestion(
        question="What club roles does Ashutosh hold at VIT Bhopal?",
        keywords=["finance manager", "content co-lead"],
        reference_answer=(
            "Finance Manager and Content Co-Lead at the Software Development Club."
        ),
        category="multi_fact",
    ),
    # --- Skills and learning ---
    TestQuestion(
        question="Which framework was recommended as the next one to learn?",
        keywords=["django"],
        reference_answer="Django, given his existing Python/FastAPI base.",
        category="direct_fact",
    ),
    TestQuestion(
        question="Which NPTEL certifications does Ashutosh have?",
        keywords=["marketing analytics", "introduction to machine learning"],
        reference_answer=(
            "NPTEL Marketing Analytics (May 2026) and NPTEL Introduction to Machine "
            "Learning (May 2025)."
        ),
        category="multi_fact",
    ),
    TestQuestion(
        question="What did the resume reviewer recommend?",
        keywords=["java", "cloud certification"],
        reference_answer=(
            "Focus on Java as a primary language, add a project, and pursue a cloud "
            "certification (AWS, Azure, or GCP)."
        ),
        category="multi_fact",
    ),
]
