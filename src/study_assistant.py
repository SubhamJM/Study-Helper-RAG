from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os

load_dotenv()

from src.RAGRetriver import get_retriever
from langchain_google_genai import ChatGoogleGenerativeAI


class MCQ(BaseModel):
    question: str
    options: List[str]
    correct_answer_index: int
    explanation: str


class StudyResponse(BaseModel):
    summary: str
    mcqs: Optional[List[MCQ]] = None


class SummaryResponse(BaseModel):
    summary: str


_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
    return _llm


def _format_context(context: list) -> str:
    return "\n\n".join(
        f"Source (page {c['metadata'].get('page', 'N/A')}, rank {c['rank']}):\n{c['content']}"
        for c in context
    )


def get_study_material(topic: str, include_mcqs: bool = False) -> StudyResponse:
    if not topic or not topic.strip():
        return StudyResponse(summary="Please enter a valid topic.", mcqs=None)

    context = get_retriever().retrive_context(topic)

    if not context:
        return StudyResponse(
            summary=f"No information found about '{topic}' in the textbook. Try a different topic.",
            mcqs=None
        )

    context_text = _format_context(context)

    base_prompt = (
        "You are a Theory of Computation tutor. Based strictly on the context below "
        "from the textbook \"Introduction to Automata Theory, Languages, and Computation\" "
        f"by Hopcroft, Motwani, and Ullman, answer the following.\n\n"
        f"Topic: {topic}\n\n"
        f"Context from textbook:\n{context_text}\n\n"
        "Instructions:\n"
        "1. Provide a comprehensive summary of this topic based ONLY on the provided context.\n"
        "2. The summary should be detailed and structured, covering key concepts, definitions, "
        "and examples from the text.\n"
    )

    llm = _get_llm()

    if include_mcqs:
        mcq_instruction = (
            "3. Also create exactly 5 multiple choice questions based on the topic and context above.\n"
            "   Each MCQ must have:\n"
            "   - A clear question\n"
            "   - 4 options (A, B, C, D)\n"
            "   - The index of the correct option (0-based, where 0=A, 1=B, 2=C, 3=D)\n"
            "   - A brief explanation of why the correct answer is right\n"
            "   - Questions should test understanding, not just memorization.\n"
        )
        prompt = base_prompt + mcq_instruction
        structured_llm = llm.with_structured_output(StudyResponse)
        return structured_llm.invoke(prompt)

    prompt = base_prompt
    structured_llm = llm.with_structured_output(SummaryResponse)
    result = structured_llm.invoke(prompt)
    return StudyResponse(summary=result.summary, mcqs=None)
