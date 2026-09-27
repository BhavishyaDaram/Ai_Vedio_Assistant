import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "open-mistral-7b")


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
    return ChatMistralAI(model=MISTRAL_MODEL, api_key=api_key)


def extract_action_items(transcript: str) -> str:
    """Extract action items and assigned tasks from the transcript."""
    if not transcript or not transcript.strip():
        return "No action items found."

    llm = get_llm()
    prompt = (
        "From the following transcript, extract all action items, tasks, and follow-ups. "
        "List them clearly as bullet points. If there are none, state 'No explicit action items found.'\n\n"
        f"{transcript}"
    )
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error extracting action items: {e}")
        return "Failed to extract action items."


def extract_key_decisions(transcript: str) -> str:
    """Extract key decisions made in the transcript."""
    if not transcript or not transcript.strip():
        return "No key decisions found."

    llm = get_llm()
    prompt = (
        "From the following transcript, extract all key decisions, agreements, or conclusions made. "
        "List them clearly as bullet points. If there are none, state 'No explicit key decisions found.'\n\n"
        f"{transcript}"
    )
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error extracting key decisions: {e}")
        return "Failed to extract key decisions."


def extract_questions(transcript: str) -> str:
    """Extract open or unanswered questions from the transcript."""
    if not transcript or not transcript.strip():
        return "No questions found."

    llm = get_llm()
    prompt = (
        "From the following transcript, extract all open questions, unresolved topics, or questions raised. "
        "List them clearly as bullet points. If there are none, state 'No open questions found.'\n\n"
        f"{transcript}"
    )
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error extracting questions: {e}")
        return "Failed to extract questions."
