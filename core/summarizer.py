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


def generate_title(transcript: str) -> str:
    """Generate a concise, descriptive title for the given transcript."""
    if not transcript or not transcript.strip():
        return "Untitled Video / Audio"

    llm = get_llm()
    prompt = (
        "Based on the following transcript, generate a short, clear, and descriptive title "
        "(one line only, no quotes, no extra formatting):\n\n"
        f"{transcript[:4000]}"
    )
    try:
        response = llm.invoke(prompt)
        return response.content.strip().strip('"').strip("'")
    except Exception as e:
        print(f"Error generating title: {e}")
        return "Summary of Video/Audio"


def summarize(transcript: str) -> str:
    """Generate a comprehensive summary of the given transcript."""
    if not transcript or not transcript.strip():
        return "No transcript provided."

    llm = get_llm()
    prompt = (
        "Please provide a clear, structured, and comprehensive summary of the following transcript. "
        "Highlight the key topics discussed, main takeaways, and general context:\n\n"
        f"{transcript}"
    )
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Failed to generate summary."
