# AI Video Assistant

An AI-powered video intelligence application that transforms video and audio content into searchable knowledge using **Speech-to-Text, Large Language Models, and Retrieval-Augmented Generation (RAG)**.

The application allows users to process video content, generate transcripts and AI-powered insights, and interact with the processed content through a context-aware question-answering interface.

## Overview

Long-form videos such as lectures, meetings, interviews, webinars, and educational content often contain valuable information that is difficult to search manually.

**AI Video Assistant** solves this by converting video content into structured, searchable information.

The pipeline processes the input through multiple stages:

**Video / Audio → Audio Processing → Transcription → Summarization → Insight Extraction → Embeddings → Vector Store → RAG → AI Q&A**

This enables users to move from passively watching videos to actively querying their content.

## Key Features

* **Video & Audio Processing**

  * Process video/audio inputs through an automated pipeline.
  * Supports media processing and audio extraction.

* **Automatic Speech-to-Text**

  * Uses OpenAI Whisper for transcription.
  * Converts spoken content into searchable text.

* **AI Summarization**

  * Generates concise summaries from processed transcripts.
  * Helps users understand long-form content quickly.

* **AI-Generated Titles**

  * Generates meaningful titles based on the content of the transcript.

* **Action Item Extraction**

  * Identifies actionable tasks discussed within the content.

* **Key Decision Extraction**

  * Extracts important decisions and conclusions from the transcript.

* **Question Extraction**

  * Identifies important or unresolved questions from the processed content.

* **RAG-Powered Question Answering**

  * Converts transcript chunks into vector embeddings.
  * Stores embeddings in ChromaDB.
  * Retrieves relevant context for user queries.
  * Generates answers using retrieved transcript context.

* **Interactive Streamlit Interface**

  * Provides a user-friendly interface for processing content and interacting with the AI assistant.

* **Export Support**

  * Includes PDF/TXT generation capabilities for processed information.

## RAG Architecture

The core of the application is the Retrieval-Augmented Generation pipeline.

```text
                ┌─────────────────────┐
                │   Video / Audio      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Audio Processing    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Whisper STT        │
                │   Transcription      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     Transcript       │
                └──────────┬──────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        Summarization   Extraction    RAG Pipeline
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ Text Chunking       │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ HuggingFace         │
                              │ Embeddings          │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ ChromaDB Vector     │
                              │ Store               │
                              └──────────┬──────────┘
                                         │
                              User Question
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ Similarity Search   │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ Retrieved Context   │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ Mistral LLM         │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              Context-Aware Answer
```

## How RAG Works

1. The video/audio content is processed and transcribed.
2. The transcript is divided into smaller chunks.
3. Each chunk is converted into a numerical vector using Hugging Face embeddings.
4. The embeddings are stored in a local ChromaDB vector store.
5. When a user asks a question, the system searches for the most relevant transcript chunks.
6. The retrieved context is passed to the language model.
7. The LLM generates a response based on the retrieved information.

This approach allows the assistant to answer questions using the processed video content instead of relying only on the model's general knowledge.

## Tech Stack

| Category                 | Technology                         |
| ------------------------ | ---------------------------------- |
| Programming Language     | Python                             |
| UI                       | Streamlit                          |
| Speech-to-Text           | OpenAI Whisper                     |
| LLM                      | Mistral AI                         |
| LLM Orchestration        | LangChain                          |
| Embeddings               | Hugging Face Sentence Transformers |
| Vector Database          | ChromaDB                           |
| Audio Processing         | PyDub, FFmpeg                      |
| Video/YouTube Processing | yt-dlp                             |
| ML Backend               | PyTorch                            |
| Translation              | deep-translator                    |
| Document Export          | ReportLab, FPDF2                   |
| Environment Management   | python-dotenv                      |

## Project Structure

```text
Ai_Vedio_Assistant/
│
├── app.py
├── main.py
├── test.py
├── test_streamlit.py
├── requirements.txt
├── .gitignore
│
├── core/
│   ├── transcriber.py
│   ├── summarizer.py
│   ├── extractor.py
│   ├── rag_engine.py
│   └── vector_store.py
│
└── utils/
    └── audio_processor.py
```

### Core Components

**`app.py`**

* Streamlit application interface
* Connects the different processing stages
* Provides the interactive user experience

**`core/transcriber.py`**

* Handles Whisper-based speech transcription.

**`core/summarizer.py`**

* Generates AI-powered titles and summaries.

**`core/extractor.py`**

* Extracts action items, key decisions, and questions.

**`core/vector_store.py`**

* Generates embeddings and manages vector storage.

**`core/rag_engine.py`**

* Builds the RAG pipeline.
* Retrieves relevant transcript context.
* Generates context-aware answers.

**`utils/audio_processor.py`**

* Handles audio/video preprocessing and input processing.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/BhavishyaDaram/Ai_Vedio_Assistant.git
cd Ai_Vedio_Assistant
```

### 2. Create a virtual environment

#### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

FFmpeg is required for audio/video processing.

Verify the installation:

```bash
ffmpeg -version
```

### 5. Configure environment variables

Create a `.env` file in the project root and add the required API credentials.

```env
MISTRAL_API_KEY=your_api_key_here
```

Do not commit your `.env` file or expose API keys publicly.

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Usage

### Step 1 — Provide Video/Audio Input

Upload or provide the supported video/audio content through the application.

### Step 2 — Process the Content

The application processes the input and performs speech-to-text transcription.

### Step 3 — Generate Insights

The system generates:

* Transcript
* AI-generated title
* Summary
* Action items
* Key decisions
* Questions

### Step 4 — Ask Questions

Use the RAG-powered chat interface to ask questions about the processed content.

For example:

```text
What are the main topics discussed?

What decisions were made?

Explain the concept discussed in the second section.

What action items were mentioned?

What was the conclusion?
```

## Why RAG?

A traditional LLM can answer general questions, but it does not automatically have access to the specific information contained inside a user's video.

RAG provides a way to connect the language model with the application's own knowledge source.

In this project:

```text
Video
   ↓
Transcript
   ↓
Embeddings
   ↓
ChromaDB
   ↓
Relevant Context
   ↓
LLM
   ↓
Answer
```

This makes the assistant more useful for content-specific question answering.

## Use Cases

* Educational video analysis
* Lecture summarization
* Meeting analysis
* Interview analysis
* Webinar analysis
* YouTube content exploration
* Podcast transcription
* Knowledge extraction from long-form content
* Interactive question answering over video content

## Engineering Concepts Demonstrated

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation
* Vector databases
* Semantic search
* Text embeddings
* Large Language Models
* Speech-to-text systems
* Prompt-based information extraction
* LangChain pipelines
* Streamlit application development
* Audio/video preprocessing
* Modular Python architecture
* AI-powered document generation

## Future Improvements

Potential improvements include:

* Timestamp-aware answers
* Source citations for retrieved transcript sections
* Multi-video knowledge bases
* Persistent user sessions
* Advanced reranking for retrieval
* Multimodal video understanding
* Speaker identification
* Cloud deployment optimization
* Improved evaluation metrics for RAG retrieval and answer quality

## Project Goal

The goal of this project was to explore how **Generative AI and Retrieval-Augmented Generation can be applied to unstructured video content** and turn long-form media into an interactive knowledge source.

Instead of simply watching a video, users can **search, summarize, extract insights, and interact with its content using natural language.**
---

If you found this project useful, feel free to explore the repository and share your feedback.
