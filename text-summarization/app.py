import os
import validators
import streamlit as st
from dotenv import load_dotenv
from traceback import format_exc

# 1. Load environment and set user agent
load_dotenv()
os.environ["USER_AGENT"] = "gen-ai-course-app"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# 2. Prompt for API key if missing
if not GROQ_API_KEY:
    GROQ_API_KEY = st.sidebar.text_input("Groq API Key", type="password")
    if not GROQ_API_KEY:
        st.info("Please enter your Groq API Key.")

# 3. Initialize the Groq LLM
from langchain_groq import ChatGroq
llm = ChatGroq(
    model_name="Llama3-8b-8192",
    groq_api_key=GROQ_API_KEY,
    streaming=True
)

# 4. Streamlit UI setup
st.set_page_config(page_title="YouTube Summarizer", page_icon="🤖")
st.title("YouTube/Text Summarization")
generic_url = st.text_input("Enter URL to summarize")

def normalize_youtube_url(url: str) -> str:
    """Convert short youtu.be URLs to full watch URLs."""
    if "youtu.be/" in url:
        vid = url.split("youtu.be/")[-1].split("?")[0]
        return f"https://www.youtube.com/watch?v={vid}"
    return url

# 5. Summarization logic
if st.button("Summarize"):
    if not generic_url or not GROQ_API_KEY:
        st.error("Please enter both a URL and your Groq API Key.")
    elif not validators.url(generic_url):
        st.error("Invalid URL.")
    else:
        try:
            docs = None
            st.write("Input URL:", generic_url)

            # ── YouTube branch ──
            if "youtube.com" in generic_url or "youtu.be" in generic_url:
                normalized = normalize_youtube_url(generic_url)
                st.write("Normalized:", normalized)
                video_id = normalized.split("v=")[-1].split("&")[0]
                st.write("Video ID:", video_id)

                st.write("Fetching transcript via youtube-transcript-api...")
                from youtube_transcript_api import YouTubeTranscriptApi
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                from langchain_core.documents import Document

                try:
                    # Fetch English captions (manual or auto‑generated)
                    ytt = YouTubeTranscriptApi()
                    segments = ytt.fetch(video_id)  # returns FetchedTranscriptSnippet list :contentReference[oaicite:5]{index=5}
                    st.write(f"Fetched {len(segments)} segments")
                except Exception as ex:
                    st.error(f"Transcript fetch error: {ex}")
                    segments = []

                if segments:
                    # Join using .text attribute on each snippet :contentReference[oaicite:6]{index=6}
                    text = " ".join(seg.text for seg in segments)
                    st.write("Transcript preview:", text[:200])
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=100
                    )  # recursive chunking :contentReference[oaicite:7]{index=7}
                    chunks = splitter.split_text(text)
                    st.write("Number of chunks:", len(chunks))
                    docs = [Document(page_content=chunk) for chunk in chunks]
                else:
                    st.error("No transcript available for this video.")

            # ── Generic URL branch ──
            else:
                st.write("Loading generic URL content via UnstructuredURLLoader...")
                from langchain_community.document_loaders import UnstructuredURLLoader

                loader = UnstructuredURLLoader(
                    urls=[generic_url],
                    ssl_verify=False,
                    headers={"User-Agent": "Mozilla/5.0"},
                )  # docs loader :contentReference[oaicite:8]{index=8}
                docs = loader.load()  # returns List[Document] :contentReference[oaicite:9]{index=9}
                st.write("Docs count:", len(docs))

            # ── Summarization ──
            if docs:
                from langchain.chains.summarize import load_summarize_chain

                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="stuff",
                    verbose=True
                )  # summarization chain :contentReference[oaicite:10]{index=10}
                summary = chain.run(docs)
                st.success(summary)
            else:
                st.error("No content to summarize.")

        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.write(format_exc())
