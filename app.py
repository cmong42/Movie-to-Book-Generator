# app.py ---------------------------------------------------------------
import os
import json
import requests
from mediawiki import MediaWiki # <-- needed for Wikipedia lookup
import wikipediaapi   # <-- needed for Wikipedia lookup
from flask import (
    Flask,
    render_template,
    request,
    Response,
    stream_with_context,
    jsonify,
)
from dotenv import load_dotenv

load_dotenv()  # loads .env for any environment vars you may want later

app = Flask(__name__)

# ----------------------------------------------------------------------
# Build prompt ----------------------------------------------------------
def build_prompt(page_summary: str, fmt: str, pages: int, instr: str) -> str:
    """
    The prompt now contains the *summary* of the Wikipedia article,
    not just the URL.
    """
    prompt = (
        f"Read the following summary of a Wikipedia article, "
        f"and generate the text for a {fmt}. "
        f"Ensure your output contains at least {pages*250} words, and is written in a style that is appropriate for the format."
        f"Ensure your output is detailed, and engaging. Try to avoid outputting simple plot summaries, opting for a more creative and deeper experience."
        f"Also, ensure your output has originality and emotional depth to it, and is not just a rehash of the Wikipedia summary. Add elements such as character development, and dialogue, conveying the emotions of the characters and offering a view into their innermost thoughts."
        f"Avoid using generic phrases and cliches, and instead focus on creating a unique and compelling narrative that captures the essence of the story."
        f"In addition, avoid using simplistic language and repetitive wording. Instead use rich and descriptive language that paints a vivid picture of the story's world and characters."
        f"Again, your output MUST BE AT LEAST {pages*250} WORDS."
    )
    if instr:
        prompt += (
            f", and ensure the following instructions are taken into account when generating your text: {instr}"
        )
    prompt += ".\n\n--- Article summary ---\n"
    prompt += page_summary
    return prompt

# ----------------------------------------------------------------------
# Ollama streaming -----------------------------------------------
def stream_ollama(prompt: str):
    """Connect to Ollama, stream the response and yield only the `response` field."""
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {"model": "llama3", "prompt": prompt, "stream": True}

    try:
        with requests.post(url, headers=headers, json=data, stream=True, timeout=120) as r:
            r.raise_for_status()

            for raw in r.iter_lines():
                if not raw:
                    continue
                try:
                    chunk = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if "response" in chunk:
                    yield chunk["response"]

    except Exception as e:
        yield f"Ollama request failed: {e}\n"

# ----------------------------------------------------------------------
# Wikipedia helper -------------------------------------------------------
def get_wikipedia_summary(title: str) -> str:
    """
    Return the summary of the Wikipedia page (first paragraph).
    Returns empty string if the page does not exist.
    """
    wiki = wikipediaapi.Wikipedia(user_agent="Movie-to-Book-Generator/1.0", language="en")
    page = wiki.page(title)
    if not page.exists():
        return ""
    return page.summary.strip()

# ----------------------------------------------------------------------
# Routes ---------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ----------------------------------------------------------------------
# Wikipedia lookup endpoint --------------------------------------------
@app.route("/wikipedia_lookup")
def wikipedia_lookup():
    """
    Accepts `q=` query param.  Returns a JSON list of up to 5
    article titles that match the query.
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])

    wiki = MediaWiki()
    # WikipediaAPI has a fast search method
    results = wiki.search(query, results=5)
    return jsonify(results)

# ----------------------------------------------------------------------
# Streaming endpoint --------------------------------------------
@app.route("/generate-stream", methods=["POST"])
def generate_stream():
    article_title = request.form.get("wikipedia_article", "").strip()
    num_pages = request.form.get("num_pages", "").strip()
    fmt = request.form.get("format", "").strip()
    instructions = request.form.get("additional-instructions", "").strip()

    if not all([article_title, num_pages, fmt]):
        return Response("Missing required fields", status=400)
    try:
        pages_int = int(num_pages)
    except ValueError:
        return Response("Number of pages must be an integer", status=400)

    # Fetch the Wikipedia summary
    page_summary = get_wikipedia_summary(article_title)
    if not page_summary:
        return Response(f"Could not find Wikipedia page '{article_title}'.", status=404)

    prompt = build_prompt(page_summary, fmt, pages_int, instructions)

    def generator():
        yield "--- Ollama start ---\n"
        for chunk in stream_ollama(prompt):
            yield chunk
        yield "\n--- Ollama end ---\n"

    return Response(stream_with_context(generator()), mimetype="text/plain; charset=utf-8")

# ----------------------------------------------------------------------
# Run ---------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)