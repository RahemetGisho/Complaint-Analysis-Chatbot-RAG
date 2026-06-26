from typing import Optional

import gradio as gr

from src.config import (
    ENABLE_METADATA_FILTERING,
    GENERATION_BACKEND,
    TOP_K,
)
from src.rag_pipeline import RAGPipeline

PRODUCT_CATEGORIES = [
    "All products",
    "Credit Card",
    "Personal Loan",
    "Savings Account",
    "Money Transfer",
]

EXAMPLE_QUESTIONS = [
    "Why are customers unhappy with money transfers?",
    "Most common credit card billing disputes?",
    "Unauthorized charges on savings accounts?",
    "Complaints about closing an account?",
]

REDACTION_TEXTURE_B64 = "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0ODAiIGhlaWdodD0iMTYwIiB2aWV3Qm94PSIwIDAgNDgwIDE2MCI+CjxyZWN0IHg9IjE2NSIgeT0iMzgiIHdpZHRoPSI3NCIgaGVpZ2h0PSI0IiByeD0iMiIgZmlsbD0iI0Y1QTYyMyIgb3BhY2l0eT0iMC4xMiIvPgo8cmVjdCB4PSIzNyIgeT0iMTM3IiB3aWR0aD0iMzYiIGhlaWdodD0iNCIgcng9IjIiIGZpbGw9IiNGNUE2MjMiIG9wYWNpdHk9IjAuMDkiLz4KPHJlY3QgeD0iMjkiIHk9IjEyOSIgd2lkdGg9IjUxIiBoZWlnaHQ9IjQiIHJ4PSIyIiBmaWxsPSIjRjVBNjIzIiBvcGFjaXR5PSIwLjA1Ii8+CjxyZWN0IHg9IjIyMiIgeT0iMTA3IiB3aWR0aD0iMzIiIGhlaWdodD0iNCIgcng9IjIiIGZpbGw9IiNGNUE2MjMiIG9wYWNpdHk9IjAuMDgiLz4KPHJlY3QgeD0iMjgyIiB5PSIxMDgiIHdpZHRoPSIzMSIgaGVpZ2h0PSI0IiByeD0iMiIgZmlsbD0iI0Y1QTYyMyIgb3BhY2l0eT0iMC4xNCIvPgo8cmVjdCB4PSI2MyIgeT0iNTciIHdpZHRoPSIzMSIgaGVpZ2h0PSI0IiByeD0iMiIgZmlsbD0iI0Y1QTYyMyIgb3BhY2l0eT0iMC4xMSIvPgo8cmVjdCB4PSIyMDMiIHk9IjEyIiB3aWR0aD0iNTIiIGhlaWdodD0iNCIgcng9IjIiIGZpbGw9IiNGNUE2MjMiIG9wYWNpdHk9IjAuMDYiLz4KPHJlY3QgeD0iNDM5IiB5PSIzNCIgd2lkdGg9IjYxIiBoZWlnaHQ9IjQiIHJ4PSIyIiBmaWxsPSIjRjVBNjIzIiBvcGFjaXR5PSIwLjEiLz4KPHJlY3QgeD0iMjc2IiB5PSIzMCIgd2lkdGg9IjYzIiBoZWlnaHQ9IjQiIHJ4PSIyIiBmaWxsPSIjRjVBNjIzIiBvcGFjaXR5PSIwLjExIi8+Cjwvc3ZnPg=="
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.gray,
).set(
    body_background_fill="#F7F5F0",
    block_background_fill="#FFFFFF",
    block_border_color="#E5E1D8",
    block_radius="14px",
    button_primary_background_fill="#F5A623",
    button_primary_background_fill_hover="#E0951A",
    button_primary_text_color="#14171C",
    input_radius="10px",
)

CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');

:root {{
    --ink: #14171C;
    --ink-elevated: #1F242C;
    --paper: #F7F5F0;
    --amber: #F5A623;
    --trust-blue: #3B6E8F;
    --sage: #4F7A5B;
    --muted-on-dark: #9AA1AC;
}}

.gradio-container {{ background: var(--paper) !important; }}

/* ---------- Hero ---------- */
#hero {{
    background-color: var(--ink);
    background-image:
        linear-gradient(180deg, rgba(20,23,28,0.0) 60%, rgba(20,23,28,0.35) 100%),
        url("data:image/svg+xml;base64,{REDACTION_TEXTURE_B64}");
    background-repeat: no-repeat, repeat;
    background-size: cover, 480px 160px;
    border-radius: 18px;
    padding: 36px 40px 32px 40px;
    margin-bottom: 28px;
}}
#topbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 34px;
}}
#brand {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 17px;
    color: #F2EFE8;
    letter-spacing: 0.01em;
}}
#brand .accent {{ color: var(--amber); }}
#brand .sub {{ color: var(--muted-on-dark); font-weight: 400; margin-left: 6px; }}
#staff-badge {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--muted-on-dark);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 999px;
    padding: 5px 12px;
    letter-spacing: 0.04em;
}}
#headline {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 500;
    font-size: 34px;
    line-height: 1.28;
    color: #F7F5F0;
    max-width: 640px;
    margin: 0 0 26px 0;
}}
#headline .accent {{ color: var(--amber); font-style: italic; }}

#search-row {{ gap: 10px !important; }}
#hero-search textarea, #hero-search input {{
    background: #1B1F26 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #F2EFE8 !important;
    border-radius: 999px !important;
    padding: 14px 22px !important;
    font-size: 15px !important;
}}
#hero-search textarea::placeholder {{ color: #6B7280 !important; }}
#hero-search textarea:focus, #hero-search input:focus {{
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px rgba(245,166,35,0.18) !important;
}}
#ask-btn {{
    border-radius: 999px !important;
    font-weight: 600 !important;
    padding-left: 26px !important;
    padding-right: 26px !important;
    border: none !important;
}}
#disclaimer {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px;
    color: var(--muted-on-dark);
    margin: 10px 2px 22px 2px;
    line-height: 1.5;
}}

#chip-row {{ gap: 8px !important; flex-wrap: wrap !important; }}
.chip {{
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #D7D3C8 !important;
    border-radius: 999px !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    box-shadow: none !important;
    min-width: 0 !important;
    flex: 0 0 auto !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.3 !important;
}}
.chip:hover {{
    border-color: var(--amber) !important;
    color: #F7F5F0 !important;
    background: rgba(245,166,35,0.1) !important;
}}

/* ---------- Mobile ---------- */
@media (max-width: 700px) {{
    #hero {{ padding: 26px 20px 24px 20px; }}
    #topbar {{ flex-direction: column; align-items: flex-start; gap: 10px; margin-bottom: 24px; }}
    #staff-badge {{ white-space: nowrap; }}
    #headline {{ font-size: 26px; max-width: 100%; }}
    #search-row {{ flex-direction: column; }}
    #ask-btn {{ width: 100%; }}
}}

/* ---------- Console (light area) ---------- */
#controls-card, #answer-card, #sources-card {{
    background: #FFFFFF;
    border: 1px solid #E5E1D8;
    border-radius: 14px;
    padding: 22px 24px;
}}
.section-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--trust-blue);
    font-weight: 600;
    margin-bottom: 10px;
}}
#controls-card .section-label {{ margin-bottom: 16px; }}

#answer-card {{ margin-bottom: 20px; min-height: 90px; }}
#answer-card p {{ font-size: 15.5px; line-height: 1.65; color: #1F2937; }}

.evidence-card {{
    border-left: 3px solid var(--amber);
    background: #FCFAF5;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-bottom: 12px;
}}
.evidence-card:last-child {{ margin-bottom: 0; }}
.evidence-tag {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    color: var(--trust-blue);
    margin-bottom: 6px;
}}
.evidence-tag .sim {{
    color: var(--sage);
    font-weight: 600;
}}
.evidence-text {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13.5px;
    color: #374151;
    line-height: 1.5;
}}
.no-sources {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13.5px;
    color: #9CA3AF;
    font-style: italic;
}}

#clear-btn {{
    border-radius: 10px !important;
}}

/* Sources accordion - match the evidence-card aesthetic */
#sources-accordion {{
    border: 1px solid #E5E1D8 !important;
    border-radius: 14px !important;
    background: #FFFFFF !important;
}}
#sources-accordion > .label-wrap {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--trust-blue) !important;
    font-weight: 600 !important;
    padding: 18px 22px !important;
}}
"""

_pipeline: Optional[RAGPipeline] = None
_pipeline_error: Optional[str] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline, _pipeline_error
    if _pipeline is None and _pipeline_error is None:
        try:
            print("Loading RAG pipeline (embedding model + vector store) - this happens once...")
            _pipeline = RAGPipeline()
            print("Pipeline ready.")
        except Exception as e:
            _pipeline_error = str(e)
    if _pipeline_error is not None:
        raise RuntimeError(
            f"RAG pipeline failed to initialize: {_pipeline_error}\n\n"
            "Check that vector_store/ has been built and, if using the default "
            "backend, that HF_TOKEN is set."
        )
    return _pipeline


def format_sources_html(retrieved_chunks: list) -> str:
    if not retrieved_chunks:
        return "<div class='no-sources'>No evidence retrieved yet \u2014 ask a question above.</div>"

    cards = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk.get("metadata", {}) or {}
        product = meta.get("product_category", "Unknown product")
        company = meta.get("company", "Unknown company")
        complaint_id = meta.get("complaint_id", "?")
        similarity = chunk.get("similarity")
        if isinstance(similarity, (int, float)):
            sim_str = f"{max(0.0, similarity):.0%} match"
        else:
            sim_str = "match strength n/a"
        text = (chunk.get("chunk_text") or "").strip()

        cards.append(
            f"<div class='evidence-card'>"
            f"<div class='evidence-tag'>[{i}] {product} \u00b7 {company} \u00b7 "
            f"#{complaint_id} \u00b7 <span class='sim'>{sim_str}</span></div>"
            f"<div class='evidence-text'>\u201c{text}\u201d</div>"
            f"</div>"
        )
    return "".join(cards)


def ask_stream(question: str, k: int, product_filter: str, backend: str):
    if not question or not question.strip():
        yield "_Enter a question above to get started._", "<div class='no-sources'>No evidence retrieved yet.</div>", gr.update(label="Sources")
        return

    where = None
    if ENABLE_METADATA_FILTERING and product_filter and product_filter != "All products":
        where = {"product_category": product_filter}

    try:
        pipeline = get_pipeline()
    except Exception as e:
        yield f"\u26a0\ufe0f {e}", "<div class='no-sources'>No evidence retrieved.</div>", gr.update(label="Sources")
        return

    if backend and backend != pipeline.generation_backend:
        pipeline.generation_backend = backend

    answer_so_far = ""
    sources_html = "<div class='no-sources'>Retrieving evidence...</div>"
    accordion_update = gr.update(label="Sources")
    yield answer_so_far, sources_html, accordion_update

    for event in pipeline.answer_stream(question, k=int(k), where=where):
        event_type = event.get("type")

        if event_type == "sources":
            chunks = event.get("retrieved_chunks", [])
            sources_html = format_sources_html(chunks)
            accordion_update = gr.update(label=f"Sources ({len(chunks)})", open=True)
            yield answer_so_far, sources_html, accordion_update

        elif event_type == "token":
            answer_so_far += event.get("text", "")
            yield answer_so_far, sources_html, accordion_update

        elif event_type == "done":
            answer_so_far = event.get("answer", answer_so_far)
            yield answer_so_far, sources_html, accordion_update

        elif event_type == "error":
            answer_so_far = event.get("answer", "An error occurred.")
            yield answer_so_far, sources_html, accordion_update


def clear_all():
    return "", "_Your answer will appear here._", "<div class='no-sources'>No evidence retrieved yet.</div>", gr.update(label="Sources")


def fill_question(q):
    return q


# -----------------------------------------------------------------------
# UI layout
# -----------------------------------------------------------------------
with gr.Blocks(title="CrediTrust Complaint Insights") as demo:

    with gr.Column(elem_id="hero"):
        gr.HTML(
            "<div id='topbar'>"
            "<div id='brand'>&#9679; CrediTrust <span class='accent'>Complaint Insights</span>"
            "<span class='sub'>internal research console</span></div>"
            "<div id='staff-badge'>STAFF TOOL &middot; v1.0</div>"
            "</div>"
            "<div id='headline'>Every answer, traced back to the<br>"
            "<span class='accent'>complaint that made it true.</span></div>"
        )

        with gr.Row(elem_id="search-row"):
            question_box = gr.Textbox(
                show_label=False,
                placeholder="Ask about a complaint pattern, e.g. \u201cWhy are customers unhappy with money transfers?\u201d",
                lines=1,
                scale=5,
                elem_id="hero-search",
                container=False,
            )
            ask_button = gr.Button("Ask", variant="primary", scale=1, elem_id="ask-btn")

        gr.HTML(
            "<div id='disclaimer'>Answers are generated only from retrieved complaint excerpts "
            "below \u2014 never from outside knowledge. Verify evidence before relying on an answer.</div>"
        )

        with gr.Row(elem_id="chip-row"):
            chip_buttons = [gr.Button(q, elem_classes="chip", size="sm") for q in EXAMPLE_QUESTIONS]

    with gr.Row():
        with gr.Column(scale=1, elem_id="controls-card"):
            gr.HTML("<div class='section-label'>Console</div>")
            k_slider = gr.Slider(minimum=1, maximum=10, value=TOP_K, step=1, label="Evidence chunks (k)")
            backend_dropdown = gr.Dropdown(
                choices=["hf_inference_api", "local_pipeline"],
                value=GENERATION_BACKEND,
                label="Generation backend",
            )
            product_dropdown = gr.Dropdown(
                choices=PRODUCT_CATEGORIES,
                value="All products",
                label="Filter by product",
                visible=ENABLE_METADATA_FILTERING,
            )
            clear_button = gr.Button("Clear", elem_id="clear-btn")

        with gr.Column(scale=2):
            with gr.Column(elem_id="answer-card"):
                gr.HTML("<div class='section-label'>Answer</div>")
                answer_box = gr.Markdown(value="_Your answer will appear here._")

            with gr.Accordion("Sources", open=True, elem_id="sources-accordion") as sources_accordion:
                sources_box = gr.HTML(value="<div class='no-sources'>No evidence retrieved yet.</div>")

    ask_inputs = [question_box, k_slider, product_dropdown, backend_dropdown]
    ask_outputs = [answer_box, sources_box, sources_accordion]

    ask_button.click(fn=ask_stream, inputs=ask_inputs, outputs=ask_outputs)
    question_box.submit(fn=ask_stream, inputs=ask_inputs, outputs=ask_outputs)

    for chip, question_text in zip(chip_buttons, EXAMPLE_QUESTIONS):
        chip.click(fn=fill_question, inputs=[gr.State(question_text)], outputs=question_box).then(
            fn=ask_stream, inputs=ask_inputs, outputs=ask_outputs
        )

    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[question_box, answer_box, sources_box, sources_accordion],
    )


if __name__ == "__main__":
    demo.queue().launch(theme=theme, css=CUSTOM_CSS)