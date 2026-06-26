
# ROLE
ROLE = """
You are a senior financial complaint analyst working for a digital banking company.
You analyze customer complaints across credit cards, loans, savings accounts, and money transfers.
"""

# INSTRUCTION (STRONG GROUNDING)
INSTRUCTION = """
You MUST follow these rules strictly:

1. Use ONLY the provided context.
2. If the context is not relevant to the question:
   - DO NOT summarize it
   - DO NOT mention it
   - Respond ONLY with the refusal message below

Refusal message (must be exact):
"There is not enough information in the provided complaints to answer this question."

3. Never attempt to force an answer from unrelated context.
"""

# FALLBACK RULE (OUT OF DOMAIN)
FALLBACK = """
If the question is unrelated to banking services or customer complaints:
- Politely refuse
- Explain that you only analyze complaint data from the dataset
"""

# FORMAT CONTROL (IMPROVED FOR EVALUATION)
FORMAT_CUE = """
If and only if context is relevant:

Answer in structured format:

1. Direct Answer (1–2 sentences)
2. Key Complaint Themes
3. Evidence

If context is not relevant:
Return only the refusal message.
"""

# FULL PROMPT TEMPLATE
PROMPT_TEMPLATE = """
{role}

{instruction}

{fallback}

Context:
{context}

Question:
{question}

{format_cue}

Answer:
"""


# -----------------------------
# CONTEXT FORMATTER
# -----------------------------
def format_context(retrieved_chunks: list) -> str:
    """
    Converts retrieved Chroma chunks into structured context.
    """
    if not retrieved_chunks:
        return "No relevant complaint excerpts were retrieved."

    formatted = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.get("metadata", {}) or {}

        product = meta.get("product_category", "Unknown Product")
        company = meta.get("company", "Unknown Company")
        cid = meta.get("complaint_id", "?")

        # FIX: sometimes your loader uses "document" instead of "chunk_text"
        text = (
            chunk.get("chunk_text")
            or chunk.get("document")
            or ""
        ).strip()

        formatted.append(
            f"[{i}] ({product} | {company} | complaint #{cid})\n{text}"
        )

    return "\n\n".join(formatted)


# -----------------------------
# BUILD FINAL PROMPT
# -----------------------------
def build_prompt(question: str, retrieved_chunks: list) -> str:
    """
    Assembles final RAG prompt.
    """

    context = format_context(retrieved_chunks)

    return PROMPT_TEMPLATE.format(
        role=ROLE,
        instruction=INSTRUCTION,
        fallback=FALLBACK,
        context=context,
        question=question,
        format_cue=FORMAT_CUE,
    )