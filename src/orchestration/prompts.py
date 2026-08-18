"""
Prompts, versioned.

Prompts are treated as configuration, not as string literals buried in
functions. Each one carries a version string that gets written into every
trace, so when answer quality changes you can tell whether it was a prompt
edit, a retrieval change, or a model change. Without this, comparing
"before and after" is guesswork.

Three separate prompt problems live here, and they are genuinely different:

  1. Answering from retrieved documents without inventing anything
  2. Behaving sensibly when retrieval comes back empty
  3. Weighing long-term memory against the current question
"""

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a personal assistant that answers questions using \
the user's own documents.

Rules:
- Answer only from the provided context and memories. Do not use outside knowledge.
- Cite sources by their bracketed number, like [1] or [2].
- If the context does not contain the answer, say so plainly. Do not guess.
- Be concise. Do not pad the answer.
"""

# Used when retrieval found something. The instruction to prefer context over
# prior knowledge is doing real work here: without it the model happily blends
# the two and you cannot tell which parts came from the documents.
RAG_PROMPT = """{system}

{memory_block}Context from the user's documents:
{context}

Question: {question}

Answer using only the context above. Cite sources as [1], [2], and so on."""

# Used when retrieval came back empty or too weak. Telling the model there is
# no context is far better than sending an empty context block, which reads as
# an accident and invites the model to fill the gap itself.
NO_CONTEXT_PROMPT = """{system}

{memory_block}No relevant documents were found for this question.

Question: {question}

Tell the user you could not find anything in their documents about this. \
Do not answer from general knowledge. Keep it to one or two sentences."""

MEMORY_BLOCK = """What you remember about this user:
{memories}

"""


def build_prompt(question: str, context: str, memories: str) -> tuple:
    """
    Assemble the final prompt.

    Returns (prompt_text, prompt_name) so the trace records which variant ran.
    Which prompt fired is often the answer to "why did it respond like that".
    """
    memory_block = MEMORY_BLOCK.format(memories=memories) if memories else ""

    if context.strip():
        prompt = RAG_PROMPT.format(
            system=SYSTEM_PROMPT,
            memory_block=memory_block,
            context=context,
            question=question,
        )
        return prompt, "rag"

    prompt = NO_CONTEXT_PROMPT.format(
        system=SYSTEM_PROMPT,
        memory_block=memory_block,
        question=question,
    )
    return prompt, "no_context"
