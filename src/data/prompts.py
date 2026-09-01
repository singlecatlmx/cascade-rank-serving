SYSTEM_MESSAGE = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)
INSTRUCTION = (
    "Given a math question, its correct answer, and an incorrect answer, "
    "retrieve the misconception that best explains the incorrect answer."
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def user_content(query, document, variant):
    if variant == "a1_document_last":
        return f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
    if variant == "a0_document_first":
        return f"<Document>: {document}\n<Instruct>: {INSTRUCTION}\n<Query>: {query}"
    raise ValueError(f"unknown prompt variant: {variant}")


def prompt_token_ids(tokenizer, query, document, variant):
    content = user_content(query, document, variant)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": content},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not rendered.endswith(SUFFIX):
        raise ValueError(
            "enable_thinking=False did not produce the official empty thinking suffix; "
            f"rendered tail={rendered[-120:]!r}"
        )
    if variant == "a1_document_last":
        if content.rfind("<Document>:") < content.rfind("<Query>:"):
            raise ValueError("A1 prompt must place Document after Query")
    elif content.find("<Document>:") > content.find("<Query>:"):
        raise ValueError("A0 prompt must place Document before Query")
    return tokenizer.encode(rendered, add_special_tokens=False)
