SYSTEM_MESSAGE = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)
INSTRUCTION = (
    "Given a math question, its correct answer, and an incorrect answer, "
    "retrieve the misconception that best explains the incorrect answer."
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ENABLE_THINKING = False
PREFIX = (
    f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
    "<|im_start|>user\n"
)


def user_content(query, document, variant):
    if variant == "a1_document_last":
        return f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
    if variant == "a0_document_first":
        return f"<Document>: {document}\n<Instruct>: {INSTRUCTION}\n<Query>: {query}"
    raise ValueError(f"unknown prompt variant: {variant}")


def prompt_token_ids(tokenizer, query, document, variant):
    assert ENABLE_THINKING is False
    content = user_content(query, document, variant)
    rendered = PREFIX + content + SUFFIX
    if not rendered.endswith(SUFFIX):
        raise ValueError("prompt must end with the official empty thinking suffix")
    if variant == "a1_document_last":
        if content.rfind("<Document>:") < content.rfind("<Query>:"):
            raise ValueError("A1 prompt must place Document after Query")
    elif content.find("<Document>:") > content.find("<Query>:"):
        raise ValueError("A0 prompt must place Document before Query")
    return tokenizer.encode(rendered, add_special_tokens=False)
