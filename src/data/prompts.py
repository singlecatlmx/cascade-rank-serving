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
        return f"<Instruct>: {INSTRUCTION}\n\n<Query>: {query}\n\n<Document>: {document}"
    if variant == "a0_document_first":
        return f"<Document>: {document}\n\n<Instruct>: {INSTRUCTION}\n\n<Query>: {query}"
    raise ValueError(f"unknown prompt variant: {variant}")


def prompt_token_ids(tokenizer, query, document, variant, suffix_tokens):
    content = user_content(query, document, variant)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": content},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    assert "<think>" not in rendered
    if variant == "a1_document_last":
        assert content.rfind("<Document>:") > content.rfind("<Query>:")
    else:
        assert content.find("<Document>:") < content.find("<Query>:")
    return tokenizer.encode(rendered, add_special_tokens=False) + suffix_tokens
