"""The classification prompt, framed as a yes/no question so one next-token distribution gives a score."""

QUESTION = ('This is a meme. The text written on the image is: "{text}". '
            "Taking the image and the text together, is this meme hateful? Answer Yes or No.")
YES, NO = "Yes", "No"


def answer_word(label: int) -> str:
    return YES if label == 1 else NO


def build_messages(text: str, label: int | None = None) -> list[dict]:
    """Chat messages in the Hugging Face multimodal format. With a label, the assistant turn is included (training)."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION.format(text=text)}]}]
    if label is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer_word(label)}]})
    return msgs
