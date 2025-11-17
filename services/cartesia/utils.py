from typing import Optional, Tuple
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab")

SENTENCE_ENDING_PUNCTUATION = (
    # Latin script punctuation (most European languages, Filipino, etc.)
    ".",
    "!",
    "?",
    ";",
    "…",
    # East Asian punctuation (Chinese (Traditional & Simplified), Japanese, Korean)
    "。",  # Ideographic full stop
    "？",  # Full-width question mark
    "！",  # Full-width exclamation mark
    "；",  # Full-width semicolon
    "．",  # Full-width period
    "｡",  # Halfwidth ideographic period
    # Indic scripts punctuation (Hindi, Sanskrit, Marathi, Nepali, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Oriya, Assamese)
    "।",  # Devanagari danda (single vertical bar)
    "॥",  # Devanagari double danda (double vertical bar)
    # Arabic script punctuation (Arabic, Persian, Urdu, Pashto)
    "؟",  # Arabic question mark
    "؛",  # Arabic semicolon
    "۔",  # Urdu full stop
    "؏",  # Arabic sign misra (classical texts)
    # Thai
    "।",  # Thai uses Devanagari-style punctuation in some contexts
    # Myanmar/Burmese
    "၊",  # Myanmar sign little section
    "။",  # Myanmar sign section
    # Khmer
    "។",  # Khmer sign khan
    "៕",  # Khmer sign bariyoosan
    # Lao
    "໌",  # Lao cancellation mark (used as period)
    "༎",  # Tibetan mark delimiter tsheg bstar (also used in Lao contexts)
    # Tibetan
    "།",  # Tibetan mark intersyllabic tsheg
    "༎",  # Tibetan mark delimiter tsheg bstar
    # Armenian
    "։",  # Armenian full stop
    "՜",  # Armenian exclamation mark
    "՞",  # Armenian question mark
    # Ethiopic script (Amharic)
    "።",  # Ethiopic full stop
    "፧",  # Ethiopic question mark
    "፨",  # Ethiopic paragraph separator
)

TAG_ENDING_PUNCTUATION = (
    "]",
    ">",
)


def aggreagate_sentences(speakable_text: str, incomplete_text: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Aggregate speakable text into sentences. If there is an incomplete text, append it to the last sentence.
    """
    sentences = sent_tokenize(speakable_text)
    if len(sentences) == 0:
        return speakable_text, incomplete_text
    if sentences[-1].endswith(SENTENCE_ENDING_PUNCTUATION) or sentences[-1].endswith(TAG_ENDING_PUNCTUATION):
        return speakable_text, incomplete_text
    else:
        return "".join(sentences[:-1]), (incomplete_text if incomplete_text else "") + sentences[-1]


def parse_speakeable_text(text: str) -> Tuple[str, Optional[str]]:
    """
    Parse text looking for XML-like tags starting with '<'.

    If a tag is incomplete (has '<' but no closing '>'), return:
    - First element: text before the '<'
    - Second element: the incomplete tag (from '<' onwards)

    If all tags are complete, return:
    - First element: the entire text
    - Second element: None

    Examples:
    - "Sure! <break time=\"250ms\" />" -> ("Sure! <break time=\"250ms\" />", None)
    - "Sure! <break" -> ("Sure! ", "<break")
    - "Sure! [laughter]" -> ("Sure! [laughter]", None)
    """
    start_tag = text.find("<")

    # No tag found, return entire text
    if start_tag == -1:
        return text, None

    end_tag = text.find(">", start_tag)

    # Tag not closed, return text before tag and incomplete tag
    if end_tag == -1:
        return text[:start_tag], text[start_tag:]

    # Tag is closed, check if there are more tags after this one
    # We need to recursively check the remaining text
    remaining_text = text[end_tag + 1 :]
    if remaining_text:
        speakeable, incomplete = parse_speakeable_text(remaining_text)
        if incomplete is not None:
            # Found an incomplete tag in the remaining text
            return text[:start_tag] + text[start_tag : end_tag + 1] + speakeable, incomplete

    # All tags are complete
    return text, None
