"""Inverse Text Normalization (ITN) for VoiceInk.

Converts spoken number words to Arabic numerals:
  - Chinese: uses wetext (WeNet ITN) — context-aware, preserves idioms
  - English: uses word2number — handles multi-word numbers and percentages
"""

import logging
import re

log = logging.getLogger("voiceinput")

# Chinese: fix cn2an false positives where 一 is a word, not the number 1
_CN_REVERT = re.compile(r"1(些|下子?|起|直|定|样|般|切|边|块儿?|会儿?|共|向|旦|时|味|概)")

# English number words recognized by word2number
_EN_NUM_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "trillion", "and",
}
_EN_SCALE_WORDS = {"hundred", "thousand", "million", "billion", "trillion"}


def _en_itn(text):
    """English Inverse Text Normalization using word2number library."""
    try:
        from word2number import w2n
    except ImportError:
        return text

    words = text.split()
    result = []
    i = 0
    while i < len(words):
        w_clean = words[i].lower().rstrip(".,;:!?").replace("-", " ").split()

        if w_clean and w_clean[0] in _EN_NUM_WORDS and w_clean[0] != "and":
            # Collect consecutive number words
            raw_span = [words[i]]
            j = i + 1
            while j < len(words):
                wj_parts = words[j].lower().rstrip(".,;:!?").replace("-", " ").split()
                if wj_parts and all(p in _EN_NUM_WORDS for p in wj_parts):
                    raw_span.append(words[j])
                    j += 1
                else:
                    break

            # Build the number phrase for word2number
            phrase = " ".join(raw_span).rstrip(".,;:!?")
            phrase_words = [w for w in phrase.lower().replace("-", " ").split() if w != "and"]

            # Only convert if: multi-word, contains scale, or followed by percent
            has_scale = any(w in _EN_SCALE_WORDS for w in phrase_words)
            followed_by_pct = j < len(words) and words[j].lower().rstrip(".,;:!?") == "percent"

            if len(phrase_words) >= 2 or has_scale or followed_by_pct:
                try:
                    value = w2n.word_to_num(phrase)
                    suffix = ""
                    if followed_by_pct:
                        suffix = "%"
                        j += 1
                    result.append(str(value) + suffix)
                    i = j
                    continue
                except ValueError:
                    pass

            result.append(words[i])
            i += 1
        else:
            result.append(words[i])
            i += 1
    return " ".join(result)


def normalize_numbers(text):
    """Convert spoken number words to Arabic numerals (Chinese + English).

    Chinese: uses wetext (WeNet ITN) — context-aware, preserves idioms.
    English: uses word2number — handles multi-word numbers and percentages.
    """
    # Chinese ITN via wetext (professional, context-aware)
    try:
        from wetext import Normalizer

        if not hasattr(normalize_numbers, "_zh_itn"):
            normalize_numbers._zh_itn = Normalizer(lang="zh", operator="itn")
        text = normalize_numbers._zh_itn.normalize(text)
    except ImportError:
        # Fallback to cn2an if wetext not available
        try:
            import cn2an

            text = cn2an.transform(text, "cn2an")
            text = _CN_REVERT.sub(r"一\1", text)
        except (ImportError, Exception):
            pass
    except Exception as e:
        log.debug("wetext ITN error: %s", e)

    # English ITN
    text = _en_itn(text)
    return text
