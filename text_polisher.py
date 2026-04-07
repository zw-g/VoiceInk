"""LLM-based text post-processing for VoiceInk.

Contains the TextPolisher class, system prompts, and the _needs_polish heuristic.
"""

import logging
import re
import threading

log = logging.getLogger("voiceinput")

_LLM_MODEL_ID = "Qwen/Qwen3-8B-MLX-4bit"

_LLM_SYSTEM_PROMPT = """\
You are a voice transcription post-processor. Output ONLY the cleaned text.

CRITICAL: PRESERVE the original language of every word. NEVER translate between languages.
CRITICAL: Do NOT modify sentences that are already well-formed. Only fix formatting issues.

Rules:
1. Convert spoken numbers to digits:
   - Chinese: 百分之三十二→32%, 三百七十六→376, 零点五→0.5
   - English: thirty-eight→38, twelve point five→12.5
   - Dates: 二零二六年四月三号→2026年4月3号, April third→April 3rd
2. Convert math/symbols in ALL languages:
   - Chinese: 大于→>, 小于→<, 等于→=, 加→+, 减→-, 乘以→×, 除以→÷, 大于等于→≥, 小于等于→≤, 不等于→≠, 的平方→², 根号→√
   - English: is greater than→>, is less than→<, equals→=, plus→+, minus→-, times→×, divided by→÷, is greater than or equal to→≥, squared→², square root→√, to the power of→superscript
3. Convert time expressions:
   - Chinese: 下午两点半→下午2:30, 上午九点十五→上午9:15, 三点四十五→3:45
   - English: three thirty pm→3:30 PM, ten fifteen am→10:15 AM, noon→12:00 PM
4. Convert ordinals:
   - Chinese: 第三→第3, 第二十一→第21, 第一百→第100
   - English: first→1st, second→2nd, third→3rd, twenty-first→21st, forty-second→42nd
5. Convert currency amounts:
   - Chinese: 三百五十块→350块, 两千元→2000元
   - English: fifty dollars→$50, twenty five cents→$0.25
6. Convert measurements:
   - Chinese: 三十公里→30公里, 五百克→500克
   - English: twenty miles→20 miles, fifteen pounds→15 pounds
7. Remove filler words ONLY:
   Chinese: 呃, 嗯, 那个, 就是说, 然后呢
   English: um, uh, like (as filler), you know (as filler), so basically
8. Add punctuation
9. Preserve idioms (三心二意, 不管三七二十一)
10. Do NOT rephrase, reword, or modify meaningful content

Example 1: 呃就是说这个东西嗯还不错然后呢我们看看
Output 1: 这个东西还不错，我们看看

Example 2: 二零二六年四月三号下午两点半我们开会
Output 2: 2026年4月3号下午2:30我们开会

Example 3: 呃这个model的performance大概百分之九十五然后呢还不错
Output 3: 这个model的performance大概95%，还不错

Example 4: um I think it costs about fifty dollars and twenty five cents
Output 4: I think it costs about $50 and $0.25"""

_DICT_CLASSIFY_PROMPT = (
    "You classify voice transcription corrections. Given the ASR output and "
    "user's correction, decide if the corrected word should be added to the "
    "speech recognition dictionary.\n"
    "Answer ONLY 'YES word' or 'NO'.\n\n"
    "ADD (YES):\n"
    "- Proper nouns, brand names, technical terms, product names, person names\n"
    "- Chinese proper nouns the ASR consistently gets wrong\n"
    "- Bilingual code-switching: English words the ASR mistranscribes as Chinese "
    "characters (e.g., user says 'mean' but ASR writes '命'). These MUST be added "
    "because the ASR cannot distinguish them without dictionary hints.\n\n"
    "SKIP (NO): typo/grammar fixes, punctuation changes, "
    "single characters, rephrasing, capitalization-only changes.\n\n"
    "Be PICKY for common words in monolingual context. But for cross-language "
    "errors (Chinese char where English word was intended, or vice versa), "
    "ALWAYS say YES — these are exactly what the dictionary is for.\n\n"
    "Examples:\n"
    '- ASR: "pie torch" -> User: "PyTorch" => YES PyTorch\n'
    '- ASR: "anthrobic" -> User: "Anthropic" => YES Anthropic\n'
    '- ASR: "the model are good" -> User: "the model is good" => NO\n'
    '- ASR: "kuda" -> User: "CUDA" => YES CUDA\n'
    '- ASR: "he go" -> User: "he went" => NO\n'
    '- ASR: "法布里凯特" -> User: "Phabricator" => YES Phabricator\n'
    '- ASR: "非常命的" -> User: "非常mean的" => YES mean\n'
    '- ASR: "好的哈" -> User: "好的ha" => NO (common interjection)\n'
    '- ASR: "我要看你" -> User: "我要看repo" => YES repo\n'
)


# Pre-compiled regex patterns for _needs_polish (avoid re-compiling on every call)
_RE_MATH_CN = re.compile(r'大于等于|小于等于|不等于|大于|小于|等于|乘以|除以')
_RE_MATH_EN = re.compile(r'\b(greater than|less than|equals|squared|divided by)\b', re.IGNORECASE)
_RE_FILLER_FINAL = re.compile(r'啊[，。？！\s]|啊$|对吧|是吧')
_RE_FILLER_CN = re.compile(r'呃|嗯|就是说|然后呢|那个')
_RE_FILLER_EN = re.compile(r'\bum\b|\buh\b|\bso basically\b', re.IGNORECASE)
_RE_FILLER_LIKE = re.compile(r',\s*like,|, you know,|^like ', re.IGNORECASE)
_RE_ORDINAL = re.compile(r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|twentieth|thirtieth)\b', re.IGNORECASE)
_RE_TIME_CN = re.compile(r'[一二三四五六七八九十两]+点[半一二三四五六七八九十]*')
_RE_NUM_CN = re.compile(r'百分之|零点|[一二三四五六七八九十百千万亿]{2,}')
_RE_NUM_EN = re.compile(r'\b(thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b', re.IGNORECASE)
_RE_CURRENCY = re.compile(r'块|元|美元|dollars?|bucks|cents?|公里|公斤|米|pounds?|miles?|kilometers?', re.IGNORECASE)
_RE_NO_PUNCT = re.compile(r'[.,!?;:，。！？；：]')


def _needs_polish(text):
    """Quick heuristic: does this text need LLM polishing?

    Returns False for short, clean text that would come back identical
    from the LLM, saving 0.8-2.6s of latency.
    """
    # Unconverted math expressions (Chinese) — check before length gate
    # so short expressions like "大于" still get polished
    if _RE_MATH_CN.search(text):
        return True
    # Unconverted math expressions (English)
    if _RE_MATH_EN.search(text):
        return True
    # Chinese sentence-final fillers / tag questions — check before length gate
    # so short phrases like "你说对吧" still get polished
    if _RE_FILLER_FINAL.search(text):
        return True
    # Very short text — not worth the LLM overhead
    if len(text) < 8:
        return False
    # Chinese filler words
    if _RE_FILLER_CN.search(text):
        return True
    # English filler words
    if _RE_FILLER_EN.search(text):
        return True
    # "like" / "you know" as fillers (with surrounding commas or at start)
    if _RE_FILLER_LIKE.search(text):
        return True
    # English ordinals
    if _RE_ORDINAL.search(text):
        return True
    # Currency and measurement words
    if _RE_CURRENCY.search(text):
        return True
    # Chinese time patterns (e.g., 两点半, 三点十五)
    if _RE_TIME_CN.search(text):
        return True
    # Unconverted Chinese number words (百分之, 零点, or consecutive number characters)
    if _RE_NUM_CN.search(text):
        return True
    # Unconverted English number words
    if _RE_NUM_EN.search(text):
        return True
    # Long text with no punctuation — likely needs punctuation from LLM
    if len(text) > 20 and not _RE_NO_PUNCT.search(text):
        return True
    # No obvious issues found — skip polish
    return False


class TextPolisher:
    """LLM-based text post-processor using Qwen3-8B on MLX (r4_three_targeted_examples, 95.2% benchmark)."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._sampler = None
        self._loaded = False
        self._load_failed = False
        self._inference_lock = threading.Lock()
        self._notify_fn = None

    def set_notify(self, fn):
        """Set the notification callback to avoid importing voice_input."""
        self._notify_fn = fn

    def load(self):
        """Load the LLM model. Call from background thread."""
        try:
            from mlx_lm import load as mlx_load
            from mlx_lm.sample_utils import make_sampler

            log.info("Loading text polish model %s", _LLM_MODEL_ID)
            self._model, self._tokenizer = mlx_load(_LLM_MODEL_ID)
            self._sampler = make_sampler(temp=0.3, top_p=0.8, top_k=20)
            self._loaded = True
            log.info("Text polish model loaded")
        except Exception as e:
            log.warning("Text polish model failed to load: %s", e, exc_info=True)
            self._loaded = False
            self._load_failed = True
            if self._notify_fn:
                self._notify_fn("VoiceInk", "Text polish model failed to load")

    def polish(self, text):
        """Polish text using the LLM. Returns original text on failure."""
        if not self._loaded or not text.strip():
            return text
        try:
            from mlx_lm import generate

            # /no_think is a Qwen3-specific directive that disables the model's
            # internal chain-of-thought reasoning, reducing latency for simple tasks.
            messages = [
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": text + "\n/no_think"},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            with self._inference_lock:
                raw = generate(
                    self._model, self._tokenizer, prompt=prompt,
                    max_tokens=max(len(text) * 3, 200),
                    sampler=self._sampler, verbose=False,
                )
            # Strip thinking block if present
            clean = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL)
            clean = re.sub(r"<think>.*", "", clean, flags=re.DOTALL).strip()

            # Safety: if output is empty or wildly different length, use original
            if not clean or len(clean) < len(text) * 0.3 or len(clean) > len(text) * 2:
                log.warning("Text polish output rejected (len %d→%d), using original", len(text), len(clean))
                return text
            return clean
        except Exception as e:
            log.warning("Text polish failed: %s", e, exc_info=True)
            return text

    def classify_correction(self, original, corrected, dictionary_words):
        """Classify whether a correction is dictionary-worthy. Returns (bool, word)."""
        if not self._loaded or not corrected.strip():
            return False, None
        try:
            from mlx_lm import generate
            dict_sample = ", ".join(dictionary_words[:50])
            messages = [
                {"role": "system", "content": _DICT_CLASSIFY_PROMPT + f"\nCurrent dictionary: {dict_sample}"},
                {"role": "user", "content": f"ASR: \"{original}\"\nUser: \"{corrected}\"\n/no_think"},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            with self._inference_lock:
                raw = generate(
                    self._model, self._tokenizer, prompt=prompt,
                    max_tokens=30, sampler=self._sampler, verbose=False,
                )
            clean = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL)
            clean = re.sub(r"<think>.*", "", clean, flags=re.DOTALL).strip()
            log.info("Dict classify: '%s' -> '%s' => %s", original, corrected, clean)
            if clean.upper().startswith("YES"):
                parts = clean.split(None, 1)
                word = parts[1].strip() if len(parts) > 1 else corrected.strip()
                return True, word
            return False, None
        except Exception as e:
            log.warning("Dict classify failed: %s", e)
            return False, None
