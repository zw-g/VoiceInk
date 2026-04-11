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

CRITICAL: PRESERVE the original language. NEVER translate Chinese to English (e.g. 我们 must NOT become Our/We).
CRITICAL: MINIMIZE punctuation. Add comma ONLY for context shifts (e.g. 改成90天之前是→改成90天，之前是).
CRITICAL: Remove ALL spaces between Chinese and English words. 帮我 ping 一下→帮我ping一下, 这个 feature→这个feature. Exception: keep space after Literally, Actually, Right, Basically before Chinese.
CRITICAL: Numbers inside Chinese idioms are NOT real numbers. NEVER convert them.
CRITICAL: 万/千 MUST expand fully: 三万=30000, 两千=2000. BUT keep 万 as unit: 两百万=200万.
CRITICAL: In pure Chinese narrative sentences describing past events with multiple numbers, preserve ALL original number forms (八千块=八千块, 两万四=两万四).

Rules:
1. Numbers to digits:
   - Chinese: 百分之三十二=32%, 三百七十六=376, 零点五=0.5, 两千=2000, 三万=30000, 两百万=200万
   - English: thirty-eight=38, twelve point five=12.5, seventy percent=70%, twenty thousand=20,000
   - Dates: April third=April 3rd, January first=January 1st, September first=September 1st
   - Compound: fifty thousand=50,000, twelve hundred=1,200
   - Counters: 两碗=2碗, 五个=5个, 四百度=400度
2. Math (spaces around operators): 小于等于=≤, 大于等于=≥, 不等于=≠, 大于=>, 小于=<, 等于==
3. Time: 下午两点半=下午2:30, three thirty pm=3:30 PM, two pm=2:00 PM, seven am=7 AM
4. Ordinals: 第三=第3, first=1st, sixth=6th, twentieth=20th, forty-second=42nd
5. Currency: 三百五十块=350块, fifty dollars=$50, 三万块=30000块, four hundred dollars=$400
6. Measurements: 三十公里=30公里, twenty miles=20 miles, three hundred miles=300 miles
7. Remove fillers: 呃, 嗯, 那个(filler), 就是说, 然后呢, 啊(start) | um, uh, like(filler), so basically, you know(filler)
   Keep meaningful: yeah, yep, okay, sure
8. FROZEN idioms: 三心二意, 不管三七二十一, 一举两得, 接二连三, 一目十行, 三言两语
9. Do NOT convert: 一个/一些/一下/一点/一直/一般/一共, 七折/八折/九折
10. Hyphenate: twelve hour=12-hour. Do NOT rephrase or translate.

Example 1: 那个嗯你能不能帮我看看这段代码 Output: 你能不能帮我看看这段代码
Example 2: 他三心二意地做了一百道题 Output: 他三心二意地做了100道题
Example 3: How about 我们先吃饭 then continue working 吃完再说 Output: How about我们先吃饭then continue working吃完再说
Example 4: 年终奖发了三万块 Output: 年终奖发了30000块
Example 5: Literally 他真的就站在那里一动不动 Output: Literally 他真的就站在那里一动不动
Example 6: you know we should probably handle that edge case Output: we should probably handle that edge case
Example 7: y小于等于一百 Output: y ≤ 100
Example 8: 嗯 the sixth iteration 提升了 twenty percent Output: the 6th iteration提升了20%
Example 9: 我们的test coverage是seventy eight percent Output: 我们的test coverage是78%
Example 10: 嗯 training 从 September first 开始嗯到月底 Output: training从September 1st开始到月底
Example 11: 嗯大概 eighty four percent 的覆盖率呃 Output: 大概84%的覆盖率
Example 12: uh cost four hundred dollars drove three hundred miles first day visited five attractions seventy percent walking did twenty thousand steps Output: cost $400 drove 300 miles 1st day visited 5 attractions 70% walking did 20,000 steps"""

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
    '- ASR: "I like the desine" -> User: "I like the design" => NO\n'
)


# Pre-compiled regex patterns for _needs_polish (avoid re-compiling on every call)
_RE_MATH_CN = re.compile(r'大于等于|小于等于|不等于|大于|小于|等于|乘以|除以')
_RE_MATH_EN = re.compile(r'\b(greater than|less than|equals|squared|divided by)\b', re.IGNORECASE)
_RE_FILLER_FINAL = re.compile(r'啊[，。？！\s]|啊$|对吧|是吧')
_RE_FILLER_CN = re.compile(r'呃|嗯|就是说|然后呢|那个')
_RE_FILLER_EN = re.compile(r'\bum\b|\buh\b|\bso basically\b', re.IGNORECASE)
_RE_FILLER_LIKE = re.compile(r',\s*like,|, you know,|^like ', re.IGNORECASE)
_RE_ORDINAL = re.compile(r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|thirtieth|fortieth|fiftieth|sixtieth|seventieth|eightieth|ninetieth|hundredth|thousandth)\b', re.IGNORECASE)
_RE_TIME_CN = re.compile(r'[一二三四五六七八九十两]+点[半一二三四五六七八九十]*')
_RE_NUM_CN = re.compile(r'百分之|零点|[一二三四五六七八九十百千万亿]{2,}')
_RE_NUM_EN = re.compile(r'\b(thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b', re.IGNORECASE)
_RE_TIME_EN = re.compile(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(am|pm|o.clock|thirty|fifteen|forty.five)\b', re.IGNORECASE)
_RE_CURRENCY_MEASURE = re.compile(
    r'(?:\d|[一二三四五六七八九十百千万])\s*(?:块|元|美元|dollars?|bucks|cents?|公里|公斤|米|pounds?|miles?|kilometers?)',
    re.IGNORECASE
)
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
    if _RE_CURRENCY_MEASURE.search(text):
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
    # English time expressions (e.g., "two pm", "twelve thirty", "one o'clock")
    if _RE_TIME_EN.search(text):
        return True
    # Long text with no punctuation — likely needs punctuation from LLM
    if len(text) > 20 and not _RE_NO_PUNCT.search(text):
        return True
    # No obvious issues found — skip polish
    return False


class TextPolisher:
    """LLM-based text post-processor using Qwen3-8B on MLX (r8_h2_p1_fixes_plus_adversarial_guard, 99.6% benchmark)."""

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
            self._sampler = make_sampler(temp=0.1, top_p=0.9, top_k=10)
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
            if not clean or len(clean) < len(text) * 0.15 or len(clean) > len(text) * 2:
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
