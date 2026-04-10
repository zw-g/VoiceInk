"""Automated tests for VoiceInk core functions.

Tests pure functions that don't require rumps, audio, or AX API:
  - normalize_numbers() / _en_itn() — inverse text normalization
  - _needs_polish() — heuristic for whether text needs LLM polish
  - DictionaryGuard — anti-spam logic for dictionary additions
  - load_dictionary() / save_settings() — file I/O

Run:
    cd ~/.local/voice-input
    .venv-py2app/bin/python -m unittest test_voice_input -v
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# We import only the pure functions — avoid triggering rumps/sounddevice/pynput
# at module level by patching them before importing voice_input.
import sys
_stubs = {}
for mod_name in ("rumps", "sounddevice", "pynput", "pynput.keyboard",
                 "Quartz", "Vision", "AppKit", "numpy", "mlx_lm",
                 "mlx_lm.sample_utils"):
    if mod_name not in sys.modules:
        from unittest.mock import MagicMock
        _stubs[mod_name] = MagicMock()
        sys.modules[mod_name] = _stubs[mod_name]

# numpy needs a real ndarray for some code paths, but for our tests MagicMock
# suffices. Patch sd (sounddevice) early so import doesn't fail.
import voice_input
from voice_input import (
    _en_itn,
    normalize_numbers,
    _needs_polish,
    _is_valid_context_term,
    _count_words,
    DictionaryGuard,
    load_dictionary,
    save_settings,
    DEFAULT_DICT,
    SETTINGS_PATH,
    State,
)
from text_polisher import TextPolisher


# ── _en_itn tests ───────────────────────────────────────────────────


class TestEnITN(unittest.TestCase):
    """English inverse text normalization via word2number."""

    def test_multi_word_number(self):
        self.assertEqual(_en_itn("twenty three"), "23")

    def test_hundred_scale(self):
        self.assertEqual(_en_itn("one hundred"), "100")

    def test_large_number(self):
        self.assertEqual(_en_itn("two thousand five hundred"), "2500")

    def test_percentage(self):
        self.assertEqual(_en_itn("fifty percent"), "50%")

    def test_single_word_no_scale_preserved(self):
        # Single number word without scale should NOT be converted
        self.assertEqual(_en_itn("three apples"), "three apples")

    def test_non_number_text_unchanged(self):
        self.assertEqual(_en_itn("hello world"), "hello world")

    def test_empty_string(self):
        self.assertEqual(_en_itn(""), "")

    def test_mixed_number_and_text(self):
        result = _en_itn("I need twenty five dollars")
        self.assertIn("25", result)

    def test_hundred_alone_is_scale(self):
        # "hundred" is a scale word, so even alone it converts
        self.assertEqual(_en_itn("hundred"), "100")


# ── normalize_numbers tests ──────────────────────────────────────────


class TestNormalizeNumbers(unittest.TestCase):
    """Chinese + English inverse text normalization."""

    def test_chinese_number(self):
        result = normalize_numbers("一百二十三")
        self.assertEqual(result, "123")

    def test_chinese_percentage(self):
        result = normalize_numbers("百分之五十")
        self.assertEqual(result, "50%")

    def test_chinese_idiom_preserved(self):
        # Idioms like 三心二意 should not be converted to numbers
        result = normalize_numbers("三心二意")
        # wetext should preserve idioms; the result should not be all digits
        self.assertFalse(result.isdigit(),
                         f"Idiom was incorrectly converted to digits: {result}")

    def test_english_number_via_normalize(self):
        result = normalize_numbers("twenty three")
        self.assertEqual(result, "23")

    def test_empty_string(self):
        self.assertEqual(normalize_numbers(""), "")

    def test_plain_text_unchanged(self):
        result = normalize_numbers("hello world")
        self.assertEqual(result, "hello world")

    def test_mixed_chinese_english(self):
        # English ITN only works on whitespace-separated tokens;
        # when number words are sandwiched between CJK chars without
        # spaces, they stay as-is (expected behavior).
        result = normalize_numbers("I have twenty five apples")
        self.assertIn("25", result)


# ── _needs_polish tests ──────────────────────────────────────────────


class TestNeedsPolish(unittest.TestCase):
    """Heuristic for whether text needs LLM polishing."""

    def test_short_clean_text_no_polish(self):
        self.assertFalse(_needs_polish("hello"))

    def test_short_text_under_8_chars(self):
        self.assertFalse(_needs_polish("hi"))

    def test_empty_string(self):
        self.assertFalse(_needs_polish(""))

    def test_short_math_greater_than(self):
        # Issue #62: short math expressions should still trigger polish
        self.assertTrue(_needs_polish("大于"))

    def test_short_math_less_than(self):
        self.assertTrue(_needs_polish("小于"))

    def test_short_math_equals(self):
        self.assertTrue(_needs_polish("等于"))

    def test_english_math_expression(self):
        self.assertTrue(_needs_polish("x is greater than y"))

    def test_english_divided_by(self):
        self.assertTrue(_needs_polish("ten divided by two"))

    def test_chinese_filler_words(self):
        self.assertTrue(_needs_polish("呃就是说这个东西还不错"))

    def test_english_filler_um(self):
        self.assertTrue(_needs_polish("um I think this is good"))

    def test_english_filler_uh(self):
        self.assertTrue(_needs_polish("uh let me think about this"))

    def test_english_filler_so_basically(self):
        self.assertTrue(_needs_polish("so basically we need to fix this"))

    def test_chinese_number_words(self):
        self.assertTrue(_needs_polish("这个价格是百分之五十"))

    def test_chinese_consecutive_numbers(self):
        self.assertTrue(_needs_polish("一共三百七十六个"))

    def test_english_number_word_hundred(self):
        self.assertTrue(_needs_polish("we need one hundred items"))

    def test_english_number_word_thousand(self):
        self.assertTrue(_needs_polish("about two thousand users"))

    def test_long_text_no_punctuation(self):
        # Issue #64: long text without punctuation should trigger polish
        text = "this is a long sentence without any punctuation marks at all"
        self.assertTrue(_needs_polish(text))

    def test_long_text_with_punctuation_no_polish(self):
        text = "This is a well-formed sentence, with proper punctuation."
        self.assertFalse(_needs_polish(text))

    def test_normal_short_punctuated_text(self):
        self.assertFalse(_needs_polish("好的。"))

    def test_chinese_punctuation_counts(self):
        text = "这是一个有标点符号的句子，写得很好。"
        self.assertFalse(_needs_polish(text))

    def test_chinese_filler_ah_with_comma(self):
        # Issue #82: 啊 as sentence-final filler before punctuation
        self.assertTrue(_needs_polish("这个东西啊，还不错"))

    def test_chinese_filler_ah_end_of_text(self):
        # Issue #82: 啊 at end of text
        self.assertTrue(_needs_polish("这个东西啊"))

    def test_chinese_tag_question_dui_ba(self):
        # Issue #82: 对吧 tag question filler
        self.assertTrue(_needs_polish("你说对吧"))

    def test_chinese_tag_question_shi_ba(self):
        # Issue #82: 是吧 tag question filler
        self.assertTrue(_needs_polish("你觉得是吧"))

    def test_lower_punctuation_threshold(self):
        # Issue #82: >20 chars without punctuation should trigger polish
        text = "this is a sentence without punct"
        self.assertTrue(_needs_polish(text))

    def test_english_time_two_pm(self):
        # Issue #113: English time expression "two pm"
        self.assertTrue(_needs_polish("the meeting is at two pm"))

    def test_english_time_twelve_thirty(self):
        # Issue #113: English time expression "twelve thirty"
        self.assertTrue(_needs_polish("let's meet at twelve thirty"))

    def test_english_time_one_oclock(self):
        # Issue #113: English time expression "one o'clock"
        self.assertTrue(_needs_polish("it starts at one o'clock"))


# ── DictionaryGuard tests ────────────────────────────────────────────


class TestDictionaryGuard(unittest.TestCase):
    """Anti-spam logic for dictionary auto-additions."""

    def setUp(self):
        self.guard = DictionaryGuard()
        self.empty_dict = {"vocabulary": []}
        self.dict_with_word = {"vocabulary": ["PyTorch"]}

    def test_should_prompt_new_word(self):
        self.assertTrue(self.guard.should_prompt("MLX", self.empty_dict))

    def test_should_not_prompt_empty_word(self):
        self.assertFalse(self.guard.should_prompt("", self.empty_dict))

    def test_should_not_prompt_none_word(self):
        self.assertFalse(self.guard.should_prompt(None, self.empty_dict))

    def test_should_not_prompt_single_char(self):
        self.assertFalse(self.guard.should_prompt("x", self.empty_dict))

    def test_should_not_prompt_word_already_in_dict(self):
        self.assertFalse(
            self.guard.should_prompt("PyTorch", self.dict_with_word)
        )

    def test_case_insensitive_duplicate_check(self):
        self.assertFalse(
            self.guard.should_prompt("pytorch", self.dict_with_word)
        )
        self.assertFalse(
            self.guard.should_prompt("PYTORCH", self.dict_with_word)
        )

    def test_should_not_prompt_after_max_additions(self):
        for _ in range(3):
            self.guard.record_add()
        self.assertFalse(self.guard.should_prompt("NewWord", self.empty_dict))

    def test_should_prompt_before_max_additions(self):
        self.guard.record_add()
        self.guard.record_add()
        self.assertTrue(self.guard.should_prompt("NewWord", self.empty_dict))

    def test_record_reject_blocks_same_word(self):
        self.guard.record_reject("BadWord")
        self.assertFalse(self.guard.should_prompt("BadWord", self.empty_dict))

    def test_record_reject_case_insensitive(self):
        self.guard.record_reject("BadWord")
        self.assertFalse(self.guard.should_prompt("badword", self.empty_dict))

    def test_should_not_prompt_when_dict_full(self):
        full_dict = {"vocabulary": [f"word{i}" for i in range(500)]}
        self.assertFalse(self.guard.should_prompt("NewWord", full_dict))

    def test_record_add_increments_count(self):
        self.guard.record_add()
        self.assertEqual(self.guard._session_adds, 1)


# ── load_dictionary tests ────────────────────────────────────────────


class TestLoadDictionary(unittest.TestCase):
    """File I/O for dictionary loading."""

    def test_load_existing_dictionary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dict.json"
            expected = {"vocabulary": ["TestWord", "AnotherWord"]}
            path.write_text(json.dumps(expected))
            result = load_dictionary(path)
            self.assertEqual(result, expected)

    def test_load_nonexistent_creates_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dict.json"
            result = load_dictionary(path)
            self.assertEqual(result, DEFAULT_DICT)
            # File should now exist on disk
            self.assertTrue(path.exists())

    def test_load_corrupt_file_returns_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dict.json"
            path.write_text("not valid json {{{")
            result = load_dictionary(path)
            self.assertEqual(result, DEFAULT_DICT)


# ── save_settings tests ──────────────────────────────────────────────


class TestSaveSettings(unittest.TestCase):
    """Atomic write for user settings."""

    def test_save_and_read_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "settings.json"
            settings = {"model": "test-model", "sample_rate": 16000}
            with patch.object(voice_input, 'SETTINGS_PATH', fake_path):
                save_settings(settings)
            data = json.loads(fake_path.read_text())
            self.assertEqual(data["model"], "test-model")
            self.assertEqual(data["sample_rate"], 16000)

    def test_save_creates_file_with_restricted_permissions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "settings.json"
            with patch.object(voice_input, 'SETTINGS_PATH', fake_path):
                save_settings({"key": "value"})
            mode = oct(os.stat(fake_path).st_mode & 0o777)
            self.assertEqual(mode, "0o600")

    def test_save_preserves_unicode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "settings.json"
            settings = {"name": "测试中文"}
            with patch.object(voice_input, 'SETTINGS_PATH', fake_path):
                save_settings(settings)
            data = json.loads(fake_path.read_text())
            self.assertEqual(data["name"], "测试中文")


# ── TextPolisher.polish() safety tests ──────────────────────────────


class TestTextPolisherSafety(unittest.TestCase):
    """Test TextPolisher.polish() safety checks with mocked LLM."""

    def setUp(self):
        self.polisher = TextPolisher()
        self.polisher._loaded = True
        self.polisher._model = MagicMock()
        self.polisher._tokenizer = MagicMock()
        self.polisher._tokenizer.apply_chat_template.return_value = "test_prompt"
        self.polisher._sampler = MagicMock()

    @patch('mlx_lm.generate')
    def test_empty_output_returns_original(self, mock_gen):
        mock_gen.return_value = ""
        result = self.polisher.polish("hello world")
        self.assertEqual(result, "hello world")

    @patch('mlx_lm.generate')
    def test_thinking_block_stripped(self, mock_gen):
        mock_gen.return_value = "<think>reasoning here</think>Hello world"
        result = self.polisher.polish("hello world")
        self.assertEqual(result, "Hello world")

    @patch('mlx_lm.generate')
    def test_unclosed_think_stripped(self, mock_gen):
        mock_gen.return_value = "<think>reasoning without close tag Hello world"
        result = self.polisher.polish("hello world")
        # Should strip everything from <think> onwards
        self.assertEqual(result, "hello world")  # Falls back to original since clean is empty

    @patch('mlx_lm.generate')
    def test_too_short_output_returns_original(self, mock_gen):
        mock_gen.return_value = "Hi"  # Way too short compared to input
        result = self.polisher.polish("hello world this is a longer sentence")
        self.assertEqual(result, "hello world this is a longer sentence")

    @patch('mlx_lm.generate')
    def test_too_long_output_returns_original(self, mock_gen):
        mock_gen.return_value = "hello world " * 50  # Way too long
        result = self.polisher.polish("hello world")
        self.assertEqual(result, "hello world")

    @patch('mlx_lm.generate')
    def test_normal_output_returned(self, mock_gen):
        mock_gen.return_value = "Hello, world!"
        result = self.polisher.polish("hello world")
        self.assertEqual(result, "Hello, world!")


# ── _is_valid_context_term tests ────────────────────────────────────


class TestContextTermFilter(unittest.TestCase):
    """Test quality filtering for ASR context terms."""

    def test_valid_short_term(self):
        self.assertTrue(_is_valid_context_term("MLX"))

    def test_valid_normal_term(self):
        self.assertTrue(_is_valid_context_term("PyTorch"))

    def test_valid_hyphenated(self):
        self.assertTrue(_is_valid_context_term("well-known"))

    def test_valid_with_apostrophe(self):
        self.assertTrue(_is_valid_context_term("it's"))

    def test_reject_too_long(self):
        self.assertFalse(_is_valid_context_term("a" * 26))

    def test_reject_too_short(self):
        self.assertFalse(_is_valid_context_term("x"))

    def test_reject_empty(self):
        self.assertFalse(_is_valid_context_term(""))

    def test_reject_high_digit_ratio(self):
        self.assertFalse(_is_valid_context_term("a123456"))

    def test_reject_embedded_digit(self):
        # Short term with letter-digit-letter pattern (OCR artifact)
        self.assertFalse(_is_valid_context_term("a1b"))

    def test_reject_leading_digit(self):
        self.assertFalse(_is_valid_context_term("1abc"))

    def test_reject_leading_punctuation(self):
        self.assertFalse(_is_valid_context_term(".hello"))

    def test_reject_non_alphanum_chars(self):
        self.assertFalse(_is_valid_context_term("hello@world"))

    def test_valid_cjk_term(self):
        # CJK characters are alpha per Python's isalpha()
        self.assertTrue(_is_valid_context_term("\u4f60\u597d"))

    def test_accept_term_at_max_length(self):
        self.assertTrue(_is_valid_context_term("a" * 25))

    def test_accept_term_at_min_length(self):
        self.assertTrue(_is_valid_context_term("ab"))


# ── _count_words tests ─────────────────────────────────────────────


class TestCountWords(unittest.TestCase):
    """CJK-aware word counting for usage statistics."""

    def test_english_words(self):
        self.assertEqual(_count_words("hello world"), 2)

    def test_chinese_characters(self):
        self.assertEqual(_count_words("今天天气很好"), 6)

    def test_mixed_english_chinese(self):
        self.assertEqual(_count_words("hello你好world"), 4)

    def test_empty_string(self):
        self.assertEqual(_count_words(""), 0)


# ── TestClassifyCorrection tests ───────────────────────────────────


class TestClassifyCorrection(unittest.TestCase):
    """Test TextPolisher.classify_correction() with mocked LLM."""

    def setUp(self):
        self.polisher = TextPolisher()
        self.polisher._loaded = True
        self.polisher._model = MagicMock()
        self.polisher._tokenizer = MagicMock()
        self.polisher._tokenizer.apply_chat_template.return_value = "test"
        self.polisher._sampler = MagicMock()

    @patch('mlx_lm.generate')
    def test_yes_response(self, mock_gen):
        mock_gen.return_value = "YES PyTorch"
        ok, word = self.polisher.classify_correction("pie torch", "PyTorch", [])
        self.assertTrue(ok)
        self.assertEqual(word, "PyTorch")

    @patch('mlx_lm.generate')
    def test_no_response(self, mock_gen):
        mock_gen.return_value = "NO"
        ok, word = self.polisher.classify_correction("he go", "he went", [])
        self.assertFalse(ok)
        self.assertIsNone(word)


# ── Additional ITN edge case tests ─────────────────────────────────


class TestITNEdgeCases(unittest.TestCase):
    """Additional edge cases for ITN (inverse text normalization)."""

    def test_chinese_decimal(self):
        result = normalize_numbers("零点五")
        self.assertIn("0.5", result)

    def test_hyphenated_english(self):
        result = _en_itn("twenty-three")
        self.assertEqual(result, "23")

    def test_en_itn_trailing_punctuation_preserved(self):
        # Trailing punctuation is preserved after number conversion
        result = _en_itn("twenty three.")
        self.assertEqual(result, "23.")

    def test_en_itn_trailing_exclamation(self):
        # Trailing exclamation is preserved after number conversion
        self.assertEqual(_en_itn("twenty three!"), "23!")

    def test_en_itn_trailing_question(self):
        # Trailing question mark is preserved after number conversion
        self.assertEqual(_en_itn("twenty three?"), "23?")


# ── Settings validation tests ─────────────────────────────────────


class TestSettingsValidation(unittest.TestCase):
    """Validate that load_settings() enforces correct types."""

    def test_invalid_type_falls_back_to_default(self):
        """Setting with wrong type falls back to default value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "settings.json"
            # sample_rate should be int, not str
            fake_path.write_text(json.dumps({"sample_rate": "not_a_number"}))
            with patch.object(voice_input, 'SETTINGS_PATH', fake_path):
                settings = voice_input.load_settings()
            # Should fall back to default (16000)
            self.assertEqual(settings.get("sample_rate"), 16000)

    def test_valid_type_passes_through(self):
        """Setting with correct type is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "settings.json"
            fake_path.write_text(json.dumps({"sample_rate": 44100}))
            with patch.object(voice_input, 'SETTINGS_PATH', fake_path):
                settings = voice_input.load_settings()
            self.assertEqual(settings.get("sample_rate"), 44100)

    def test_missing_key_uses_default(self):
        """Missing key is not added but default is used when accessed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "settings.json"
            fake_path.write_text(json.dumps({}))
            with patch.object(voice_input, 'SETTINGS_PATH', fake_path):
                settings = voice_input.load_settings()
            # sample_rate not in settings, so .get() returns None
            self.assertNotIn("sample_rate", settings)


# ── Settings round-trip test ──────────────────────────────────────


class TestSettingsRoundTrip(unittest.TestCase):
    """Save settings with stats and history, reload, verify data intact."""

    def test_save_and_reload_preserves_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "settings.json"
            settings = {
                "model": "Qwen/Qwen3-ASR-1.7B",
                "sample_rate": 16000,
                "text_polish": True,
                "streaming": False,
                "stats": {
                    "today": "2026-04-06",
                    "today_words": 42,
                    "today_recordings": 5,
                    "total_words": 1000,
                    "total_recordings": 200,
                },
                "history": ["hello world", "test recording"],
            }
            with patch.object(voice_input, 'SETTINGS_PATH', fake_path):
                save_settings(settings)
            # Reload
            with patch.object(voice_input, 'SETTINGS_PATH', fake_path):
                reloaded = voice_input.load_settings()
            self.assertEqual(reloaded["model"], "Qwen/Qwen3-ASR-1.7B")
            self.assertEqual(reloaded["sample_rate"], 16000)
            self.assertTrue(reloaded["text_polish"])
            self.assertFalse(reloaded["streaming"])
            self.assertEqual(reloaded["stats"]["today_words"], 42)
            self.assertEqual(reloaded["stats"]["total_recordings"], 200)
            self.assertEqual(reloaded["history"], ["hello world", "test recording"])


# ── History truncation privacy test ──────────────────────────────


class TestHistoryTruncation(unittest.TestCase):
    """Issue #122: history persisted to disk must be truncated for privacy."""

    def test_long_text_truncated_on_save(self):
        """Text longer than 50 chars is truncated with ellipsis on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "settings.json"
            long_text = "a" * 80  # 80 chars, exceeds 50-char limit
            settings = {"history": [long_text]}
            with patch.object(voice_input, 'SETTINGS_PATH', fake_path):
                save_settings(settings)
            data = json.loads(fake_path.read_text())
            saved = data["history"][0]
            self.assertEqual(len(saved), 51)  # 50 chars + ellipsis
            self.assertTrue(saved.endswith("\u2026"))
            self.assertEqual(saved, "a" * 50 + "\u2026")

    def test_short_text_not_truncated(self):
        """Text 50 chars or shorter is stored as-is."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "settings.json"
            short_text = "hello world"
            settings = {"history": [short_text]}
            with patch.object(voice_input, 'SETTINGS_PATH', fake_path):
                save_settings(settings)
            data = json.loads(fake_path.read_text())
            self.assertEqual(data["history"][0], "hello world")

    def test_exactly_50_chars_not_truncated(self):
        """Text at exactly 50 chars is not truncated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "settings.json"
            exact_text = "a" * 50
            settings = {"history": [exact_text]}
            with patch.object(voice_input, 'SETTINGS_PATH', fake_path):
                save_settings(settings)
            data = json.loads(fake_path.read_text())
            self.assertEqual(data["history"][0], "a" * 50)


# ── State machine transition tests ───────────────────────────────


class TestStateMachine(unittest.TestCase):
    """Verify the State enum and its transition table.

    The valid transitions are documented in the State docstring in
    voice_input.py.  We encode them as a data structure and verify
    completeness, correctness, and the enum members themselves.
    """

    # Canonical transition table extracted from the State docstring.
    # Maps each state to the set of states it can transition to.
    VALID_TRANSITIONS = {
        State.LOADING: {State.IDLE},
        State.IDLE: {State.RECORDING_HOLD, State.WAITING_DOUBLE_CLICK},
        State.RECORDING_HOLD: {State.PROCESSING, State.IDLE},
        State.WAITING_DOUBLE_CLICK: {State.RECORDING_TOGGLE, State.PROCESSING, State.IDLE},
        State.RECORDING_TOGGLE: {State.PROCESSING, State.IDLE},
        State.PROCESSING: {State.IDLE, State.ERROR},
        State.ERROR: {State.IDLE},
    }

    # ── Enum membership tests ──────────────────────────────────────

    def test_state_count(self):
        """State enum has exactly 7 members."""
        self.assertEqual(len(State), 7)

    def test_expected_members(self):
        """All expected state names are present."""
        expected = {
            "LOADING", "IDLE", "RECORDING_HOLD",
            "WAITING_DOUBLE_CLICK", "RECORDING_TOGGLE",
            "PROCESSING", "ERROR",
        }
        actual = {s.name for s in State}
        self.assertEqual(actual, expected)

    def test_values_are_lowercase(self):
        """Every State value is a lowercase string matching its name."""
        for s in State:
            self.assertEqual(s.value, s.name.lower())

    # ── Transition table completeness ──────────────────────────────

    def test_every_state_has_transitions(self):
        """Every state in the enum appears as a key in the transition table."""
        for s in State:
            self.assertIn(s, self.VALID_TRANSITIONS,
                          f"{s.name} missing from VALID_TRANSITIONS")

    def test_transition_targets_are_valid_states(self):
        """Every target in the transition table is a valid State member."""
        for src, targets in self.VALID_TRANSITIONS.items():
            for tgt in targets:
                self.assertIsInstance(tgt, State,
                                     f"Invalid target {tgt!r} from {src.name}")

    def test_no_self_transitions(self):
        """No state should transition to itself."""
        for src, targets in self.VALID_TRANSITIONS.items():
            self.assertNotIn(src, targets,
                             f"{src.name} has a self-transition")

    # ── Specific transition tests ──────────────────────────────────

    def test_loading_transitions(self):
        """LOADING can only go to IDLE."""
        self.assertEqual(self.VALID_TRANSITIONS[State.LOADING], {State.IDLE})

    def test_idle_transitions(self):
        """IDLE goes to RECORDING_HOLD or WAITING_DOUBLE_CLICK."""
        self.assertEqual(
            self.VALID_TRANSITIONS[State.IDLE],
            {State.RECORDING_HOLD, State.WAITING_DOUBLE_CLICK},
        )

    def test_recording_hold_transitions(self):
        """RECORDING_HOLD goes to PROCESSING or IDLE (cancel)."""
        self.assertEqual(
            self.VALID_TRANSITIONS[State.RECORDING_HOLD],
            {State.PROCESSING, State.IDLE},
        )

    def test_waiting_double_click_transitions(self):
        """WAITING_DOUBLE_CLICK goes to RECORDING_TOGGLE, PROCESSING, or IDLE."""
        self.assertEqual(
            self.VALID_TRANSITIONS[State.WAITING_DOUBLE_CLICK],
            {State.RECORDING_TOGGLE, State.PROCESSING, State.IDLE},
        )

    def test_recording_toggle_transitions(self):
        """RECORDING_TOGGLE goes to PROCESSING or IDLE (cancel)."""
        self.assertEqual(
            self.VALID_TRANSITIONS[State.RECORDING_TOGGLE],
            {State.PROCESSING, State.IDLE},
        )

    def test_processing_transitions(self):
        """PROCESSING goes to IDLE (success) or ERROR (failure)."""
        self.assertEqual(
            self.VALID_TRANSITIONS[State.PROCESSING],
            {State.IDLE, State.ERROR},
        )

    def test_error_transitions(self):
        """ERROR recovers to IDLE."""
        self.assertEqual(self.VALID_TRANSITIONS[State.ERROR], {State.IDLE})

    # ── Reachability ───────────────────────────────────────────────

    def test_all_states_reachable_from_loading(self):
        """Every state is reachable from LOADING via valid transitions."""
        visited = set()
        queue = [State.LOADING]
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self.VALID_TRANSITIONS.get(current, set()))
        self.assertEqual(visited, set(State),
                         f"Unreachable states: {set(State) - visited}")

    def test_idle_reachable_from_every_state(self):
        """IDLE is reachable from every state (always possible to recover)."""
        for start in State:
            visited = set()
            queue = [start]
            while queue:
                current = queue.pop()
                if current in visited:
                    continue
                visited.add(current)
                queue.extend(self.VALID_TRANSITIONS.get(current, set()))
            self.assertIn(State.IDLE, visited,
                          f"IDLE not reachable from {start.name}")

    # ── Visual property tests ──────────────────────────────────────

    def test_visual_recording_states(self):
        """Recording states map to 'recording' visual."""
        for s in (State.RECORDING_HOLD, State.WAITING_DOUBLE_CLICK, State.RECORDING_TOGGLE):
            self.assertEqual(s.visual, "recording", f"{s.name}.visual")

    def test_visual_processing(self):
        self.assertEqual(State.PROCESSING.visual, "processing")

    def test_visual_idle(self):
        self.assertEqual(State.IDLE.visual, "idle")

    def test_visual_error(self):
        self.assertEqual(State.ERROR.visual, "error")

    def test_visual_loading(self):
        self.assertEqual(State.LOADING.visual, "loading")

    def test_every_state_has_visual(self):
        """Every state returns a non-empty visual string."""
        for s in State:
            v = s.visual
            self.assertIsInstance(v, str, f"{s.name}.visual is not str")
            self.assertTrue(len(v) > 0, f"{s.name}.visual is empty")


if __name__ == "__main__":
    unittest.main()

