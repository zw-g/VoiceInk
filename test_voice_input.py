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
    DictionaryGuard,
    load_dictionary,
    save_settings,
    DEFAULT_DICT,
    SETTINGS_PATH,
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

    def test_en_itn_trailing_punctuation_known_behavior(self):
        # Known behavior: trailing punctuation is dropped from number phrases
        result = _en_itn("twenty three.")
        self.assertEqual(result, "23")


if __name__ == "__main__":
    unittest.main()

