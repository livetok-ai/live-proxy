from providers.cartesia.utils import aggreagate_sentences, parse_speakeable_text


class TestParseSpeakeableText:
    """Test cases for parse_speakeable_text function."""

    def test_complete_self_closing_tag(self):
        """Test with complete self-closing tag."""
        text = 'Sure! <break time="250ms" />'
        result = parse_speakeable_text(text)
        assert result == ('Sure! <break time="250ms" />', None)

    def test_incomplete_tag(self):
        """Test with incomplete tag."""
        text = "Sure! <break"
        result = parse_speakeable_text(text)
        assert result == ("Sure! ", "<break")

    def test_complete_tag_with_closing_tag(self):
        """Test with complete tag that has a closing tag."""
        text = 'Sure! <break time="250ms" /> <prosody rate="0.85">Hee hee hee. </prosody>Why'
        result = parse_speakeable_text(text)
        assert result == ('Sure! <break time="250ms" /> <prosody rate="0.85">Hee hee hee. </prosody>Why', None)

    def test_incomplete_tag_with_space(self):
        """Test with incomplete tag after space."""
        text = 'Sure! <break time="250ms" /> <prosody ra'
        result = parse_speakeable_text(text)
        assert result == ('Sure! <break time="250ms" /> ', "<prosody ra")

    def test_square_bracket_notation(self):
        """Test with square bracket notation (like [laughter])."""
        text = "Sure! [laughter]"
        result = parse_speakeable_text(text)
        assert result == ("Sure! [laughter]", None)

    def test_incomplete_square_bracket(self):
        """Test with incomplete square bracket notation."""
        text = "Sure! [laught"
        result = parse_speakeable_text(text)
        assert result == ("Sure! [laught", None)

    def test_no_tags(self):
        """Test with text that has no tags."""
        text = "This is plain text"
        result = parse_speakeable_text(text)
        assert result == ("This is plain text", None)

    def test_empty_string(self):
        """Test with empty string."""
        text = ""
        result = parse_speakeable_text(text)
        assert result == ("", None)

    def test_only_tag(self):
        """Test with only a tag."""
        text = '<break time="250ms" />'
        result = parse_speakeable_text(text)
        assert result == ('<break time="250ms" />', None)

    def test_incomplete_tag_at_start(self):
        """Test with incomplete tag at the start."""
        text = '<break time="250ms"'
        result = parse_speakeable_text(text)
        assert result == ("", '<break time="250ms"')


class TestAggreateSentences:
    """Test cases for aggreagate_sentences function."""

    def test_complete_sentence_with_period(self):
        """Test with a complete sentence ending with period."""
        speakable = "Hello world."
        incomplete = None
        result = aggreagate_sentences(speakable, incomplete)
        assert result == ("Hello world.", None)

    def test_complete_sentence_with_question_mark(self):
        """Test with a complete sentence ending with question mark."""
        speakable = "How are you?"
        incomplete = None
        result = aggreagate_sentences(speakable, incomplete)
        assert result == ("How are you?", None)

    def test_complete_sentence_with_exclamation(self):
        """Test with a complete sentence ending with exclamation mark."""
        speakable = "Hello there!"
        incomplete = None
        result = aggreagate_sentences(speakable, incomplete)
        assert result == ("Hello there!", None)

    def test_incomplete_sentence_no_punctuation(self):
        """Test with incomplete sentence (no ending punctuation)."""
        speakable = "Hello world"
        incomplete = None
        result = aggreagate_sentences(speakable, incomplete)
        assert result == ("", "Hello world")

    def test_multiple_complete_sentences(self):
        """Test with multiple complete sentences."""
        speakable = "Hello world. How are you?"
        incomplete = None
        result = aggreagate_sentences(speakable, incomplete)
        assert result == ("Hello world. How are you?", None)

    def test_mixed_complete_and_incomplete(self):
        """Test with complete sentence followed by incomplete."""
        speakable = "Hello world. How are"
        incomplete = None
        result = aggreagate_sentences(speakable, incomplete)
        assert result == ("Hello world.", "How are")

    def test_incomplete_with_existing_incomplete_text(self):
        """Test aggregating incomplete sentence with existing incomplete text."""
        speakable = "you doing"
        incomplete = "How are "
        result = aggreagate_sentences(speakable, incomplete)
        assert result == ("", "How are you doing")

    def test_complete_with_existing_incomplete_text(self):
        """Test complete sentence with existing incomplete text (returns as-is)."""
        speakable = "you doing?"
        incomplete = "How are "
        result = aggreagate_sentences(speakable, incomplete)
        # The function doesn't combine incomplete with speakable - it just returns them as-is
        assert result == ("you doing?", "How are ")

    def test_sentence_ending_with_tag_close(self):
        """Test sentence ending with > tag marker."""
        speakable = "Hello <prosody rate='0.85'>world</prosody>"
        incomplete = None
        result = aggreagate_sentences(speakable, incomplete)
        assert result == ("Hello <prosody rate='0.85'>world</prosody>", None)

    def test_sentence_ending_with_square_bracket(self):
        """Test sentence ending with ] square bracket."""
        speakable = "Hello [laughter]"
        incomplete = None
        result = aggreagate_sentences(speakable, incomplete)
        assert result == ("Hello [laughter]", None)

    def test_empty_speakable_text(self):
        """Test with empty speakable text."""
        speakable = ""
        incomplete = None
        result = aggreagate_sentences(speakable, incomplete)
        assert result == ("", None)

    def test_east_asian_punctuation_period(self):
        """Test with East Asian full stop (。)."""
        speakable = "こんにちは。"
        incomplete = None
        result = aggreagate_sentences(speakable, incomplete)
        assert result == ("こんにちは。", None)

    def test_east_asian_punctuation_question(self):
        """Test with East Asian question mark (？)."""
        speakable = "元気ですか？"
        incomplete = None
        result = aggreagate_sentences(speakable, incomplete)
        assert result == ("元気ですか？", None)

    def test_arabic_punctuation(self):
        """Test with Arabic question mark (؟)."""
        speakable = "كيف حالك؟"
        incomplete = None
        result = aggreagate_sentences(speakable, incomplete)
        assert result == ("كيف حالك؟", None)

    def test_devanagari_danda(self):
        """Test with Devanagari danda (।)."""
        speakable = "नमस्ते।"
        incomplete = None
        result = aggreagate_sentences(speakable, incomplete)
        assert result == ("नमस्ते।", None)

    def test_ellipsis_punctuation(self):
        """Test with ellipsis (…)."""
        speakable = "Wait for it…"
        incomplete = None
        result = aggreagate_sentences(speakable, incomplete)
        assert result == ("Wait for it…", None)

    def test_multiple_sentences_last_incomplete(self):
        """Test with multiple sentences where last is incomplete."""
        speakable = "First sentence. Second sentence. Third incomplete"
        incomplete = None
        result = aggreagate_sentences(speakable, incomplete)
        # sent_tokenize joins without space, so we get the concatenated result
        assert result == ("First sentence.Second sentence.", "Third incomplete")
