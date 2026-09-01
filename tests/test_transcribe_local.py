import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from helpers.transcribe import call_faster_whisper


class FasterWhisperAdapterTests(unittest.TestCase):
    def test_converts_words_to_existing_transcript_format(self):
        segment = SimpleNamespace(words=[
            SimpleNamespace(word=" hola", start=1.0, end=1.4),
            SimpleNamespace(word=" mundo", start=1.5, end=2.0),
        ])
        model = SimpleNamespace(transcribe=lambda *args, **kwargs: (iter([segment]), None))
        module = SimpleNamespace(WhisperModel=lambda *args, **kwargs: model)

        with patch.dict(sys.modules, {"faster_whisper": module}):
            result = call_faster_whisper(Path("audio.wav"), "es", "tiny")

        self.assertEqual(result, {"words": [
            {"type": "word", "text": "hola", "start": 1.0, "end": 1.4},
            {"type": "word", "text": "mundo", "start": 1.5, "end": 2.0},
        ]})


if __name__ == "__main__":
    unittest.main()
