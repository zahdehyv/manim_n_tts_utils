from pathlib import Path

from manim import logger
from manim_voiceover.helper import remove_bookmarks, wav2mp3
from manim_voiceover.services.base import SpeechService
import wave

try:
    from piper.voice import PiperVoice
except ImportError:
    logger.error("Missing packages. Run `pip install TTS` to use CoquiService.")


class PiperService(SpeechService):
    """Speech service for Coqui TTS.
    Default model: ``tts_models/en/ljspeech/tacotron2-DDC``.
    """

    def __init__(
        self,
        model_path: str = "./",
        progress_bar: bool = True,
        gpu=False,
        **kwargs,
    ):
        self.tts = PiperVoice.load(model_path)

        # Run TTS
        
        self.init_kwargs = kwargs
        SpeechService.__init__(self, **kwargs)

    def generate_from_text(
        self, text: str, cache_dir: str = None, path: str = None, **kwargs
    ) -> dict:
        if cache_dir is None:
            cache_dir = self.cache_dir

        input_text = remove_bookmarks(text)
        input_data = {"input_text": text, "service": "piper"}

        cached_result = self.get_cached_result(input_data, cache_dir)
        if cached_result is not None:
            return cached_result

        if path is None:
            audio_path = self.get_audio_basename(input_data) + ".mp3"
        else:
            audio_path = path

        if not kwargs:
            kwargs = self.init_kwargs

        output_path = str(Path(cache_dir) / audio_path)
        wav_path = Path(output_path).with_suffix(".wav")

        # Text to speech to a file
        wav_file = wave.open(str(wav_path), 'w')
        # text = "This is an example of text-to-speech using Piper TTS."
        audio = self.tts.synthesize(input_text,wav_file)

        wav2mp3(wav_path, output_path)

        json_dict = {
            "input_text": text,
            "input_data": input_data,
            "original_audio": audio_path,
            # "word_boundaries": word_boundaries,
        }

        return json_dict