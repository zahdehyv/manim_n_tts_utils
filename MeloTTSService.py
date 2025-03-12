from pathlib import Path

from manim import logger
from manim_voiceover.helper import remove_bookmarks, wav2mp3
from manim_voiceover.services.base import SpeechService
import wave

try:
    from melo.api import TTS
except ImportError:
    logger.error("Missing packages. Run `pip install TTS` to use CoquiService.")


class MeloService(SpeechService):
    """Speech service for Coqui TTS.
    Default model: ``tts_models/en/ljspeech/tacotron2-DDC``.
    """

    def __init__(
        self,
        speaker_id = 'EN-BR',
        device: str = "cuda",
        speed = 1.0,
        progress_bar: bool = True,
        gpu=False,
        **kwargs,
    ):
        self.tts = TTS(language='EN', device=device)
        speaker_ids = self.tts.hps.data.spk2id
        self.speaker_id = speaker_ids[speaker_id]
        self.speed = speed
        # Run TTS
        
        self.init_kwargs = kwargs
        SpeechService.__init__(self, **kwargs)

    def generate_from_text(
        self, text: str, cache_dir: str = None, path: str = None, **kwargs
    ) -> dict:
        if cache_dir is None:
            cache_dir = self.cache_dir

        input_text = remove_bookmarks(text)
        input_data = {"input_text": text, "service": "melo"}

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
        self.tts.tts_to_file(input_text, self.speaker_id, str(wav_path), speed=self.speed)

        wav2mp3(wav_path, output_path)

        json_dict = {
            "input_text": text,
            "input_data": input_data,
            "original_audio": audio_path,
            # "word_boundaries": word_boundaries,
        }

        return json_dict