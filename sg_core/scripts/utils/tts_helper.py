# -*- coding: utf-8 -*-
import datetime
import os
import re
import time

from google.cloud import texttospeech


class TTSHelper:
    """ helper class for google TTS
    set the environment variable GOOGLE_APPLICATION_CREDENTIALS first
    GOOGLE_APPLICATION_CREDENTIALS = 'path to json key file'
    """

    cache_folder = './cached_wav/'

    def __init__(self, cache_path=None):
        if cache_path is not None:
            self.cache_folder = cache_path

        # create cache folder
        try:
            os.makedirs(self.cache_folder)
        except OSError:
            pass

        # init tts
        self.client = texttospeech.TextToSpeechClient()
        self.voice_en_standard = texttospeech.types.VoiceSelectionParams(
            language_code='en-US', name='en-US-Standard-B')
        self.voice_en_female = texttospeech.types.VoiceSelectionParams(
            language_code='en-US', name='en-US-Wavenet-F')
        self.voice_en_female_2 = texttospeech.types.VoiceSelectionParams(
            language_code='en-US', name='en-US-Wavenet-C')
        self.voice_gb_female = texttospeech.types.VoiceSelectionParams(
            language_code='en-GB', name='en-GB-Wavenet-C')
        self.voice_en_male = texttospeech.types.VoiceSelectionParams(
            language_code='en-US', name='en-US-Wavenet-D')
        self.voice_en_male_2 = texttospeech.types.VoiceSelectionParams(
            language_code='en-US', name='en-US-Wavenet-A')
        # self.voice_en_male_2 = texttospeech.types.VoiceSelectionParams(
        #     language_code='en-US', name='en-AU-Wavenet-B')
        self.voice_ko_female = texttospeech.types.VoiceSelectionParams(
            language_code='ko-KR', name='ko-KR-Wavenet-A')
        self.voice_ko_male = texttospeech.types.VoiceSelectionParams(
            language_code='ko-KR', name='ko-KR-Wavenet-D')
        self.audio_config_en = texttospeech.types.AudioConfig(
            # speaking_rate=0.67,
            speaking_rate=1.0,
            audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16)  # using WAV takes more time than MP3 (about 0.Xs)
        self.audio_config_en_slow = texttospeech.types.AudioConfig(
            speaking_rate=0.85,
            audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16)  # using WAV takes more time than MP3 (about 0.Xs)
        self.audio_config_kr = texttospeech.types.AudioConfig(
            speaking_rate=1.0,
            audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16)

        # clean up cache folder
        self._cleanup_cachefolder()

    def _cleanup_cachefolder(self):
        """ remove least accessed files in the cache """
        dir_to_search = self.cache_folder
        for dirpath, dirnames, filenames in os.walk(dir_to_search):
            for file in filenames:
                curpath = os.path.join(dirpath, file)
                file_accessed = datetime.datetime.fromtimestamp(os.path.getatime(curpath))
                if datetime.datetime.now() - file_accessed > datetime.timedelta(days=30):
                    os.remove(curpath)

    def _string2numeric_hash(self, text):
        import hashlib
        return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:16], 16)

    def synthesis(self, ssml_text, voice_name='en-female', verbose=False):
        if not ssml_text.startswith(u'<speak>'):
            ssml_text = u'<speak>' + ssml_text + u'</speak>'

        filename = os.path.join(self.cache_folder, str(self._string2numeric_hash(voice_name + ssml_text)) + '.wav')

        # load or synthesis audio
        if not os.path.exists(filename):
            if verbose:
                start = time.time()

            # let's synthesis
            if voice_name == 'en-female':
                voice = self.voice_en_female
                audio_config = self.audio_config_en
            elif voice_name == 'en-female_2':
                voice = self.voice_en_female_2
                audio_config = self.audio_config_en_slow
                # audio_config = self.audio_config_en
            elif voice_name == 'gb-female':
                voice = self.voice_gb_female
                audio_config = self.audio_config_en
            elif voice_name == 'en-male':
                voice = self.voice_en_male
                audio_config = self.audio_config_en
            elif voice_name == 'en-male_2':
                voice = self.voice_en_male_2
                # audio_config = self.audio_config_en_slow
                audio_config = self.audio_config_en
            elif voice_name == 'kr-female':
                voice = self.voice_ko_female
                audio_config = self.audio_config_kr
            elif voice_name == 'kr-male':
                voice = self.voice_ko_male
                audio_config = self.audio_config_kr
            elif voice_name == 'en-standard':
                voice = self.voice_en_standard
                audio_config = self.audio_config_en
            else:
                raise ValueError

            synthesis_input = texttospeech.types.SynthesisInput(ssml=ssml_text)
            response = self.client.synthesize_speech(synthesis_input, voice, audio_config)

            if verbose:
                print('synthesis: took {0:.2f} seconds'.format(time.time() - start))
                start = time.time()

            # save to a file
            with open(filename, 'wb') as out:
                out.write(response.audio_content)
                if verbose:
                    print('written to file "{}"'.format(filename))

            if verbose:
                print('save wav file: took {0:.2f} seconds'.format(time.time() - start))
        else:
            if verbose:
                print('use cached file "{}"'.format(filename))

        return filename


def test_tts_helper():
    tts = TTSHelper()

    voice = 'en-male'  # 'kr'
    text = '<prosody rate="slow">load a new sound buffer from a filename</prosody>, a <emphasis level="strong">python</emphasis> file object'

    # voice = 'kr'
    # text = u'나는 나오입니다 <break time="1s"/> 안녕하세요.'

    # split into sentences
    sentences = list(filter(None, re.split("[.,!?:\-]+", text)))
    sentences = [s.strip().lower() for s in sentences]
    print(sentences)

    # synthesis
    filenames = []
    for s in sentences:
        filenames.append(tts.synthesis(s, voice_name=voice, verbose=True))

    # play
    for f in filenames:
        sound_obj, duration = tts.get_sound_obj(f)
        tts.play(sound_obj)
        print('playing... {0:.2f} seconds'.format(duration))
        time.sleep(duration)


if __name__ == '__main__':
    test_tts_helper()
