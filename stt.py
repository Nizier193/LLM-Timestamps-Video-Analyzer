import librosa
import random
import string
from pydub import AudioSegment
import whisper_timestamped as whisper
import os

def generate_random_string(length: int) -> str:
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

class WhisperSTT:
    def __init__(self, model: str = "tiny"):
        self.duration = 0 # save the duration for keep the timing during the merge
        self.model = whisper.load_model(model, device="cpu")

    def get_transcript(self, audio_path: str) -> list[tuple[str, float, float]]:
        result = []
        result_string = ""

        for chunk in self.chunks_audio(audio_path):
            transcript = self.__call_whisper__(chunk[0])
            for (path, text, start, end) in self.clean_transcript(chunk[0], transcript, result):
                result.append((path, text, start, end))
                result_string += f"{text}: {start[0]} - {end[-1]}\n"
        
        self.__clean_global__()
        return result, result_string, AudioSegment.from_file(audio_path).duration_seconds

    def clean_transcript(self, audio_path, transcript, prev_transcript) -> list[tuple[str, str, list, list]]:
        print(f"Cleaning the STT...")

        result = []
        
        if len(prev_transcript) > 1:
            audio, sample_rate = librosa.load(prev_transcript[-1][0])
            self.duration += librosa.get_duration(y=audio, sr=sample_rate)

        for i in range(0, len(transcript['segments']), 5):
            segments = transcript['segments'][i:i+5]
            for segment in segments:
                    words = ' '.join([x['text'] for x in segment['words']])
                    words_start_times = [x['start'] + self.duration for x in segment['words']]
                    words_end_times = [x['end'] + self.duration for x in segment['words']]
                    result.append((audio_path, words, words_start_times, words_end_times))

        print(f"Process completed.")
        return result
            

    def get_transcript_v2(self, audio_path: str, n_words_chunk: int = 4):
        transcript = self.__call_whisper__(audio_path)

        result = []
        subtitle_number = 1  # Инициализируем счетчик субтитров

        for chunk in transcript['segments']:
            for word_index in range(0, len(chunk['words']), n_words_chunk):
                word_chunk = chunk['words'][word_index:word_index + n_words_chunk]
                start_time = word_chunk[0]['start']
                end_time = word_chunk[-1]['end']
                word = ' '.join([x['text'] for x in word_chunk])
                confidence = sum([x['confidence'] for x in word_chunk]) / len(word_chunk)

                result.append(
                    {
                        "subtitle_number": subtitle_number,  # Добавляем номер субтитра
                        "start_timecode": start_time,
                        "end_timecode": end_time,
                        "subtitle": word,
                        "confidence": confidence
                    }
                )
                subtitle_number += 1  # Увеличиваем счетчик для следующего субтитра

        return result
    
    def chunks_audio(self, audio_path: str):
        audio = AudioSegment.from_file(audio_path)
        size = self.calc_chunks_size(audio.duration_seconds)
        chunk_duration = audio.duration_seconds / size
        chunks = []

        print(f"\nChunking the audio...")
        print(f"Duration: {audio.duration_seconds}")
        print(f"Single Chunk Duration: {chunk_duration}")
        print(f"Size: {size}\n")

        # Create 'tmp' directory if it doesn't exist
        os.makedirs('tmp', exist_ok=True)

        for i in range(size):
            start = i * chunk_duration * 1000  # pydub works with milliseconds
            end = (i + 1) * chunk_duration * 1000
            tmp_audio = f"tmp/{generate_random_string(16)}.mp3"
            audio_chunk = audio[start:end]
            audio_chunk.export(tmp_audio, format="mp3")
            chunks.append((tmp_audio, start/1000, end/1000))

        return chunks
    
    @staticmethod
    def calc_chunks_size(duration):
        return max(1, int((duration / 60) / 15))

    def __call_whisper__(self, audio_path):
        print(f'\nLoading audio {audio_path}...')
        audio = whisper.load_audio(audio_path)
        transcript = whisper.transcribe(self.model, audio, language="ru", verbose=False)
        return transcript

    def __clean_global__(self):
        self.duration = 0