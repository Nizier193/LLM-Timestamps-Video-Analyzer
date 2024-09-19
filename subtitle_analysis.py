from typing import List

from pydantic import BaseModel

from settings import Source
from settings import OUTPUT_FILES
from stt import WhisperSTT

import re

from youtube_transcript_api import YouTubeTranscriptApi
from pydub import AudioSegment
from aicorrection import AICorrection

import ffmpeg
import json
import os

class YoutubeTranscript():
    def extract_video_id(self, yt_vid_url: str) -> str:
        # Updated regex pattern to match various YouTube URL formats
        pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/)|youtu\.be\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})(?:\S+)?'

        match = re.search(pattern, yt_vid_url)
        if match:
            return match.group(1)
        else:
            return None
        
    def get_transcript(self, yt_video_url: str, srt_save_path: str = OUTPUT_FILES) -> List[dict]:
        yt_video_id = self.extract_video_id(yt_video_url)
        transcript = YouTubeTranscriptApi.get_transcript(yt_video_id)

        json_content = []
        for entry in transcript:
            start = entry['start']
            duration = entry['duration']
            text = entry['text']

            start_timecode = self.format_timecode(start)
            end_timecode = self.format_timecode(start + duration)

            json_content.append({
                "start_timecode": start_timecode,
                "end_timecode": end_timecode,
                "subtitle": text
            })

        # Ensure the output directory exists
        os.makedirs(srt_save_path, exist_ok=True)

        with open(os.path.join(srt_save_path, 'subtitles.json'), 'w', encoding='utf-8') as file:
            json.dump(json_content, file, ensure_ascii=False, indent=4)

        return json_content

    def format_timecode(self, seconds: float) -> str:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

class SubtitlesAnalysis():
    """
    Анализ субтитров.
    Берёт субтитры и анализирует их.
    """

    def get_audio(self, video_path: str) -> str:
        output_path = video_path.rsplit('.', 1)[0] + '.mp3'
        
        try:
            audio = AudioSegment.from_file(video_path, format="mp4")
            audio.export(output_path, format="mp3")
            return output_path
        except Exception as e:
            print(f'Error occurred: {str(e)}')
            return None
        
    def get_youtube_subtitles(self, youtube_video_url: str):
        # Works ok

        extractor = YoutubeTranscript()
        subtitles = extractor.get_transcript(youtube_video_url)

        return subtitles

    def get_local_subtitles(self, video_path: str):
        audio_path = self.get_audio(video_path)

        extractor = WhisperSTT("tiny")
        transcript = extractor.get_transcript_v2(
            audio_path=audio_path,
            n_words_chunk=10
        )

        return transcript

    def AI_analysis(self, subtitles: List[dict]):
        corrector = AICorrection()
        corrected_subtitles = corrector.run(subtitles)

        return corrected_subtitles

    def run(self, output_json: str, source: str = Source.Local, video_path: str = None, youtube_video_url: str = None) -> List[dict]:
        """
        Анализ субтитров. Возвращает json-объект с субтитрами.
        """

        if source == Source.Local:
            subtitles = self.get_local_subtitles(video_path)
        elif source == Source.Youtube:
            subtitles = self.get_youtube_subtitles(youtube_video_url)
        else:
            raise ValueError(f"Invalid source: {source}")

        analysis = self.AI_analysis(subtitles)

        with open(output_json, 'w', encoding='utf-8') as json_file:
            json.dump(analysis, json_file, ensure_ascii=False, indent=4)

        return analysis