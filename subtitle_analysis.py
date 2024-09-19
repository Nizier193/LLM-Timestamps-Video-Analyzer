from typing import List

from pydantic import BaseModel

from settings import Source
from settings import OUTPUT_FILES

import re

from youtube_transcript_api import YouTubeTranscriptApi

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
        """
        Извлекает аудиодорожку из видеофайла в формате ogg.
        """
        output_path = video_path.rsplit('.', 1)[0] + '.ogg'
        
        try:
            (
                ffmpeg
                .input(video_path)
                .output(output_path, acodec='libvorbis', audio_bitrate='128k')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return output_path
        except ffmpeg.Error as e:
            print(f'Error occurred: {e.stderr.decode()}')
            return None
        
    def get_youtube_subtitles(self, youtube_video_url: str):
        # Works ok

        extractor = YoutubeTranscript()
        subtitles = extractor.get_transcript(youtube_video_url)

        return subtitles

    def get_local_subtitles(self, video_path: str):
        audio_path = self.get_audio(video_path)

        subtitles = {
            "start_timecode": "00:00:00,000",
            "end_timecode": "00:00:00,000",
            "analysis": "Анализ субтитров"
        }

        return subtitles

    def AI_analysis(self, subtitles: List[dict]):
        # No logic yet
        # AI Analysis works weird

        return subtitles

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
