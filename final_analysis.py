import base64
import json
import os
import io
from datetime import timedelta
from typing import List

from pydantic import BaseModel
from openai import OpenAI

from dotenv import load_dotenv
from PIL import Image
import cv2
import ffmpeg

from uuid import uuid4

from settings import Source

from image_analysis import ImageAnalysis
from video_analysis import VideoAnalysis
from subtitle_analysis import SubtitlesAnalysis
from settings import InterestingMoments

import yt_dlp
from pathlib import Path

load_dotenv()

def format_analysis_text(concat_analysis, start_fragment: int = 1, n_fragments: int = 20):
    analysis_template = """
Fragment {index}:
{separator}
Video: {video_start} --> {video_end}
---
{video_analysis}
---
Subtitles:
{subtitles}
"""
    
    separator = '=' * 40
    
    analysis_fragments = []
    for index, item in enumerate(concat_analysis[start_fragment - 1:start_fragment+n_fragments], start_fragment):
        video = item["video_analysis"]
        subtitles = item["subtitles_analysis"]
        
        if subtitles:
            subtitles_text = "\n".join(
                "{start} --> {end}: {text}".format(
                    start=sub['start_timecode'],
                    end=sub['end_timecode'],
                    text=sub['subtitle']
                ) for sub in subtitles
            )
        else:
            subtitles_text = "No subtitles for this fragment."
        
        fragment = analysis_template.format(
            index=index,
            separator=separator,
            video_start=video['start_timecode'],
            video_end=video['end_timecode'],
            video_analysis=video['analysis'],
            subtitles=subtitles_text
        )
        analysis_fragments.append(fragment)
                
    return "\n\n".join(analysis_fragments)

class VideoAnalysisBySubtitles():
    # TODO: Rewrite prompt to be more clear and more functional
    analysis_prompt = """
Ты - ассистент, который анализирует видео и субтитры.

Тебе нужно выбрать ТРИ самых интересных фрагмента из видео и дать оценку на интересность от 1 до 5.

Правила к выбору фрагментов:
* Если есть видео и субтитры - отдаём приоритет видео с субтитрами
* Если есть субтитры, но нет видео - отдаём приоритет субтитров
* Если есть красивое видео - отдаём приоритет видео

Ты можешь и должен объединять субтитры для видео, если это сделает видео более интересным.
Не делай видео более минуты.

ЕСЛИ ЕСТЬ СУБТИТРЫ ОБРЕЗАТЬ ТОЛЬКО ПО НИМ
МОЖНО ОБРЕЗАТЬ ВИДЕО
    """

    def __init__(self):
        self.video_analysis = VideoAnalysis(
            interval_seconds=10,
            resize_factor=30
        )
        self.subtitles_analysis = SubtitlesAnalysis()

    def video_subtitles_concat(self, video_analysis_json: str, subtitles_json: str, output_json: str) -> None:
        # Чтение JSON файлов с анализом видео и субтитрами
        with open(video_analysis_json, 'r', encoding='utf-8') as va_file:
            video_analysis = json.load(va_file)
        
        with open(subtitles_json, 'r', encoding='utf-8') as st_file:
            subtitles_analysis = json.load(st_file)
        
        # Объединение анализов на основе частично совпадающих таймкодов
        combined_analysis = []
        for va in video_analysis:
            va_start = self.timecode_to_seconds(va["start_timecode"])
            va_end = self.timecode_to_seconds(va["end_timecode"])
            
            matching_subtitles = []
            for st in subtitles_analysis:
                st_start = self.timecode_to_seconds(st["start_timecode"])
                st_end = self.timecode_to_seconds(st["end_timecode"])
                
                # Проверка на частичное совпадение таймкодов
                if (va_start <= st_start <= va_end) or (va_start <= st_end <= va_end) or (st_start <= va_start <= st_end):
                    matching_subtitles.append(st)
            
            combined_analysis.append({
                "video_analysis": va,
                "subtitles_analysis": matching_subtitles
            })
        
        # Сохранение объединенного анализа в JSON файл
        with open(output_json, 'w', encoding='utf-8') as output_file:
            json.dump(combined_analysis, output_file, ensure_ascii=False, indent=4)

        return combined_analysis

    @staticmethod
    def timecode_to_seconds(timecode: str) -> float:
        """Преобразует SRT таймкод в секунды."""
        hours, minutes, seconds = timecode.split(':')
        seconds, milliseconds = seconds.split(',')
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

    def local_analysis(self, output_dir: str, video_path: str) -> None:
        uuid = str(uuid4())
        os.makedirs(output_dir, exist_ok=True)

        video_analysis = self.video_analysis.run(
            source=Source.Local,
            video_path=video_path,
            output_json=f"{output_dir}/{uuid}-video.json"
        )

        subtitles_analysis = self.subtitles_analysis.run(
            source=Source.Local,
            video_path=video_path,
            output_json=f"{output_dir}/{uuid}-subtitles.json"
        )

        return uuid

    def youtube_analysis(self, output_dir: str, youtube_video_url: str) -> None:
        uuid = str(uuid4())
        os.makedirs(output_dir, exist_ok=True)

        video_analysis = self.video_analysis.run(
            output_json=f"{output_dir}/{uuid}-video.json",
            source=Source.Youtube,
            youtube_video_url=youtube_video_url
        )

        audio_analysis = self.subtitles_analysis.run(
            output_json=f"{output_dir}/{uuid}-subtitles.json",
            source=Source.Youtube,
            youtube_video_url=youtube_video_url
        )

        return uuid

    def run(self, 
            output_dir: str, 
            source: str = Source.Local, 
            video_path: str = None, 
            youtube_video_url: str = None) -> None:
        
        if source == Source.Local:
            uuid = self.local_analysis(output_dir, video_path)
        elif source == Source.Youtube:
            uuid = self.youtube_analysis(output_dir, youtube_video_url)
        else:
            raise ValueError(f"Invalid source: {source}")
            
        output_json_video = f"{output_dir}/{uuid}-video.json"
        output_json_subtitles = f"{output_dir}/{uuid}-subtitles.json"
        output_json_concat = f"{output_dir}/{uuid}-concat.json"

        analysis = self.video_subtitles_concat(
            video_analysis_json=f"{output_dir}/{uuid}-video.json",
            subtitles_json=f"{output_dir}/{uuid}-subtitles.json",
            output_json=f"{output_dir}/{uuid}-concat.json"
        )

        concat_analysis = json.load(open(output_json_concat, 'r', encoding='utf-8'))
        analysis_text = format_analysis_text(concat_analysis)

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": self.analysis_prompt
                },
                {
                    "role": "user",
                    "content": analysis_text
                }
            ],
            response_format=InterestingMoments
        )

        return completion.choices[0].message.parsed
    

def yt_download(yt_vid_url: str, mp4_dir_save_path: str) -> str:
    # Create the directory if it doesn't exist
    os.makedirs(mp4_dir_save_path, exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(mp4_dir_save_path, '%(title)s.%(ext)s'),
        'restrictfilenames': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(yt_vid_url, download=False)
        filename = ydl.prepare_filename(info)
        ydl.download([yt_vid_url])

    video_file = Path(filename)

    # Ensure the file has a .mp4 extension
    if not video_file.suffix == '.mp4':
        new_video_file = video_file.with_suffix('.mp4')
        video_file.rename(new_video_file)
        video_file = new_video_file

    return str(video_file)