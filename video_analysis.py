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
from settings import Source, OUTPUT_FILES

import yt_dlp
from pathlib import Path

from image_analysis import ImageAnalysis

load_dotenv()


class VideoAnalysis():
    """
    Анализ видео.
    Берёт кадры из видео и анализирует их.
    """
    def __init__(self, interval_seconds: int = 1, api_key: str = None, resize_factor: int = 10):
        self.image_analysis = ImageAnalysis(
            api_key=api_key,
            resize_factor=resize_factor
        )
        self.interval_seconds = interval_seconds

    def yt_download(self, yt_vid_url: str, mp4_dir_save_path: str) -> str:
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

        return str(video_file)  # Return path to file

    def run(self, 
            output_json: str, 
            source: str = Source.Local, 
            video_path: str = None, 
            youtube_video_url: str = None) -> List[dict]:
        # works ok

        if source == Source.Local:
            cap = cv2.VideoCapture(video_path)
        elif source == Source.Youtube:
            video_path = self.yt_download(
                youtube_video_url,
                mp4_dir_save_path=OUTPUT_FILES
            )
            cap = cv2.VideoCapture(video_path)
        else:
            raise ValueError(f"Invalid source: {source}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_step = int(fps * self.interval_seconds)  # Calculate frame_step based on interval_seconds
        frame_count = 0
        previous_scene = None
        analysis_results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_step == 0:
                # Преобразуем кадр из BGR (OpenCV) в RGB (PIL)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # Анализируем кадр
                analysis = self.image_analysis.analyze(pil_image, scene=previous_scene)
                
                # Формируем таймкоды
                start_time = self.format_timecode(frame_count / fps)
                end_time = self.format_timecode((frame_count + frame_step) / fps)

                # Добавляем результат анализа в список
                analysis_results.append({
                    "start_timecode": start_time,
                    "end_timecode": end_time,
                    "analysis": analysis
                })

                previous_scene = analysis

            frame_count += 1

        cap.release()

        # Сохраняем результаты анализа в JSON файл
        with open(output_json, 'w', encoding='utf-8') as json_file:
            json.dump(analysis_results, json_file, ensure_ascii=False, indent=4)

        return analysis_results

    @staticmethod
    def format_timecode(seconds: float) -> str:
        """Форматирует время в секундах в формат SRT таймкода."""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = td.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"