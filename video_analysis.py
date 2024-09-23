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
    def __init__(self, api_key: str = None, resize_factor: int = 1):
        self.image_analysis = ImageAnalysis(
            api_key=api_key,
            resize_factor=resize_factor
        )
        print("[VideoAnalysis] - [__init__] - Initialized VideoAnalysis class")

    def yt_download(self, yt_vid_url: str, mp4_dir_save_path: str) -> str:
        print(f"[VideoAnalysis] - [yt_download] - Downloading video from {yt_vid_url}")
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

        print(f"[VideoAnalysis] - [yt_download] - Downloaded video to {video_file}")
        return str(video_file)  # Return path to file

    def make_analysis_text(self, analysis_results: List[dict]) -> str:
        analysis_text = ""
        for fragment in analysis_results:
            analysis_text += f"Стартовый таймкод: {fragment['start_timecode']}\n"
            analysis_text += f"Конечный таймкод: {fragment['end_timecode']}\n"
            analysis_text += f"Анализ: {fragment['analysis']}\n"
            analysis_text += "\n"
        return analysis_text

    def run(self, 
            output_json: str, 
            source: str = Source.Local, 
            video_path: str = None, 
            youtube_video_url: str = None,
            resize_factor: int = 1,
            interval_seconds: int = 1) -> List[dict]:
        print(f"[VideoAnalysis] - [run] - Starting video analysis with source: {source}")

        if source == Source.Local:
            cap = cv2.VideoCapture(video_path)
            print(f"[VideoAnalysis] - [run] - Opened local video file: {video_path}")
        elif source == Source.Youtube:
            video_path = self.yt_download(
                youtube_video_url,
                mp4_dir_save_path=OUTPUT_FILES
            )
            cap = cv2.VideoCapture(video_path)
            print(f"[VideoAnalysis] - [run] - Downloaded and opened YouTube video file: {video_path}")
        else:
            raise ValueError(f"Invalid source: {source}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval_frames = int(fps * interval_seconds)
        frames_per_analysis = 16
        frame_step = interval_frames // frames_per_analysis
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        analysis_results = []
        
        total_tokens = 0
        for start_frame in range(0, total_frames, interval_frames):
            combined_image = Image.new('RGB', (frame_width * 4, frame_height * 4))
            
            timecodes = []
            for i in range(frames_per_analysis):
                frame_position = start_frame + i * frame_step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"[VideoAnalysis] - [run] - End of video reached at frame {frame_position}")
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Расположение кадра в сетке 4x4
                x = (i % 4) * frame_width
                y = (i // 4) * frame_height
                combined_image.paste(pil_image, (x, y))
                
                # Добавляем таймкод для текущего кадра
                timecode = self.format_timecode(frame_position / fps)
                timecodes.append(timecode)
            
            # Анализ объединенного изображения
            combined_image.show()
            analysis, total_tokens_per_image = self.image_analysis.analyze(
                combined_image,
                prompt_params={
                    "scene": self.make_analysis_text(analysis_results),
                    "timecodes": timecodes
                },
                resize_factor=resize_factor
            )
            total_tokens += total_tokens_per_image
            
            start_time = self.format_timecode(start_frame / fps)
            end_time = self.format_timecode((start_frame + interval_frames) / fps)
            
            analysis_results.append({
                "start_timecode": start_time,
                "end_timecode": end_time,
                "analysis": analysis,
                "image_timecodes": timecodes
            })
            
            print(f"[VideoAnalysis] - [run] - Analyzed combined frame from {start_time} to {end_time}")

        cap.release()
        print("[VideoAnalysis] - [run] - Released video capture")

        # Сохраняем результаты анализа в JSON файл
        with open(output_json, 'w', encoding='utf-8') as json_file:
            json.dump(analysis_results, json_file, ensure_ascii=False, indent=4)
            print(f"[VideoAnalysis] - [run] - Saved analysis results to {output_json}")

        return analysis_results, total_tokens

    @staticmethod
    def format_timecode(seconds: float) -> str:
        """Форматирует время в секундах в формат SRT таймкода."""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = td.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
