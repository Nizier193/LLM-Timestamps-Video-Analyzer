import base64
import json
import os
import io
from datetime import timedelta
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import cv2
import ffmpeg

from uuid import uuid4

load_dotenv()

# Анализ картинки с соотнесением к предыдущему кадру
# Выдаёт описание картинки без таймкодов
class ImageAnalysis():
    analysis_prompt = """
Ты - ассистент, который анализирует картинку и предполагает, что на ней происходит.
Фрагмент взят из большого видео.

Ты должен предположить, что изображено на картинке в формате:
- Ситуация
- Действующие лица (если есть)
- Чем интересен кадр? Если нет чего-то интересного, то не упоминай это.
- Какие действия происходят в соотнесении с предыдущим кадром, если есть.

Описание предыдущего кадра:
<scene>
{scene}
</scene>

Ответ не более 3 предложений, которые должны включать в себя все перечисленные пункты.


НЕ ВКЛЮЧАЙ В ОТВЕТ ДРУГИХ РАССУЖДЕНИЙ ИЛИ МЫСЛЕЙ
НЕ ДЕЛАЙ ОТВЕТ ОЧЕНЬ БОЛЬШИМ
ОТВЕТ НА РУССКОМ ЯЗЫКЕ
"""

    def __init__(self, api_key: str = None, resize_factor: int = 10):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.resize_factor = resize_factor

    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Параметр resize_factor отвечает за уменьшение изображения.
        Этот параметр нужен для уменьшения затрат на обработку изображения.
        """

        width, height = image.size
        new_size = (width // self.resize_factor, height // self.resize_factor)
        new_image = image.resize(new_size, Image.LANCZOS)
        return new_image

    def encode_image(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def analyze(self, image: Image.Image, scene: str = None) -> str:
        """
        Анализ картинки с соотнесением к предыдущему кадру.
        scene - предыдущий кадр, по умолчанию None.
        """

        resized_image = self.resize_image(image)
        base64_image = self.encode_image(resized_image)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": self.analysis_prompt.format(scene=scene)
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content
    

class VideoAnalysis():
    """
    Анализ видео.
    Берёт кадры из видео и анализирует их.
    """
    def __init__(self, video_path: str, image_analysis: ImageAnalysis, interval_seconds: int = 1):
        self.video_path = video_path
        self.image_analysis = image_analysis
        self.interval_seconds = interval_seconds

    def analyze(self, output_json: str) -> List[dict]:
        cap = cv2.VideoCapture(self.video_path)
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

    def analyze(self, output_json: str) -> List[dict]:
        audio_path = self.get_audio(self.video_path)
        # // TODO: Добавить анализ субтитров 

        analysis_results = [
            {
                "start_timecode": "00:00:00,000",
                "end_timecode": "00:00:00,000",
                "analysis": "Анализ субтитров"
            }
        ]

        with open(output_json, 'w', encoding='utf-8') as json_file:
            json.dump(analysis_results, json_file, ensure_ascii=False, indent=4)

        return analysis_results

class VideoAnalysisBySubtitles():
    """
    Анализ видео по субтитрам.
    Берёт фрагменты, которые содержат субтитры, и анализирует их.
    Субтитры должны содержать таймкоды, иначе анализ не будет работать.

    ImageAnalysis - класс, который анализирует изображение.
    VideoAnalysis - класс, который анализирует видео.
    VideoAnalysisBySubtitles - класс, который анализирует субтитры по таймкодам и добавляет эти субтитры в .srt файл анализа видео.
    """
    def __init__(self):
        self.video_analysis = VideoAnalysis(
            video_path="source/example.mp4", 
            image_analysis=ImageAnalysis(
                resize_factor=10
            ), 
            interval_seconds=5
        )
        self.subtitles_analysis = SubtitlesAnalysis(
            video_path="source/example.mp4",
        )

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

    def analyze(self, output_dir: str) -> None:
        uuid = str(uuid4())
        os.makedirs(output_dir, exist_ok=True)

        subtitles = self.subtitles_analysis.analyze(output_json=f"{output_dir}/{uuid}-subtitles.json")  
        video = self.video_analysis.analyze(output_json=f"{output_dir}/{uuid}-video.json")  

        analysis = self.video_subtitles_concat(
            video_analysis_json=f"{output_dir}/{uuid}-video.json",
            subtitles_json=f"{output_dir}/{uuid}-subtitles.json",
            output_json=f"{output_dir}/{uuid}-concat.json"
        )

        # // TODO: Добавить общий анализ + оценку на вирусность

        return {
            "analysis": analysis,
            "subtitles": subtitles,
            "video": video
        }


analyzer = VideoAnalysisBySubtitles()
analyzer.analyze(output_dir="output")