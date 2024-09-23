from openai import OpenAI

from dotenv import load_dotenv
from PIL import Image
import base64
import io
import os
import json
load_dotenv()

from pydantic import BaseModel
from pprint import pprint

class ImageAnalysisModel(BaseModel):
    scene_and_main_characters: str
    what_is_happening: str
    what_is_interesting: str
    is_interesting: bool
    most_interesting_fragment: list[str, str]

class ImageAnalysis():
    analysis_prompt = """
Ты - опытный монтажёр видео, который анализирует видеоряд и помогает найти самые интересные фрагменты.
Тебе представлено 16 картинок видео в 4x4 последовательно.

Твоя задача - описать, что на картинках, и как кадр изменился по сравнению с предыдущими описаниями кадров.
Максимально объективно расскажи, является ли фрагмент вирусным.

Расскажи объективно об этом видео:
1. Главные лица
2. Действие
3. Красоту кадра (если нет, так и скажи)
4. Что-либо интересное (если нет, так и скажи)
5. Является ли интересным для публикации? (True или False)
6. Временной отрезок, который является самым интересным. Формат:  [start_time, end_time] 2 строки

ВЫВЕДИ ОТВЕТ В JSON ФОРМАТЕ
НЕ ВКЛЮЧАЙ В ОТВЕТ ЛЮБЫЕ КОММЕНТАРИИ
ТЫ ОБЯЗАН ВЫБРАТЬ ОДИН ВРЕМЕННЫЙ ОТРЕЗОК, КОТОРЫЙ БУДЕТ САМЫМ ИНТЕРЕСНЫМ
"""

    task_prompt = """
Как опытный монтажер, расскажи что происходит на видеоряде.
Нужно понять, интересен ли фрагмент для публикации (вирусное ли видео).

Тебе представлено 16 картинок видео в 4x4 последовательно.
Всё это кадры видео, которые являются частью фрагмента видео.
Расскажи объективно об этом видео:
1. Главные лица
2. Действие
3. Красоту кадра (если нет, так и скажи)
4. Что-либо интересное (если нет, так и скажи)
5. Является ли интересным для публикации? (True или False)
6. Временной отрезок, который является самым интересным

Предыдущие кадры:
<last_scenes>
{scene_from_last_frames}
</last_scenes>

Формат ответа:
(
    "scene_and_main_characters": "...", # Описание сцены и главных действующих лиц
    "what_is_happening": "...", # Описание действия
    "what_is_interesting": "...", # Описание интересного
    "is_interesting": "..." # Является ли фрагмент интересным для публикации
    "most_interesting_fragment": "..." # Временной отрезок самого интересного промежутка. Формат:  [start_time, end_time] 2 строки
)

ВЫВЕДИ ОТВЕТ В JSON ФОРМАТЕ
НЕ ВКЛЮЧАЙ В ОТВЕТ ЛЮБЫЕ КОММЕНТАРИИ
ТЫ ОБЯЗАН ВЫБРАТЬ ОДИН ВРЕМЕННЫЙ ОТРЕЗОК, КОТОРЫЙ БУДЕТ САМЫМ ИНТЕРЕСНЫМ
    """

    def __init__(self, api_key: str = None, resize_factor: int = 10, black_and_white: bool = False):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.resize_factor = resize_factor
        self.black_and_white = black_and_white

    def convert_to_black_and_white(self, image: Image.Image) -> Image.Image:
        return image.convert("L")

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

    def analyze(self, image: Image.Image, prompt_params: dict = None, resize_factor: int = 1) -> str:
        """
        Анализ картинки с соотнесением к предыдущему кадру.
        scene - предыдущий кадр, по умолчанию None.
        """
        self.resize_factor = resize_factor

        resized_image = self.resize_image(image)

        if self.black_and_white:
            resized_image = self.convert_to_black_and_white(resized_image)

        base64_image = self.encode_image(resized_image)

        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": self.analysis_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.task_prompt.format(
                                scene_from_last_frames=prompt_params["scene"],
                            )
                        },
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
            response_format=ImageAnalysisModel
        )

        completion_tokens = response.usage.completion_tokens
        prompt_tokens = response.usage.prompt_tokens
        total_tokens = response.usage.total_tokens

        return json.loads(response.choices[0].message.content), total_tokens
    