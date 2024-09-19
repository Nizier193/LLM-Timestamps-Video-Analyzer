from openai import OpenAI

from dotenv import load_dotenv
from PIL import Image
import base64
import io
import os

load_dotenv()

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

НЕ УКАЗЫВАЙ ЗАГОЛОВКИ ПУНКТОВ
НЕ ВКЛЮЧАЙ В ОТВЕТ ДРУГИХ РАССУЖДЕНИЙ ИЛИ МЫСЛЕЙ
НЕ ДЕЛАЙ ОТВЕТ ОЧЕНЬ БОЛЬШИМ
ОТВЕТ НА РУССКОМ ЯЗЫКЕ
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

    def analyze(self, image: Image.Image, scene: str = None) -> str:
        """
        Анализ картинки с соотнесением к предыдущему кадру.
        scene - предыдущий кадр, по умолчанию None.
        """

        resized_image = self.resize_image(image)

        if self.black_and_white:
            resized_image = self.convert_to_black_and_white(resized_image)

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

        completion_tokens = response.usage.completion_tokens
        prompt_tokens = response.usage.prompt_tokens
        total_tokens = response.usage.total_tokens

        return response.choices[0].message.content
    