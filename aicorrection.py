from typing import List
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import json

from settings import OUTPUT_FILES

load_dotenv()

class SingleSubtitles(BaseModel):
    start_timecode: float
    end_timecode: float
    subtitle: str
    confidence: float
    subtitle_number: int

class Subtitles(BaseModel):
    subtitles: List[SingleSubtitles]

class AICorrection:
    correct_prompt = """
Experienced subtitler who writes captions or subtitles that accurately represent the audio, including dialogue, sound effects, and music. 
The subtitles need to be properly timed with the video using correct time codes.

The output should be in JSON format.
You must correct the words in the subtitles if they do not exist in the Russian language or if they contain errors.
"""

    task_prompt = """
As an experienced subtitler, help me with the subtitles:

Here are the subtitles:
{subtitles}

The subtitles need to be properly timed with the video using correct time codes.
You must correct the words in the subtitles if they do not exist in the English language or if they contain errors.
Numerate the subtitles.

Answer in JSON Format.
"""

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def run(self, analysis: dict):
        text = ""
        for analysis_item in analysis:
            text += f"Subtitle Number: {analysis_item["subtitle_number"]} Start time: {analysis_item['start_timecode']}, End time: {analysis_item['end_timecode']}, Subtitle: {analysis_item['subtitle']}, Confidence: {analysis_item['confidence']}\n\n"

        print(text)

        aicorrection_assistant_prompt = self.task_prompt.format(subtitles=text)
        open(f"aicorrection_output/aicorrection_assistant_prompt.txt", "w", encoding="utf-8").write(aicorrection_assistant_prompt)
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": self.correct_prompt
                },
                {
                    "role": "user",
                    "content": aicorrection_assistant_prompt
                }
            ],
            response_format=Subtitles
        )
        
        completion_tokens = response.usage.completion_tokens
        prompt_tokens = response.usage.prompt_tokens
        total_tokens = response.usage.total_tokens

        analysis = response.choices[0].message.content
        parsed_response_content = json.loads(analysis)

        return parsed_response_content, total_tokens
