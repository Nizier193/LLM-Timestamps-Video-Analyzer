import json
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from langchain_ru_llms.yandexllm import YandexChatModel

from subtitle_analysis import SubtitlesAnalysis
from settings import OUTPUT_FILES, Source
import os

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

prompt = """
You will be provided with a transcription extract from a video clip and the full content of an transcript corresponding to that clip. Your task is to match the transcription extract to the subtitle segment it best aligns with and return the results in a specific format.

Here is the full content of the transcript:
<transcript>
{subtitles}
</transcript>

Please follow these steps:
1. Carefully read through the transcription excerpt within the <transcript> tags.
2. Given the extract, search through the <transcript> content to find the subtitle segment that best matches the extract. To determine the best match, look for segments that contain the most overlapping words or phrases with the extract.
3. Once you've found the best matching subtitle segment for the excerpt, format the match as follows:
[segment number]
[start time] --> [end time] 
[matched transcription extract]
5. After processing the extract, combine the formatted matches into a single block of text. This should resemble a valid transcript, with each match separated by a blank line.

Please note: transcript files have a specific format that must be followed exactly in order for them to be readable. Therefore, it is crucial that you do not include any extra content beyond the raw transcript data itself. This means:
- No comments explaining your work
- No notes about which extracts matched which segments
- No additional text that isn't part of the subtitle segments

Simply return the matches, properly formatted, as the entire contents of your response.
"""

backstory = """
Experienced subtitler who writes captions or subtitles that accurately represent the audio, including dialogue, sound effects, and music. 
The subtitles need to be properly timed with the video using correct time codes.
Match a list of extracts from a video clip with the corresponding timed subtitles. 
Given the segments found by the Digital Producer, find the segment timings within the file and return each segment as an subtitle segment.
You have to return all subtitles given by Digital Producer.
"""

"""
[
    {
        "start_timecode": "00:00:00,000",
        "end_timecode": "00:00:00,000",
        "text": "Hello, how are you today?"
    },
    {
        "start_timecode": "00:00:00,000",
        "end_timecode": "00:00:00,000",
        "text": "Hello, how are you today?"
    }
]
"""

from pydantic import BaseModel

class Subtitle(BaseModel):
    start_timecode: str
    end_timecode: str
    text: str

class Subtitles(BaseModel):
    subtitles: List[Subtitle]

def parse_subtitles(subtitles: str) -> List[str]:
    # Works in some weird way
    
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    subtitles = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": backstory},
            {"role": "user", "content": prompt.format(subtitles=subtitles)}
        ],
        response_format=Subtitles,
    )

    return subtitles

def json_to_text(json_data: dict) -> str:
    text = ""

    for subtitle in json_data:
        text += f"{subtitle['start_timecode']} --> {subtitle['end_timecode']} \n{subtitle['subtitle']}\n\n"

    return text