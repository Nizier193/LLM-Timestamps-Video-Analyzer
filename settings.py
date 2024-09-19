from typing import List
from pydantic import BaseModel

INPUT_FILES = "input_files"
OUTPUT_FILES = "output_files"

class Source:
    Youtube = "Youtube"
    Local = "Local"

class Moment(BaseModel):
    start_timecode: str
    end_timecode: str
    analysis: str

class Fragment(BaseModel):
    video_analysis: Moment
    subtitles_analysis: Moment
    rating_by_interest: int

class InterestingMoments(BaseModel):
    fragments: List[Fragment]