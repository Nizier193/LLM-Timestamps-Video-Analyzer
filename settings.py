from typing import List
from pydantic import BaseModel

INPUT_FILES = "input_files"
OUTPUT_FILES = "output_files"

class Source:
    Youtube = "Youtube"
    Local = "Local"