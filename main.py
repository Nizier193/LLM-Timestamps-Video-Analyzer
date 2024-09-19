from final_analysis import VideoAnalysisBySubtitles

from settings import Source
from pprint import pprint

analyzer = VideoAnalysisBySubtitles()

pprint(analyzer.run(
    output_dir='test_output',
    source=Source.Local,
    video_path='input_files/example3.mp4'
))
