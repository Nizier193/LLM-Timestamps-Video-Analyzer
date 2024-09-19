from final_analysis import VideoAnalysisBySubtitles

from settings import Source

analyzer = VideoAnalysisBySubtitles()
result = analyzer.run(
    output_dir="output_test_latest",
    source=Source.Youtube,
    youtube_video_url="https://www.youtube.com/watch?v=BQ4XkEi5Z5o"
)

from pprint import pprint
pprint(result)
