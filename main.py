from final_analysis import VideoAnalysisBySubtitles

from settings import Source

from pprint import pprint


analyzer = VideoAnalysisBySubtitles(
    video_interval=10, # Кадр берётся каждые 10 секунд
    resize_factor=30 # Уменьшение размерности в 30 раз
)

pprint(analyzer.run(
    output_dir='test_output',
    source=Source.Local,
    video_path='input_files/example3.mp4'
))

"""
If you want to analyze youtube video, uncomment this code

pprint(analyzer.run(
    output_dir='test_output',
    source=Source.Youtube,
    youtube_video_url='https://www.youtube.com/watch?v=dQw4w9WgXcQ'
))
"""