from final_analysis import VideoAnalysisBySubtitles

from settings import Source

from pprint import pprint


analyzer = VideoAnalysisBySubtitles(
    video_interval=240, # Кадр берётся каждые 10 секунд
    resize_factor=4 # Уменьшение размерности в 4 раза
)

# TODO: Fix weird bug with subtitles
# TODO: Make LLM follow client_wants more strictly < weird bug
# TODO: Make cut_every_seconds working

pprint(analyzer.run(
    output_dir='pitch_2_output',
    source=Source.Local,
    video_path='input_files/pitch_1.mp4',
    client_wants="Сделай интересные видео из фрагмента где говорится про API и где ответы на вопросы, но лучше про вопросы. Каждое видео должно быть не меньше 30 секунд."
))