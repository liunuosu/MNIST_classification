from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def snipe_mp4(input_file, output_file, start_time, end_time):
    try:
        clip = VideoFileClip(input_file)
        duration = clip.duration

        # Convert time in HH:MM:SS format to seconds
        start_time_seconds = sum(x * int(t) for x, t in zip([3600, 60, 1], start_time.split(":")))
        end_time_seconds = sum(x * int(t) for x, t in zip([3600, 60, 1], end_time.split(":")))

        # Check if the specified time range is within the duration of the video
        if end_time_seconds > duration:
            print("End time exceeds video duration.")
            return

        # Snipe the portion of the video
        ffmpeg_extract_subclip(input_file, start_time_seconds, end_time_seconds, targetname=output_file)

        print(f"Sniping successful. The portion from {start_time} to {end_time} has been saved as {output_file}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace the following variables with your desired values
input_file = r'F:/GitHub/mnist_classificaton_tutorial/MNIST_classification/blurry_vs_brickhits.mp4'
output_file = r'F:/GitHub/mnist_classificaton_tutorial/MNIST_classification/blurry_vs_brickhits_cut.mp4'
start_time = "1:04:50"
end_time = "1:55:30"

snipe_mp4(input_file, output_file, start_time, end_time)
