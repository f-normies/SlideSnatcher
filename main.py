import cv2
import numpy as np
from PIL import Image
import argparse
import sys
import os
from tqdm import tqdm
import glob

def select_video(video_path):
    video_formats = ['*.mkv', '*.mp4']
    video_files = []
    for format in video_formats:
        video_files.extend(glob.glob(f'{video_path}/{format}'))

    if not video_files:
        print("No video files found in the directory.")
        sys.exit()
    
    print("Select video to process:")
    for index, file in enumerate(video_files, 1):
        print(f"({index}) - {os.path.basename(file)}")
    
    choice = input("Enter the number of the video to process: ")
    try:
        selected_video = video_files[int(choice) - 1]
    except (IndexError, ValueError):
        print("Invalid selection.")
        sys.exit()
    
    return selected_video

def main():
    parser = argparse.ArgumentParser(description='Extract slides from a video file.')
    parser.add_argument('-v', '--video_path', type=str, default='./videos', help='Directory containing video files')
    parser.add_argument('-o', '--output_path', type=str, default='./slides', help='Directory to save the extracted slides')
    args = parser.parse_args()

    # Video selection
    video_file = select_video(args.video_path)
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_directory = os.path.join(args.output_path, video_name)
    os.makedirs(output_directory, exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error opening video file {video_file}")
        sys.exit()

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    skip_frames = frame_rate * 2

    # Progress bar setup
    pbar = tqdm(total=total_frames, unit='frame', position=0, leave=True)
    tqdm.write("Processing Video... Please wait.")

    slide_number = 0
    ret, prev_frame = cap.read()
    if not ret:
        tqdm.write("Failed to read video")
        sys.exit()

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames logic
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + skip_frames - 1)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_frame_gray, gray_frame)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        if np.sum(thresh) > 1000000:
            slide_number += 1
            output_filename = f'{output_directory}/slide_{slide_number}.png'
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(output_filename)
            tqdm.write(f'Slide {slide_number} saved as ' + output_filename.replace("\\", "/"))
            prev_frame_gray = gray_frame

        pbar.update(skip_frames)

    cap.release()
    pbar.close()

if __name__ == '__main__':
    main()