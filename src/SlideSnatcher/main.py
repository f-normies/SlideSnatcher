import cv2
import numpy as np
from PIL import Image
import argparse
import sys
from tqdm import tqdm

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract slides from a video file.')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('output_path', type=str, help='Path where the extracted slides will be saved')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error opening video file")
        sys.exit()

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Getting the frame rate of the video
    skip_frames = frame_rate * 2  # Skip every 2 seconds worth of video

    # Progress bar setup
    pbar = tqdm(total=total_frames, unit='frame', position=1, leave=True)
    print("Processing Video... Please wait.")

    slide_number = 0
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video")
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
            output_filename = f'{args.output_path}/slide_{slide_number}.png'
            Image.fromarray(frame).save(output_filename)
            print(f'Slide {slide_number} saved as {output_filename}')
            prev_frame_gray = gray_frame

        pbar.update(skip_frames)

    cap.release()
    pbar.close()