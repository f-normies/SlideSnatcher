import cv2
import numpy as np
from PIL import Image
import argparse
import sys

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract slides from a video file.')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('output_path', type=str, help='Path where the extracted slides will be saved')

    # Parse arguments
    args = parser.parse_args()

    # Open the video file
    cap = cv2.VideoCapture(args.video_path)

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video")
        sys.exit()
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Slide counter
    slide_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute difference and threshold
        diff = cv2.absdiff(prev_frame_gray, gray_frame)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Check if the difference is significant enough to consider it a new slide
        if np.sum(thresh) > 1000000:  # This threshold might need adjustment
            slide_number += 1
            output_filename = f'{args.output_path}/slide_{slide_number}.png'
            Image.fromarray(frame).save(output_filename)
            print(f'Slide {slide_number} saved as {output_filename}')
            prev_frame_gray = gray_frame

    cap.release()