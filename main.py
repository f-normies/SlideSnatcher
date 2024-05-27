import cv2
import numpy as np
from PIL import Image
import argparse
import sys
import os
from tqdm import tqdm
import glob
import requests
from urllib.parse import urlparse

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

def download_video_from_url(url, output_dir):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        url_path = urlparse(url).path
        video_name = os.path.basename(url_path)
        video_path = os.path.join(output_dir, video_name)
        total_size = int(response.headers.get('content-length', 0))

        with open(video_path, 'wb') as f, tqdm(
            desc=video_name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        return video_path
    else:
        print(f"Failed to download video from URL: {url}")
        sys.exit()

def create_pdf_from_images(image_folder, output_pdf_path):
    image_files = sorted(glob.glob(f'{image_folder}/*.png'))
    if not image_files:
        print("No images found to compile into a PDF.")
        return

    images = [Image.open(image_file).convert('RGB') for image_file in image_files]
    images[0].save(output_pdf_path, save_all=True, append_images=images[1:])
    print(f"PDF saved as {output_pdf_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract slides from a video file.')
    parser.add_argument('-v', '--video_path', type=str, default='./videos', help='Directory containing video files or URL of the video file')
    parser.add_argument('-o', '--output_path', type=str, default='./slides', help='Directory to save the extracted slides')
    args = parser.parse_args()

    # Check if the video path is a URL
    video_path = args.video_path
    output_directory = args.output_path
    if urlparse(video_path).scheme in ('http', 'https'):
        print("Downloading video from URL...")
        video_path = download_video_from_url(video_path, './videos')
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    else:
        # Video selection
        video_file = select_video(video_path)
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        video_path = video_file

    output_directory = os.path.join(output_directory, video_name)
    os.makedirs(output_directory, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
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
            prev_frame_gray = gray_frame

        pbar.update(skip_frames)

    cap.release()
    pbar.close()

    # Create PDF
    while True:
        choice = input("Do you wish to make a compiled PDF? (Y/n): ").strip().lower()
        if choice == 'y':
            output_pdf_path = os.path.join(output_directory, f"{video_name}_slides.pdf")
            create_pdf_from_images(output_directory, output_pdf_path)
            break
        elif choice == 'n':
            break

if __name__ == '__main__':
    main()
