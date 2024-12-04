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
from concurrent.futures import ThreadPoolExecutor

def convert_to_mp4(input_video, output_video):
    try:
        audio_param = '-an'
        
        encoders = [
            # NVIDIA GPU
            {
                'codec': 'h264_nvenc',
                'params': '-preset p1 -tune fastdecode -rc:v vbr_hq'
            },
            # AMD GPU
            {
                'codec': 'h264_amf',
                'params': '-quality speed -rc vbr_peak'
            },
            # Intel QuickSync
            {
                'codec': 'h264_qsv',
                'params': '-preset veryfast'
            },
            # CPU fallback
            {
                'codec': 'libx264',
                'params': '-preset ultrafast -tune fastdecode'
            }
        ]
        
        success = False
        for encoder in encoders:
            command = (
                f'ffmpeg -i "{input_video}" '
                f'-c:v {encoder["codec"]} '
                f'{encoder["params"]} '
                f'-b:v 5M '
                f'-maxrate 10M '
                f'-bufsize 10M '
                f'{audio_param}'
                f'-fps_mode passthrough '
                f'-threads 8 '
                f'-y "{output_video}"'
            )
            
            try:
                result = os.system(command)
                if result == 0:
                    success = True
                    break
            except Exception as e:
                continue
        
        if not success:
            print("Conversion attempt failed")
            return False
            
        return True
    except Exception as e:
        print(f"Error converting video: {e}")
        return False

def prepare_video(video_path):
    _, ext = os.path.splitext(video_path)
    if ext.lower() == '.avi':
        print("Converting AVI to MP4 for better performance...")
        mp4_path = video_path.replace('.avi', '_converted.mp4')
        if convert_to_mp4(video_path, mp4_path):
            return mp4_path
    return video_path

def select_video(video_path):
    video_formats = ['*.mkv', '*.mp4', '*.avi']
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
            for chunk in response.iter_content(chunk_size=8192):
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

def save_frame(frame, output_filename):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        image.save(output_filename)
        return True
    except Exception as e:
        print(f"Error saving frame to {output_filename}: {e}")
        return False

def process_video(video_path, output_directory, threshold):
    try:
        os.makedirs(output_directory, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        
        skip_frames = min(frame_rate * 2, max(1, total_frames // 200))
        
        pbar = tqdm(total=total_frames, unit='frame', desc="Processing frames")
        
        slide_number = 0
        ret, prev_frame = cap.read()
        if not ret:
            print("Failed to read first frame")
            return False

        slide_number += 1
        first_output_filename = os.path.join(output_directory, f'slide_{slide_number:03d}.png')
        if not save_frame(prev_frame, first_output_filename):
            print(f"Failed to save first frame!")
            return False

        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_frame_gray = cv2.resize(prev_frame_gray, (0, 0), fx=0.5, fy=0.5)
        
        window_size = 5
        recent_diffs = [0] * window_size
        window_idx = 0
        
        while True:
            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if current_pos + skip_frames >= total_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + skip_frames)
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.resize(frame_gray, (0, 0), fx=0.5, fy=0.5)
            
            diff = cv2.absdiff(prev_frame_gray, frame_gray)
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            
            recent_diffs[window_idx] = mean_diff
            window_idx = (window_idx + 1) % window_size
            local_mean = np.mean(recent_diffs)
            
            is_significant_change = (
                mean_diff > threshold and
                mean_diff > local_mean * 1.5 and
                std_diff > threshold * 0.5
            )
            
            if is_significant_change:
                slide_number += 1
                output_filename = os.path.join(output_directory, f'slide_{slide_number:03d}.png')
                if not save_frame(frame, output_filename):
                    print(f"Warning: Failed to save frame {slide_number}")
                    continue
            
            prev_frame_gray = frame_gray
            pbar.update(skip_frames)
        
        saved_files = glob.glob(os.path.join(output_directory, "*.png"))
        
        return True

    except Exception as e:
        print(f"Error during video processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cap.release()
        pbar.close()

def create_pdf_from_images(image_folder, output_pdf_path):
    image_files = sorted(glob.glob(f'{image_folder}/*.png'))
    if not image_files:
        print("No images found to compile into a PDF.")
        return False

    print("Creating PDF from extracted slides...")
    try:
        def load_image(image_file):
            return Image.open(image_file).convert('RGB')

        with ThreadPoolExecutor() as executor:
            images = list(executor.map(load_image, image_files))
        
        images[0].save(
            output_pdf_path, 
            save_all=True, 
            append_images=images[1:],
            quality=95,
            optimize=True
        )
        print(f"PDF saved as {output_pdf_path}")
        return True
    except Exception as e:
        print(f"Error creating PDF: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract slides from a video file with optimized performance.')
    parser.add_argument('-v', '--video_path', type=str, default='./videos', 
                       help='Directory containing video files or URL of the video file')
    parser.add_argument('-o', '--output_path', type=str, default='./slides', 
                       help='Directory to save the extracted slides')
    parser.add_argument('-nc', '--no-convert', action='store_true',
                       help='Disable automatic AVI to MP4 conversion')
    parser.add_argument('-t','--threshold', type=float, default=0.8,
                       help='Threshold for slide change detection (default: 0.8)')
    args = parser.parse_args()

    try:
        base_output_path = os.path.abspath(args.output_path)
        os.makedirs(base_output_path, exist_ok=True)
        os.makedirs('./videos', exist_ok=True)

        video_path = args.video_path
        if urlparse(video_path).scheme in ('http', 'https'):
            print("Downloading video from URL...")
            video_path = download_video_from_url(video_path, './videos')
            video_name = os.path.splitext(os.path.basename(video_path))[0]
        else:
            video_file = select_video(video_path)
            video_name = os.path.splitext(os.path.basename(video_file))[0]
            video_path = video_file

        if not args.no_convert and video_path.lower().endswith('.avi'):
            video_path = prepare_video(video_path)
            
        safe_video_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_'))
        output_directory = os.path.join(base_output_path, safe_video_name)
        
        os.makedirs(output_directory, exist_ok=True)
        
        print(f"\nProcessing video: {os.path.basename(video_path)}")
        print(f"Output directory: {output_directory}")

        if process_video(video_path, output_directory, args.threshold):
            print("\nSlide extraction completed successfully!")
            
            image_files = sorted(glob.glob(os.path.join(output_directory, "*.png")))
            if not image_files:
                print(f"Warning: No images found in {output_directory}")
                return
                
            print(f"Found {len(image_files)} images")
            
            while True:
                choice = input("\nDo you wish to create a compiled PDF? (Y/n): ").strip().lower()
                if choice in ['y', '']:
                    output_pdf_path = os.path.join(output_directory, f"{safe_video_name}_slides.pdf")
                    break
                elif choice == 'n':
                    print("\nSlides saved as individual images.")
                    break
        else:
            print("\nError occurred during video processing.")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
