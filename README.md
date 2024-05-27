# SlideSnatcher
## Description

SlideSnatcher is a tool designed to automatically extract slides from video. This utility was developed to address the common problem of accessing presentation materials that are not readily available in slide format after a lecture or meeting.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

To install SlideSnatcher, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/SlideSnatcher.git
    ```
2. Navigate to the project directory:
    ```bash
    cd SlideSnatcher
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use SlideSnatcher, run the following command from the root of the project directory:

```python
python main.py -v path_to_videos_folder -o path_to_output_folder
```

If you want to download and process a video directly from a URL, you can pass the URL as the `-v` argument:

```python
python main.py -v video_url -o path_to_output_folder
```

You can also run it without any arguments to use default paths:

```python
python main.py
```

This will start the application, which prompts you to select a video from the default video directory. After selecting a video, the application will extract the slides and save them into the specified output directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.