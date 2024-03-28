import json
from typing import Any, Optional

import fire
from tqdm import tqdm

from data_juicer.utils.mm_utils import (image_path_to_base64,
                                        video_path_to_base64)
from tools.mm_eval.gpt4v.lib import (check_missing_keys, compare_image_to_text,
                                     compare_text_to_image,
                                     compare_text_to_video,
                                     compare_video_to_text, parse_ini)


def image_to_text(input: str,
                  output: str,
                  *,
                  check_key: str = '',
                  **kwargs: Any):
    '''
    Evaluates image-to-text generation using pairwise comparison.

    :param input: Path for the JSONL file containing evaluation entries.
    :param output: Path to the JSONL file for saving evaluation results.
    :param check_key: Key path to check for existence in the evaluation
        results. Separated by dots for nested keys.
    :param kwargs: Extra keyword arguments passed to the OpenAI API request.
    '''

    expected_keys = ['image', 'text_0', 'text_1']

    with open(input) as fin:
        lines = fin.readlines()

    with open(output, 'w') as fout:
        for line in tqdm(lines):
            line = line.strip()
            entry = json.loads(line)
            missing = check_missing_keys(entry, expected_keys)
            if missing:
                result = f'Missing keys: {" ".join(missing)}'
            else:
                image = image_path_to_base64(entry['image'], mime_type='auto')
                text_0 = entry['text_0'].strip()
                text_1 = entry['text_1'].strip()
                result_raw = compare_image_to_text(image, text_0, text_1,
                                                   **kwargs)
                if not result_raw:
                    result = 'API Request error'
                else:
                    result = parse_ini(result_raw, check_key)
            fout.write(json.dumps(result) + '\n')


def text_to_image(input: str,
                  output: str,
                  *,
                  check_key: str = '',
                  **kwargs: Any):
    '''
    Evaluates text-to-image generation using pairwise comparison.

    :param input: Path to the JSONL file containing evaluation entries.
    :param output: Path to the JSONL file for saving evaluation results.
    :param check_key: Key path to check for existence in the evaluation
        results. Separated by dots for nested keys.
    :param kwargs: Extra keyword arguments passed to the OpenAI API request.
    '''

    expected_keys = ['text', 'image_0', 'image_1']

    with open(input) as fin:
        lines = fin.readlines()

    with open(output, 'w') as fout:
        for line in tqdm(lines):
            line = line.strip()
            entry = json.loads(line)
            missing = check_missing_keys(entry, expected_keys)
            if missing:
                result = f'Missing keys: {" ".join(missing)}'
            else:
                text = entry['text'].strip()
                image_0 = image_path_to_base64(entry['image_0'],
                                               mime_type='auto')
                image_1 = image_path_to_base64(entry['image_1'],
                                               mime_type='auto')
                result_raw = compare_text_to_image(text, image_0, image_1,
                                                   **kwargs)
                if not result_raw:
                    result = 'API Request error'
                else:
                    result = parse_ini(result_raw, check_key)
            fout.write(json.dumps(result) + '\n')


def video_to_text(
    input: str,
    output: str,
    *,
    frame_num: Optional[int] = None,
    fps: Optional[float] = None,
    check_key: str = '',
    **kwargs: Any,
):
    '''
    Evaluates video-to-text generation using pairwise comparison.

    :param input: Path to the JSONL file containing evaluation entries.
    :param output: Path to the JSONL file for saving evaluation results.
    :param frame_num: The number of frames to sample from each video.
    :param fps: The sampling rate in frames per second.
    :param check_key: Key path to check for existence in the evaluation
        results. Separated by dots for nested keys.
    :param kwargs: Extra keyword arguments passed to the OpenAI API request.

    Note: `frame_num` and `fps` are mutually exclusive; only one may be set.
    '''

    if not ((frame_num is None) ^ (fps is None)):
        raise ValueError("The parameters 'frame_num' and 'fps' are \
                mutually exclusive; only one may be set.")

    expected_keys = ['video', 'text_0', 'text_1']

    with open(input) as fin:
        lines = fin.readlines()

    with open(output, 'w') as fout:
        for line in tqdm(lines):
            line = line.strip()
            entry = json.loads(line)
            missing = check_missing_keys(entry, expected_keys)
            if missing:
                result = f'Missing keys: {" ".join(missing)}'
            else:
                frames = video_path_to_base64(entry['video'],
                                              frame_num=frame_num,
                                              fps=fps,
                                              mime_type='auto')
                text_0 = entry['text_0'].strip()
                text_1 = entry['text_1'].strip()
                result_raw = compare_video_to_text(frames, text_0, text_1,
                                                   **kwargs)
                if not result_raw:
                    result = 'API Request error'
                else:
                    result = parse_ini(result_raw, check_key)
            fout.write(json.dumps(result) + '\n')


def text_to_video(
    input: str,
    output: str,
    *,
    frame_num: Optional[int] = None,
    fps: Optional[float] = None,
    check_key: str = '',
    **kwargs: Any,
):
    '''
    Evaluates text-to-video generation using pairwise comparison.

    :param input: Path to the JSONL file containing evaluation entries.
    :param output: Path to the JSONL file for saving evaluation results.
    :param frame_num: The number of frames to sample from each video.
    :param fps: The sampling rate in frames per second.
    :param check_key: Key path to check for existence in the evaluation
        results. Separated by dots for nested keys.
    :param kwargs: Extra keyword arguments passed to the OpenAI API request.

    Note: `frame_num` and `fps` are mutually exclusive; only one may be set.
    '''

    if not ((frame_num is None) ^ (fps is None)):
        raise ValueError("The parameters 'frame_num' and 'fps' are \
                mutually exclusive; only one may be set.")

    expected_keys = ['text', 'video_0', 'video_1']

    with open(input) as fin:
        lines = fin.readlines()

    with open(output, 'w') as fout:
        for line in tqdm(lines):
            line = line.strip()
            entry = json.loads(line)
            missing = check_missing_keys(entry, expected_keys)
            if missing:
                result = 'Missing keys: {" ".join(missing)}'
            else:
                text = entry['text'].strip()
                frames_0 = video_path_to_base64(entry['video_0'],
                                                frame_num=frame_num,
                                                fps=fps,
                                                mime_type='auto')
                frames_1 = video_path_to_base64(entry['video_1'],
                                                frame_num=frame_num,
                                                fps=fps,
                                                mime_type='auto')
                result_raw = compare_text_to_video(text, frames_0, frames_1,
                                                   **kwargs)
                if not result_raw:
                    result = 'API Request error'
                else:
                    result = parse_ini(result_raw, check_key)
            fout.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    fire.Fire({
        'image_to_text': image_to_text,
        'text_to_image': text_to_image,
        'video_to_text': video_to_text,
        'text_to_video': text_to_video,
    })
