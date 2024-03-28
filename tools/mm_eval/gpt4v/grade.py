import json
from typing import Any, Optional

import fire
from tqdm import tqdm

from data_juicer.utils.mm_utils import (image_path_to_base64,
                                        video_path_to_base64)
from tools.mm_eval.gpt4v.lib import (check_missing_keys, grade_image_to_text,
                                     grade_text_to_image, grade_text_to_video,
                                     grade_video_to_text, parse_ini)


def image_to_text(input: str,
                  output: str,
                  *,
                  check_key: str = '',
                  **kwargs: Any):
    '''
    Evaluates text-to-image generation using single-answer grading.

    :param input: Path to the JSONL file containing evaluation entries.
    :param output: Path to the JSONL file for saving evaluation results.
    :param check_key: Key path to check for existence in the evaluation
        results. Separated by dots for nested keys.
    :param kwargs: Extra keyword arguments passed to the OpenAI API request.
    '''

    expected_keys = ['image', 'text']

    with open(input) as fin:
        lines = fin.readlines()

    with open(output, 'w') as fout:
        for line in tqdm.tqdm(lines):
            line = line.strip()
            entry = json.loads(line)
            missing = check_missing_keys(entry, expected_keys)
            if missing:
                result = f'Missing keys: {" ".join(missing)}'
            else:
                image = image_path_to_base64(entry['image'], mime_type='auto')
                text = entry['text'].strip()
                result_raw = grade_image_to_text(image, text, **kwargs)
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
    Evaluates text-to-image generation using single-answer grading.

    :param input: Path to the JSONL file containing evaluation entries.
    :param output: Path to the JSONL file for saving evaluation results.
    :param check_key: Key path to check for existence in the evaluation
        results. Separated by dots for nested keys.
    :param kwargs: Extra keyword arguments passed to the OpenAI API request.
    '''

    expected_keys = ['text', 'image']

    with open(input) as fin:
        lines = fin.readlines()

    with open(output, 'w') as fout:
        for line in tqdm.tqdm(lines):
            line = line.strip()
            entry = json.loads(line)
            missing = check_missing_keys(entry, expected_keys)
            if missing:
                result = f'Missing keys: {" ".join(missing)}'
            else:
                text = entry['text'].strip()
                image = image_path_to_base64(entry['image'], mime_type='auto')
                result_raw = grade_text_to_image(text, image, **kwargs)
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
    Evaluates video-to-text generation using single-answer grading.

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

    expected_keys = ['video', 'text']

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
                text = entry['text'].strip()
                result_raw = grade_video_to_text(frames, text, **kwargs)
                if not result_raw:
                    result = 'API Request error'
                else:
                    result = parse_ini(result_raw, check_key)
            fout.write(json.dumps(result) + '\n')


# text --> video
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
    Evaluates text-to-video generation using single-answer grading.

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

    expected_keys = ['text', 'video']
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
                frames = video_path_to_base64(entry['video'],
                                              frame_num=frame_num,
                                              fps=fps,
                                              mime_type='auto')
                result_raw = grade_text_to_video(text, frames, **kwargs)
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
