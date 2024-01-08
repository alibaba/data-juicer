import re

import numpy as np
from datasets import Audio, Image

from data_juicer.utils.constant import DEFAULT_PREFIX


# A class to keep special tokens for multimodal information in the texts
# The tokens in this class can be updated by corresponding arguments in config
class SpecialTokens(object):
    # modality
    image = f'<{DEFAULT_PREFIX}image>'
    audio = f'<{DEFAULT_PREFIX}audio>'

    # others
    eoc = f'<|{DEFAULT_PREFIX}eoc|>'


def get_special_tokens():
    special_token_dict = {
        key: value
        for key, value in SpecialTokens.__dict__.items()
        if not key.startswith('__')
    }
    return special_token_dict


def remove_special_tokens(text):
    for value in get_special_tokens().values():
        text = text.replace(value, '').strip()
    return text


def remove_non_special_tokens(text):
    special_tokens = get_special_tokens().values()
    patterns = '|'.join(re.escape(token) for token in special_tokens)
    special_tokens_found = re.findall(patterns, text)
    text_with_only_special_tokens = ''.join(special_tokens_found)

    return text_with_only_special_tokens


# Image
def load_images(paths):
    return [load_image(path) for path in paths]


def load_image(path):
    img_feature = Image()
    img = img_feature.decode_example(img_feature.encode_example(path))
    return img


def pil_to_opencv(pil_image):
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    numpy_image = np.array(pil_image)
    # RGB to BGR
    opencv_image = numpy_image[:, :, ::-1]
    return opencv_image


def get_image_size(path, ):
    import os
    return os.path.getsize(path)


def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    ix_min = max(x1_min, x2_min)
    ix_max = min(x1_max, x2_max)
    iy_min = max(y1_min, y2_min)
    iy_max = min(y1_max, y2_max)
    intersection = max(0, (ix_max - ix_min) * (iy_max - iy_min))
    union = area1 + area2 - intersection
    return 1.0 * intersection / union


# Audio
def load_audios(paths):
    return [load_audio(path) for path in paths]


def load_audio(path, sampling_rate=None):
    aud_feature = Audio(sampling_rate)
    aud = aud_feature.decode_example(aud_feature.encode_example(path))
    return aud['array'], aud['sampling_rate']


# Others
def size_to_bytes(size):
    alphabets_list = [char for char in size if char.isalpha()]
    numbers_list = [char for char in size if char.isdigit()]

    if len(numbers_list) == 0:
        raise ValueError(f'Your input `size` does not contain numbers: {size}')

    size_numbers = int(float(''.join(numbers_list)))

    if len(alphabets_list) == 0:
        # by default, if users do not specify the units, the number will be
        # regarded as in bytes
        return size_numbers

    suffix = ''.join(alphabets_list).lower()

    if suffix == 'kb' or suffix == 'kib':
        return size_numbers << 10
    elif suffix == 'mb' or suffix == 'mib':
        return size_numbers << 20
    elif suffix == 'gb' or suffix == 'gib':
        return size_numbers << 30
    elif suffix == 'tb' or suffix == 'tib':
        return size_numbers << 40
    elif suffix == 'pb' or suffix == 'pib':
        return size_numbers << 50
    elif suffix == 'eb' or suffix == 'eib':
        return size_numbers << 60
    elif suffix == 'zb' or suffix == 'zib':
        return size_numbers << 70
    elif suffix == 'yb' or suffix == 'yib':
        return size_numbers << 80
    else:
        raise ValueError(f'You specified unidentifiable unit: {suffix}, '
                         f'expected in [KB, MB, GB, TB, PB, EB, ZB, YB, '
                         f'KiB, MiB, GiB, TiB, PiB, EiB, ZiB, YiB], '
                         f'(case insensitive, counted by *Bytes*).')


def insert_texts_after_placeholders(original_string,
                                    placeholders,
                                    new_texts,
                                    delimiter_in_insert_pos=' '):
    if len(placeholders) != len(new_texts):
        raise ValueError(
            'The number of placeholders and new_texts must be equal')

    modified_string = original_string
    for placeholder, new_text in zip(placeholders, new_texts):
        # Find the index of the next occurrence of the placeholder
        index = modified_string.find(placeholder)
        if index == -1:
            raise ValueError(
                f"Placeholder '{placeholder}' not found in the string")
        # Insert new_text at the found index position
        modified_string = \
            modified_string[:index + len(placeholder)] + \
            new_text + \
            delimiter_in_insert_pos + \
            modified_string[index + len(placeholder):]

    return modified_string
