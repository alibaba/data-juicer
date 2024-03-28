import base64
import datetime
import imghdr
import io
import mimetypes
import os
import re
import shutil
from typing import List, Union

import av
import numpy as np
from datasets import Audio, Image
from loguru import logger

from data_juicer.utils.constant import DEFAULT_PREFIX, Fields
from data_juicer.utils.file_utils import add_suffix_to_filename

# suppress most warnings from av
av.logging.set_level(av.logging.PANIC)


# A class to keep special tokens for multimodal information in the texts
# The tokens in this class can be updated by corresponding arguments in config
class SpecialTokens(object):
    # modality
    image = f'<{DEFAULT_PREFIX}image>'
    audio = f'<{DEFAULT_PREFIX}audio>'
    video = f'<{DEFAULT_PREFIX}video>'

    # others
    eoc = f'<|{DEFAULT_PREFIX}eoc|>'


AV_STREAM_THREAD_TYPE = 'AUTO'
"""
    av stream thread type support "SLICE", "FRAME", "AUTO".

        "SLICE": Decode more than one part of a single frame at once

        "FRAME": Decode more than one frame at once

        "AUTO": Using both "FRAME" and "SLICE"
        AUTO is faster when there are no video latency.

"""


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


def load_data_with_context(sample, context, loaded_data_keys, load_func):
    """
    The unified loading function with contexts for multimodal data.
    """
    data = {}
    for loaded_data_key in loaded_data_keys:
        if context and loaded_data_key in sample[Fields.context]:
            # load from context
            data[loaded_data_key] = sample[Fields.context][loaded_data_key]
        else:
            if loaded_data_key not in data:
                # avoid load the same data
                data_item = load_func(loaded_data_key)
                data[loaded_data_key] = data_item
                if context:
                    # store the data into context
                    sample[Fields.context][loaded_data_key] = data_item
    return sample, data


# Images
def load_images(paths):
    return [load_image(path) for path in paths]


def load_images_byte(paths):
    return [load_image_byte(path) for path in paths]


def load_image(path):
    img_feature = Image()
    img = img_feature.decode_example(img_feature.encode_example(path))
    img = img.convert('RGB')
    return img


def load_image_byte(path):
    with open(path, 'rb') as image_file:
        image_data = image_file.read()
    return image_data


def image_path_to_base64(image_path, mime_type=None):
    with open(image_path, 'rb') as image_file:
        image_byte = image_file.read()
    return image_byte_to_base64(image_byte, mime_type)


def image_byte_to_base64(image_byte, mime_type=None):
    image_base64 = base64.b64encode(image_byte).decode('utf-8')

    if mime_type is None:
        return image_base64

    if mime_type == 'auto':
        # guess correct mime_type
        image_type = imghdr.what(None, h=image_byte)
        mime_type = mimetypes.types_map.get(f'.{image_type}')
        if mime_type is None:
            mime_type = 'image/jpeg'

    # https://www.rfc-editor.org/rfc/rfc2397
    return f'data:{mime_type};base64,{image_base64}'


def pil_to_opencv(pil_image):
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    numpy_image = np.array(pil_image)
    # RGB to BGR
    opencv_image = numpy_image[:, :, ::-1]
    return opencv_image


def get_file_size(path):
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


# Audios
def load_audios(paths):
    return [load_audio(path) for path in paths]


def load_audio(path, sampling_rate=None):
    aud_feature = Audio(sampling_rate)
    aud = aud_feature.decode_example(aud_feature.encode_example(path))
    return aud['array'], aud['sampling_rate']


# Videos
def load_videos(paths):
    return [load_video(path) for path in paths]


def load_video(path):
    """
    Load a video using its path.

    :param path: the path to this video.
    :return: a container object form PyAv library, which contains all streams
        in this video (video/audio/...) and can be used to decode these streams
        to frames.
    """
    container = av.open(path)
    return container


def video_path_to_base64(video_path, frame_num=None, fps=None, mime_type=None):
    duration = get_video_duration(video_path)

    if fps is not None:
        valid_frame_num = round(duration * fps)
        if frame_num is not None:
            valid_frame_num = min(valid_frame_num, frame_num)
    else:
        valid_frame_num = frame_num or 0

    frames_base64 = []
    if frame_num > 0:
        frames = extract_video_frames_uniformly(video_path, valid_frame_num)
        for frame in frames:
            img = frame.to_image()
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            buf.seek(0)
            frame_base64 = image_byte_to_base64(buf.read(), mime_type)
            frames_base64.append(frame_base64)
    return frames_base64


def get_video_duration(input_video: Union[str, av.container.InputContainer],
                       video_stream_index=0):
    """
    Get the video's duration from the container

    :param input_video: the container object form PyAv library, which
        contains all streams in this video (video/audio/...) and can be used
        to decode these streams to frames.
    :param video_stream_index: the video stream index to decode,
        default set to 0.
    :return: duration of the video in second
    """
    if isinstance(input_video, str):
        container = av.open(input_video)
    elif isinstance(input_video, av.container.InputContainer):
        container = input_video
    else:
        raise ValueError(f'Unsupported type of input_video. Should be one of '
                         f'[str, av.container.InputContainer], but given '
                         f'[{type(input_video)}].')

    input_video_stream = container.streams.video[video_stream_index]
    duration = input_video_stream.duration * input_video_stream.time_base
    return float(duration)


def get_decoded_frames_from_video(
        input_video: Union[str, av.container.InputContainer],
        video_stream_index=0):
    """
    Get the video's frames from the container

    :param input_video: the container object form PyAv library, which
        contains all streams in this video (video/audio/...) and can be used
        to decode these streams to frames.
    :param video_stream_index: the video stream index to decode,
        default set to 0.
    :return: an iterator of all the frames of the video
    """
    if isinstance(input_video, str):
        container = av.open(input_video)
    elif isinstance(input_video, av.container.InputContainer):
        container = input_video
    stream = container.streams.video[video_stream_index]
    # use "AUTO" thread_type for faster decode
    stream.thread_type = AV_STREAM_THREAD_TYPE
    return container.decode(stream)


def cut_video_by_seconds(
    input_video: Union[str, av.container.InputContainer],
    output_video: str,
    start_seconds: float,
    end_seconds: float = None,
):
    """
    Cut a video into several segments by times in second.

    :param input_video: the path to input video or the video container.
    :param output_video: the path to output video.
    :param start_seconds: the start time in second.
    :param end_seconds: the end time in second. If it's None, this function
        will cut the video from the start_seconds to the end of the video.
    :return: a boolean flag indicating whether the video was successfully
        cut or not.
    """
    # open the original video
    if isinstance(input_video, str):
        container = av.open(input_video)
    else:
        container = input_video

    # create the output video
    output_container = av.open(output_video, 'w')

    # add the video stream into the output video according to input video
    input_video_stream = container.streams.video[0]
    codec_name = input_video_stream.codec_context.name
    fps = input_video_stream.base_rate
    output_video_stream = output_container.add_stream(codec_name,
                                                      rate=str(fps))
    output_video_stream.width = input_video_stream.codec_context.width
    output_video_stream.height = input_video_stream.codec_context.height
    output_video_stream.pix_fmt = input_video_stream.codec_context.pix_fmt

    # add the audio stream into the output video with template of input audio
    if len(container.streams.audio) == 0:
        input_audio_stream = None
    else:
        input_audio_stream = container.streams.audio[0]
        output_container.add_stream(template=input_audio_stream)

    # seek to the start time, time must be in microsecond if no
    # stream is specified
    container.seek(int(start_seconds * 1000000),
                   any_frame=False,
                   backward=True)

    # copy the video and audio streams until the end time
    # NOTICE: for different streams, the time have to be converted to be
    # in the corresponding time base.
    video_at_the_end = False
    # compute the start/end pts for video/audio streams
    video_start_pts = int(start_seconds / input_video_stream.time_base)
    video_end_pts = (end_seconds / input_video_stream.time_base
                     if end_seconds else input_video_stream.duration)
    if input_audio_stream is not None:
        audio_start_pts = int(start_seconds / input_audio_stream.time_base)
        audio_end_pts = (end_seconds / input_audio_stream.time_base
                         if end_seconds else input_audio_stream.duration)
    for packet in container.demux(input_video_stream, input_audio_stream):
        if packet.stream.type == 'video':
            for frame in packet.decode():
                if frame.pts < video_start_pts:
                    continue
                if frame.pts > video_end_pts:
                    # continue to check until the next P/I frame
                    if frame.pict_type in {'P', 'I'}:
                        video_at_the_end = True
                        break
                    continue
                frame.pts -= video_start_pts  # timestamp alignment
                for inter_packet in output_video_stream.encode(frame):
                    output_container.mux(inter_packet)
        elif packet.stream.type == 'audio':
            if packet.pts is None or packet.dts is None:
                continue
            if packet.pts < audio_start_pts or packet.pts > audio_end_pts:
                continue
            packet.pts -= audio_start_pts
            packet.dts -= audio_start_pts
            output_container.mux(packet)
        if video_at_the_end:
            break

    # flush all packets
    for packet in output_video_stream.encode():
        output_container.mux(packet)

    # close the output videos
    if isinstance(input_video, str):
        container.close()
    output_container.close()
    if not os.path.exists(output_video):
        logger.warning(f'This video could not be successfully cut in '
                       f'[{start_seconds}, {end_seconds}] seconds. '
                       f'Please set more accurate parameters.')
    return os.path.exists(output_video)


def process_each_frame(input_video: Union[str, av.container.InputContainer],
                       output_video: str, frame_func):
    """
    Process each frame in video by replacing each frame by
    `frame_func(frame)`.

    :param input_video: the path to input video or the video container.
    :param output_video: the path to output video.
    :param frame_func: a function which inputs a frame and outputs another
        frame.
    """
    frame_modified = False

    # open the original video
    if isinstance(input_video, str):
        container = av.open(input_video)
    else:
        container = input_video

    # create the output video
    output_container = av.open(output_video, 'w')

    # add the audio stream into the output video with template of input audio
    for input_audio_stream in container.streams.audio:
        output_container.add_stream(template=input_audio_stream)

    # add the video stream into the output video according to input video
    for input_video_stream in container.streams.video:
        # search from the beginning
        container.seek(0, backward=False, any_frame=True)

        codec_name = input_video_stream.codec_context.name
        fps = input_video_stream.base_rate
        output_video_stream = output_container.add_stream(codec_name,
                                                          rate=str(fps))
        output_video_stream.pix_fmt = input_video_stream.codec_context.pix_fmt
        output_video_stream.width = input_video_stream.codec_context.width
        output_video_stream.height = input_video_stream.codec_context.height

        for packet in container.demux(input_video_stream):
            for frame in packet.decode():
                new_frame = frame_func(frame)
                if new_frame != frame:
                    frame_modified = True
                # for resize cases
                output_video_stream.width = new_frame.width
                output_video_stream.height = new_frame.height
                for inter_packet in output_video_stream.encode(new_frame):
                    output_container.mux(inter_packet)

        # flush all packets
        for packet in output_video_stream.encode():
            output_container.mux(packet)

    # close the output videos
    if isinstance(input_video, str):
        container.close()
    output_container.close()

    if frame_modified:
        return output_video
    else:
        shutil.rmtree(output_video, ignore_errors=True)
        return (input_video
                if isinstance(input_video, str) else input_video.name)


def extract_key_frames(input_video: Union[str, av.container.InputContainer]):
    """
    Extract key frames from the input video.

    :param input_video: input video path or container.
    :return: a list of key frames.
    """
    # load the input video
    if isinstance(input_video, str):
        container = av.open(input_video)
    elif isinstance(input_video, av.container.InputContainer):
        container = input_video
    else:
        raise ValueError(f'Unsupported type of input_video. Should be one of '
                         f'[str, av.container.InputContainer], but given '
                         f'[{type(input_video)}].')

    key_frames = []
    input_video_stream = container.streams.video[0]
    ori_skip_method = input_video_stream.codec_context.skip_frame
    input_video_stream.codec_context.skip_frame = 'NONKEY'
    # restore to the beginning of the video
    container.seek(0, backward=False, any_frame=False)
    for frame in container.decode(input_video_stream):
        key_frames.append(frame)
    # restore to the original skip_type
    input_video_stream.codec_context.skip_frame = ori_skip_method

    if isinstance(input_video, str):
        container.close()
    return key_frames


def get_key_frame_seconds(input_video: Union[str,
                                             av.container.InputContainer]):
    """
    Get seconds of key frames in the input video.
    """
    key_frames = extract_key_frames(input_video)
    ts = [float(f.pts * f.time_base) for f in key_frames]
    ts.sort()
    return ts


def extract_video_frames_uniformly(
    input_video: Union[str, av.container.InputContainer],
    frame_num: int,
):
    """
    Extract a number of video frames uniformly within the video duration.

    :param input_video: input video path or container.
    :param frame_num: The number of frames to be extracted. If it's 1, only the
        middle frame will be extracted. If it's 2, only the first and the last
        frames will be extracted. If it's larger than 2, in addition to the
        first and the last frames, other frames will be extracted uniformly
        within the video duration.
    :return: a list of extracted frames.
    """
    # load the input video
    if isinstance(input_video, str):
        container = av.open(input_video)
    elif isinstance(input_video, av.container.InputContainer):
        container = input_video
    else:
        raise ValueError(f'Unsupported type of input_video. Should be one of '
                         f'[str, av.container.InputContainer], but given '
                         f'[{type(input_video)}].')

    input_video_stream = container.streams.video[0]
    total_frame_num = input_video_stream.frames
    if total_frame_num < frame_num:
        logger.warning('Number of frames to be extracted is larger than the '
                       'total number of frames in this video. Set it to the '
                       'total number of frames.')
        frame_num = total_frame_num
    # calculate the frame seconds to be extracted
    duration = input_video_stream.duration * input_video_stream.time_base
    if frame_num == 1:
        extract_seconds = [duration / 2]
    else:
        step = duration / (frame_num - 1)
        extract_seconds = [step * i for i in range(0, frame_num)]

    # group durations according to the seconds of key frames
    key_frame_seconds = get_key_frame_seconds(container)
    if 0.0 not in key_frame_seconds:
        key_frame_seconds = [0.0] + key_frame_seconds
    if len(key_frame_seconds) == 1:
        second_groups = [extract_seconds]
    else:
        second_groups = []
        idx = 0
        group_id = 0
        curr_group = []
        curr_upper_bound_ts = key_frame_seconds[group_id + 1]
        while idx < len(extract_seconds):
            curr_ts = extract_seconds[idx]
            if curr_ts < curr_upper_bound_ts:
                curr_group.append(curr_ts)
                idx += 1
            else:
                second_groups.append(curr_group)
                group_id += 1
                curr_group = []
                if group_id >= len(key_frame_seconds) - 1:
                    break
                curr_upper_bound_ts = key_frame_seconds[group_id + 1]
        if len(curr_group) > 0:
            second_groups.append(curr_group)
        if idx < len(extract_seconds):
            second_groups.append(extract_seconds[idx:])

    # extract frames by their group's key frames
    extracted_frames = []
    time_base = input_video_stream.time_base
    for i, second_group in enumerate(second_groups):
        key_frame_second = key_frame_seconds[i]
        if len(second_group) == 0:
            continue
        if key_frame_second == 0.0:
            # search from the beginning
            container.seek(0, backward=False, any_frame=True)
            search_idx = 0
            curr_pts = second_group[search_idx] / time_base
            for frame in container.decode(input_video_stream):
                if frame.pts >= curr_pts:
                    extracted_frames.append(frame)
                    search_idx += 1
                    if search_idx >= len(second_group):
                        break
                    curr_pts = second_group[search_idx] / time_base
        else:
            # search from a key frame
            container.seek(int(key_frame_second * 1e6))
            search_idx = 0
            curr_pts = second_group[search_idx] / time_base
            find_all = False
            for packet in container.demux(input_video_stream):
                for frame in packet.decode():
                    if frame.pts >= curr_pts:
                        extracted_frames.append(frame)
                        search_idx += 1
                        if search_idx >= len(second_group):
                            find_all = True
                            break
                        curr_pts = second_group[search_idx] / time_base
                if find_all:
                    break
            if not find_all and frame is not None:
                # add the last frame
                extracted_frames.append(frame)

    # if the container is opened in this function, close it
    if isinstance(input_video, str):
        container.close()
    return extracted_frames


def extract_audio_from_video(
    input_video: Union[str, av.container.InputContainer],
    output_audio: str = None,
    start_seconds: int = 0,
    end_seconds: int = None,
    stream_indexes: Union[int, List[int]] = None,
):
    """
    Extract audio data for the given video.

    :param input_video: input video. Can be a video path or an
        av.container.InputContainer.
    :param output_audio: output audio path. If it's None, the audio data won't
        be written to file. If stream_indexes is not None, it will output
        multiple audio files with original filename and the stream indexes.
        Default: None.
    :param start_seconds: the start seconds to extract audio data. Default: 0,
        which means extract from the start of the video.
    :param end_seconds: the end seconds to stop extracting audio data. If it's
        None, the extraction won't stop until the end of the video. Default:
        None.
    :param stream_indexes: there might be multiple audio streams in the video,
        so we need to decide which audio streams with stream_indexes will be
        extracted. It can be a single index or a list of indexes. If it's None,
        all audio streams will be extracted. Default: None.
    """
    if isinstance(input_video, str):
        input_container = av.open(input_video)
    elif isinstance(input_video, av.container.InputContainer):
        input_container = input_video
    else:
        raise ValueError(f'Unsupported type of input_video. Should be one of '
                         f'[str, av.container.InputContainer], but given '
                         f'[{type(input_video)}].')

    if output_audio and not output_audio.endswith('mp3'):
        raise ValueError(f'Now we only support export the audios into `mp3` '
                         f'format, but given '
                         f'[{os.path.splitext(output_audio)[1]}')

    # no audios in the video
    num_audio_streams = len(input_container.streams.audio)
    if stream_indexes is None:
        valid_stream_indexes = list(range(num_audio_streams))
    elif isinstance(stream_indexes, int):
        valid_stream_indexes = [stream_indexes]
    else:
        # remove indexes that are larger than the total number of audio streams
        valid_stream_indexes = [
            idx for idx in stream_indexes if idx < num_audio_streams
        ]
    # no valid expected audio streams
    if len(valid_stream_indexes) == 0:
        return [], [], valid_stream_indexes

    audio_data_list = []
    audio_sampling_rate_list = []
    for idx in valid_stream_indexes:
        # read the current audio stream
        input_audio_stream = input_container.streams.audio[idx]
        # get the sampling rate
        audio_sampling_rate_list.append(float(1 /
                                              input_audio_stream.time_base))

        if output_audio:
            # if the output_audio is not None, prepare the output audio file
            this_output_audio = add_suffix_to_filename(output_audio, f'_{idx}')
            output_container = av.open(this_output_audio, 'w')
            output_stream = output_container.add_stream('mp3')

        # get the start/end pts
        start_pts = int(start_seconds / input_audio_stream.time_base)
        end_pts = (end_seconds /
                   input_audio_stream.time_base if end_seconds else None)

        audio_data = []
        for frame in input_container.decode(input_audio_stream):
            if frame.pts is None or frame.dts is None:
                continue
            if frame.pts < start_pts:
                continue
            if end_pts and frame.pts > end_pts:
                break
            # get frame data
            array = frame.to_ndarray()[0]
            audio_data.append(array)

            if output_audio:
                # compute the right pts when writing an audio file
                frame.pts -= start_pts
                frame.dts -= start_pts
                for packet in output_stream.encode(frame):
                    output_container.mux(packet)

        # flush
        if output_audio:
            for packet in output_stream.encode(None):
                output_container.mux(packet)

        if isinstance(input_video, str):
            input_container.close()
        if output_audio:
            output_container.close()
        audio_data_list.append(np.concatenate(audio_data))

    return audio_data_list, audio_sampling_rate_list, valid_stream_indexes


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
            delimiter_in_insert_pos + \
            new_text + \
            delimiter_in_insert_pos + \
            modified_string[index + len(placeholder):]

    return modified_string


def timecode_string_to_seconds(timecode: str):
    """
    Convert a timecode string to the float seconds.

    :param timecode: the input timecode string. Must in "HH:MM:SS.fff(fff)"
        format.
    """
    # parse the timecode string
    dt = datetime.datetime.strptime(timecode, '%H:%M:%S.%f')

    # compute the start/end time in second
    pts = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    return pts


def parse_string_to_roi(roi_string, roi_type='pixel'):
    """
    Convert a roi string to four number x1, y1, x2, y2 stand for the region.
    When the type is 'pixel', (x1, y1), (x2, y2) are the locations of pixels
    in the top left corner and the bottom right corner respectively. If the
    roi_type is 'ratio', the coordinates are normalized by wights and
    heights.

    :param roi_string: the roi string
    :patam roi_type: the roi string type
    return tuple of (x1, y1, x2, y2) if roi_string is valid, else None
    """
    if not roi_string:
        return None

    pattern = r'^\s*[\[\(]?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\]\)]?\s*$'  # noqa: E501

    match = re.match(pattern, roi_string)

    if match:
        if roi_type == 'pixel':
            return tuple(int(num) for num in match.groups())
        elif roi_type == 'ratio':
            return tuple(min(1.0, float(num)) for num in match.groups())
        else:
            logger.warning('The roi_type must be "pixel" or "ratio".')
            return None
    else:
        logger.warning(
            'The roi_string must be four no negative numbers in the '
            'format of "x1, y1, x2, y2", "(x1, y1, x2, y2)", or '
            '"[x1, y1, x2, y2]".')
        return None
    return None
