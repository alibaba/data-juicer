import copy

from jsonargparse.typing import ClosedUnitInterval
from loguru import logger

from data_juicer.utils.mm_utils import (SpecialTokens, image_byte_to_base64,
                                        insert_texts_after_placeholders,
                                        load_image_byte,
                                        remove_non_special_tokens,
                                        remove_special_tokens)
from data_juicer.utils.model_utils import call_gpt_vision_api

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_IMAGES

SYSTEM_PROMPTS = {
    'resoning':
    "You are an AI visual assistant that can analyze a single image. The task is to use the provided image, create a plausible question about the image, and provide the answer in detail.\n\nYou can create complex questions beyond describing the scene. Make the question challenging by not including the visual content details in the question so that the user needs to reason about that first.\n\nTo answer such questions, you should require first understanding the visual content, then based on the background knowledge or reasoning, either explain why the things are happening that way, or provide guides and help to user's request. \n\nPlease give the Q&A content directly and separate questions and answers with Q and A.",  # noqa: E501
    'description':
    'You are an AI visual assistant that can analyze a single image. The task is to use the provided image, create a reasonable question that describes the content of the image, and provide the answer in detail.\n\nPlease give the Q&A content directly and separate questions and answers with Q and A.',  # noqa: E501
    'conversation':
    'You are an AI visual assistant, and you are seeing a single image.\n\nDesign a conversation between you and a person asking about this image. The answers should be in a tone that a visual AI assistant is seeing the image and answering the question. Ask diverse questions and give corresponding answers.\n\nInclude questions asking about the visual content of the image, including the object types, counting the objects, object actions, object locations, relative positions between objects, etc. Only include questions that have definite answers:\n(1) one can see the content in the image that the question asks about and can answer confidently;\n(2) one can determine confidently from the image that it is not in the image.\nDo not ask any question that cannot be answered confidently.\n\nConversation also include complex questions that are relevant to the content in the image, for example, asking about background knowledge of the objects in the image, asking to discuss about events happening in the image, etc. Again, do not ask about uncertain details.\nProvide detailed answers when answering complex questions. For example, give detailed examples or reasoning steps to make the content more convincing and well-organized. Please give the content of the conversation directly and separate questions and answers with Q and A'  # noqa: E501
}


@OPERATORS.register_module('image_captioning_from_gpt4v_mapper')
@LOADED_IMAGES.register_module('image_captioning_from_gpt4v_mapper')
class ImageCaptioningFromGPT4VMapper(Mapper):
    """Mapper to generate samples whose texts are generated based on
    gpt-4-visison and the image."""

    def __init__(self,
                 mode: str = 'description',
                 api_key: str = None,
                 max_tokens: int = 500,
                 temperature: ClosedUnitInterval = 1.0,
                 system_prompt: str = '',
                 user_prompt: str = '',
                 user_prompt_key: str = None,
                 keep_original_sample: bool = True,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param mode: mode of text generated from images, can be one of
            ['resoning', 'description', 'conversation', 'custom']
        :param api_key: the API key to authenticate the request.
        :param max_tokens: the maximum number of tokens to generate.
            Default is 500.
        :param temperature: controls the randomness of the output (range
            from 0 to 1). Default is 0.
        :param system_prompt: a string prompt used to set the context of a
            conversation and provide global guidance or rules for the
            gpt4-vision so that it can  generate responses in the expected way.
            If `mode` set to `custom`, the parameter will be used.
        :param user_prompt: a string prompt to guide the generation of
            gpt4-vision for each samples. It's "" in default, which means no
            prompt provided.
        :param uers_prompt_key: the key name of fields in samples to store
            prompts for each sample. It's used for set different prompts for
            different samples. If it's none, use prompt in parameter "prompt".
            It's None in default.
        :param keep_original_sample: whether to keep the original sample. If
            it's set to False, there will be only generated text in the
            final datasets and the original text will be removed. It's True
            in default.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._batched_op = True
        if mode not in ['resoning', 'description', 'conversation', 'custom']:
            raise ValueError(
                f'Mode [{mode}] is not supported. '
                f'Can only be one of '
                f'["resoning", "description", "conversation", "custom"].')

        if mode == 'custom':
            self.system_prompt = system_prompt
            logger.info('The parameter `mode` set to `[custom]`. Data-Juicer '
                        'will use `system_prompt` to generate text.')
        else:
            self.system_prompt = SYSTEM_PROMPTS[mode]
            logger.info(
                f'The parameter `mode` set to [{mode}]. Data-Juicer will '
                f'use default prompt to generate text.')

        self.mode = mode
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.user_prompt = user_prompt
        self.user_prompt_key = user_prompt_key
        self.keep_original_sample = keep_original_sample
        self.any_or_all = any_or_all
        self.extra_args = kwargs

        # report a warning when both user_prompt and user_prompt_key are set
        if self.user_prompt and self.user_prompt_key:
            logger.warning(
                'Both the parameter `user_prompt` and `user_prompt_key` are '
                'set. Data-Juicer will consider `user_prompt_key` first.')

    def _process_single_sample(self, sample):
        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            return []

        # the generated results
        generated_sample = copy.deepcopy(sample)
        generated_sample[self.text_key] = ''

        # load all image(s)
        loaded_image_keys = sample[self.image_key]
        images = {}
        for loaded_image_key in loaded_image_keys:
            if loaded_image_key not in images:
                # avoid loading the same images
                image = load_image_byte(loaded_image_key)
                images[loaded_image_key] = image

        # construct user prompts
        if self.user_prompt_key and isinstance(sample[self.user_prompt_key],
                                               str):
            # check user_prompt_key is not None, and it's a str in the sample
            prompt_texts = sample[self.user_prompt_key]
        elif self.user_prompt and isinstance(self.user_prompt, str):
            # check prompt is not None, and it's a str
            prompt_texts = self.user_prompt
        else:
            prompt_texts = ''

        offset = 0
        # do generation for each image chunk by chunk
        for chunk in sample[self.text_key].split(SpecialTokens.eoc):
            # skip empty chunks or contents after the last eoc token
            if not chunk.strip():
                continue

            else:
                img_count = chunk.count(SpecialTokens.image)
                text_with_only_special_tokens = remove_non_special_tokens(
                    chunk)
                generated_text_single_chunk = []
                for image_key in loaded_image_keys[offset:offset + img_count]:
                    image = images[image_key]
                    res = call_gpt_vision_api(self.system_prompt,
                                              prompt_texts,
                                              image_byte_to_base64(
                                                  image, 'image/jpeg'),
                                              api_key=self.api_key,
                                              max_tokens=self.max_tokens,
                                              temperature=self.temperature,
                                              **self.extra_args)
                    generated_text_single_chunk.append(res)
                if self.any_or_all == 'all' and not all(
                        generated_text_single_chunk):
                    return []

                # insert the generated text according to given mode
                place_holders = [SpecialTokens.image] * img_count
                new_generated_text_per_chunk = insert_texts_after_placeholders(
                    original_string=text_with_only_special_tokens,
                    placeholders=place_holders,
                    new_texts=generated_text_single_chunk)
                generated_sample[
                    self.
                    text_key] += f'{new_generated_text_per_chunk}{SpecialTokens.eoc}'  # noqa: E501
                offset += img_count
        if self.any_or_all == 'any' and not remove_special_tokens(
                generated_sample[self.text_key]):
            return []

        return [generated_sample]

    def process(self, samples):
        # reconstruct samples from "dict of lists" to "list of dicts"
        reconstructed_samples = []
        for i in range(len(samples[self.text_key])):
            reconstructed_samples.append(
                {key: samples[key][i]
                 for key in samples})
        samples_after_generation = []
        # do generation for each sample within the batch
        for ori_sample in reconstructed_samples:
            if self.keep_original_sample:
                samples_after_generation.append(ori_sample)
            generated_samples = self._process_single_sample(ori_sample)
            if len(generated_samples) != 0:
                samples_after_generation.extend(generated_samples)
        # reconstruct samples from "list of dicts" to "dict of lists"
        keys = samples_after_generation[0].keys()
        res_samples = {}
        for key in keys:
            res_samples[key] = [s[key] for s in samples_after_generation]

        return res_samples
