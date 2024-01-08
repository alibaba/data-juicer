import copy
import random

import numpy as np
from jsonargparse.typing import PositiveInt

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import (SpecialTokens,
                                        insert_texts_after_placeholders,
                                        load_image, remove_non_special_tokens,
                                        remove_special_tokens)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..deduplicator.document_simhash_deduplicator import \
    DocumentSimhashDeduplicator
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'generate_caption_mapper'

with AvailabilityChecking(['torch', 'transformers'], OP_NAME):
    import torch
    import transformers  # noqa: F401

    # avoid hanging when calling blip2 in multiprocessing
    torch.set_num_threads(1)


def jaccard_similarity_np(hash_val1, hash_val2):
    equal_hashes = np.sum(hash_val1 == hash_val2)
    total_hashes = len(hash_val1)  # Assume both vectors are the same length
    return equal_hashes / total_hashes


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class GenerateCaptionMapper(Mapper):
    """Mapper to generate samples whose captions are generated based on
    another model and the figure."""

    def __init__(self,
                 hf_blip2='Salesforce/blip2-opt-2.7b',
                 caption_num: PositiveInt = 1,
                 keep_candidate_mode: str = 'random_any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_blip2: blip2 model name on huggingface to generate caption
        :param caption_num: how many candidate captions to generate
        for each image
        :param keep_candidate_mode: retain strategy for the generated
        $caption_num$ candidates.
            'random_any': Retain the random one from generated captions
            'similar_one': Retain the generated one that is most similar to the
                original caption
            'all': Retain all generated captions by concatenation
        Note: This is a batched_OP, whose input and output type are
            both list. Suppose there are $N$ list of input samples, whose batch
            size is $b$, and denote caption_num as $M$.
            The number of total samples after generation is $2Nb$
            for 'random_any' and 'similar_one' mode,
             and $(1+M)Nb$ for 'all' mode.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.is_batched_op = True
        if keep_candidate_mode not in [
                'random_any', 'similar_one_simhash', 'all'
        ]:
            raise ValueError(
                f'Keep strategy [{keep_candidate_mode}] is not supported. '
                f'Can only be one of '
                f'["random_any", "similar_one_simhash", "all"].')

        self.model_key = prepare_model(model_type='hf_blip',
                                       model_key=hf_blip2,
                                       usage='conditional_generation')
        model, img_processor = get_model(model_key=self.model_key,
                                         usage='conditional_generation')
        self.model_in_ctx = model
        self.img_processor_in_ctx = img_processor
        self.caption_num = caption_num
        self.keep_candidate_mode = keep_candidate_mode
        self.extra_args = kwargs

        if keep_candidate_mode in ['random_any', 'similar_one_simhash']:
            self.num_newly_generated_samples = 1
        elif keep_candidate_mode in ['all']:
            self.num_newly_generated_samples = self.caption_num
        else:
            self.num_newly_generated_samples = 0

    def _process_single_sample(self, ori_sample):
        """

        :param ori_sample: a single data sample before applying generation
        :return: batched results after generation
        """
        # there is no image in this sample
        if self.image_key not in ori_sample or \
                not ori_sample[self.image_key]:
            return []

        # the generated results
        generated_samples = [
            copy.deepcopy(ori_sample)
            for _ in range(self.num_newly_generated_samples)
        ]
        for generated_sample in generated_samples:
            generated_sample[self.text_key] = ''

        # 1. load all image(s)
        loaded_image_keys = ori_sample[self.image_key]
        images = {}
        for loaded_image_key in loaded_image_keys:
            # load from context
            if loaded_image_key in ori_sample[Fields.context]:
                images[loaded_image_key] = \
                    ori_sample[Fields.context][loaded_image_key]
            elif loaded_image_key not in images:
                # avoid load the same images
                image = load_image(loaded_image_key)
                images[loaded_image_key] = image

        offset = 0

        # we follow such assumption:
        # all text/img/video/audio data within a chunk are correlated.
        # As a result,
        # the original text will be removed,
        # the generated text will be placed following each SpecialTokens.img
        # and the original special tokens are kept in an order-preserving way.

        # do generation for each image chunk by chunk
        for chunk in ori_sample[self.text_key].split(SpecialTokens.eoc):
            img_count = chunk.count(SpecialTokens.image)
            text_with_only_special_tokens = remove_non_special_tokens(chunk)
            image_chunk = []
            for image_key in loaded_image_keys[offset:offset + img_count]:
                image = images[image_key]
                image_chunk.append(image)

            # 2. generate candidate caption(s) in batch manner
            generated_text_candidates_single_chunk = \
                [[] for _ in range(img_count)]
            # an assistant 2-D array,
            # generated_text_candidates_single_chunk[i][j] indicates
            # the $i$-th generated candidate for the $j$-th image

            inputs = self.img_processor_in_ctx(images=image_chunk, )
            for i in range(self.caption_num):
                generated_ids = self.model_in_ctx.generate(**inputs)
                generated_text = self.img_processor_in_ctx.batch_decode(
                    generated_ids, skip_special_tokens=True)
                generated_text_candidates_single_chunk[i] = generated_text

            # 3. insert a list of generated captions into the positions of
            # subsequent placeholders in the original string
            new_generated_text_all_images = \
                [[] for _ in range(self.num_newly_generated_samples)]
            # new_generated_text_all_images is an helper array, element [i][j]
            # denotes the reduced $i$-th result for the $j$-th image

            # reduce the captions according to given mode image by image
            for j in range(img_count):
                new_generated_text_per_image = self._reduce_captions_per_image(
                    chunk, generated_text_candidates_single_chunk[j])
                assert self.num_newly_generated_samples == \
                       len(new_generated_text_per_image)
                for i in range(len(new_generated_text_per_image)):
                    new_generated_text_all_images[i].extend(
                        new_generated_text_per_image[i])

            # insert the captions according to given mode
            place_holders = [SpecialTokens.image] * img_count
            for i in range(self.num_newly_generated_samples):
                new_generated_text_per_chunk = insert_texts_after_placeholders(
                    original_string=text_with_only_special_tokens,
                    placeholders=place_holders,
                    new_texts=new_generated_text_all_images[i])
                generated_samples[i][self.text_key] += \
                    f'{new_generated_text_per_chunk}{SpecialTokens.eoc}'

            offset += img_count

        return generated_samples

    def _reduce_captions_per_image(self, chunk,
                                   generated_text_candidates_single_chunk):
        new_generated_text_per_chunk = []
        if self.keep_candidate_mode == 'random_any':
            new_generated_text_per_chunk.append(
                random.choice(generated_text_candidates_single_chunk))
        elif self.keep_candidate_mode == 'all':
            new_generated_text_per_chunk.extend(new_generated_text_per_chunk)
        elif self.keep_candidate_mode == 'similar_one_simhash':
            ori_normal_text = remove_special_tokens(chunk)
            # using a simhash OP to calculate their similarity
            op_simhash = DocumentSimhashDeduplicator(**self.extra_args)
            ori_text_hash = op_simhash.compute_hash(ori_normal_text)
            generated_text_hashes = [
                op_simhash.compute_hash(candidate_text)
                for candidate_text in generated_text_candidates_single_chunk
            ]
            similarity_scores = [
                jaccard_similarity_np(ori_text_hash, generated_text_hash)
                for generated_text_hash in generated_text_hashes
            ]
            max_index = max(range(len(similarity_scores)),
                            key=similarity_scores.__getitem__)
            new_generated_text_per_chunk.append(
                generated_text_candidates_single_chunk[max_index])
        return new_generated_text_per_chunk

    def process(self, samples):
        """
        Note: This is a batched_OP, whose the input and output type are
            both list. Suppose there are $N$ input sample list with batch
            size as $b$, and denote caption_num as $M$.
            the number of total samples after generation is $2Nb$
            for 'random_any' and 'similar_one' mode,
             and $(1+M)Nb$ for 'all' mode.
        :param samples:
        :return:
        """
        samples_after_generation = []
        # do generation for each sample within the batch
        for ori_sample in samples:
            samples_after_generation.append(ori_sample)
            generated_samples = self._process_single_sample(ori_sample)
            if len(generated_samples) != 0:
                samples_after_generation.extend(generated_samples)

        return samples_after_generation
