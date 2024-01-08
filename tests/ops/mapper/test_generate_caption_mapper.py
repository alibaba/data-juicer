import os
import unittest

from datasets import Dataset

from data_juicer.ops.mapper.generate_caption_mapper import \
    GenerateCaptionMapper
from data_juicer.utils.mm_utils import SpecialTokens


class GenerateCaptionMapperTest(unittest.TestCase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')

    cat_path = os.path.join(data_path, 'cat.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')

    def _run_mapper(self, dataset: Dataset, op, num_proc=1):

        dataset = dataset.map(op.compute_stats, num_proc=num_proc)
        dataset = dataset.filter(op.process, num_proc=num_proc)
        dataset_list = dataset.select_columns(column_names=['text']).to_list()
        # assert the caption is generated successfully in terms of not_none
        # as the generated content is not deterministic
        self.assertNotEqual(len(dataset_list), 0)

    def test_no_eoc_special_token(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = GenerateCaptionMapper(hf_blip2='Salesforce/blip2-opt-2.7b',
                                   caption_num=1,
                                   keep_candidate_mode='random_any')
        self._run_mapper(dataset, op)

    def test_eoc_special_token(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat{SpecialTokens.eoc}',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella{SpecialTokens.eoc}',
            'images': [self.img3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = GenerateCaptionMapper(hf_blip2='Salesforce/blip2-opt-2.7b',
                                   caption_num=1,
                                   keep_candidate_mode='random_any')
        self._run_mapper(dataset, op)

    def test_multi_candidate_keep_random_any(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = GenerateCaptionMapper(hf_blip2='Salesforce/blip2-opt-2.7b',
                                   caption_num=4,
                                   keep_candidate_mode='random_any')
        self._run_mapper(dataset, op)

    def test_multi_candidate_keep_all(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = GenerateCaptionMapper(hf_blip2='Salesforce/blip2-opt-2.7b',
                                   caption_num=4,
                                   keep_candidate_mode='all')
        self._run_mapper(dataset, op)

    def test_multi_candidate_keep_similar_one(self):
        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = GenerateCaptionMapper(hf_blip2='Salesforce/blip2-opt-2.7b',
                                   caption_num=4,
                                   keep_candidate_mode='similar_one_simhash')
        self._run_mapper(dataset, op)

    def test_multi_process(self):
        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }] * 10

        dataset = Dataset.from_list(ds_list)
        op = GenerateCaptionMapper(hf_blip2='Salesforce/blip2-opt-2.7b',
                                   caption_num=1,
                                   keep_candidate_mode='random_any')
        self._run_mapper(dataset, op, num_proc=4)


if __name__ == '__main__':
    unittest.main()
