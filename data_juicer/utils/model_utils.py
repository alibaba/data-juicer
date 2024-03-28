import fnmatch
import os
from functools import partial
from typing import Any, List, Optional, Union

import multiprocess as mp
import requests
import wget
from loguru import logger

from data_juicer import cuda_device_count, use_cuda

from .cache_utils import DATA_JUICER_MODELS_CACHE as DJMC

MODEL_ZOO = {}

# Default cached models links for downloading
MODEL_LINKS = ('https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/'
               'data_juicer/models/')

# Backup cached models links for downloading
BACKUP_MODEL_LINKS = {
    # language identification model from fasttext
    'lid.176.bin':
    'https://dl.fbaipublicfiles.com/fasttext/supervised-models/',

    # tokenizer and language model for English from sentencepiece and KenLM
    '*.sp.model':
    'https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/',
    '*.arpa.bin':
    'https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/',

    # sentence split model from nltk punkt
    'punkt.*.pickle':
    'https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/'
    'data_juicer/models/',
}


def get_backup_model_link(model_name):
    for pattern, url in BACKUP_MODEL_LINKS.items():
        if fnmatch.fnmatch(model_name, pattern):
            return url
    return None


def check_model(model_name, force=False):
    """
    Check whether a model exists in DATA_JUICER_MODELS_CACHE.
    If exists, return its full path.
    Else, download it from cached models links.

    :param model_name: a specified model name
    :param force: Whether to download model forcefully or not, Sometimes
        the model file maybe incomplete for some reason, so need to
        download again forcefully.
    """
    # check for local model
    if os.path.exists(model_name):
        return model_name

    if not os.path.exists(DJMC):
        os.makedirs(DJMC)

    # check if the specified model exists. If it does not exist, download it
    cached_model_path = os.path.join(DJMC, model_name)
    if force:
        if os.path.exists(cached_model_path):
            os.remove(cached_model_path)
            logger.info(
                f'Model [{cached_model_path}] invalid, force to downloading...'
            )
        else:
            logger.info(
                f'Model [{cached_model_path}] not found . Downloading...')

        try:
            model_link = os.path.join(MODEL_LINKS, model_name)
            wget.download(model_link, cached_model_path, bar=None)
        except:  # noqa: E722
            try:
                backup_model_link = os.path.join(
                    get_backup_model_link(model_name), model_name)
                wget.download(backup_model_link, cached_model_path, bar=None)
            except:  # noqa: E722
                logger.error(
                    f'Downloading model [{model_name}] error. '
                    f'Please retry later or download it into {DJMC} '
                    f'manually from {model_link} or {backup_model_link} ')
                exit(1)
    return cached_model_path


def prepare_fasttext_model(model_name='lid.176.bin'):
    """
    Prepare and load a fasttext model.

    :param model_name: input model name
    :return: model instance.
    """
    import fasttext

    logger.info('Loading fasttext language identification model...')
    try:
        ft_model = fasttext.load_model(check_model(model_name))
    except:  # noqa: E722
        ft_model = fasttext.load_model(check_model(model_name, force=True))
    return ft_model


def prepare_sentencepiece_model(model_path):
    """
    Prepare and load a sentencepiece model.

    :param model_path: input model path
    :return: model instance
    """
    import sentencepiece

    logger.info('Loading sentencepiece model...')
    sentencepiece_model = sentencepiece.SentencePieceProcessor()
    try:
        sentencepiece_model.load(check_model(model_path))
    except:  # noqa: E722
        sentencepiece_model.load(check_model(model_path, force=True))
    return sentencepiece_model


def prepare_sentencepiece_for_lang(lang, name_pattern='{}.sp.model'):
    """
    Prepare and load a sentencepiece model for specific langauge.

    :param lang: language to render model name
    :param name_pattern: pattern to render the model name
    :return: model instance.
    """

    model_name = name_pattern.format(lang)
    return prepare_sentencepiece_model(model_name)


def prepare_kenlm_model(lang, name_pattern='{}.arpa.bin'):
    """
    Prepare and load a kenlm model.

    :param model_name: input model name in formatting syntax.
    :param lang: language to render model name
    :return: model instance.
    """
    import kenlm

    model_name = name_pattern.format(lang)

    logger.info('Loading kenlm language model...')
    try:
        kenlm_model = kenlm.Model(check_model(model_name))
    except:  # noqa: E722
        kenlm_model = kenlm.Model(check_model(model_name, force=True))
    return kenlm_model


def prepare_nltk_model(lang, name_pattern='punkt.{}.pickle'):
    """
    Prepare and load a nltk punkt model.

    :param model_name: input model name in formatting syntax
    :param lang: language to render model name
    :return: model instance.
    """
    from nltk.data import load

    nltk_to_punkt = {
        'en': 'english',
        'fr': 'french',
        'pt': 'portuguese',
        'es': 'spanish'
    }
    assert (lang in nltk_to_punkt.keys()
            ), 'lang must be one of the following: {}'.format(
                list(nltk_to_punkt.keys()))
    model_name = name_pattern.format(nltk_to_punkt[lang])

    logger.info('Loading nltk punkt split model...')
    try:
        nltk_model = load(check_model(model_name))
    except:  # noqa: E722
        nltk_model = load(check_model(model_name, force=True))
    return nltk_model


def prepare_video_blip_model(pretrained_model_name_or_path,
                             return_model=True,
                             trust_remote_code=False):
    """
    Prepare and load a video-clip model with the correspoding processor.

    :param pretrained_model_name_or_path: model name or path
    :param return_model: return model or not
    :param trust_remote_code: passed to transformers
    :return: a tuple (model, input processor) if `return_model` is True;
        otherwise, only the processor is returned.
    """
    import torch
    import torch.nn as nn
    from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                              Blip2Config, Blip2ForConditionalGeneration,
                              Blip2QFormerModel, Blip2VisionModel)
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    class VideoBlipVisionModel(Blip2VisionModel):
        """A simple, augmented version of Blip2VisionModel to handle
        videos."""

        def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[tuple, BaseModelOutputWithPooling]:
            """Flatten `pixel_values` along the batch and time dimension,
            pass it through the original vision model,
            then unflatten it back.

            :param pixel_values: a tensor of shape
            (batch, channel, time, height, width)

            :returns:
                last_hidden_state: a tensor of shape
                (batch, time * seq_len, hidden_size)
                pooler_output: a tensor of shape
                (batch, time, hidden_size)
                hidden_states:
                    a tuple of tensors of shape
                    (batch, time * seq_len, hidden_size),
                    one for the output of the embeddings +
                    one for each layer
                attentions:
                    a tuple of tensors of shape
                    (batch, time, num_heads, seq_len, seq_len),
                    one for each layer
            """
            if pixel_values is None:
                raise ValueError('You have to specify pixel_values')

            batch, _, time, _, _ = pixel_values.size()

            # flatten along the batch and time dimension to create a
            # tensor of shape
            # (batch * time, channel, height, width)
            flat_pixel_values = pixel_values.permute(0, 2, 1, 3,
                                                     4).flatten(end_dim=1)

            vision_outputs: BaseModelOutputWithPooling = super().forward(
                pixel_values=flat_pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            # now restore the original dimensions
            # vision_outputs.last_hidden_state is of shape
            # (batch * time, seq_len, hidden_size)
            seq_len = vision_outputs.last_hidden_state.size(1)
            last_hidden_state = vision_outputs.last_hidden_state.view(
                batch, time * seq_len, -1)
            # vision_outputs.pooler_output is of shape
            # (batch * time, hidden_size)
            pooler_output = vision_outputs.pooler_output.view(batch, time, -1)
            # hidden_states is a tuple of tensors of shape
            # (batch * time, seq_len, hidden_size)
            hidden_states = (tuple(
                hidden.view(batch, time * seq_len, -1)
                for hidden in vision_outputs.hidden_states)
                             if vision_outputs.hidden_states is not None else
                             None)
            # attentions is a tuple of tensors of shape
            # (batch * time, num_heads, seq_len, seq_len)
            attentions = (tuple(
                hidden.view(batch, time, -1, seq_len, seq_len)
                for hidden in vision_outputs.attentions)
                          if vision_outputs.attentions is not None else None)
            if return_dict:
                return BaseModelOutputWithPooling(
                    last_hidden_state=last_hidden_state,
                    pooler_output=pooler_output,
                    hidden_states=hidden_states,
                    attentions=attentions,
                )
            return (last_hidden_state, pooler_output, hidden_states,
                    attentions)

    class VideoBlipForConditionalGeneration(Blip2ForConditionalGeneration):

        def __init__(self, config: Blip2Config) -> None:
            # HACK: we call the grandparent super().__init__() to bypass
            # Blip2ForConditionalGeneration.__init__() so we can replace
            # self.vision_model
            super(Blip2ForConditionalGeneration, self).__init__(config)

            self.vision_model = VideoBlipVisionModel(config.vision_config)

            self.query_tokens = nn.Parameter(
                torch.zeros(1, config.num_query_tokens,
                            config.qformer_config.hidden_size))
            self.qformer = Blip2QFormerModel(config.qformer_config)

            self.language_projection = nn.Linear(
                config.qformer_config.hidden_size,
                config.text_config.hidden_size)
            if config.use_decoder_only_language_model:
                language_model = AutoModelForCausalLM.from_config(
                    config.text_config)
            else:
                language_model = AutoModelForSeq2SeqLM.from_config(
                    config.text_config)
            self.language_model = language_model

            # Initialize weights and apply final processing
            self.post_init()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
    if return_model:
        model_class = VideoBlipForConditionalGeneration
        model = model_class.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
    return (model, processor) if return_model else processor


def prepare_simple_aesthetics_model(pretrained_model_name_or_path,
                                    return_model=True):
    """
    Prepare and load a simple aesthetics model.

    :param pretrained_model_name_or_path: model name or path
    :param return_model: return model or not
    :return: a tuple (model, input processor) if `return_model` is True;
        otherwise, only the processor is returned.
    """
    from aesthetics_predictor import (AestheticsPredictorV1,
                                      AestheticsPredictorV2Linear,
                                      AestheticsPredictorV2ReLU)
    from transformers import CLIPProcessor

    processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path)
    if not return_model:
        return processor
    else:
        if 'v1' in pretrained_model_name_or_path:
            model = AestheticsPredictorV1.from_pretrained(
                pretrained_model_name_or_path)
        elif ('v2' in pretrained_model_name_or_path
              and 'linear' in pretrained_model_name_or_path):
            model = AestheticsPredictorV2Linear.from_pretrained(
                pretrained_model_name_or_path)
        elif ('v2' in pretrained_model_name_or_path
              and 'relu' in pretrained_model_name_or_path):
            model = AestheticsPredictorV2ReLU.from_pretrained(
                pretrained_model_name_or_path)
        else:
            raise ValueError(
                'Not support {}'.format(pretrained_model_name_or_path))
        return (model, processor)


def prepare_huggingface_model(pretrained_model_name_or_path,
                              return_model=True,
                              trust_remote_code=False):
    """
    Prepare and load a HuggingFace model with the correspoding processor.

    :param pretrained_model_name_or_path: model name or path
    :param return_model: return model or not
    :param trust_remote_code: passed to transformers
    :return: a tuple (model, input processor) if `return_model` is True;
        otherwise, only the processor is returned.
    """
    import transformers
    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code)

    if return_model:
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
        if hasattr(config, 'auto_map'):
            class_name = next(
                (k for k in config.auto_map if k.startswith('AutoModel')),
                'AutoModel')
        else:
            # TODO: What happens if more than one
            class_name = config.architectures[0]

        model_class = getattr(transformers, class_name)
        model = model_class.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code)

    return (model, processor) if return_model else processor


def prepare_spacy_model(lang, name_pattern='{}_core_web_md-3.5.0'):
    """
    Prepare spacy model for specific language.

    :param lang: language of sapcy model. Should be one of ["zh",
        "en"]
    :return: corresponding spacy model
    """
    import spacy

    assert lang in ['zh', 'en'], 'Diversity only support zh and en'
    model_name = name_pattern.format(lang)
    logger.info(f'Loading spacy model [{model_name}]...')
    compressed_model = '{}.zip'.format(model_name)

    # decompress the compressed model if it's not decompressed
    def decompress_model(compressed_model_path):
        decompressed_model_path = compressed_model_path.replace('.zip', '')
        if os.path.exists(decompressed_model_path) and os.path.isdir(
                decompressed_model_path):
            return decompressed_model_path
        import zipfile

        with zipfile.ZipFile(compressed_model_path) as zf:
            zf.extractall(DJMC)
        return decompressed_model_path

    try:
        diversity_model = spacy.load(
            decompress_model(check_model(compressed_model)))
    except:  # noqa: E722
        diversity_model = spacy.load(
            decompress_model(check_model(compressed_model, force=True)))
    return diversity_model


def prepare_diffusion_model(pretrained_model_name_or_path,
                            diffusion_type,
                            torch_dtype='fp32',
                            revision='main'):
    """
        Prepare and load an Diffusion model from HuggingFace.

        :param pretrained_model_name_or_path: input Diffusion model name
            or local path to the model
        :param diffusion_type: the use of the diffusion model. It can be
            'image2image', 'text2image', 'inpainting'
        :param torch_dtype: the floating point to load the diffusion
            model. Can be one of ['fp32', 'fp16', 'bf16']
        :param revision: The specific model version to use. It can be a
            branch name, a tag name, a commit id, or any identifier allowed
            by Git.
        :return: a Diffusion model.
    """
    import torch
    from diffusers import (AutoPipelineForImage2Image,
                           AutoPipelineForInpainting,
                           AutoPipelineForText2Image)

    diffusion_type_to_pipeline = {
        'image2image': AutoPipelineForImage2Image,
        'text2image': AutoPipelineForText2Image,
        'inpainting': AutoPipelineForInpainting
    }

    if diffusion_type not in diffusion_type_to_pipeline.keys():
        raise ValueError(
            f'Not support {diffusion_type} diffusion_type for diffusion '
            'model. Can only be one of '
            '["image2image", "text2image", "inpainting"].')

    if torch_dtype not in ['fp32', 'fp16', 'bf16']:
        raise ValueError(
            f'Not support {torch_dtype} torch_dtype for diffusion '
            'model. Can only be one of '
            '["fp32", "fp16", "bf16"].')

    if not use_cuda() and (torch_dtype == 'fp16' or torch_dtype == 'bf16'):
        raise ValueError(
            'In cpu mode, only fp32 torch_dtype can be used for diffusion'
            ' model.')

    pipeline = diffusion_type_to_pipeline[diffusion_type]
    if torch_dtype == 'bf16':
        torch_dtype = torch.bfloat16
    elif torch_dtype == 'fp16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = pipeline.from_pretrained(pretrained_model_name_or_path,
                                     revision=revision,
                                     torch_dtype=torch_dtype)

    return model


def prepare_recognizeAnything_model(
        pretrained_model_name_or_path='ram_plus_swin_large_14m.pth',
        input_size=384):
    """
    Prepare and load recognizeAnything model.

    :param model_name: input model name.
    :param input_size: the input size of the model.
    """
    from ram.models import ram_plus

    logger.info('Loading recognizeAnything model...')
    try:
        model = ram_plus(pretrained=check_model(pretrained_model_name_or_path),
                         image_size=input_size,
                         vit='swin_l')
    except:  # noqa: E722
        model = ram_plus(pretrained=check_model(pretrained_model_name_or_path,
                                                force=True),
                         image_size=input_size,
                         vit='swin_l')
    model.eval()
    return model


MODEL_FUNCTION_MAPPING = {
    'fasttext': prepare_fasttext_model,
    'sentencepiece': prepare_sentencepiece_for_lang,
    'kenlm': prepare_kenlm_model,
    'nltk': prepare_nltk_model,
    'huggingface': prepare_huggingface_model,
    'simple_aesthetics': prepare_simple_aesthetics_model,
    'spacy': prepare_spacy_model,
    'diffusion': prepare_diffusion_model,
    'video_blip': prepare_video_blip_model,
    'recognizeAnything': prepare_recognizeAnything_model
}


def prepare_model(model_type, **model_kwargs):
    assert (model_type in MODEL_FUNCTION_MAPPING.keys()
            ), 'model_type must be one of the following: {}'.format(
                list(MODEL_FUNCTION_MAPPING.keys()))
    global MODEL_ZOO
    model_func = MODEL_FUNCTION_MAPPING[model_type]
    model_key = partial(model_func, **model_kwargs)
    # always instantiate once for possible caching
    model_objects = model_key()
    MODEL_ZOO[model_key] = model_objects
    return model_key


def move_to_cuda(model, rank):
    # Assuming model can be either a single module or a tuple of modules
    if not isinstance(model, tuple):
        model = (model, )

    for module in model:
        if callable(getattr(module, 'to', None)):
            logger.info(
                f'Moving {module.__class__.__name__} to CUDA device {rank}')
            module.to(f'cuda:{rank}')


def get_model(model_key=None, rank=None):
    if model_key is None:
        return None

    global MODEL_ZOO
    if model_key not in MODEL_ZOO:
        logger.debug(
            f'{model_key} not found in MODEL_ZOO ({mp.current_process().name})'
        )
        MODEL_ZOO[model_key] = model_key()
    if use_cuda():
        rank = 0 if rank is None else rank
        rank = rank % cuda_device_count()
        move_to_cuda(MODEL_ZOO[model_key], rank)
    return MODEL_ZOO[model_key]


def call_gpt_vision_api(
    system_prompt: str = '',
    user_prompt: str = '',
    images: Union[str, List[str], None] = None,
    *,
    api_key: str = None,
    model: str = 'gpt-4-vision-preview',
    max_tokens: int = 500,
    temperature: float = 0.0,
    **kwargs: Any,
):
    images = [images] if isinstance(images, str) else (images or [])

    api_url = 'https://api.openai.com/v1/chat/completions'

    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        logger.error(
            'The api_key must be set either by passing it to the function '
            'call or by setting the OPENAI_API_KEY environment variable')
        return ''

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    image_payload = [{
        'type': 'image_url',
        'image_url': {
            'url': url,
            'detail': 'low'
        }
    } for url in images]
    data = {
        'model':
        model,
        'messages': [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role':
                'user',
                'content': [
                    {
                        'type': 'text',
                        'text': user_prompt
                    },
                    *image_payload,
                ],
            },
        ],
        'max_tokens':
        max_tokens,
        'temperature':
        temperature,
        **kwargs,
    }

    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        if 'choices' in result and result['choices']:
            return result['choices'][0]['message']['content']
        else:
            logger.warning('No results returned from the API.')
            return ''
    except requests.exceptions.HTTPError as errh:
        if errh.response.status_code == 401:
            logger.warning('Invalid API key provided.')
        elif errh.response.status_code == 429:
            logger.warning(
                'API request limit has been reached. Please try again later.')
        else:
            logger.warning(f'HTTP error occurred: {errh}')
    except requests.exceptions.ConnectionError:
        logger.warning('Network error occurred. Please check your connection.')
    except requests.exceptions.Timeout:
        logger.warning('The request timed out. Please try again later.')
    except requests.exceptions.RequestException as err:
        logger.warningt(f'An error occurred: {err}')
    except Exception as e:
        logger.warning(f'An unexpected error occurred: {e}')

    logger.warning('API request failed.')
    return ''
