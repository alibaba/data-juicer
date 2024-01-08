import os

import wget
from loguru import logger

from .cache_utils import DATA_JUICER_MODELS_CACHE

# Default directory to store models
MODEL_PATH = DATA_JUICER_MODELS_CACHE

# Default backup cached models links for downloading
BACKUP_MODEL_LINKS = {
    # language identification model from fasttext
    'lid.176.bin':
    'https://dl.fbaipublicfiles.com/fasttext/supervised-models/',

    # tokenizer and language model for English from sentencepiece and KenLM
    '%s.sp.model':
    'https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/',
    '%s.arpa.bin':
    'https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/',

    # sentence split model from nltk punkt
    'punkt.%s.pickle':
    'https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/'
    'data_juicer/models/'
}

# Default cached models links for downloading
MODEL_LINKS = 'https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/' \
               'data_juicer/models/'

MODEL_ZOO = {}


def check_model(model_name, args=(), force=False):
    """
    Check whether a model exists in MODEL_PATH. If exists, return its full path
    Else, download it from cached models links.

    :param model_name: a specified model name
    :param args: optional extra args of model.
    :param force: Whether to download model forcefully or not, Sometimes
        the model file maybe incomplete for some reason, so need to
        download again forcefully.
    """
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    # check if the specified model exists. If it does not exist, download it
    true_model_name = model_name % args
    mdp = os.path.join(MODEL_PATH, true_model_name)
    if force:
        if os.path.exists(mdp):
            os.remove(mdp)
            logger.info(
                f'Model [{true_model_name}] invalid, force to downloading...')
        else:
            logger.info(
                f'Model [{true_model_name}] not found . Downloading...')

        try:
            model_link = os.path.join(MODEL_LINKS, true_model_name)
            wget.download(model_link, mdp, bar=None)
        except:  # noqa: E722
            try:
                backup_model_link = os.path.join(
                    BACKUP_MODEL_LINKS[model_name], true_model_name)
                wget.download(backup_model_link, mdp, bar=None)
            except:  # noqa: E722
                logger.error(
                    f'Downloading model [{true_model_name}] error. '
                    f'Please retry later or download it into {MODEL_PATH} '
                    f'manually from {model_link} or {backup_model_link} ')
                exit(1)
    return mdp


def prepare_fasttext_model(model_name):
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


def prepare_sentencepiece_model(model_name, lang):
    """
    Prepare and load a sentencepiece model.

    :param model_name: input model name in formatting syntax
    :param lang: language to render model name
    :return: model instance.
    """
    import sentencepiece
    logger.info('Loading sentencepiece model...')
    sentencepiece_model = sentencepiece.SentencePieceProcessor()
    try:
        sentencepiece_model.load(check_model(model_name, lang))
    except:  # noqa: E722
        sentencepiece_model.load(check_model(model_name, lang, force=True))
    return sentencepiece_model


def prepare_kenlm_model(model_name, lang):
    """
    Prepare and load a kenlm model.

    :param model_name: input model name in formatting syntax.
    :param lang: language to render model name
    :return: model instance.
    """
    import kenlm
    logger.info('Loading kenlm language model...')
    try:
        kenlm_model = kenlm.Model(check_model(model_name, lang))
    except:  # noqa: E722
        kenlm_model = kenlm.Model(check_model(model_name, lang, force=True))
    return kenlm_model


def prepare_nltk_model(model_name, lang):
    """
    Prepare and load a nltk punkt model.

    :param model_name: input model name in formatting syntax
    :param lang: language to render model name
    :return: model instance.
    """

    nltk_to_punkt = {
        'en': 'english',
        'fr': 'french',
        'pt': 'portuguese',
        'es': 'spanish'
    }
    assert lang in nltk_to_punkt.keys(
    ), 'lang must be one of the following: {}'.format(
        list(nltk_to_punkt.keys()))

    from nltk.data import load
    logger.info('Loading nltk punkt split model...')
    try:
        nltk_model = load(check_model(model_name, nltk_to_punkt[lang]))
    except:  # noqa: E722
        nltk_model = load(
            check_model(model_name, nltk_to_punkt[lang], force=True))
    return nltk_model


def prepare_huggingface_tokenizer(tokenizer_name):
    """
    Prepare and load a tokenizer from HuggingFace.

    :param tokenizer_name: input tokenizer name
    :return: a tokenizer instance.
    """
    from transformers import AutoTokenizer
    logger.info(f'Loading tokenizer {tokenizer_name} from HuggingFace...')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              trust_remote_code=True)
    return tokenizer


def prepare_huggingface_clip(clip_name):
    """
    Prepare and load a clip and processor from HuggingFace.

    :param clip_name: input clip name
    :return: a pair of clip instance and processor instance.
    """
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(clip_name)
    processor = CLIPProcessor.from_pretrained(clip_name)
    logger.info(f'Loading clip and processor {clip_name} from HuggingFace...')

    return model, processor


def prepare_huggingface_blip(
    blip_name,
    usage=None,
):
    """
    Prepare and load a blip and processor from HuggingFace.

    :param blip_name: input blip name in huggingface hup
    :param usage: a string indicating the type for processor and model wrapper
    :return: a pair of blip instance and processor instance.
    """
    model = None
    processor = None
    if usage is None:
        usage = 'image_text_retrieval'
    if 'blip2' in blip_name:
        if usage == 'conditional_generation':
            from transformers import (Blip2ForConditionalGeneration,
                                      Blip2Processor)
            model = Blip2ForConditionalGeneration.from_pretrained(blip_name)
            processor = Blip2Processor.from_pretrained(blip_name)
    elif 'blip' in blip_name:
        if usage == 'image_text_retrieval':
            from transformers import BlipForImageTextRetrieval, BlipProcessor
            model = BlipForImageTextRetrieval.from_pretrained(blip_name)
            processor = BlipProcessor.from_pretrained(blip_name)

    if model is None or processor is None:
        raise NotImplementedError('Unsupported model preparing behavior for '
                                  f'your given blip_name={blip_name} and '
                                  f'usage={usage}')

    logger.info(f'Loaded blip and processor {blip_name} from HuggingFace...')
    return model, processor


def prepare_huggingface_owlvit(owlvit_name):
    """
    Prepare and load an OwlViT and processor from HuggingFace.

    :param owlvit_name: input OwlViT name
    :return: a pair of OwlViT instance and processor instance.
    """
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    model = OwlViTForObjectDetection.from_pretrained(owlvit_name)
    processor = OwlViTProcessor.from_pretrained(owlvit_name)
    logger.info(f'Loading OwlViT and processor {owlvit_name} from '
                f'HuggingFace...')

    return (model, processor)


def prepare_diversity_model(model_name, lang):
    """
    Prepare diversity model for specific language.

    :param model_name: the model name to be loaded.
    :param lang: language of diversity model. Should be one of ["zh",
        "en"]
    :return: corresponding diversity model
    """
    import spacy
    assert lang in ['zh', 'en'], 'Diversity only support zh and en'
    model_name = model_name % lang
    logger.info(f'Loading spacy model [{model_name}]...')
    compressed_model = '%s.zip' % model_name

    # decompress the compressed model if it's not decompressed
    def decompress_model(compressed_model_path):
        decompressed_model_path = compressed_model_path.replace('.zip', '')
        if os.path.exists(decompressed_model_path) \
                and os.path.isdir(decompressed_model_path):
            return decompressed_model_path
        import zipfile
        with zipfile.ZipFile(compressed_model_path) as zf:
            zf.extractall(MODEL_PATH)
        return decompressed_model_path

    try:
        diversity_model = spacy.load(
            decompress_model(check_model(compressed_model)))
    except:  # noqa: E722
        diversity_model = spacy.load(
            decompress_model(check_model(compressed_model, force=True)))
    return diversity_model


def prepare_model(lang='en',
                  model_type='sentencepiece',
                  model_key=None,
                  usage=None):
    """
    Prepare and load a model or a tokenizer from MODEL_ZOO.

    :param lang: which lang model to load
    :param model_type: model or tokenizer type
    :param model_key: tokenizer name, only used when
        prepare HuggingFace tokenizer
    :param usage: detailed usage to indicate some specific type
        of the model or the tokenizer
    :return: a model or tokenizer instance
    """

    type_to_name = {
        'fasttext': ('lid.176.bin', prepare_fasttext_model),
        'sentencepiece': ('%s.sp.model', prepare_sentencepiece_model),
        'kenlm': ('%s.arpa.bin', prepare_kenlm_model),
        'nltk': ('punkt.%s.pickle', prepare_nltk_model),
        'huggingface': ('%s', prepare_huggingface_tokenizer),
        'hf_clip': ('%s', prepare_huggingface_clip),
        'hf_blip': ('%s', prepare_huggingface_blip),
        'hf_owlvit': ('%s', prepare_huggingface_owlvit),
        'spacy': ('%s_core_web_md-3.5.0', prepare_diversity_model),
    }
    assert model_type in type_to_name.keys(
    ), 'model_type must be one of the following: {}'.format(
        list(type_to_name.keys()))

    if model_key is None:
        model_key = model_type + '_' + lang
    if model_key not in MODEL_ZOO.keys():
        model_name, model_func = type_to_name[model_type]
        if model_type in ['fasttext']:
            MODEL_ZOO[model_key] = model_func(model_name)
        elif model_type in ['huggingface', 'hf_clip']:
            MODEL_ZOO[model_key] = model_func(model_key)
        elif model_type in ['hf_blip']:
            MODEL_ZOO[model_key] = model_func(model_key, usage)
        elif model_type == 'hf_owlvit':
            MODEL_ZOO[model_key] = model_func(model_key)
        else:
            MODEL_ZOO[model_key] = model_func(model_name, lang)
    return model_key


def get_model(model_key, lang='en', model_type='sentencepiece', usage=None):
    """
    Get a model or a tokenizer from MODEL_ZOO.

    :param model_key: name of the model or tokenzier
    """
    if model_key is None:
        return None
    if model_key not in MODEL_ZOO:
        prepare_model(lang=lang,
                      model_type=model_type,
                      model_key=model_key,
                      usage=usage)
    return MODEL_ZOO.get(model_key, None)
