# GPT-4V(ision) as a Generalist Evaluator

This evaluation suit implements GPT-4V(ision) as a generalist evaluator to assess multimodal large language models, covering the following tasks:
- **image-to-text** generation (captioning)
- **text-to-image** generation
- **video-to-text** generation (captioning)
- **text-to-video** generation

For each task, two evaluation methods are provided:
- **Single-answer grading**: Assess the quality of model generation from multiple dimensions and assign corresponding scores.
- **Pairwise comparison**: Compare the generative capabilities of two models and determine the winner (or tie)

## Input

An input file in JSON Lines (`.jsonl`) format is required to supply the entries for evaluation, with each line comprising a valid JSON object. Depending on the task, the required keys in these JSON objects will vary.

### Grading

For single-answer grading evaluation, each JSON object comprise at leat **two** keys corresponding to the required modalities.

Taking the image-to-text generation task as an example, each JSON object should include the keys `image` and `text`. The input file should appear as follows:

```JSON
{"image": "/path/to/image0", "text": "generated caption"}
{"image": "/path/to/image1", "text": "generated caption"}
...
```

Similarly,
- For text-to-image grading, `text` and `image` keys are required.
- For video-to-text grading, `video` and `text` keys are required.
- For text-to-video grading, `text` and `video` keys are required.

### Comparison

For pairwise comparison evaluation, each JSON object is expected to include **three** modality keys.

Also taking the image-to-text generation task as an example, each JSON object now should contain one `image` key along with two text keys, designated as `text_0` and `text_1`. The input file is expected to have the following structure:

```JSON
{"image": "/path/to/image0", "text_0": "model_A generated caption", "text_1": "model_B generated caption"}
{"image": "/path/to/image1", "text_0": "model_A generated caption", "text_1": "model_B generated caption"}
...
```

Similarly,
- For text-to-image comparision, `text` and `image_0, image_1` keys are required.
- For video-to-text comparision, `video` and `text_0, text_1` keys are required.
- For text-to-video comparision, `text` and `video_0, video_1` keys are required.

## Execution

For grading evaluation, run `python grade.py <task> <input.jsonl> <output.jsonl>`

For comparison evaluation, run `python compare.py <task> <input.jsonl> <output.jsonl>`

Avaliable `<task>` are:
- image_to_text
- text_to_image
- video_to_text
- text_to_video


You can always get usage and configurable arguments by running:
```shell
python grade.py <task> --help
python compare.py <task> --help
```

## Output

The evaluaiton results are also formatted in the JSON Lines structure, where each line directly corresponds to the respective input entry.

### Grading

For all tasks, the evaluation results will be stored under the `overall` key within each JSON object, featuring a `score` and, where applicable, a `rationale`.

For image and video generation, a more detailed set of criteria is also included, comprising `relevance`, `clarity`, `accuracy`, and, in the case of video, `coherence`. For each of these criteria, there will be an associated `score` and `rationale`.

### Comparison

For all comparison evaluation, two keys are provided: `winner` and `rationale`.