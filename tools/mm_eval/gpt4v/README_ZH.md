# GPT-4V(ision) as a Generalist Evaluator

这个评估套件实现了 GPT-4V 作为一个通才评估器，用于评价多模态大模型在以下任务上的表现：
- 图像到文本生成（图像描述）
- 文本到图像生成
- 视频到文本生成（视频描述）
- 文本到视频生成

对于每项任务，都提供了两种评估方法：
- 单一答案评分：从多个维度评估模型生成的质量，并相应打分。
- 成对比较：比较两个模型的生成能力，并确定胜者（或平局）。

## 输入

需要一个 JSON Lines（`.jsonl`）格式的输入文件来提供评估的条目，每一行包含一个有效的 JSON 对象。根据任务的不同，这些 JSON 对象中需要包含的键也会有所不同。

### 打分评测

打分评测要求每个 JSON 对象包含两个与任务模态相关的键。

以图像到文本（image-to-text）的生成任务为例，每个 JSON 对象应该包括 `image` 和 `text` 键。样例输入文件格式如下：

```JSON
{"image": "/path/to/image0", "text": "generated caption"}
{"image": "/path/to/image1", "text": "generated caption"}
...
```

类似地，
- 对于文本到图像（text-to-image）评分，需要 `text` 和 `image` 键。
- 对于视频到文本（video-to-text）评分，需要 `video` 和 `text` 键。
- 对于文本到视频（text-to-video）评分，需要 `text` 和 `video` 键。


### 比较评测

对于成对比较评测，每个 JSON 对象需要包含三个模态相关的键。

同样以图像到文本生成（image-to-text）为例，每个 JSON 对象会包含一个 `image` 键和两个文本键，必须分别命名为 `text_0` 和 `text_1`。输入格式如下：

```JSON
{"image": "/path/to/image0", "text_0": "model_A generated caption", "text_1": "model_B generated caption"}
{"image": "/path/to/image1", "text_0": "model_A generated caption", "text_1": "model_B generated caption"}
...
```

类似地，
- 对于文本到图像（text-to-image）的比较评测，需要 `text` 和 `image_0, image_1` 键。
- 对于视频到文本（vide-to-text）的比较评测，需要 `video` 和 `text_0, text_1`键。
- 对于文本到视频（text-to-vide）的比较评测，需要 `text` 和 `video_0, video_1` 键。

## 执行评测

对于打分评测，运行 `python grade.py <task> <input.jsonl> <output.jsonl>`

对于比较评测，运行 `python compare.py <task> <input.jsonl> <output.jsonl>`

可选的 `<task>` 包括:
- image_to_text
- text_to_image
- video_to_text
- text_to_video

执行以下命令获取各任务的使用说明和可配置参数：
```shell
python grade.py <task> --help
python compare.py <task> --help
```

## 输出

评估结果也以 JSON Lines 格式保存，其中每行对应于相应的输入条目。

### 打分评测

对于所有任务，评测结果将会保存在 JSON 对象的 `overall` 键中，包含具体的 `score`（分数）以及可能的 `rationale`（理由）。

对于图像和视频生成任务，会额外提供更详细的多维度评测结果，包括 `relevance`（相关性）、`clarity`（清晰度）、`accuracy`（准确性），以及视频的 `coherence`（连贯性）。对于每个维度，都有对应的 `score`和 `rationale`。

### 比较评测

对于所有的比较评测，都会在每个 JSON 的 `overall` 键中提供 `winner` 和 `rationale`。
