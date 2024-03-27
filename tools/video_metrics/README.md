# Metrics for video generation

This folder contains some postprocess scripts for evaluation of generated videos.

## Usage

### Compute FVD/ISV

Use [calc_metrics_for_dataset.py](calc_metrics_for_dataset.py) to compute FVD/ISV for generated videos.

```shell
python tools/video_metrics/calc_metrics_for_dataset.py        \
    --real_data_path    <real_data_path>        \
    --fake_data_path    <fake_data_path>        \
    --metric            <metric>                \
    --detector_path     <detector_path>         \
    --result_path       <result_path>           \
    --num_runs          <num_runs>              \
    --height            <height>                \
    --width             <width>                 \
    --replace_cache     <replace_cache>         \
    --verbose           <verbose>               \

# get help
python tools/video_metrics/calc_metrics_for_dataset.py --help
```

- `real_data_path`: The path to ground truth dataset. Only support for `jsonl` format. The video paths are put in the list under `videos` keys. Required when computing FVD.
- `fake_data_path`: The path to generated dataset. Only support for `jsonl` format. The video paths are put in the list under `videos` keys.
- `metric`: The name of metric applied, currently support `fvd2048_16f`, `fvd2048_128f`, `fvd2048_128f_subsample8f`, `isv2048_ucf`.
    - `fvd2048_16f`: Compute FVD, sample 2048 times in dataset, 16 adjacent frames each time.
    - `fvd2048_128f`:  compute FVD, sample 2048 times in dataset, 128 adjacent frames each time.
    - `fvd2048_128f_subsample8f`: compute FVD, sample 2048 times in dataset, 16 frames each time, sample 1 frame every adjacent 8 frames.
    - `isv2048_ucf`: Compute Inception Scorecompute, sample 2048 times in dataset, 16 frames each time, split to 10 subset to compute ISs and return the mean and std of ISs.
- `detector_path`: Path to the corresponding detection model. Download the model from web if it is None.
- `result_path`: Path to JSON filename for saving results.
- `num_runs`: How many runs of the metric to average over.
- `height`: Sampled frames will be resized to this height.
- `width`: Sampled frames will be resized to this width.
- `replace_cache`: Whether to replace the dataset stats cache.
- `verbose`: Whether to log progress.
