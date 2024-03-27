import fire
import json
import torch
from pathlib import Path

from tools.video_metrics.metrics import metric_main
from tools.video_metrics.util import EasyDict


def calc_metrics(
    real_data_path: str,
    fake_data_path: str,
    metric: str = "fvd2048_16f",
    detector_path: str = None,
    result_path: str = None,
    num_runs: int = 1,
    height: int = 240,
    width: int = 320,
    replace_cache: bool = False,
    verbose: bool = False,
):
    """
        Call the FID/FVD metrics for image/video generation

        :param real_data_path: The path to ground truth dataset. 
            Only support for `jsonl` format. The video paths are put 
            in the list under `videos` keys. Required when computing FVD.
        :param fake_data_path: The path to generated dataset. Only support 
            for `jsonl` format. The video paths are put in the list under 
            `videos` keys.
        :param metric: Metric to compute, can be one of 
            ['fvd2048_16f', 'fvd2048_128f', 'fvd2048_128f_subsample8f', 
            'isv2048_ucf']
        :param detector_path: Path to the corresponding detection model.
            Download the model from web if it is None.
        :param result_path: Path to JSON filename for saving results
        :param num_runs: How many runs of the metric to average over
        :param height: Sampled frames will be resized to this height
        :param width: Sampled frames will be resized to this width
        :param replace_cache: Whether to replace the dataset stats cache
        :param verbose: Whether to log progress
    """
    print(f"Metric: {metric}")

    if result_path is not None:
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)

    _metrics = list(metric_main._metric_dict.keys())
    if metric not in _metrics:
        raise ValueError(f'Metric [{metric}] is not supported. '
                         f'Can only be one of {_metrics}.')

    # assert torch.cuda.device_count() >= 1, 'must be executed in CUDA'

    # Initialize dataset options for real data.
    dataset_kwargs = EasyDict(
        dataset_path=real_data_path,
        seq_length=1,
        height=height,
        width=width,
    )

    # Initialize dataset options for fake data.
    gen_dataset_kwargs = EasyDict(
        dataset_path=fake_data_path,
        seq_length=1,
        height=height,
        width=width,
    )

    result_dict = metric_main.calc_metric(
        metric=metric,
        dataset_kwargs=dataset_kwargs,
        gen_dataset_kwargs=gen_dataset_kwargs,
        generator_as_dataset=True,
        replace_cache=replace_cache,
        verbose=verbose,
        num_runs=num_runs,
        detector_path=detector_path
    )


    json_line = json.dumps(
        dict(
            result_dict,
            real_data_path=real_data_path,
            fake_data_path=fake_data_path,
        )
    )
    print(json_line)

    if result_path is not None:
        with open(result_path, "at") as fp:
            fp.write(f"{json_line}\n")


if __name__ == "__main__":
    fire.Fire(calc_metrics)
