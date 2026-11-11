import json

import pytest
import torch

from taxon import hardware as hw
from taxon.labeling.cli import Workspace
from taxon.reproduce import write_result
from taxon.rl.grpo import GrpoSettings, grpo_arguments
from taxon.rl.rewards import RewardWeights
from taxon.training import cli, measure
from taxon.training.measure import StepMeter, overnight_steps
from taxon.training.train import TrainSettings, run_device, training_arguments


@pytest.mark.parametrize(
    "cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, None)]
)
def test_the_accelerator_prefers_nvidia_then_apple(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert hw.accelerator() == expected


def test_the_hardware_record_names_the_machine_and_device():
    record = hw.hardware("cpu")
    assert {"os", "machine", "chip", "cpu_cores", "memory_gb", "device"} <= set(record)
    assert record["device"] == "cpu"
    assert hw.hardware("mps")["gpu"].endswith("GPU")


def test_a_real_run_trains_on_the_accelerator_and_a_tiny_one_on_cpu(monkeypatch):
    monkeypatch.setattr("taxon.training.train.accelerator", lambda: "mps")
    assert run_device(True) == "mps"
    assert run_device(False) == "cpu"


def test_every_reproduced_result_records_the_hardware(tmp_path):
    write_result("x", {"value": 1}, tmp_path)
    saved = json.loads((tmp_path / "x.json").read_text())
    assert saved["value"] == 1
    assert saved["hardware"]["device"] == "cpu"


def test_labeling_results_record_the_hardware(tmp_path):
    space = Workspace(tmp_path)
    space.write_result(tmp_path / "r.json", {"spent": 0})
    assert "hardware" in json.loads((tmp_path / "r.json").read_text())


def test_apple_s_gpu_trains_in_bf16_without_pinned_memory(tmp_path):
    arguments = training_arguments(TrainSettings(gpu=True), tmp_path, "mps")
    assert (arguments.use_cpu, arguments.bf16, arguments.dataloader_pin_memory) == (
        False,
        True,
        False,
    )


def test_grpo_uses_vllm_only_on_nvidia(tmp_path):
    weights = RewardWeights()
    assert grpo_arguments(GrpoSettings(gpu=True), tmp_path, weights, "cuda").use_vllm is True
    assert grpo_arguments(GrpoSettings(gpu=True), tmp_path, weights, "mps").use_vllm is False
    assert grpo_arguments(GrpoSettings(), tmp_path, weights, "cpu").use_vllm is False


def test_the_meter_leaves_the_warm_up_step_out_of_the_median():
    meter = StepMeter("cpu")
    meter.seconds = [30.0, 2.0, 4.0, 3.0]
    summary = meter.summary()
    assert summary["first_step_s"] == 30.0
    assert summary["median_step_s"] == 3.0
    assert summary["peak_allocated_gb"] is None


def test_the_meter_times_steps(monkeypatch):
    clock = iter([10.0, 12.5])
    monkeypatch.setattr(measure.time, "perf_counter", lambda: next(clock))
    meter = StepMeter("cpu")
    meter.on_step_begin(None, None, None)
    meter.on_step_end(None, None, None)
    assert meter.seconds == [2.5]


def test_an_overnight_run_keeps_a_tenth_spare():
    assert overnight_steps(36.0, hours=8.0) == 720
    assert overnight_steps(7.0, hours=1.0) == 462


def test_peak_memory_is_reported_in_gigabytes():
    assert 0 < measure.max_rss_gb() < 1024


def test_measuring_needs_a_gpu(monkeypatch, capsys, tmp_path, taxonomy):
    from taxon.products import write_jsonl

    write_jsonl(tmp_path / "products.jsonl", [])
    write_jsonl(tmp_path / "gold_queue.jsonl", [])
    monkeypatch.setattr(cli, "accelerator", lambda: None)
    code = cli.main(
        ["measure"],
        derived=tmp_path,
        processor_factory=lambda: None,
        load_taxonomy=lambda: taxonomy,
    )
    assert code == cli.EXIT_REFUSED
    assert "needs a GPU" in capsys.readouterr().out
