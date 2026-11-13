# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fnmatch
import multiprocessing as mp
import os
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from vllm import LLM, SamplingParams
from vllm.config import LoadConfig
from vllm.model_executor.model_loader import ShardedStateLoader
from vllm.transformers_utils.repo_utils import hf_api

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    ignore_eos=True,
)


def test_filter_subtensors():
    state_dict = {
        "a": torch.empty(2),
        "b": torch.empty((2, 4)),
        "c": torch.empty((2, 4, 8)),
    }
    state_dict.update(
        {
            "x": state_dict["b"],
            "y": state_dict["c"][1, 2, :],
            "z": state_dict["c"][1, :, 4],
        }
    )
    filtered_state_dict = ShardedStateLoader._filter_subtensors(state_dict)
    assert tuple(filtered_state_dict.keys()) == ("a", "b", "c")
    for key, tensor in filtered_state_dict.items():
        # NOTE: don't use `equal` here, as the tensor might contain NaNs
        assert tensor is state_dict[key]


def _mock_pp_group(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    *,
    pp_rank: int = 0,
    tp_rank: int = 0,
    tp_world_size: int = 1,
) -> None:
    """Stub the parallel-state accessors used by the sharded state loader."""
    monkeypatch.setattr(
        "vllm.distributed.get_pp_group",
        lambda: SimpleNamespace(world_size=world_size, rank_in_group=pp_rank),
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tp_group",
        lambda: SimpleNamespace(world_size=tp_world_size, rank_in_group=tp_rank),
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_rank",
        lambda: tp_rank,
    )
    monkeypatch.setattr(
        "vllm.distributed.get_pipeline_model_parallel_rank",
        lambda: pp_rank,
    )


@pytest.mark.parametrize(
    "model,model_weights,repo_id",
    [("org/model", None, "org/model"), ("local/dir", "org/weights", "org/weights")],
)
def test_load_weights_resolves_hf_repo_id(
    monkeypatch, tmp_path, model, model_weights, repo_id
):
    """HF repo IDs are downloaded with the requested revision before loading."""
    _mock_pp_group(monkeypatch, 1)
    weights = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    save_file({"weight": weights}, tmp_path / "model-rank-0-part-0.safetensors")
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.sharded_state_loader.download_weights_from_hf",
        download,
    )
    load_config = LoadConfig(
        load_format="sharded_state",
        download_dir=str(tmp_path / "cache"),
        ignore_patterns=["unused/*"],
    )
    model_config = SimpleNamespace(
        model=model, model_weights=model_weights, revision="checkpoint-revision"
    )
    linear = torch.nn.Linear(2, 2, bias=False)

    ShardedStateLoader(load_config).load_weights(linear, model_config)

    torch.testing.assert_close(linear.weight, weights)
    download.assert_called_once_with(
        repo_id,
        load_config.download_dir,
        ["*.safetensors"],
        "checkpoint-revision",
        ignore_patterns=load_config.ignore_patterns,
    )


def test_load_model_round_trips_post_processed_transposed_weights(
    monkeypatch, tmp_path
):
    """Saved sharded checkpoints contain post-processed weights (e.g.
    transposed FP8 weights or cast scales). Reloading must accept the saved
    shapes/dtypes and enter weights_already_processed() so
    process_weights_after_loading does not transform them a second time.
    """
    from unittest.mock import MagicMock

    import torch
    from safetensors.torch import save_file

    from vllm.config.load import LoadConfig
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.utils import is_weights_pre_processed

    _mock_pp_group(monkeypatch, 1)

    class _TransposeQuantMethod(QuantizeMethodBase):
        supports_pre_processed_weights = True

        def create_weights(self, layer, *args, **kwargs):
            pass

        def apply(self, layer, *args, **kwargs):
            pass

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            layer.kernel_ready = True
            if is_weights_pre_processed():
                return
            layer.weight = torch.nn.Parameter(
                layer.weight.t().contiguous(), requires_grad=False
            )
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.to(torch.float32).squeeze(-1),
                requires_grad=False,
            )

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Module()
            self.linear.quant_method = _TransposeQuantMethod()
            self.linear.kernel_ready = False
            self.linear.weight = torch.nn.Parameter(
                torch.zeros(4, 8, dtype=torch.float16), requires_grad=False
            )
            self.linear.weight_scale = torch.nn.Parameter(
                torch.ones(4, 1, dtype=torch.float16), requires_grad=False
            )

    src_model = _DummyModel()
    expected_weight = torch.arange(32, dtype=torch.float16).reshape(4, 8)
    expected_scale = torch.linspace(1.0, 4.0, 4, dtype=torch.float16).reshape(4, 1)
    src_model.linear.weight.data.copy_(expected_weight)
    src_model.linear.weight_scale.data.copy_(expected_scale)
    src_model.linear.quant_method.process_weights_after_loading(src_model.linear)
    assert src_model.linear.weight.shape == (8, 4)
    assert src_model.linear.weight_scale.shape == (4,)
    assert src_model.linear.weight_scale.dtype == torch.float32

    save_file(
        ShardedStateLoader._filter_subtensors(src_model.state_dict()),
        str(tmp_path / "model-rank-0-part-0.safetensors"),
    )

    reloaded_model = _DummyModel()
    model_config = MagicMock()
    model_config.model = str(tmp_path)
    model_config.model_weights = None
    model_config.revision = None
    model_config.dtype = torch.float16
    vllm_config = MagicMock()
    vllm_config.device_config.device = "cpu"
    vllm_config.load_config = LoadConfig(load_format="sharded_state")
    vllm_config.model_config = model_config

    loader = ShardedStateLoader(vllm_config.load_config)
    monkeypatch.setattr(loader, "create_model", lambda **kwargs: reloaded_model)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.base_loader.finalize_layerwise_processing",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.base_loader.log_online_quantization_time",
        lambda *args, **kwargs: None,
    )
    loaded = loader.load_model(vllm_config, model_config)

    assert loaded.linear.kernel_ready
    assert loaded.linear.weight.shape == (8, 4)
    assert torch.equal(loaded.linear.weight, expected_weight.t().contiguous())
    assert loaded.linear.weight_scale.shape == (4,)
    assert loaded.linear.weight_scale.dtype == torch.float32
    assert torch.equal(
        loaded.linear.weight_scale, expected_scale.to(torch.float32).squeeze(-1)
    )


def test_load_weights_narrows_lora_padded_tensor(monkeypatch, tmp_path):
    """Strictly smaller matching-dtype tensors still narrow for LoRA padding."""
    from unittest.mock import MagicMock

    import torch
    from safetensors.torch import save_file

    from vllm.config.load import LoadConfig

    _mock_pp_group(monkeypatch, 1)
    save_file(
        {"weight": torch.ones(3, 8, dtype=torch.float16)},
        str(tmp_path / "model-rank-0-part-0.safetensors"),
    )

    class _PaddedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.zeros(5, 8, dtype=torch.float16), requires_grad=False
            )

    model = _PaddedModel()
    model_config = MagicMock()
    model_config.model = str(tmp_path)
    model_config.model_weights = None
    model_config.revision = None
    ShardedStateLoader(LoadConfig(load_format="sharded_state")).load_weights(
        model, model_config
    )
    assert model.weight.shape == (5, 8)
    assert torch.equal(model.weight[:3], torch.ones(3, 8, dtype=torch.float16))
    assert torch.equal(model.weight[3:], torch.zeros(2, 8, dtype=torch.float16))


@pytest.mark.parametrize(
    "world_size,expected",
    [
        (1, ShardedStateLoader.TP_ONLY_PATTERN),
        (2, ShardedStateLoader.PP_AND_TP_PATTERN),
    ],
)
def test_get_default_pattern(monkeypatch, world_size, expected):
    """The default pattern is chosen from the pipeline parallel group size."""
    _mock_pp_group(monkeypatch, world_size)
    assert ShardedStateLoader._get_default_pattern() == expected


def test_render_default_pattern_default_tp(monkeypatch):
    """Without a custom pattern and PP=1, legacy TP-only names are rendered."""
    _mock_pp_group(monkeypatch, 1)
    assert (
        ShardedStateLoader._render_default_pattern(part=0)
        == "model-rank-0-part-0.safetensors"
    )


def test_render_default_pattern_default_pp(monkeypatch):
    """Without a custom pattern and PP>1, pp/tp-aware names are rendered."""
    _mock_pp_group(monkeypatch, 2)
    assert (
        ShardedStateLoader._render_default_pattern(part=0)
        == "model-pp-0-tp-0-part-0.safetensors"
    )


def test_render_default_pattern_custom_tp_only(monkeypatch):
    """A custom TP-only pattern is honored when pipeline parallelism is off."""
    _mock_pp_group(monkeypatch, 1)
    assert (
        ShardedStateLoader._render_default_pattern(
            ShardedStateLoader.TP_ONLY_PATTERN, part=0
        )
        == "model-rank-0-part-0.safetensors"
    )


def test_render_default_pattern_tp_only_with_pp_fails(monkeypatch):
    """TP-only filenames would collide across pipeline stages; fail fast."""
    _mock_pp_group(monkeypatch, 2)
    with pytest.raises(ValueError, match="does not embed the pipeline parallel"):
        ShardedStateLoader._render_default_pattern(
            ShardedStateLoader.TP_ONLY_PATTERN, part=0
        )


def test_render_default_pattern_pp_pattern_without_pp_fails(monkeypatch):
    """A pp/tp-aware pattern cannot be rendered when pipeline parallelism is off."""
    _mock_pp_group(monkeypatch, 1)
    with pytest.raises(ValueError, match="requires pipeline parallelism"):
        ShardedStateLoader._render_default_pattern(
            ShardedStateLoader.PP_AND_TP_PATTERN, part=0
        )


def test_render_default_pattern_custom_pp(monkeypatch):
    """A custom pp/tp-aware pattern is honored when pipeline parallelism is on."""
    _mock_pp_group(monkeypatch, 2)
    assert (
        ShardedStateLoader._render_default_pattern(
            ShardedStateLoader.PP_AND_TP_PATTERN, part=0
        )
        == "model-pp-0-tp-0-part-0.safetensors"
    )


@pytest.mark.parametrize(
    "pattern,match",
    [
        # tp field present but pp field missing: pipeline stages with the
        # same tp rank would write the same file.
        (
            "model-tp-{tp_rank}-part-{part}.safetensors",
            "does not embed the pipeline parallel",
        ),
        # pp field present but tp field missing: TP workers would overwrite
        # each other's shards.
        (
            "model-pp-{pp_rank}-part-{part}.safetensors",
            "does not embed the tensor parallel",
        ),
    ],
)
def test_render_default_pattern_incomplete_custom_patterns_fail(
    monkeypatch, pattern, match
):
    """A custom pattern must distinguish every active rank (pp and tp)."""
    _mock_pp_group(monkeypatch, 2, pp_rank=1, tp_rank=0, tp_world_size=2)
    with pytest.raises(ValueError, match=match):
        ShardedStateLoader._render_default_pattern(pattern, part=0)


def test_render_default_pattern_no_rank_field_with_tp_fails(monkeypatch):
    """Without any rank field, TP workers would collide even when PP is off."""
    _mock_pp_group(monkeypatch, 1, tp_world_size=2)
    with pytest.raises(ValueError, match="does not embed the tensor parallel"):
        ShardedStateLoader._render_default_pattern(
            "model-part-{part}.safetensors", part=0
        )


def test_render_default_pattern_custom_pp_tp1_allows_pp_only(monkeypatch):
    """With TP=1 a pp-only custom pattern is safe and must be accepted."""
    _mock_pp_group(monkeypatch, 2, pp_rank=1, tp_world_size=1)
    assert (
        ShardedStateLoader._render_default_pattern(
            "model-pp-{pp_rank}-part-{part}.safetensors", part=0
        )
        == "model-pp-1-part-0.safetensors"
    )


@pytest.mark.parametrize(
    "pattern,match",
    [
        # {{pp_rank}} renders as literal text: both stages would write the
        # same file, so it must not count as a pp field.
        (
            "model-{{pp_rank}}-tp-{tp_rank}-part-{part}.safetensors",
            "does not embed the pipeline parallel",
        ),
        # Same for an escaped tp field: TP workers would overwrite each
        # other's shards.
        (
            "model-pp-{pp_rank}-tp-{{tp_rank}}-part-{part}.safetensors",
            "does not embed the tensor parallel",
        ),
    ],
)
def test_render_default_pattern_escaped_field_not_rank_field(
    monkeypatch, pattern, match
):
    """Escaped braces are literal text and cannot pose as rank fields."""
    _mock_pp_group(monkeypatch, 2, pp_rank=1, tp_rank=0, tp_world_size=2)
    with pytest.raises(ValueError, match=match):
        ShardedStateLoader._render_default_pattern(pattern, part=0)


@pytest.mark.parametrize(
    "pattern,world_size,tp_world_size,pp_rank,tp_rank,expected",
    [
        # Mixed tp alias works when pipeline parallelism is on.
        (
            "model-pp-{pp_rank}-rank-{rank}-part-{part}.safetensors",
            2,
            2,
            1,
            0,
            "model-pp-1-rank-0-part-0.safetensors",
        ),
        # tp_rank alias works for TP-only checkpoints too.
        (
            "model-tp-{tp_rank}-part-{part}.safetensors",
            1,
            2,
            0,
            3,
            "model-tp-3-part-0.safetensors",
        ),
    ],
)
def test_render_default_pattern_tp_aliases_accepted(
    monkeypatch, pattern, world_size, tp_world_size, pp_rank, tp_rank, expected
):
    """Both tp aliases are supplied at render time, so either works."""
    _mock_pp_group(
        monkeypatch,
        world_size,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
    )
    assert ShardedStateLoader._render_default_pattern(pattern, part=0) == expected


def test_render_default_pattern_unsupported_field_fails(monkeypatch):
    """Unknown placeholders are rejected instead of KeyError-ing later."""
    _mock_pp_group(monkeypatch, 2, pp_rank=1, tp_world_size=2)
    with pytest.raises(ValueError, match="unsupported placeholder"):
        ShardedStateLoader._render_default_pattern(
            "model-{pp_rank}-{tp_rank}-{epoch}-part-{part}.safetensors", part=0
        )


@pytest.mark.parametrize(
    "pp_rank,tp_rank,expected",
    [
        (0, 0, "model-pp-0-tp-0-part-0.safetensors"),
        (1, 2, "model-pp-1-tp-2-part-0.safetensors"),
        (3, 5, "model-pp-3-tp-5-part-0.safetensors"),
    ],
)
def test_render_default_pattern_non_zero_ranks(monkeypatch, pp_rank, tp_rank, expected):
    """The rendered filename must embed the actual pp/tp ranks, not zeros."""
    _mock_pp_group(monkeypatch, 2, pp_rank=pp_rank, tp_rank=tp_rank)
    assert (
        ShardedStateLoader._render_default_pattern(
            ShardedStateLoader.PP_AND_TP_PATTERN, part=0
        )
        == expected
    )


def test_render_default_pattern_part_wildcard(monkeypatch):
    """The glob wildcard '*' passes through as the part token for file lookup."""
    _mock_pp_group(monkeypatch, 1, tp_rank=4)
    assert (
        ShardedStateLoader._render_default_pattern(part="*")
        == "model-rank-4-part-*.safetensors"
    )


def test_save_model_pp_aware_filenames(monkeypatch, tmp_path):
    """save_model writes pp/tp-aware filenames for pipeline-parallel models."""
    _mock_pp_group(monkeypatch, 2, pp_rank=1, tp_rank=2)

    saved: list[str] = []

    def _capture(tensors, path, **kwargs):
        saved.append(os.path.basename(path))

    monkeypatch.setattr("safetensors.torch.save_file", _capture)

    model = SimpleNamespace(
        state_dict=lambda: {"w": torch.zeros(4, dtype=torch.float32)}
    )
    ShardedStateLoader.save_model(model, str(tmp_path))
    assert saved == ["model-pp-1-tp-2-part-0.safetensors"]


def test_save_model_part_split(monkeypatch, tmp_path):
    """save_model rolls over to a new part when max_size is exceeded."""
    _mock_pp_group(monkeypatch, 1, tp_rank=0)

    saved: list[str] = []

    def _capture(tensors, path, **kwargs):
        saved.append(os.path.basename(path))

    monkeypatch.setattr("safetensors.torch.save_file", _capture)

    # Two float32[4] tensors (16 B each); max_size=20 fits one but not two.
    model = SimpleNamespace(
        state_dict=lambda: {
            "a": torch.zeros(4, dtype=torch.float32),
            "b": torch.zeros(4, dtype=torch.float32),
        }
    )
    ShardedStateLoader.save_model(model, str(tmp_path), max_size=20)
    assert saved == [
        "model-rank-0-part-0.safetensors",
        "model-rank-0-part-1.safetensors",
    ]


def test_save_model_rejects_bad_pattern_before_overwrite(monkeypatch, tmp_path):
    """An unsafe pattern is rejected before any shard file is written."""
    _mock_pp_group(monkeypatch, 2, pp_rank=1, tp_rank=0, tp_world_size=2)

    # With the escaped pp field this pattern renders both stages to the
    # same literal name; the validation must fire before any write.
    colliding = tmp_path / "model-{pp_rank}-tp-0-part-0.safetensors"
    colliding.write_bytes(b"existing shard")
    before = sorted(p.name for p in tmp_path.iterdir())

    model = SimpleNamespace(
        state_dict=lambda: {"w": torch.zeros(4, dtype=torch.float32)}
    )
    with pytest.raises(ValueError, match="does not embed the pipeline parallel"):
        ShardedStateLoader.save_model(
            model,
            str(tmp_path),
            pattern="model-{{pp_rank}}-tp-{tp_rank}-part-{part}.safetensors",
        )

    # The pre-existing shard is untouched and nothing new was written.
    assert colliding.read_bytes() == b"existing shard"
    assert sorted(p.name for p in tmp_path.iterdir()) == before


@pytest.fixture(scope="module")
def llama_3p2_1b_files():
    input_dir = hf_api().snapshot_download(
        "meta-llama/Llama-3.2-1B-Instruct", ignore_patterns=["*.bin*", "original/*"]
    )

    yield input_dir


def _run_writer(input_dir, output_dir, weights_patterns, **kwargs):
    llm_sharded_writer = LLM(model=input_dir, **kwargs)

    # Dump worker states to output directory
    llm_sharded_writer.llm_engine.engine_core.save_sharded_state(path=output_dir)

    # Copy metadata files to output directory
    for file in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, file)):
            shutil.copytree(
                os.path.join(input_dir, file), os.path.join(output_dir, file)
            )
        elif not any(fnmatch.fnmatch(file, ext) for ext in weights_patterns):
            shutil.copy(os.path.join(input_dir, file), output_dir)


def _run_generate(input_dir, queue: mp.Queue, **kwargs):
    llm = LLM(model=input_dir, **kwargs)
    gen = llm.generate(prompts, sampling_params)
    queue.put([g.outputs[0].__dict__ for g in gen])
    queue.close()
    queue.join_thread()


@pytest.mark.parametrize("enable_lora", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_sharded_state_loader(
    enable_lora, tp_size, num_gpus_available, llama_3p2_1b_files
):
    if num_gpus_available < tp_size:
        pytest.skip(f"Not enough GPUs for tensor parallelism {tp_size}")

    weights_patterns = ("*.safetensors",)
    gpu_memory_utilization = 0.8
    input_dir = llama_3p2_1b_files
    ctx = mp.get_context("spawn")

    # Keep batching deterministic: this test compares exact greedy outputs
    # across separate engine processes, whose schedulers may otherwise form
    # different batches depending on prompt-processing timing.
    platform_args = {"max_num_seqs": 1}

    # Run in separate processes for memory & CUDA isolation
    with TemporaryDirectory() as output_dir:
        p = ctx.Process(
            target=_run_writer,
            args=(input_dir, output_dir, weights_patterns),
            kwargs=dict(
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=True,
                **platform_args,
            ),
        )
        p.start()
        p.join()

        queue = ctx.Queue()

        p = ctx.Process(
            target=_run_generate,
            args=(input_dir, queue),
            kwargs=dict(
                enable_lora=enable_lora,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tp_size,
                **platform_args,
            ),
        )
        p.start()
        # Call queue.get() before p.join() to prevent deadlock:
        # If p.join() is called before queue.get() and the queue is full,
        # the child process may block while writing to the queue and never
        # terminate, causing the parent to wait indefinitely on p.join().
        # See: https://github.com/vllm-project/vllm/pull/22371#discussion_r2257773814
        out_before = queue.get()
        p.join()
        queue.close()
        queue.join_thread()

        queue = ctx.Queue()

        p = ctx.Process(
            target=_run_generate,
            args=(output_dir, queue),
            kwargs=dict(
                enable_lora=enable_lora,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tp_size,
                load_format="sharded_state",
                **platform_args,
            ),
        )
        p.start()
        # Call queue.get() before p.join() to prevent deadlock:
        # If p.join() is called before queue.get() and the queue is full,
        # the child process may block while writing to the queue and never
        # terminate, causing the parent to wait indefinitely on p.join().
        # See: https://github.com/vllm-project/vllm/pull/22371#discussion_r2257773814
        out_after = queue.get()
        p.join()
        queue.close()
        queue.join_thread()

        assert out_before == out_after


def _run_sharded_state_round_trip(
    input_dir: str,
    *,
    tp_size: int,
    pp_size: int,
    num_gpus_available: int,
) -> None:
    """Save a TP/PP sharded checkpoint, reload it, and compare outputs.

    Args:
        input_dir: directory containing the original (un-sharded) model.
        tp_size: tensor parallel size used for both save and load.
        pp_size: pipeline parallel size used for both save and load.
        num_gpus_available: number of GPUs on the current machine.

    """
    if num_gpus_available < tp_size * pp_size:
        pytest.skip(f"Not enough GPUs for TP={tp_size}, PP={pp_size}")

    weights_patterns = ("*.safetensors",)
    gpu_memory_utilization = 0.8
    ctx = mp.get_context("spawn")

    # Keep batching deterministic: this test compares exact greedy outputs
    # across separate engine processes, whose schedulers may otherwise form
    # different batches depending on prompt-processing timing.
    platform_args = {"max_num_seqs": 1}
    parallel_args = dict(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )

    with TemporaryDirectory() as output_dir:
        p = ctx.Process(
            target=_run_writer,
            args=(input_dir, output_dir, weights_patterns),
            kwargs=dict(
                **parallel_args,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=True,
                **platform_args,
            ),
        )
        p.start()
        p.join()
        assert p.exitcode == 0, f"writer process failed with exit code {p.exitcode}"

        if pp_size > 1:
            # PP checkpoints must use the pp/tp-aware naming scheme so that
            # different pipeline stages write distinct files.
            sharded_files = [
                f for f in os.listdir(output_dir) if f.endswith(".safetensors")
            ]
            assert sharded_files, "no sharded checkpoint files were written"
            assert any(
                fnmatch.fnmatch(f, "model-pp-*-tp-*-part-*.safetensors")
                for f in sharded_files
            ), f"expected PP-aware filenames, got {sharded_files}"

        queue = ctx.Queue()
        p = ctx.Process(
            target=_run_generate,
            args=(input_dir, queue),
            kwargs=dict(
                enable_lora=False,
                gpu_memory_utilization=gpu_memory_utilization,
                **parallel_args,
                **platform_args,
            ),
        )
        p.start()
        # Call queue.get() before p.join() to prevent deadlock:
        # If p.join() is called before queue.get() and the queue is full,
        # the child process may block while writing to the queue and never
        # terminate, causing the parent to wait indefinitely on p.join().
        # See: https://github.com/vllm-project/vllm/pull/22371#discussion_r2257773814
        out_before = queue.get()
        p.join()
        assert p.exitcode == 0, f"generate process failed with exit code {p.exitcode}"
        queue.close()
        queue.join_thread()

        queue = ctx.Queue()
        p = ctx.Process(
            target=_run_generate,
            args=(output_dir, queue),
            kwargs=dict(
                enable_lora=False,
                gpu_memory_utilization=gpu_memory_utilization,
                load_format="sharded_state",
                **parallel_args,
                **platform_args,
            ),
        )
        p.start()
        out_after = queue.get()
        p.join()
        assert p.exitcode == 0, f"generate process failed with exit code {p.exitcode}"
        queue.close()
        queue.join_thread()

        assert out_before == out_after


def test_sharded_state_loader_pp(num_gpus_available, llama_3p2_1b_files):
    """Round-trip sharded state save/load with pipeline parallelism (PP=2)."""
    _run_sharded_state_round_trip(
        llama_3p2_1b_files,
        tp_size=1,
        pp_size=2,
        num_gpus_available=num_gpus_available,
    )


def test_sharded_state_loader_pp_tp(num_gpus_available, llama_3p2_1b_files):
    """Round-trip sharded state save/load with pipeline + tensor parallelism
    (PP=2, TP=2)."""
    _run_sharded_state_round_trip(
        llama_3p2_1b_files,
        tp_size=2,
        pp_size=2,
        num_gpus_available=num_gpus_available,
    )
