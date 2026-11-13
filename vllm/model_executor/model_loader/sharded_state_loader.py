# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import collections
import glob
import os
import string
import time
from collections.abc import Generator
from copy import copy
from typing import Any

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    runai_safetensors_weights_iterator,
)
from vllm.model_executor.utils import weights_already_processed
from vllm.transformers_utils.s3_utils import glob as s3_glob
from vllm.transformers_utils.utils import is_s3

logger = init_logger(__name__)


class ShardedStateLoader(BaseModelLoader):
    """Model loader that directly loads each worker's model state dict, which
    enables a fast load path for large tensor-parallel/pipeline-parallel models
    where each worker only needs to read its own shard rather than the entire
    checkpoint. See `examples/features/sharded_state/save_sharded_state_offline.py`
    for creating a sharded checkpoint.
    """

    TP_ONLY_PATTERN = "model-rank-{rank}-part-{part}.safetensors"
    PP_AND_TP_PATTERN = "model-pp-{pp_rank}-tp-{tp_rank}-part-{part}.safetensors"

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        extra_config = (
            {}
            if load_config.model_loader_extra_config is None
            else copy(load_config.model_loader_extra_config)
        )
        # A custom shard filename pattern can be supplied via
        # model_loader_extra_config; when omitted (None), the default pattern
        # is selected at render time based on the parallel configuration.
        self.pattern = extra_config.pop("pattern", None)
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: {extra_config.keys()}"
            )

    @staticmethod
    def _get_default_pattern() -> str:
        from vllm.distributed import get_pp_group

        return (
            ShardedStateLoader.PP_AND_TP_PATTERN
            if get_pp_group().world_size > 1
            else ShardedStateLoader.TP_ONLY_PATTERN
        )

    @staticmethod
    def _render_default_pattern(pattern: str | None = None, *, part: int | str) -> str:
        from vllm.distributed import (
            get_pipeline_model_parallel_rank,
            get_pp_group,
            get_tensor_model_parallel_rank,
            get_tp_group,
        )

        # _get_default_pattern is called here because _render_default_pattern
        # can be invoked from save_model without instantiating a
        # ShardedStateLoader instance.
        if pattern is None:
            pattern = ShardedStateLoader._get_default_pattern()

        # Parse the real format fields instead of matching placeholder
        # text: escaped braces like {{pp_rank}} render as literal text
        # and must not count as a rank field.
        fields = {
            field
            for _, field, _, _ in string.Formatter().parse(pattern)
            if field is not None
        }
        unsupported = fields - {"pp_rank", "tp_rank", "rank", "part"}
        if unsupported:
            raise ValueError(
                f"Pattern '{pattern}' uses unsupported placeholder(s) "
                f"{sorted(unsupported)}; supported fields are pp_rank, "
                "tp_rank, rank and part"
            )
        tp_rank = get_tensor_model_parallel_rank()
        pp_rank = get_pipeline_model_parallel_rank()
        pp_world_size = get_pp_group().world_size
        tp_world_size = get_tp_group().world_size
        # The pattern must carry a rank field for every active
        # parallelism dimension, otherwise shards from different ranks
        # collide on the same filename and silently overwrite each other.
        has_pp_field = "pp_rank" in fields
        has_tp_field = "tp_rank" in fields or "rank" in fields
        if pp_world_size > 1:
            if not has_pp_field:
                raise ValueError(
                    f"Pattern '{pattern}' does not embed the pipeline parallel "
                    "rank; with pipeline parallelism every stage would write the "
                    f"same file. Use '{ShardedStateLoader.PP_AND_TP_PATTERN}' "
                    "instead."
                )
            if tp_world_size > 1 and not has_tp_field:
                raise ValueError(
                    f"Pattern '{pattern}' does not embed the tensor parallel "
                    "rank; TP workers would overwrite each other's shards."
                )
            # Supply both tp aliases so mixed-alias patterns render.
            return pattern.format(
                pp_rank=pp_rank, tp_rank=tp_rank, rank=tp_rank, part=part
            )
        if has_pp_field:
            raise ValueError(
                f"Pattern '{pattern}' requires pipeline parallelism, but the "
                "pipeline parallel size is 1"
            )
        if tp_world_size > 1 and not has_tp_field:
            raise ValueError(
                f"Pattern '{pattern}' does not embed the tensor parallel rank; "
                "TP workers would overwrite each other's shards."
            )
        return pattern.format(tp_rank=tp_rank, rank=tp_rank, part=part)

    @staticmethod
    def _filter_subtensors(
        tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Filter out all tensors that share the same memory or a subset of the
        memory of another tensor.
        """
        same_storage_groups: dict[Any, list[tuple[str, torch.Tensor]]] = (
            collections.defaultdict(list)
        )
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))

        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        result: dict[str, torch.Tensor] = {}
        for group in same_storage_groups.values():
            for k, t in group:
                a, b = t.data_ptr(), get_end_ptr(t)
                for k2, t2 in group:
                    if not t2.is_contiguous():
                        continue
                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t.is_contiguous():
                        break  # t2 covers strictly more memory than t.
                    if k2 < k:
                        # Same tensors, keep the one with the smaller key.
                        break
                else:
                    result[k] = t
        return result

    def _prepare_weights(self, model_name_or_path: str, revision: str | None):
        if is_s3(model_name_or_path) or os.path.isdir(model_name_or_path):
            return model_name_or_path
        else:
            allow_patterns = ["*.safetensors"]
            return download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model, model_config.revision)

    @staticmethod
    def _all_quant_methods_support_pre_processed_weights(model: nn.Module) -> bool:
        return all(
            quant_method.supports_pre_processed_weights
            for module in model.modules()
            if isinstance(
                quant_method := getattr(module, "quant_method", None),
                QuantizeMethodBase,
            )
        )

    def _process_weights_after_loading(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        target_device: torch.device,
    ) -> None:
        # ShardedStateLoader.save_model exports model.state_dict() after
        # process_weights_after_loading has already run. Skip redundant weight
        # transforms (such as FP8 weight transposes) so loaded checkpoints are
        # not transformed a second time.
        if self._all_quant_methods_support_pre_processed_weights(model):
            with weights_already_processed():
                super()._process_weights_after_loading(
                    model, model_config, target_device
                )
        else:
            super()._process_weights_after_loading(model, model_config, target_device)

    @staticmethod
    def _set_post_processed_tensor(
        model: nn.Module,
        key: str,
        tensor: torch.Tensor,
        default_device: torch.device,
        *,
        preserve_param_object: bool = True,
    ) -> None:
        mod_path, _, leaf = key.rpartition(".")
        try:
            module = model.get_submodule(mod_path) if mod_path else model
        except AttributeError as exc:
            raise KeyError(key) from exc

        if leaf in module._parameters:
            existing = module._parameters[leaf]
            device = (
                existing.device
                if existing is not None and not existing.is_meta
                else default_device
            )
            loaded = tensor.to(device=device)
            if preserve_param_object and existing is not None and not existing.is_meta:
                existing.data = loaded
            else:
                module.register_parameter(
                    leaf, nn.Parameter(loaded, requires_grad=False)
                )
        elif leaf in module._buffers:
            existing = module._buffers[leaf]
            device = (
                existing.device
                if existing is not None and not existing.is_meta
                else default_device
            )
            module.register_buffer(leaf, tensor.to(device=device))
        else:
            module.register_parameter(
                leaf,
                nn.Parameter(tensor.to(device=default_device), requires_grad=False),
            )

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        local_model_path = self._prepare_weights(model_weights, model_config.revision)

        rendered_pattern = ShardedStateLoader._render_default_pattern(
            self.pattern, part="*"
        )
        filepaths = []
        if is_s3(local_model_path):
            file_pattern = f"*{rendered_pattern}"
            filepaths = s3_glob(path=local_model_path, allow_pattern=[file_pattern])
        else:
            filepaths = glob.glob(
                os.path.join(
                    local_model_path,
                    rendered_pattern,
                )
            )
        if not filepaths:
            # TODO: support un-sharded checkpoints too
            raise ValueError(
                f"Could not find checkpoint files "
                f"'{os.path.join(local_model_path, rendered_pattern)}', "
                f"only pre-sharded checkpoints are currently supported!"
            )
        state_dict = self._filter_subtensors(model.state_dict())
        default_device = next(
            (t.device for t in state_dict.values() if not t.is_meta),
            torch.device("cpu"),
        )
        counter_before_loading_weights = time.perf_counter()
        for key, tensor in self.iterate_over_files(filepaths):
            if key not in state_dict:
                # process_weights_after_loading before save_model may have added
                # persistent buffers/parameters or un-aliased tied tensors.
                self._set_post_processed_tensor(
                    model,
                    key,
                    tensor,
                    default_device,
                    preserve_param_object=False,
                )
                continue

            param = state_dict[key]
            if tensor.shape == param.shape and tensor.dtype == param.dtype:
                param.data.copy_(tensor)
            elif (
                tensor.dtype == param.dtype
                and tensor.ndim == param.ndim
                and all(
                    size <= param_size
                    for size, param_size in zip(tensor.shape, param.shape)
                )
            ):
                # If loading with LoRA enabled, additional padding may be added
                # to certain parameters. Load into a narrowed view of the data.
                param_data = param.data
                for dim, size in enumerate(tensor.shape):
                    if size < param.shape[dim]:
                        param_data = param_data.narrow(dim, 0, size)
                logger.warning(
                    "loading tensor of shape %s into parameter '%s' of shape %s",
                    tensor.shape,
                    key,
                    param.shape,
                )
                param_data.copy_(tensor)
            else:
                # ShardedStateLoader.save_model exports post-processed tensors whose
                # shape or dtype may differ from pre-processed create_model shapes
                # (e.g. transposed FP8 weights or packed quantized tensors).
                self._set_post_processed_tensor(model, key, tensor, default_device)
            state_dict.pop(key)
        counter_after_loading_weights = time.perf_counter()
        logger.info_once(
            "Loading weights took %.2f seconds",
            counter_after_loading_weights - counter_before_loading_weights,
        )
        if state_dict:
            raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")

    def iterate_over_files(
        self, paths
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        if self.load_config.load_format == "runai_streamer_sharded":
            yield from runai_safetensors_weights_iterator(paths, True)
        else:
            from safetensors.torch import safe_open

            for path in paths:
                with safe_open(path, framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(key)
                        yield key, tensor

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from safetensors.torch import save_file

        part_idx = 0
        total_size = 0
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
        state_dict_part: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            param_size = tensor.nelement() * tensor.element_size()
            if max_size is not None and total_size + param_size > max_size:
                filename = ShardedStateLoader._render_default_pattern(
                    pattern, part=part_idx
                )
                save_file(
                    state_dict_part,
                    os.path.join(path, filename),
                )
                part_idx += 1
                total_size = 0
                state_dict_part = {}
            state_dict_part[key] = tensor
            total_size += param_size
        if len(state_dict_part) > 0:
            filename = ShardedStateLoader._render_default_pattern(
                pattern, part=part_idx
            )
            save_file(
                state_dict_part,
                os.path.join(path, filename),
            )
