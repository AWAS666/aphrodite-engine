import contextlib
from typing import Type
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from aphrodite.common.config import ModelConfig
from aphrodite.modeling.models import LlamaForCausalLM, GPTJForCausalLM, GPTNeoXForCausalLM
from aphrodite.modeling.hf_downloader import initialize_dummy_weights
from aphrodite.modeling.quantize import TpGPTQQuantizer, patch_tp_linear_layer

_MODEL_REGISTRY = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,
    "GPTJForCausalLM": GPTJForCausalLM,
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
}

@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def get_model(model_config: ModelConfig, max_tokens: int) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)
    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        if model_config.quantize_config is not None:
            quantizer = TpGPTQQuantizer.from_dict(
                model_config.quantize_config.to_dict())
            patch_tp_linear_layer()
            model = model_class(model_config.hf_config)
            model = quantizer.convert_model(model)
        else:
            model = model_class(model_config.hf_config)
            model.quantize_config = model_config.quantize_config

        if model_config.load_format == "dummy":
            model = model.cuda()
            # NOTE: For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            model.load_weights(model_config.model, model_config.download_dir,
                               model_config.load_format)
            model = model.cuda()
        if model_config.quantize_config is not None:
            model = quantizer.post_init_model(model, max_tokens)
    return model.eval()