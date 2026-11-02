"""Minimal offline port of SONIC's token/feature assembly contract.

Source: NVlabs/GR00T-WholeBodyControl, gear_sonic/trl/modules/
universal_token_modules.py (assemble_all_tokens and decode), and
base_module.py (temporal input flatten / output reshape).
No policy construction, re-encoding, quantization, or simulator imports.
"""
from collections.abc import Mapping
import torch


def restore_internal_tokens(latent, *, num_tokens, token_dim):
    """Invert exported flatten(start_dim=-2), preserving batch and horizon.

    flat[b,h,n*token_dim+d] == internal[b,h,n,d].
    Noncontiguous tensors are supported; no axis transposition is performed.
    Inputs are raw post-quantization tokens, after action denormalization.
    """
    if not torch.is_tensor(latent) or latent.ndim != 3:
        raise ValueError('Expected a tensor [B,H,token_total_dim]')
    if num_tokens <= 0 or token_dim <= 0 or latent.shape[-1] != num_tokens * token_dim:
        raise ValueError('latent width does not equal num_tokens * token_dim')
    if not latent.is_floating_point() or not torch.isfinite(latent).all():
        raise ValueError('Expected finite floating-point SONIC tokens')
    return latent.reshape(*latent.shape[:2], num_tokens, token_dim)


def decode_feature_dict(decoder, decode_input_dict, input_features, output_feature_dims):
    """Port of UniversalTokenModule.decode's configured-input/output path.

    output_feature_dims must be ordered exactly like decoder_config.outputs.
    Optional conditioning/mask features are intentionally unsupported by this
    g1_kin-only port, whose matching configuration declares inputs=['token'].
    """
    if not input_features or not isinstance(output_feature_dims, Mapping) or not output_feature_dims:
        raise ValueError('Decoder input and output schemas must be nonempty')
    decoder_input = torch.cat([decode_input_dict[key] for key in input_features], dim=-1)
    output = decoder(decoder_input)
    output_dict = {}
    index = 0
    for key, dim in output_feature_dims.items():
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError('Output feature dimensions must be positive integers')
        output_dict[key] = output[..., index:index + dim]
        index += dim
    if index != output.shape[-1]:
        raise ValueError(f'Decoder output width {output.shape[-1]} differs from feature schema {index}')
    return output_dict
