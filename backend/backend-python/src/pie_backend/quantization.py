import torch

import torchao
from torchao.dtypes import AffineQuantizedTensor
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig,
    Float8WeightOnlyConfig,
)

from torchao.quantization import (
    Int4PreshuffledTensor,
    Int4Tensor,
    Int4PlainInt32Tensor,
    Int4MarlinSparseTensor,
    Int4TilePackedTo4dTensor,
    Float8Tensor,
    PerRow,
)
from torchao.quantization.quantize_.workflows import (
    Int4PackingFormat,
    Int4ChooseQParamsAlgorithm,
)



def quantize(
    x: torch.Tensor,
    config: Int4WeightOnlyConfig | Int8WeightOnlyConfig | Float8WeightOnlyConfig,
) -> torch.Tensor:
    if isinstance(config, Int4WeightOnlyConfig):
        return quantize_int4(x, config)
    elif isinstance(config, Int8WeightOnlyConfig):
        return quantize_int8(x, config)
    elif isinstance(config, Float8WeightOnlyConfig):
        return quantize_float8(x, config)
    else:
        raise TypeError(
            f"Unsupported weight-only quantization config type: {type(config)}"
        )


def quantize_int4(
    x: torch.Tensor,
    config: Int4WeightOnlyConfig,
) -> torch.Tensor:
    if x.shape[-1] % config.group_size != 0:
        raise ValueError(
            f"Cannot int4-quantize weight with shape {x.shape}: "
            f"last dim must be divisible by group_size={config.group_size}"
        )

    block_size = [1] * (x.ndim - 1) + [config.group_size]

    if (
        config.int4_choose_qparams_algorithm is Int4ChooseQParamsAlgorithm.HQQ
        and config.int4_packing_format
        not in {Int4PackingFormat.TILE_PACKED_TO_4D, Int4PackingFormat.OPAQUE}
    ):
        raise ValueError(
            "Int4ChooseQParamsAlgorithm.HQQ is only supported by "
            "Int4PackingFormat.TILE_PACKED_TO_4D and Int4PackingFormat.OPAQUE"
        )

    match config.int4_packing_format:
        case Int4PackingFormat.PRESHUFFLED:
            return Int4PreshuffledTensor.from_hp(
                x,
                block_size,
                activation_dtype=torch.bfloat16,
            )
        case Int4PackingFormat.PLAIN:
            return Int4Tensor.from_hp(x, block_size)
        case Int4PackingFormat.PLAIN_INT32:
            return Int4PlainInt32Tensor.from_hp(x, block_size)
        case Int4PackingFormat.MARLIN_SPARSE:
            return Int4MarlinSparseTensor.from_hp(x, block_size)
        case Int4PackingFormat.TILE_PACKED_TO_4D:
            return Int4TilePackedTo4dTensor.from_hp(
                x,
                block_size,
                int4_choose_qparams_algorithm=config.int4_choose_qparams_algorithm,
            )
        case _:
            raise ValueError(
                f"Unsupported int4 packing format: {config.int4_packing_format}"
            )


def quantize_int8(
    x: torch.Tensor,
    config: Int8WeightOnlyConfig,
) -> torch.Tensor:
    if config.group_size is None:
        group_size = x.shape[-1]
    else:
        group_size = config.group_size

    block_size = (1,) * (x.dim() - 1) + (group_size,)

    return AffineQuantizedTensor.from_hp_to_intx(
        x,
        torchao.quantization.MappingType.SYMMETRIC,
        block_size,
        torch.int8,
        eps=torch.finfo(torch.float32).eps,
        zero_point_dtype=torch.int64,
    )


def quantize_float8(x: torch.Tensor, config: Float8WeightOnlyConfig) -> torch.Tensor:
    return Float8Tensor.from_hp(
        x,
        float8_dtype=x.dtype,
        granularity=PerRow(),
    )
