from __future__ import annotations

from typing import Any, Sequence, Mapping

import torch
from torchvision import tv_tensors
from torch.utils._pytree import tree_flatten



class Embedding(tv_tensors.TVTensor):
    index: int

    @classmethod
    def _wrap(
            cls,
            tensor: torch.Tensor,
            index: int,
    ) -> Embedding:
        embedding = tensor.as_subclass(cls)
        embedding.index = index

        return embedding

    def __new__(
            cls,
            data: Any,
            index: torch.Tensor | int,
            *,
            dtype: torch.dtype | None = None,
            device: torch.device | str | int = None,
            requires_grad: bool | None = None,
    ):
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls._wrap(tensor, index=index)

    @classmethod
    def _wrap_output(
            cls,
            output: torch.Tensor,
            args: Sequence[Any] = (),
            kwargs: Mapping[str, Any] | None = None,
    ) -> Embedding:
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))  # type: ignore[operator]
        first_bbox_from_args = next(x for x in flat_params if isinstance(x, Embedding))
        index = first_bbox_from_args.index

        if isinstance(output, torch.Tensor) and not isinstance(output, Embedding):
            output = Embedding._wrap(output, index=index)
        elif isinstance(output, (tuple, list)):
            output = type(output)(
                Embedding._wrap(part, index=index) for part in output
            )
        return output

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr(index=self.index, tensor_contents=tensor_contents)
