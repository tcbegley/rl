# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Iterable, Union

from tensordict.nn import TensorDictSequential as _TensorDictSequential
from torch import nn

from torchrl.data import CompositeSpec
from torchrl.modules.tensordict_module.common import TensorDictModule


class TensorDictSequential(_TensorDictSequential, TensorDictModule):
    """A sequence of TensorDictModules.

    Similarly to :obj:`nn.Sequence` which passes a tensor through a chain of mappings that read and write a single tensor
    each, this module will read and write over a tensordict by querying each of the input modules.
    When calling a :obj:`TensorDictSequencial` instance with a functional module, it is expected that the parameter lists (and
    buffers) will be concatenated in a single list.

    Args:
         modules (iterable of TensorDictModules): ordered sequence of TensorDictModule instances to be run sequentially.
         partial_tolerant (bool, optional): if True, the input tensordict can miss some of the input keys.
            If so, the only module that will be executed are those who can be executed given the keys that
            are present.
            Also, if the input tensordict is a lazy stack of tensordicts AND if partial_tolerant is :obj:`True` AND if the
            stack does not have the required keys, then TensorDictSequential will scan through the sub-tensordicts
            looking for those that have the required keys, if any.

    TensorDictSequence supports functional, modular and vmap coding:
    Examples:
        >>> import functorch
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import TanhNormal, TensorDictSequential, NormalParamWrapper
        >>> from torchrl.modules.tensordict_module import ProbabilisticTensorDictModule
        >>> td = TensorDict({"input": torch.randn(3, 4)}, [3,])
        >>> spec1 = NdUnboundedContinuousTensorSpec(4)
        >>> net1 = NormalParamWrapper(torch.nn.Linear(4, 8))
        >>> fnet1, params1, buffers1 = functorch.make_functional_with_buffers(net1)
        >>> fmodule1 = TensorDictModule(fnet1, in_keys=["input"], out_keys=["loc", "scale"])
        >>> td_module1 = ProbabilisticTensorDictModule(
        ...    module=fmodule1,
        ...    spec=spec1,
        ...    dist_in_keys=["loc", "scale"],
        ...    sample_out_key=["hidden"],
        ...    distribution_class=TanhNormal,
        ...    return_log_prob=True,
        ...    )
        >>> spec2 = NdUnboundedContinuousTensorSpec(8)
        >>> module2 = torch.nn.Linear(4, 8)
        >>> fmodule2, params2, buffers2 = functorch.make_functional_with_buffers(module2)
        >>> td_module2 = TensorDictModule(
        ...    module=fmodule2,
        ...    spec=spec2,
        ...    in_keys=["hidden"],
        ...    out_keys=["output"],
        ...    )
        >>> td_module = TensorDictSequential(td_module1, td_module2)
        >>> params = params1 + params2
        >>> buffers = buffers1 + buffers2
        >>> _ = td_module(td, params=params, buffers=buffers)
        >>> print(td)
        TensorDict(
            fields={
                input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 8]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)

        >>> # The module spec aggregates all the input specs:
        >>> print(td_module.spec)
        CompositeSpec(
            hidden: NdUnboundedContinuousTensorSpec(
                 shape=torch.Size([4]),space=None,device=cpu,dtype=torch.float32,domain=continuous),
            output: NdUnboundedContinuousTensorSpec(
                 shape=torch.Size([8]),space=None,device=cpu,dtype=torch.float32,domain=continuous))

    In the vmap case:
        >>> params = tuple(p.expand(4, *p.shape).contiguous().normal_() for p in params)
        >>> buffers = tuple(b.expand(4, *b.shape).contiguous().normal_() for p in buffers)
        >>> td_vmap = td_module(td, params=params, buffers=buffers, vmap=True)
        >>> print(td_vmap)
        TensorDict(
            fields={
                input: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                scale: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([4, 3, 1]), dtype=torch.float32),
                output: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32)},
            batch_size=torch.Size([4, 3]),
            device=cpu,
            is_shared=False)


    """

    module: nn.ModuleList

    def __init__(
        self,
        *modules: TensorDictModule,
        partial_tolerant: bool = False,
    ):
        self.partial_tolerant = partial_tolerant

        in_keys, out_keys = self._compute_in_and_out_keys(modules)

        spec = CompositeSpec()
        for module in modules:
            if isinstance(module, TensorDictModule) or hasattr(module, "spec"):
                spec.update(module.spec)
            else:
                spec.update(CompositeSpec({key: None for key in module.out_keys}))

        super(_TensorDictSequential, self).__init__(
            spec=spec,
            module=nn.ModuleList(list(modules)),
            in_keys=in_keys,
            out_keys=out_keys,
        )

    def select_subsequence(
        self, in_keys: Iterable[str] = None, out_keys: Iterable[str] = None
    ) -> "TensorDictSequential":
        """Returns a new TensorDictSequential with only the modules that are necessary to compute the given output keys with the given input keys.

        Args:
            in_keys: input keys of the subsequence we want to select
            out_keys: output keys of the subsequence we want to select

        Returns:
            A new TensorDictSequential with only the modules that are necessary acording to the given input and output keys.
        """
        td_sequential = super().select_subsequence(in_keys=in_keys, out_keys=out_keys)
        return TensorDictSequential(*td_sequential.module)

    def __getitem__(self, index: Union[int, slice]) -> TensorDictModule:
        if isinstance(index, int):
            return self.module.__getitem__(index)
        else:
            return TensorDictSequential(*self.module.__getitem__(index))
