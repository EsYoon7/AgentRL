from typing import List, Any, Optional

import torch
from torch._C._distributed_c10d import ProcessGroup
from torch.distributed import _rank_not_in_group, broadcast
from torch.distributed.distributed_c10d import _group_or_default_group, _canonicalize_group_rank, _warn_not_in_group, \
    _get_object_coll_device, _object_to_tensor, _tensor_to_object


# TODO: this is merged in pytorch 2.8.0, remove this when updated
def broadcast_object_list(
        object_list: List[Any],
        src: Optional[int] = None,
        group: Optional[ProcessGroup] = None,
        device: Optional[torch.device] = None,
        group_src: Optional[int] = None,
):
    """
    Broadcasts picklable objects in ``object_list`` to the whole group.

    Similar to :func:`broadcast`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be
    broadcasted.

    Args:
        object_list (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will
            be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
            Source rank is based on global process group (regardless of ``group`` argument)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, the objects are
            serialized and converted to tensors which are moved to the
            ``device`` before broadcasting. Default is ``None``.
        group_src (int): Source rank on ``group``.  Must not specify one of ``group_src``
            and ``src`` but not both.

    Returns:
        ``None``. If rank is part of the group, ``object_list`` will contain the
        broadcasted objects from ``src`` rank.

    .. note:: For NCCL-based process groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. note:: Note that this API differs slightly from the :func:`broadcast`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. warning::
        :func:`broadcast_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`broadcast_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`broadcast` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     objects = [None, None, None]
        >>> # Assumes backend is not NCCL
        >>> device = torch.device("cpu")
        >>> dist.broadcast_object_list(objects, src=0, device=device)
        >>> objects
        ['foo', 12, {1: 2}]
    """
    group = _group_or_default_group(group)
    if src is None and group_src is None:
        src = 0
    group_src = _canonicalize_group_rank(group, src, group_src, return_global=False)
    if _rank_not_in_group(group):
        _warn_not_in_group("broadcast_object_list")
        return

    # Current device selection.
    # To preserve backwards compatibility, ``device`` is default to ``None``
    # in which case we run current logic of device selection, i.e.
    # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In the
    # case it is not ``None`` we move the size and object tensors to be
    # broadcasted to this device.
    current_device = device or _get_object_coll_device(group)
    my_group_rank = group.rank()
    # Serialize object_list elements to tensors on src rank.
    if my_group_rank == group_src:
        tensor_list, size_list = zip(
            *[_object_to_tensor(obj, current_device, group) for obj in object_list]
        )
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.empty(
            len(object_list), dtype=torch.long, device=current_device
        )

    # Broadcast object sizes
    broadcast(object_sizes_tensor, group_src=group_src, group=group)

    # Concatenate and broadcast serialized object tensors
    # Note: torch.cat will do an extra memory copy to the current device, if the tensor_list
    # has only one element, we can skip the copy.
    if my_group_rank == group_src:
        if len(tensor_list) == 1:  # type: ignore[possibly-undefined]
            object_tensor = tensor_list[0]
        else:
            object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.empty(  # type: ignore[call-overload]
            torch.sum(object_sizes_tensor).item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device=current_device,
        )

    broadcast(object_tensor, group_src=group_src, group=group)
    # Deserialize objects using their stored sizes.
    offset = 0
    if my_group_rank != group_src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset: offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size, group)
