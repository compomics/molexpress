from __future__ import annotations

import keras

from molexpress import types


def transform(
    state,
    kernel: types.Variable,
    bias: types.Variable = None,
) -> types.Array:
    """Transforms node or edge states via learnable weights.

    Args:
        state:
            The current node or edge states to be updated.
        kernel:
            The learnable kernel.
        bias:
            The learnable bias.

    Returns:
        A transformed node state.
    """

    state_transformed = keras.ops.matmul(state, kernel)
    if bias is not None:
        state_transformed += bias
    return state_transformed


def aggregate(
    node_state: types.Array,
    edge_src: types.Array,
    edge_dst: types.Array,
    edge_state: types.Array = None,
    edge_weight: types.Array = None,
) -> types.Array:
    """Aggregates node states based on edges.

    Given node A with edges AB and AC, the information (states) of nodes
    B and C will be passed to node A.

    Args:
        node_state:
            The current state of the nodes.
        edge_src:
            The indices of the source nodes.
        edge_dst:
            The indices of the destination nodes.
        edge_state:
            Optional edge states.
        edge_weight:
            Optional edge weights.

    Returns:
        Updated node states.
    """
    num_nodes = keras.ops.shape(node_state)[0]

    # Instead of casting to int, throw an error if not int?
    edge_src = keras.ops.cast(edge_src, "int32")
    edge_dst = keras.ops.cast(edge_dst, "int32")

    expected_rank = 2
    current_rank = len(keras.ops.shape(edge_src))
    for _ in range(expected_rank - current_rank):
        edge_src = keras.ops.expand_dims(edge_src, axis=-1)
        edge_dst = keras.ops.expand_dims(edge_dst, axis=-1)

    node_state_src = keras.ops.take_along_axis(node_state, edge_src, axis=0)
    if edge_weight is not None:
        node_state_src *= edge_weight

    if edge_state is not None:
        node_state_src += edge_state

    edge_dst = keras.ops.squeeze(edge_dst, axis=-1)

    node_state_updated = keras.ops.segment_sum(
        data=node_state_src, segment_ids=edge_dst, num_segments=num_nodes, sorted=False
    )
    return node_state_updated


def segment_mean(
    data: types.Array,
    segment_ids: types.Array,
    num_segments: int = None,
    sorted: bool = False,
) -> types.Array:
    """Performs a mean of data based on segment indices.

    A permutation invariant reduction of the node states to obtain an
    encoding of the graph.

    Args:
        data:
            The data to be averaged.
        segment_ids:
            The indices.
        num_segment:
            Optional number of segments.
        sorted:
            Whether segment_ids are sorted.

    Args:
        New data that has been reduced.
    """
    x = keras.ops.segment_sum(
        data=data, segment_ids=segment_ids, num_segments=num_segments, sorted=sorted
    )
    return x / keras.ops.cast(keras.ops.bincount(segment_ids), x.dtype)[:, None]
