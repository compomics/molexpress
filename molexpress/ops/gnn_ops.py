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
    if len(keras.ops.shape(kernel)) == 2: 
        # kernel.rank == state.rank == 2
        state_transformed = keras.ops.matmul(state, kernel)
    elif len(keras.ops.shape(kernel)) == len(keras.ops.shape(state)):
        # kernel.rank == state.rank == 3 
        state_transformed = keras.ops.einsum('ijh,jkh->ikh', state, kernel)
    else:
        # kernel.rank == 3 and state.rank == 2
        state_transformed = keras.ops.einsum('ij,jkh->ikh', state, kernel)
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

    expected_rank = len(keras.ops.shape(node_state))
    current_rank = len(keras.ops.shape(edge_src))
    for _ in range(expected_rank - current_rank):
        edge_src = keras.ops.expand_dims(edge_src, axis=-1)
        edge_dst = keras.ops.expand_dims(edge_dst, axis=-1)

    node_state_src = keras.ops.take_along_axis(node_state, edge_src, axis=0)
    
    if edge_weight is not None:
        node_state_src *= edge_weight

    if edge_state is not None:
        node_state_src += edge_state

    edge_dst = keras.ops.squeeze(edge_dst)

    node_state_updated = keras.ops.segment_sum(
        data=node_state_src, segment_ids=edge_dst, num_segments=num_nodes, sorted=False
    )
    return node_state_updated

def edge_softmax(score, edge_dst):
    num_segments = keras.ops.maximum(keras.ops.max(edge_dst) + 1, 1)
    score_max = keras.ops.segment_max(score, edge_dst, num_segments, sorted=False)
    score_max = gather(score_max, edge_dst)
    numerator = keras.ops.exp(score - score_max)
    denominator = keras.ops.segment_sum(numerator, edge_dst, num_segments, sorted=False)
    denominator = gather(denominator, edge_dst)
    return numerator / denominator

def gather(
    node_state: types.Array,
    edge: types.Array,      
) -> types.Array:
    expected_rank = len(keras.ops.shape(node_state))
    current_rank = len(keras.ops.shape(edge))
    for _ in range(expected_rank - current_rank):
        edge = keras.ops.expand_dims(edge, axis=-1)
    node_state_edge = keras.ops.take_along_axis(node_state, edge, axis=0)
    return node_state_edge

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
