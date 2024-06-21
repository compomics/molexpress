from __future__ import annotations

import keras

from molexpress import types
from molexpress.layers.base_layer import BaseLayer
from molexpress.ops import gnn_ops


class GINConv(BaseLayer):
    def __init__(
        self,
        units: int,
        activation: keras.layers.Activation = None,
        use_bias: bool = True,
        normalization: bool = True,
        skip_connection: bool = True,
        dropout_rate: float = 0,
        kernel_initializer: keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer = None,
        bias_regularizer: keras.regularizers.Regularizer = None,
        activity_regularizer: keras.regularizers.Regularizer = None,
        kernel_constraint: keras.constraints.Constraint = None,
        bias_constraint: keras.constraints.Constraint = None,
        **kwargs,
    ) -> None:
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.dropout_rate = dropout_rate
        self.skip_connection = skip_connection
        self.normalization = normalization

    def build(self, input_shape: dict[str, tuple[int, ...]]) -> None:
        node_state_shape = input_shape["node_state"]
        edge_state_shape = input_shape.get("edge_state")

        node_dim = node_state_shape[-1]
        if edge_state_shape is not None:
            edge_dim = edge_state_shape[-1]

        self._transform_node_state = node_dim != self.units

        if self._transform_node_state:
            self.special_node_kernel = self.add_kernel(
                name="special_node_kernel", shape=(node_dim, self.units)
            )
            node_dim = self.units

        self.node_kernel_1 = self.add_kernel(name="node_kernel_2", shape=(node_dim, self.units))
        self.node_kernel_2 = self.add_kernel(name="node_kernel_2", shape=(node_dim, self.units))

        if self.use_bias:
            self.node_bias_1 = self.add_bias(name="node_bias_1")
            self.node_bias_2 = self.add_bias(name="node_bias_2")

        self._transform_edge_state = edge_dim != node_dim

        if edge_state_shape is not None and self._transform_edge_state:
            self.special_edge_kernel = self.add_kernel(
                name="special_edge_kernel", shape=(edge_dim, node_dim)
            )

        self.epsilon = self.add_weight(name="epsilon", shape=(), initializer="zeros")

        if self.normalization:
            self.normalize = keras.layers.BatchNormalization()

        if self.dropout_rate:
            self.dropout = keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs: types.MolecularGraph) -> types.MolecularGraph:
        x = inputs.copy()

        node_state = x.pop("node_state")
        edge_src = keras.ops.cast(x["edge_src"], "int32")
        edge_dst = keras.ops.cast(x["edge_dst"], "int32")
        edge_state = x.get("edge_state")
        edge_weight = x.get("edge_weight")

        if edge_state is not None and self._transform_edge_state:
            edge_state = gnn_ops.transform(
                state=edge_state, kernel=self.special_edge_kernel, bias=None
            )

        if self._transform_node_state:
            node_state = gnn_ops.transform(
                state=node_state, kernel=self.special_node_kernel, bias=None
            )

        node_state_updated = gnn_ops.aggregate(
            node_state=node_state,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_state=edge_state,
            edge_weight=edge_weight,
        )

        node_state_updated += (1 + self.epsilon) * node_state

        node_state_updated = gnn_ops.transform(
            state=node_state_updated, kernel=self.node_kernel_1, bias=self.node_bias_1
        )

        if self.normalization:
            node_state_updated = self.normalize(node_state_updated)

        node_state_updated = self.activation(node_state_updated)

        node_state_updated = gnn_ops.transform(
            state=node_state_updated, kernel=self.node_kernel_2, bias=self.node_bias_2
        )

        if self.activation is not None:
            node_state_updated = self.activation(node_state_updated)

        if self.skip_connection:
            node_state_updated = node_state_updated + node_state

        if self.dropout_rate:
            node_state_updated = self.dropout(node_state_updated)

        return dict(node_state=node_state_updated, **x)

    def get_config(self) -> dict[str, types.Any]:
        config = super().get_config()
        config.update(
            {
                "normalization": self.normalization,
                "skip_connection": self.skip_connection,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
