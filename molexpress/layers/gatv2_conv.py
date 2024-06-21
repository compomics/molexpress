from __future__ import annotations

import keras

from molexpress import types
from molexpress.layers.base_layer import BaseLayer
from molexpress.ops import gnn_ops


class GATv2Conv(BaseLayer):
    def __init__(
        self,
        units: int,
        heads: int,
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
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.skip_connection = skip_connection
        self.normalization = normalization
        self.attention_activation = keras.activations.get('leaky_relu')
        if self.units % self.heads != 0:
            raise ValueError(
                f"units ({self.units}) needs to be divisble by heads {self.heads}")
        else:
            self.units_per_head = self.units // self.heads 

    def build(self, input_shape: dict[str, tuple[int, ...]]) -> None:
        
        node_state_shape = input_shape["node_state"]
        edge_state_shape = input_shape.get("edge_state")

        node_dim = node_state_shape[-1]

        if edge_state_shape is not None:
            edge_dim = edge_state_shape[-1]
        else:
            edge_dim = 0

        self._transform_residual = node_dim != self.units
        if self._transform_residual:
            self.residual_node_kernel = self.add_kernel(
                name="residual_node_kernel", shape=(node_dim, self.units)
            )

        self.kernel = self.add_kernel(
            name="kernel", shape=(node_dim * 2 + edge_dim, self.units_per_head, self.heads))
        if self.use_bias:
            self.bias = self.add_bias(
                name="bias", shape=(self.units_per_head, self.heads))
            
        self.attention_kernel = self.add_kernel(
            name="attention_kernel", shape=(self.units_per_head, 1, self.heads))
        if self.use_bias:
            self.attention_bias = self.add_bias(
                name="attention_bias", shape=(1, self.heads))
            

        self.node_kernel = self.add_kernel(
            name="node_kernel", shape=(node_dim, self.units_per_head, self.heads))
        if self.use_bias:
            self.node_bias = self.add_bias(
                name="node_bias", shape=(self.units_per_head, self.heads))

            
        if edge_state_shape is not None:
            self.edge_kernel = self.add_kernel(
                name="edge_kernel", shape=(
                    self.units_per_head, self.units_per_head, self.heads)
            )
            if self.use_bias:
                self.edge_bias = self.add_bias(
                    name="edge_bias", shape=(self.units_per_head, self.heads))

        if self.normalization:
            self.normalize = keras.layers.BatchNormalization()

        if self.dropout_rate:
            self.dropout = keras.layers.Dropout(self.dropout_rate)


    def call(self, inputs: types.MolecularGraph) -> types.MolecularGraph:
        x = inputs.copy()

        node_state = x.pop("node_state")
        edge_src = keras.ops.cast(x["edge_src"], "int32")
        edge_dst = keras.ops.cast(x["edge_dst"], "int32")
        edge_state = x.pop("edge_state", None)
        edge_weight = x.get("edge_weight")

        if edge_state is None:
            attention_feature = keras.ops.concatenate([
                gnn_ops.gather(node_state, edge_src),
                gnn_ops.gather(node_state, edge_dst),
            ], axis=-1)
        else:
            attention_feature = keras.ops.concatenate([
                gnn_ops.gather(node_state, edge_src),
                gnn_ops.gather(node_state, edge_dst),
                edge_state
            ], axis=-1)

        
        node_state_updated = gnn_ops.transform(
            node_state, self.node_kernel, self.node_bias)

        attention_feature = gnn_ops.transform(
            attention_feature, self.kernel, self.bias)

        if edge_state is not None:
            edge_state_updated = gnn_ops.transform(
                attention_feature, self.edge_kernel, self.edge_bias)
            edge_state_updated = keras.ops.reshape(
                edge_state_updated, (-1, self.units))
            
 
        attention_feature = self.attention_activation(attention_feature)
        attention_feature = gnn_ops.transform(
            attention_feature, self.attention_kernel, self.attention_bias
        )
        attention_score = gnn_ops.edge_softmax(attention_feature, edge_dst)

        node_state_updated = gnn_ops.aggregate(
            node_state=node_state_updated,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_state=None,
            edge_weight=attention_score,
        )

        node_state_updated = keras.ops.reshape(
            node_state_updated, (-1, self.units)
        )
        if self.activation is not None:
            node_state_updated = self.activation(node_state_updated)

        if self.skip_connection:
            if self._transform_residual:
                node_state = gnn_ops.transform(
                    node_state, self.residual_node_kernel)
            node_state_updated = node_state_updated + node_state

        if self.dropout_rate:
            node_state_updated = self.dropout(node_state_updated)

        return dict(
            node_state=node_state_updated, 
            edge_state=edge_state_updated,
            **x)

    def get_config(self) -> dict[str, types.Any]:
        config = super().get_config()
        config.update(
            {
                "heads": self.heads,
                "normalization": self.normalization,
                "skip_connection": self.skip_connection,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
