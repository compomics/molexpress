import keras

from molexpress import types 


class BaseLayer(keras.layers.Layer):

    """Base layer."""

    def __init__(
        self, 
        units: int,
        activation: keras.layers.Activation = None,
        use_bias: bool = True,
        kernel_initializer: keras.initializers.Initializer = 'glorot_uniform',
        bias_initializer: keras.initializers.Initializer = 'zeros',
        kernel_regularizer: keras.regularizers.Regularizer = None,
        bias_regularizer: keras.regularizers.Regularizer = None,
        activity_regularizer: keras.regularizers.Regularizer = None,
        kernel_constraint: keras.constraints.Constraint = None,
        bias_constraint: keras.constraints.Constraint = None,
        **kwargs
    ) -> None:
        super().__init__(
            activity_regularizer=activity_regularizer, **kwargs
        )
        self.units = units
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

    def get_config(self) -> dict[str, types.Any]:
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(
                self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(
                self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(
                self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(
                self.bias_regularizer),
            'activity_regularizer': keras.regularizers.serialize(
                self.activity_regularizer),
            'kernel_constraint': keras.constraints.serialize(
                self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(
                self.bias_constraint),
        })
        return config

    def compute_output_shape(
        self, 
        input_shape: dict[str, tuple[int, ...]]
    ) -> dict[str, tuple[int, ...]]:
        output_shape = input_shape
        output_shape['node_state'] = (
            *input_shape['node_state'][:-1], self.units
        )
        if input_shape['edge_state'] is not None:
            output_shape['edge_state'] = (
                *input_shape['edge_state'][:-1], self.units
            )
        return output_shape

    def add_kernel(
        self,
        name: str, 
        shape: tuple[int, ...], 
        dtype: str = 'float32', 
        **kwargs
    ) -> types.Variable:
        return self.add_weight(
            name=name,
            shape=shape,
            dtype=dtype,
            **self._common_weight_kwargs('kernel'),
            **kwargs,
        )
    
    def add_bias(
        self, 
        name: str, 
        shape: tuple[int, ...] = None, 
        dtype: str = 'float32', 
        **kwargs
    ) -> types.Variable:
        return self.add_weight(
            name=name,
            shape=shape if shape is not None else (self.units,),
            dtype=dtype,
            **self._common_weight_kwargs('bias'),
            **kwargs,
        )

    def _common_weight_kwargs(
        self, 
        weight_type: str
    ) -> dict[str, types.Any]:
        initializer = getattr(self, f"{weight_type}_initializer", None)
        regularizer = getattr(self, f"{weight_type}_regularizer", None)
        regularizer = None if regularizer is None else regularizer.from_config(
            regularizer.get_config()
        )
        constraint = getattr(self, f"{weight_type}_constraint", None)
        constraint = None if constraint is None else constraint.from_config(
            constraint.get_config()
        )
        return {
            'initializer': initializer,
            'regularizer': regularizer,
            'constraint': constraint,
        }
    