# L0 Regularizer and l0_dense:
## Introduction: 
-1 l0_regularizer is intended to develop into a format that can be applicable to any architecture
-2 We provide an example of dense layer to demonstrate how to incoporate l0 regularization into a model since l0 regularization induces an architectural change (probabilistic mask creation) therefore cannot be applied directly as any other L^p regularization 
## Usage:
-1 the format of l0_dense is the same as tf.python.layers.layer.dense with an additional argument: is_training to specify wheter to use the training l0 mask or prediction l0 mask.
-2 the program is built as the structure: l0_computation -> l0_regularizer -> l0_dense
-3 example:
	- regularizer = tf.l0_regularizer(reg_const)
	- layer = l0_dense(input, output_size, is_training, ...)
	- get the regularization losses by tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) and add them into the main objective

## Note:
-1 The masks are created in l0_computation and are retrieved at the l0_dense.call method.
