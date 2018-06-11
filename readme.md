# L_0 Regularizer and l0_dense:
## Introduction: 
- L_0 regularizer is intended to develop into a format that can be applicable to any architecture
- We provide an example of dense layer to demonstrate how to incoporate L_0 regularization into a model since L_0 regularization induces an architectural change (probabilistic mask creation) therefore cannot be applied directly as any other L^p regularization 
## Usage:
- the format of l0_dense is the same as tf.python.layers.layer.dense with an additional argument: is_training to specify wheter to use the training l0 mask or prediction l0 mask.
- the program is built as the structure: l0_computation -> l0_regularizer -> l0_dense
- example:
	- regularizer = tf.l0_regularizer(reg_const)
	- layer = l0_dense(input, output_size, is_training, ...)
	- get the regularization losses by tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) and add them into the main objective

## Note:
- The masks are created in l0_computation and are retrieved at the l0_dense.call method.

## To Solve/Do:
- Check. The training loss is too large after the insertion of L_0, not sure if it is normal
- Do. Develop L_0 Layer utility
- Why? failure to identify the variable created in "collection" argument in tf...dense to be initialized
- How? Better way to retrive the Tensors from l0_dense.call(...)
