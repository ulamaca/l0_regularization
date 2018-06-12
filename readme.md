# L_0 Regularizer, l0_layer and l0_dense:
## Introduction: 
- L_0 regularizer is intended to develop into a format that can be applicable to any architecture
- We provide an example of dense layer to demonstrate how to incoporate $L_0$ regularization into a model since L_0 regularization induces an architectural change (probabilistic mask creation) therefore cannot be applied directly as any other L^p regularization 

## Usage:
- the program is built as the structure: l0_computation -> l0_regularizer -> l0_dense/l0_layer and l0_computation is not recommended for user for common applications.
- l0_computation defines the computational mechanisms of l0 regularization on a tensors during which a masked conditional tensor is created and will replace the original tensor for model building. [Louizos et al. 2017](https://arxiv.org/abs/1712.01312). 
- l0_regularizer:
	- adpated from the structure of tf.contrib.layers.l2_regularizer
	- inputs: scale, scope=None
	- outputs: the (scaled) regularization loss
- l0_dense
	- adapted from the structure of tf.layers.dense
	- almost the same as the original dense function.
	- Except an additional argument is_training is added for tagging training/prediction status of the weights/bias
	- the losses created after applying l0_regularizer can be obtained by calling __tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)__
- l0_layer
	- inherited from the base.Layer class and structured much like tf.layers.dropout
	- the assumed input format is [?, D] or [D, ?]
	- inputs: input_tensor, reg_const, training, seed, name
	- outputs: the masked layer activity
	- the losses created after applying l0_regularizer can be obtained by calling __tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)__

## Note:
- The masks are created in l0_computation and are retrieved at the l0_dense.call method.

## To Solve/Do:
- Check. The training loss is too large after the insertion of L_0, not sure if it is normal
- Do(optional). Make possible for using more than one regularization mechanisms
- Do. seed argument is not developed for both l0_dense and l0_layer
- Why? failure to identify the variable created in "collection" argument in tf...dense to be initialized
- How? Better way to retrive the Tensors from l0_dense.call(...)
- How? Minimize the argument for l0_dense (regularizer_kernel, _bias can be reduced to just reg_const)
- Discussion. 1. broadcasting for l0_layer; 2. the way to check the tensor_shape in l0_computation; 3. 

