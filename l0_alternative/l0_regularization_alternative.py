

# Alternative Masking: better learning result but not theoretically correct
# todo: at l0_dense, one can use another way for masking, which potentially work (with some emirical evidence) but no theory supported
# trng_masked_kernel = get_tensor_by_name(l0_absolute_path + "/kernel" + l0_relative_path + "el0n_mask:0")
# pred_masked_kernel = get_tensor_by_name(l0_absolute_path + "/kernel" + l0_relative_path + "el0n_mask:0")
# masked_kernel = tf.cond(self.is_training,
#                         lambda: trng_masked_kernel * kernel_0,
#                         lambda: pred_masked_kernel * kernel_0, name='l0_masked_kernel')