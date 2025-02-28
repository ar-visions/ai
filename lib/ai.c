#include <import>

/// this post, post-init pattern is why its nice to have interns accessible in the module
/// that makes things far better in a language, i think.  intern is internal to the module
/// if you cannot trust your own module then thats really odd
/// with silver, intern is about module access; it actually makes things easier to shift around
/// its more free!
/// silver really is based on A, and with A 
/// you may load entire objects quite easily
/// without inventing new protocol on top of an object model

none keras_init(keras k) {
    /// construct operation map
    k->op_map = map();
    each (k->ops, op, op)
        set (k->op_map, op->name, op);

    /// create op_inputs in op, so we know our input-tensors
    each (k->ops, op, a) {
        /// create resolved op list, corresponding 
        /// with index-of the string-name from inputs
        a->op_inputs = array();
        each_ (a->inputs, string, input) {
            op res = get(k->op_map, input);
            verify (res, "could not resolve operation: %o", input);
            push (a->op_inputs, res);
        }

        a->tensor = tensor();
        each (a->op_inputs, op, i)
            each_ (i->output, quantized, q)
                push (a->tensor, q);
        verify (instanceof(op, input) || len(a->tensor) > 0, "no inputs resolved");

        // parents output is our input
        a->output = tensor();
        op parent = get(a->op_inputs, 0);
        each (a->op_inputs, op, i)
    }

}

// post-init is mostly established in keras_init 
// (at this point, we have all model data in props)
none op_init(op a) {
    return;
}

none keras_train(keras k, i32 epochs) {
}

none keras_forward(keras k, tensor input) {
    /// copy input to model input (just so we may be async with this memory)
}

// all other layers that perform relu should fuse the operation
none op_forward(op a) {
    if (a->activation == Activation_relu) {
        i64 u8_count    = shape_total(a->input_shape);
        quantized i0    = first(a->tensor);
        quantized o0    = first(a->output);
        i64 u8_actual   = A_len(i0); /// it can respond to len
        i8* input_data  = vdata(i0);
        i8* output_data = vdata(o0);
        verify(u8_count == u8_actual, "total size mismatch");
        /// we should probably assert the offsets are the same for this tensor
        verify(i0->offset == o0->offset, "expected output tensor to be quantized the same (offset)");
        verify(i0->scale  == o0->scale,  "expected output tensor to be quantized the same (scale)");
        for (i64 i = 0; i < u8_count; i++)
            output_data[i] = max(o0->offset, input_data[i]);
    }
}

none op_back(op a) {
    return;
}

// Concatenate implementation
none concatenate_init(concatenate a) {
    a->axis = -1;  // Default to last dimension
    return;
}

none concatenate_forward(concatenate a) {
    // Implement concatenation along the specified axis
    // Will need to iterate through all inputs and combine them
    return;
}

none concatenate_back(concatenate a) {
    return;
}

// Conv implementation
none conv_init(conv a) {
    return;
}

none conv_forward(conv a) {
    // Implement convolution operation
    return;
}

none conv_back(conv a) {
    // Implement backpropagation for convolution
    return;
}

// Dense implementation
none dense_init(dense a) {
    a->units = 0;
    a->kernel_initializer = NULL;
    a->weight_initializer = NULL;
    return;
}

none dense_forward(dense a) {
    // Implement dense layer forward pass
    // This will perform matrix multiplication between input and weights
    // Then add bias and apply activation if specified
    return;
}

none dense_back(dense a) {
    // Implement dense layer forward pass
    // This will perform matrix multiplication between input and weights
    // Then add bias and apply activation if specified
    return;
}

// ReLU implementation
none relu_init(relu a) {
    a->activation = Activation_relu;
    return;
}

none relu_forward(relu a) {
    // Implement ReLU activation
    // For each element in the input tensor:
    // output = max(input, threshold)
}

// Quantized implementation
none quantized_init(quantized a) {
}

// Pool implementation
none pool_init(pool a) {
    // set output shape
    verify(!a->output_shape, "unexpected output_shape");
    i64 h = get(a->output_shape, 0);
    i64 w = get(a->output_shape, 1);
    i64 c = last(a->output_shape);
    verify(A_len(a->input_shape) == 3, "unexpected input shape: %o", a->input_shape); 
    a->output_shape = shape(h / 2, w / 2, c, 0);
}

define_enum (Initializer)
define_enum (Activation)
define_class(op)
define_mod  (input,       op)
define_mod  (concatenate, op)
define_mod  (conv,        op)
define_mod  (dense,       op)
define_mod  (relu,        op)
define_mod  (output,      op)
define_class(keras)
define_meta (tensor, array, quantized)
define_meta (ops,    array, op)
define_class(quantized)
