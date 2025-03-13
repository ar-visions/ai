#include <import>
#include <math.h>

/// this post, post-init pattern is why its nice to have interns accessible in the module
/// that makes things far better in a language, i think.  intern is internal to the module
/// if you cannot trust your own module then thats really odd; the intern applies to the member space
/// but accessibility is within module

/// with silver, intern is about module access; it actually makes things easier to shift around
/// its more free,

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
        verify (instanceof(a, input) || len(a->tensor) > 0, "no inputs resolved");

        // parents output is our input
        a->output = tensor();
        op parent = get(a->op_inputs, 0);
        each (a->op_inputs, op, i) { }
    }

    // find output
    op output = get(k->op_map, string("output"));

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
    if (a->activation) {
        i64 u8_count    = shape_total(a->tensor); // this must return the tensor shape; test this
        quantized i0    = first(a->tensor);
        quantized o0    = first(a->output);
        i64 u8_actual   = A_len(i0); /// it can respond to len
        i8* input_data  = vdata(i0);
        i8* output_data = vdata(o0);
        verify(u8_count == u8_actual, "total size mismatch");
        /// we should probably assert the offsets are the same for this tensor
        verify(i0->offset == o0->offset, "expected output tensor to be quantized the same (offset)");
        verify(i0->scale  == o0->scale,  "expected output tensor to be quantized the same (scale)");

        if (a->activation == Activation_relu) {
            for (i64 i = 0; i < u8_count; i++)
                output_data[i] = max(o0->offset, input_data[i]);
        } else if (a->activation == Activation_tanh) {
            for (i64 i = 0; i < u8_count; i++) {
                // Dequantize: Convert i8 -> f32
                float x = (input_data[i] - i0->offset) * i0->scale;
                // Apply tanh function
                float y = tanhf(x);
                // Requantize: Convert f32 -> i8
                i8 quantized_y = (i8)(roundf(y / o0->scale) + o0->offset);
                // Store the result
                output_data[i] = quantized_y;
            }
        }
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
    return;
}

none dense_back(dense a) {
    return;
}

none relu_init(relu a) {
    a->activation = Activation_relu;
    return;
}
 
 static sz seek_length(FILE* f) {
    fseek(f, 0, SEEK_END);
    sz flen = ftell(f) / sizeof(float);
    fseek(f, 0, SEEK_SET);
    return flen;
 }

// called from the generic path read (json parse)
// needs to also load offset and scale
quantized quantized_with_string(quantized a, string loc) {
    path uri_f32 = form(path, "models/%o.f32", loc);
    if (exists(uri_f32)) {
        FILE* f = fopen(uri_f32->chars, "rb");;
        sz flen = seek_length(f);
        vecf res = A_valloc(typeid(vecf), typeid(f32), flen, flen, true);
        verify(fread(res, flen, 1, f) == 1, "could not read path: %o", uri_f32);
        fclose(f);
        a->realized = res;
    } else {
        path uri_i8  = form(path, "models/%o.i8", loc); /// must really contain two floats for this to make sense.  i do not want this misbound; and its required model-wise
        verify(exists(uri_i8), "i8 fallback not found");
        FILE* f = fopen(uri_i8->chars, "rb");
        sz flen = seek_length(f);
        veci8 res = A_valloc(typeid(veci8), typeid(i8), flen, flen, true);
        verify(fread(&a->scale,  sizeof(float), 1, f) == 1, "scale");
        verify(fread(&a->offset, sizeof(float), 1, f) == 1, "offset");
        verify(fread(res, flen, 1, f) == 1, "could not read path: %o", uri_i8);
        fclose(f);
        a->data = res;
    }
    return a;
}

/// construct with dimension shape (not the data)
quantized quantized_with_array(quantized a, array dims) {
    num count = len(dims);
    i64 shape[32];
    i64 index = 0;
    each (dims, object, e) {
        i64* i = instanceof(e, i64);
        shape[index++] = *i;
    }
    a->shape  = A_vec(typeid(i64), index, shape);
    i64 total = shape_total(a->shape); // improve vectors in time
    a->data   = A_valloc(typeid(i8), typeid(i8), total, total, false);
    return a;
}
 
none quantized_init(quantized a) {
}

none flatten_init(flatten a) {
}

none flatten_forward(flatten a) {
    /// tensor to reduce is at (i8*)a->input->data
}

none flatten_back(flatten a) {
    /// ?
}


none pool_init(pool a) {
    quantized i = get(a->tensor, 0);
    verify(A_len(i->shape) == 3, "unexpected input shape: %o", i->shape); 
    i64 h = shape_get(i->shape, 0);
    i64 w = shape_get(i->shape, 1);
    i64 c = shape_get(i->shape, 2);
}

none pool_forward(pool a) {
    /// tensor to reduce is at (i8*)a->input->data
}

none pool_back(pool a) {
    /// ?
}

define_enum (Initializer)
define_enum (Activation)
define_enum (Padding)
define_enum (Pooling)
define_class(op)
define_mod  (input,       op)
define_mod  (flatten,     op)
define_mod  (concatenate, op)
define_mod  (conv,        op)
define_mod  (pool,        op)
define_mod  (dense,       op)
define_mod  (relu,        op)
define_mod  (output,      op)
define_class(keras)
define_meta (tensor, array, quantized)
define_meta (ops,    array, op)
define_class(quantized)
