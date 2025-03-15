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
/// for instance: how do you import from json?
/// same way!  sometimes you take in arrays,
/// sometimes you take in string, or numeric.
/// thats unprecedented for object models in any language, let alone C

none topo_visit(keras k, op current, map visited, map in_progress) {
    bool* prog = get(in_progress, current->name);
    if (prog && *prog) {
        error("Cycle detected in network graph at operation: %s", current->name);
        return;
    }
    
    // Skip if already visited
    if (get(visited, current->name)) {
        return;
    }
    
    // Mark as in progress (for cycle detection)
    set(in_progress, current->name, A_bool(true));
    
    // Visit all dependencies (inputs) first
    each (current->op_inputs, op, input_op) {
        topo_visit(k, input_op, visited, in_progress);
    }
    
    // Mark as visited and add to execution order
    set(visited, current->name, A_bool(true));
    set(in_progress, current->name, A_bool(false));
    push(k->order, current);
}

// Helper function to build execution order
none build_exec_order(keras k) {
    k->order = array();

    // create a visited flag for each op to track progress
    map visited     = map(hsize, 32);
    map in_progress = map(hsize, 32); // To detect cycles
    
    // for each operation, ensure it's visited
    each (k->ops, op, operation) {
        if (!get(visited, operation->name)) {
            topo_visit(k, operation, visited, in_progress);
        }
    }
}

tensor find_output(keras k, op a) {
    each (k->ops, op, j)
        each (j->op_inputs, op, input_op)
            if (input_op == a) return j->tensor;

    return null;
}

/// load the ops and initialize them after
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
        each (a->inputs, string, input) {
            op res = get(k->op_map, input);
            verify (res, "could not resolve operation: %o", input);
            push (a->op_inputs, res);
        }
    }

    /// create order
    build_exec_order(k);
    op f = get(k->order, 0);
    verify(isa(f) == typeid(input), "input expected");
    k->input = hold(f->tensor); // an inputs output is our input.. it does have a lonely little input tensor sitting there unused though

    /// finalize layers
    each (k->ops, op, a) {
        a->output = find_output(k, a);
        each(a->op_inputs, op, i) {
            if (isa(i) == typeid(input)) {
                verify(compare(a->tensor->shape, i->tensor->shape) == 0, "incompatible shapes");
                drop(a->tensor);
                a->tensor = hold(i->tensor);
            }
        }
        finalize(a);
    }

    op output = get(k->op_map, string("output"));
    k->output = hold(output->tensor);
}

none keras_train(keras k, i32 epochs) {
}

tensor keras_forward(keras k, tensor input) {
    if (compare(input->shape, k->input->shape) != 0)
        resize(input, k->input);
    else {
        /// copy input tensor, and pass forward
        memcpy(k->input->realized, input->realized, total(input->shape) * sizeof(f32));
    }
    each (k->order, op, current)
        forward(current);
    /// copy output tensor -- if results are stored, we would not want those changing
    tensor res = tensor(shape, k->output->shape);
    print("keras output ident = %x", k->output);  
    memcpy(res->realized, k->output->realized, sizeof(f32) * total(k->output->shape));
    return res;
}

// post-init is mostly established in keras_init 
// (at this point, we have all model data in props)
none op_finalize(op a) {
    return;
}

// all other layers that perform relu should fuse the operation
none op_forward(op a) {
    if (a->activation) {
        i64 u8_count    = shape_total(a->tensor); // this must return the tensor shape; test this
        tensor i0       = a->tensor;
        tensor o0       = a->output;
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
none concatenate_finalize(concatenate a) {
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


none relu_init(relu a) {
    a->activation = Activation_relu;
    return;
}


none flatten_forward(flatten a) {
    /// tensor to reduce is at (i8*)a->input->data
}

none flatten_back(flatten a) {
}


define_enum (Initializer)
define_enum (Activation)
define_enum (Padding)
define_enum (Pooling)
define_class(op)
define_mod  (input,       op)
define_mod  (flatten,     op)
define_mod  (concatenate, op)
define_mod  (relu,        op)
define_mod  (output,      op)
define_class(keras)
define_meta (ops,    array, op)

