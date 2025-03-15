#include <import>

none dense_finalize(dense a) {
    verify(a->weights,        "no weights");
    verify(a->bias,           "no bias");
    verify(a->output_dim > 0, "no units");
    verify(a->input_dim > 0,  "no input_dim");
    verify(a->output->shape->data[0] == a->output_dim, "invalid output shape");
}

none dense_forward(dense a) {
    gemm(a->tensor, a->weights, a->bias, a->activation == Activation_relu, a->output);
    print("dense output identity = %x", a->output);
}

none dense_back(dense a) {
}

define_mod (dense, op)