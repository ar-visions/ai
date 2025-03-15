#include <import>
#include <immintrin.h>  // AVX2

none conv_finalize(conv c) {
    verify(c->tensor, "Conv layer has no input tensor!");

    shape     input_shape = c->tensor->shape;
    shape     out_shape   = c->output->shape;
    i64       out_total   = shape_total(out_shape);
    tensor    f           = c->tensor;
    i32       offset      = f->offset;
    f32       scale       = f->scale;
    i32       in_h        = input_shape     ->data[0];
    i32       in_w        = input_shape     ->data[1];
    i32       in_c        = input_shape     ->data[2];
    i32       filter_h    = c->kernel_size  ->data[0];
    i32       filter_w    = c->kernel_size  ->data[1];
    i32       stride_h    = c->strides      ->data[0];
    i32       stride_w    = c->strides      ->data[1];
    i32       pad_h       = (c->padding == Padding_same) ? (filter_h - 1) / 2 : 0;
    i32       pad_w       = (c->padding == Padding_same) ? (filter_w - 1) / 2 : 0;
    i32       out_h       = (in_h - filter_h + 2 * pad_h) / stride_h + 1;
    i32       out_w       = (in_w - filter_w + 2 * pad_w) / stride_w + 1;
    i32       out_c       = c->out_channels;

    verify(
        out_h == out_shape->data[0] ||
        out_w == out_shape->data[1] ||
        out_c == out_shape->data[2], "shape mismatch");
    
    i32       im2col_h     = out_h * out_w;
    i32       im2col_w     = filter_h * filter_w * in_c;
    shape     im2col_shape = new_shape(im2col_h, im2col_w, 0);

    /// our new input
    c->im2col_matrix = tensor(
        shape,  im2col_shape,
        scale,  scale,
        offset, offset);

    tensor    w = c->weights;
    shape     weights_shape  = new_shape(im2col_w, out_c, 0); // or is this out_c, im2col_w ?
    A   weights_header = A_header(weights_shape);
    i64 weights_total = shape_total(weights_shape);
    c->weights_matrix        = tensor(
        shape, weights_shape, offset, w->offset, scale, w->scale);
    f32* w_new  = c->weights_matrix->realized;
    f32* w_orig = c->weights->realized;
    for (i32 f = 0; f < out_c; f++)
        for (i32 i = 0; i < im2col_w; i++) {
            f32 v = w_orig[i * out_c + f];
            w_new[f * im2col_w + i] = v;
        }
}

none im2col(conv a, tensor input, tensor result) {
    verify(a->tensor, "Conv layer has no input tensor!");

    shape input_shape  = input->shape;
    shape output_shape = a->output->shape;
    shape im2col_shape = a->im2col_matrix->shape;
    i32   in_h         = input_shape->data[0];
    i32   in_w         = input_shape->data[1];
    i32   in_c         = input_shape->data[2];
    i32   out_h        = output_shape->data[0];
    i32   out_w        = output_shape->data[1];
    i32   out_c        = output_shape->data[2];
    i32   filter_h     = a->kernel_size->data[0];
    i32   filter_w     = a->kernel_size->data[1];
    i32   stride_h     = a->strides->data[0];
    i32   stride_w     = a->strides->data[1];
    i32   pad_h        = (a->padding == Padding_same) ? (filter_h - 1) / 2 : 0;
    i32   pad_w        = (a->padding == Padding_same) ? (filter_w - 1) / 2 : 0;
    i32   im2col_h     = im2col_shape->data[0];  // out_h * out_w
    i32   im2col_w     = im2col_shape->data[1];  // filter_h * filter_w * in_c
    f32*  input_data   = input->realized;
    f32*  im2col_data  = result->realized;
    f32*  weight_data  = a->weights_matrix->realized;
    f32*  output_data  = a->output->realized;
    f32*  bias_data    = a->bias ? a->bias->realized : NULL;


    // Total number of output points
    const i32 output_points = out_h * out_w;
    
    // Loop 1: Iterate over filter positions
    for (i32 kh = 0; kh < filter_h; kh++) {
        for (i32 kw = 0; kw < filter_w; kw++) {
            // Loop 2: Iterate over all output points
            for (i32 out_idx = 0; out_idx < output_points; out_idx++) {
                i32 oh = out_idx / out_w;
                i32 ow = out_idx % out_w;
                
                // Calculate input base position for this output point
                i32 ih = oh * stride_h - pad_h + kh;
                i32 iw = ow * stride_w - pad_w + kw;
                
                // Calculate output index in im2col_data
                i32 col_offset = out_idx * filter_h * filter_w * in_c + 
                                (kh * filter_w + kw) * in_c;
                
                // Zero-padding check
                if (ih < 0 || iw < 0 || ih >= in_h || iw >= in_w) {
                    // fill with zeros (entire channel block)
                    memset(&im2col_data[col_offset], 0, sizeof(f32) * in_c);
                } else {
                    // copy input data (entire channel block)
                    i32 input_base_idx = (ih * in_w + iw) * in_c;
                    memcpy(&im2col_data[col_offset], &input_data[input_base_idx],
                        sizeof(f32) * in_c);
                }
            }
        }
    }
}

none conv_forward(conv a) {
    verify(a->tensor, "Conv layer has no input tensor!");
    im2col(a, a->tensor, a->im2col_matrix);
    gemm(a->im2col_matrix, a->weights_matrix, a->bias, a->activation == Activation_relu, a->output);
}

none conv_back(conv a) {
}

define_mod (conv, op)

/*
typedef struct {
    int batch, in_h, in_w, in_c;
    int out_c, kernel_h, kernel_w;
    int stride;
    float *input;     // (C_in, N * H * W)
    float *weights;   // (C_out, KH, KW, C_in)
    float *bias;      // (C_out)
    float *output;    // (N, H, W, C_out)
    float *dL_dX;     // (C_in, N * H * W)
    float *dL_dW;     // (C_out, KH, KW, C_in)
    float *dL_dB;     // (C_out)
    float *dL_dY;     // (N, H, W, C_out)
} Conv2D;

void conv2d_backward_avx2(Conv2D *conv) {
    int N = conv->batch, H = conv->in_h, W = conv->in_w, C_in = conv->in_c;
    int C_out = conv->out_c, KH = conv->kernel_h, KW = conv->kernel_w;
    int stride = conv->stride;

    float *dL_dY = conv->dL_dY;
    float *dL_dX = conv->dL_dX;
    float *dL_dW = conv->dL_dW;
    float *dL_dB = conv->dL_dB;
    float *X = conv->input;
    float *W = conv->weights;

    // Zero gradients
    memset(dL_dX, 0, sizeof(float) * C_in * N * H * W);
    memset(dL_dW, 0, sizeof(float) * C_out * KH * KW * C_in);
    memset(dL_dB, 0, sizeof(float) * C_out);

    // Compute dL_dB (bias gradients) using 2-loop optimization
    for (int co = 0; co < C_out; co += 8) {
        __m256 bias_sum = _mm256_setzero_ps();
        for (int nhw = 0; nhw < N * H * W; nhw++) {  // Flattened spatial loop
            int out_idx = nhw * C_out + co;
            __m256 dY_val = _mm256_loadu_ps(&dL_dY[out_idx]);
            bias_sum = _mm256_add_ps(bias_sum, dY_val);
        }
        _mm256_storeu_ps(&dL_dB[co], bias_sum);
    }

    // Compute dL_dW (weight gradients) using 3-loop optimization
    for (int co = 0; co < C_out; co += 8) {  // Vectorized outer loop
        for (int ci = 0; ci < C_in; ci++) {  // Iterate over input channels
            __m256 w_sum[KH][KW]; // Local accumulators for kernel updates

            // Initialize accumulators
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++)
                    w_sum[kh][kw] = _mm256_setzero_ps();

            // Process all spatial locations (flattened loop)
            for (int nhw = 0; nhw < N * H * W; nhw++) {
                int out_idx = nhw * C_out + co;
                int in_idx = ci * (N * H * W) + nhw; // Access X as [ci][nhw]

                __m256 dY_val = _mm256_loadu_ps(&dL_dY[out_idx]);
                __m256 X_val = _mm256_set1_ps(X[in_idx]);

                // Accumulate gradient contributions for each kernel position
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        w_sum[kh][kw] = _mm256_fmadd_ps(dY_val, X_val, w_sum[kh][kw]);
                    }
                }
            }

            // Store accumulated weight gradients
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    int weight_idx = ((co * KH + kh) * KW + kw) * C_in + ci;
                    _mm256_storeu_ps(&dL_dW[weight_idx], w_sum[kh][kw]);
                }
            }
        }
    }

    // Compute dL_dX (input gradients) using 3-loop optimization
    for (int ci = 0; ci < C_in; ci++) { // Input channels
        for (int nhw = 0; nhw < N * H * W; nhw++) { // Flattened spatial loop
            __m256 sum = _mm256_setzero_ps();

            for (int co = 0; co < C_out; co += 8) { // Vectorized output channels
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        int out_idx = nhw * C_out + co;
                        int weight_idx = ((co * KH + kh) * KW + kw) * C_in + ci;

                        __m256 dY_val = _mm256_loadu_ps(&dL_dY[out_idx]);
                        __m256 W_val = _mm256_loadu_ps(&W[weight_idx]);
                        sum = _mm256_fmadd_ps(dY_val, W_val, sum);
                    }
                }
            }

            int in_idx = ci * (N * H * W) + nhw; // Flattened input index
            _mm256_storeu_ps(&dL_dX[in_idx], sum);
        }
    }
}

int main() {
    Conv2D conv;
    conv.batch = 1;
    conv.in_h = 32;
    conv.in_w = 32;
    conv.in_c = 3;
    conv.out_c = 64;
    conv.kernel_h = 3;
    conv.kernel_w = 3;
    conv.stride = 1;

    int H = conv.in_h, W = conv.in_w, C_in = conv.in_c, C_out = conv.out_c, KH = conv.kernel_h, KW = conv.kernel_w, N = conv.batch;
    conv.input = (float*)aligned_alloc(32, C_in * N * H * W * sizeof(float));
    conv.weights = (float*)aligned_alloc(32, C_out * KH * KW * C_in * sizeof(float));
    conv.bias = (float*)aligned_alloc(32, C_out * sizeof(float));
    conv.output = (float*)aligned_alloc(32, N * H * W * C_out * sizeof(float));
    conv.dL_dX = (float*)aligned_alloc(32, C_in * N * H * W * sizeof(float));
    conv.dL_dW = (float*)aligned_alloc(32, C_out * KH * KW * C_in * sizeof(float));
    conv.dL_dB = (float*)aligned_alloc(32, C_out * sizeof(float));
    conv.dL_dY = (float*)aligned_alloc(32, N * H * W * C_out * sizeof(float));

    conv2d_backward_avx2(&conv);

    free(conv.input);
    free(conv.weights);
    free(conv.bias);
    free(conv.output);
    free(conv.dL_dX);
    free(conv.dL_dW);
    free(conv.dL_dB);
    free(conv.dL_dY);

    return 0;
}
*/