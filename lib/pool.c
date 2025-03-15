#include <import>
#include <math.h>
#include <immintrin.h>  // AVX2

none pool_finalize(pool a) {
    tensor i = a->tensor;
    verify(i->shape->count == 3, "unexpected input shape: %o", i->shape); 
    i64 h = i->shape->data[0];
    i64 w = i->shape->data[1];
    i64 c = i->shape->data[2];
    i32 pool_h   = a->pool_size->data[0];
    i32 pool_w   = a->pool_size->data[1];
    i32 stride_h = a->strides->data[0];
    i32 stride_w = a->strides->data[1];
    i32 out_h    = (h - pool_h) / stride_h + 1;
    i32 out_w    = (w - pool_w) / stride_w + 1;

    verify(a->output->shape->data[0] == out_h && 
           a->output->shape->data[1] == out_w &&
           a->output->shape->data[2] == c, "pool output shape mismatch");
}

none pool_forward(pool a) {
    verify(a->tensor, "Pool layer has no input tensor!");

    tensor i           = a->tensor;
    tensor o           = a->output;
    i32    in_h        = i->shape->data[0];
    i32    in_w        = i->shape->data[1];
    i32    in_c        = i->shape->data[2];
    i32    pool_h      = a->pool_size->data[0];
    i32    pool_w      = a->pool_size->data[1];
    i32    stride_h    = a->strides->data[0];
    i32    stride_w    = a->strides->data[1];
    i32    out_h       = o->shape->data[0];
    i32    out_w       = o->shape->data[1];
    f32*   input_data  = i->realized;
    f32*   output_data = o->realized;

    // Number of elements to process in parallel with AVX2
    const int vec_size = 8; // AVX2 processes 8 floats at once
    
    // Process channel data in chunks of 8 at a time
    for (i32 oh = 0; oh < out_h; oh++) {
        for (i32 ow = 0; ow < out_w; ow++) {
            // Process channels in chunks of 8
            for (i32 c = 0; c < in_c; c += vec_size) {
                // Load vector of negative infinity values as initial max
                __m256 max_vals = _mm256_set1_ps(-INFINITY);
                
                // Iterate through pool window
                for (i32 kh = 0; kh < pool_h; kh++) {
                    for (i32 kw = 0; kw < pool_w; kw++) {
                        i32 ih = oh * stride_h + kh;
                        i32 iw = ow * stride_w + kw;
                        
                        if (ih < in_h && iw < in_w) {  // Ensure within bounds
                            i32 base_idx = (ih * in_w + iw) * in_c + c;
                            
                            // Load 8 channels at once from this spatial position
                            __m256 input_vec = _mm256_loadu_ps(&input_data[base_idx]);
                            
                            // Update maximum values
                            max_vals = _mm256_max_ps(max_vals, input_vec);
                        }
                    }
                }
                
                // Store results back to memory
                i32 out_base_idx = (oh * out_w + ow) * in_c + c;
                _mm256_storeu_ps(&output_data[out_base_idx], max_vals);
            }
        }
    }
}

none pool_back(pool a) {
    /// TODO: Implement max-pooling backward pass
}

define_mod (pool, op)
