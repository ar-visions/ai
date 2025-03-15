#include <import>
#include <math.h>
#include <immintrin.h>  // AVX2

static sz seek_length(FILE* f, AType of_type) {
    u64 start = ftell(f);
    fseek(f, 0, SEEK_END);
    u64 end = ftell(f);
    sz flen = (end - start) / of_type->size;
    fseek(f, start, SEEK_SET);
    return flen;
}

// called from the generic path read (json parse)
// needs to also load offset and scale
tensor tensor_with_string(tensor a, string loc) {
    path uri_f32 = form(path, "models/%o.f32", loc);
    path uri_i8  = form(path, "models/%o.i8", loc);
    FILE* f;
    bool is_f32 = exists(uri_f32);
    f = fopen(is_f32 ? uri_f32->chars : uri_i8->chars, "rb");
    a->shape = shape_read(f);
    sz flen  = seek_length(f, typeid(f32));
    if (is_f32) {
        vecf res = A_alloc2(typeid(vecf), typeid(f32), a->shape, true);
        i64 total = shape_total(a->shape);
        verify(flen == total, "f32 mismatch in size");
        verify(fread(res, sizeof(f32), flen, f) == flen, "could not read path: %o", uri_f32);
        a->realized = res;
    } else {
         /// must really contain two floats for this to make sense.  i do not want this misbound; and its required model-wise
        veci8 res = A_alloc2(typeid(veci8), typeid(i8), a->shape, true);
        verify(fread(&a->scale,  sizeof(float), 1, f) == 1, "scale");
        verify(fread(&a->offset, sizeof(float), 1, f) == 1, "offset");
        i64 total = shape_total(a->shape);
        verify(flen == total, "i8 mismatch in size");
        verify(fread(res, 1, flen, f) == flen, "could not read path: %o", uri_i8);
        a->data = res;
    }
    fclose(f);
    return a;
}

/// construct with dimension shape (not the data)
tensor tensor_with_array(tensor a, array dims) {
    num count = len(dims);
    i64 shape[32];
    i64 index = 0;
    each (dims, object, e) {
        i64* i = instanceof(e, i64);
        shape[index++] = *i;
    }
    a->shape  = shape_from(index, shape);
    a->total  = shape_total(a->shape); // improve vectors in time
    a->data   = A_valloc(typeid(i8), typeid(i8),
        a->total, a->total, false);
    return a;
}

none tensor_init(tensor a) {
    a->total    = shape_total(a->shape);
    if (!a->realized)
         a->realized = A_alloc2(typeid(vecf),  typeid(f32), a->shape, false);
    if (!a->data)
         a->data     = A_alloc2(typeid(veci8), typeid(i8),  a->shape, false);
}

// Approximate division using reciprocal multiplication
static inline __m256i avx2_div_epi16(__m256i num, __m256i denom) {
    __m256  num_f  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(num)));
    __m256  denom_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(denom)));
    
    __m256  recip = _mm256_rcp_ps(denom_f);  // Approximate 1/x
    __m256  res_f = _mm256_mul_ps(num_f, recip);
    
    __m256i res = _mm256_cvtps_epi32(res_f);
    return res;
}

none tensor_resize(tensor input, tensor output) {
    if (compare(input->shape, output->shape) == 0) {
        memcpy(input->realized, output->realized, total(input->shape) * sizeof(f32));
        memcpy(input->data,     output->data,     total(input->shape) * sizeof(i8));
        return;
    }
    // we are asserting the data is in quantized state, but that may not be the case always
    int   in_w    = input ->shape->data[1];
    int   in_h    = input ->shape->data[0];
    int   out_w   = output->shape->data[1];
    int   out_h   = output->shape->data[0];
    float scale_x = (float)in_w / out_w;
    float scale_y = (float)in_h / out_h;
    i8*   dst_i8  = output->data;
    i32*  dst_f32 = output->realized;

    for (int oy = 0; oy < out_h; oy++) {
        int ox = 0;
        for (; ox < out_w; ox += 32) {  // Process 64 pixels at a time (512-bit / 8-bit)
            __m256i sum   = _mm256_setzero_si256();
            float   fx    = ox * scale_x;
            float   fy    = oy * scale_y;
            int     sx    = (int)fx;
            int     sy    = (int)fy;
            int     ex    = (int)((ox + 1) * scale_x);
            int     ey    = (int)((oy + 1) * scale_y);
            __m256i count = _mm256_set1_epi8((int8_t)(ey - sy) * (ex - sx));

            for (int iy = sy; iy < ey; iy++)
                for (int ix = sx; ix < ex; ix++)
                    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((__m256i*)&((i8*)input->data)[iy * in_w + ix]));
            
            count       = _mm256_max_epu8(count, _mm256_set1_epi8(1));
            
            //__m256i avg = _mm256_div_epi8(sum, count); <--- avx512 is nicer

            // Convert `sum` and `count` from epi8 -> epi16 (zero extend lower 128 bits)
            __m256i sum_lo    = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(sum));
            __m256i sum_hi    = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(sum, 1));
            __m256i count_lo  = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(count));
            __m256i count_hi  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(count, 1));

            // Ensure no division by zero
            count_lo = _mm256_max_epu16(count_lo, _mm256_set1_epi16(1));
            count_hi = _mm256_max_epu16(count_hi, _mm256_set1_epi16(1));

            // Approximate integer division: (sum + (count/2)) / count for rounding
            __m256i avg_lo = avx2_div_epi16(sum_lo, count_lo);  // Manual division loop needed
            __m256i avg_hi = avx2_div_epi16(sum_hi, count_hi);  // Manual division loop needed

            // Pack results back to epi8 (saturating pack)
            __m256i avg = _mm256_packus_epi16(avg_lo, avg_hi);

            // Store result
            _mm256_storeu_si256((__m256i*)&((i8*)output->data)[oy * out_w + ox], avg);
        }
        for (; ox < out_w; ox++) {
            int   sum   = 0;
            float fx    = ox * scale_x;
            float fy    = oy * scale_y;
            int   sx    = (int)fx;
            int   sy    = (int)fy;
            int   ex    = (int)((ox + 1) * scale_x);
            int   ey    = (int)((oy + 1) * scale_y);
            for (int iy = sy; iy < ey; iy++)
                for (int ix = sx; ix < ex; ix++)
                    sum  += ((i8*)input->data)[iy * in_w + ix];
            
            int count = (ey - sy) * (ex - sx);
            ((i8*)output->data)[oy * out_w + ox] = sum / count;
        }
    }
    if (output->realized) {
        f32* dst = output->realized;
        i8*    q = output->data;
        /// can we convert to f32 using simd?
        int total_pixels = out_w * out_h;
        int i = 0;
        for (; i + 16 <= total_pixels; i += 16) { // Process 16 pixels at a time
            __m128i i8_vals = _mm_loadu_si128((__m128i*)&q[i]); // Load 16 int8 values
            __m256 float_vals = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(i8_vals)); // Convert to f32
            _mm256_storeu_ps(&dst[i], float_vals); // Store converted values
        }
        // Handle remaining pixels (non-multiple of 16)
        for (; i < total_pixels; i++)
            dst[i] = (float)q[i] / 255.0;
    }
}


none tensor_gemm(tensor input, tensor weights, tensor bias, bool relu, tensor output) {
    i32   input_h      = input   -> shape->data[0];  // out_h * out_w
    i32   input_w      = input   -> shape->data[1];  // filter_h * filter_w * in_c
    i32   output_w     = output  -> shape->data[1];
    f32*  weight_data  = weights -> realized;
    f32*  output_data  = output  -> realized;
    f32*  bias_data    = bias    ?  bias->realized : NULL;
    f32*  input_data   = input->realized;

    const i32 vec_size = 8; // AVX2 processes 8 floats per iteration
    i32 oc_aligned = output_w - (output_w % vec_size);  // Ensure we process full AVX2 blocks

    /// Perform GEMM: im2col_matrix * weights_matrix
    for (i32 ohw = 0; ohw < input_h; ohw++) {
        i32 i = 0;

        // Process 8 output channels at a time
        for (; i < oc_aligned; i += vec_size) { 
            __m256 sum = bias_data ? _mm256_loadu_ps(&bias_data[i]) : _mm256_setzero_ps();

            for (i32 j = 0; j < input_w; j++) {
                __m256 input_vec   = _mm256_set1_ps(input_data[ohw * input_w + j]); // Broadcast input
                __m256 weight_vec  = _mm256_loadu_ps(&weight_data[j * output_w + i]); // Load 8 weights
                sum = _mm256_fmadd_ps(input_vec, weight_vec, sum); // Multiply and accumulate
            }

            // Apply ReLU if needed
            if (relu) {
                __m256 zero_vec = _mm256_setzero_ps();
                sum = _mm256_max_ps(sum, zero_vec);  // ReLU: max(sum, 0)
            }

            _mm256_storeu_ps(&output_data[ohw * output_w + i], sum);
        }

        // Process remaining output channels using scalar ops
        for (; i < output_w; i++) { 
            f32 sum = bias_data ? bias_data[i] : 0.0f;
            for (i32 j = 0; j < input_w; j++) {
                sum += input_data[ohw * input_w + j] * weight_data[j * output_w + i];
            }
            output_data[ohw * output_w + i] = (!relu || sum > 0) ? sum : 0;
        }
    }
}

define_class(tensor)