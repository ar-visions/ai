#include <import>

int main(int argc, symbol args[]) {
    A_start();
    path   f      = form(path, "models/vision_base.json");
    keras  k      = read(f, typeid(keras));
    tensor input  = tensor (shape, new_shape(32, 32, 1, 0));
    memset(input->realized, 127, sizeof(f32) * 2);
    tensor output = forward(k, input);
    f32*   out    = output->realized;
    print("keras model %o, output = %.2f, %.2f, %.2f\n", k->ident, out[0], out[1], out[2]);
    return 0;
}