#include <import>

/// goal: get keras operational in native
///       0.025 accuracy error is decent for first stage target (prior to correct and feature)
///       walk the walk!
int main(int argc, symbol args[]) {
    A_start();
    path  f = form(path, "models/vision_target.json");
    keras k = read(f, typeid(keras));
    if (exists(f)) {
        print("path exists");
    }
    print("keras ident %o\n", k->ident);
    return 0;
}