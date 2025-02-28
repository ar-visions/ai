#include <import>

int main(int argc, symbol args[]) {
    path  f = form(path, "models/%s.json", k->ident);
    keras k = read(f, typeid(keras));
    print("keras ident %o\n", k->ident);
    return 0;
}