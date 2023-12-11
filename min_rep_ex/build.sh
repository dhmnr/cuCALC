nvcc   -rdc=true -c -o temp.o shared_dynamic.cu
nvcc -dlink -o shared_dynamic.o temp.o -lcudart
rm -f libshared_dynamic.a
ar cru libshared_dynamic.a shared_dynamic.o temp.o
ranlib libshared_dynamic.a
nvcc shared_main.cu  -L. -lshared_dynamic -o main -L/usr/local/cuda/lib64 -lcudart