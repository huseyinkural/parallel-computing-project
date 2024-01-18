!nvcc --version


    !pip install git+https://github.com/andreinechaev/nvcc4jupyter.git


%load_ext nvcc_plugin


nvcc pagerank.cu -o pagerank
./pagerank



%%writefile floyd.cpp
!g++ floyd.cpp -o floyd
!./floyd


