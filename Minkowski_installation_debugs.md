# make sure use conda cuda and nvcc to avoid conflict
'''
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
<!-- pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -->
<!-- conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit -->
conda install -c "nvidia/label/cuda-11.7.1" cuda-nvcc
pip install numpy==1.26.1
<!-- export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH -->
'''
# It has to be installed from source

# ModuleNotFoundError: No module named 'distutils.msvccompiler'
change setuptools to 69.5.1 could fix this
> conda install setuptools=69.5.1

# nvcc fatal   : Unknown option '-fopenmp'
> python setup.py install --blas=openblas --blas_include_dirs=${CONDA_PREFIX}/include

if the openblas is not installed:
> conda install -c anaconda openblas-devel

# gcc version. Minkowski works well with gcc 9.5
> conda install -c conda-forge gcc=9.5.0 gxx=9.5.0
> export CC=$(which gcc) CXX=$(which g++)
> export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

# cuda version better 11.x than 12.x
> conda install cudatoolkit=11.x
> export CUDA_HOME=${CONDA_PREFIX}/lib/cuda-11.x

# import Swin3D.sparse_dl.attn_cuda as attn_module

> python setup.py build_ext --inplace



# full list
conda create --name swin3d python=3.10
conda activate swin3d
conda install -y openblas-devel -c anaconda
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y -c "nvidia/label/cuda-11.7.1" cuda-nvcc
conda install -y -c "nvidia/label/cuda-11.7.1" cuda-toolkit
pip install numpy==1.26.1
conda install -y setuptools=69.5.1

conda install -y -c conda-forge gcc=9.5.0 gxx=9.5.0
export CC=$(which gcc) CXX=$(which g++)
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
conda install -y pybind11

python setup.py install --blas=openblas --blas_include_dirs=${CONDA_PREFIX}/include

python setup.py install