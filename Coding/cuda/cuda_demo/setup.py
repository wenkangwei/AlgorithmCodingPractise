import os
# 针对RTX5090 手动设置arch list，编译时会自动根据arch list选择合适的架构进行编译
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='CudaDemo',
    packages=find_packages(),
    version='0.1.1',
    ext_modules=[
        CUDAExtension(
            'sum_cuda',
            ['./ops/src/reduce_sum/sum.cpp',
             './ops/src/reduce_sum/sum_cuda.cu']
        ),
        CUDAExtension(
            'sum_double_cuda',
            ['./ops/src/sum_two_arrays/two_sum.cpp',
             './ops/src/sum_two_arrays/two_sum_cuda.cu']
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)