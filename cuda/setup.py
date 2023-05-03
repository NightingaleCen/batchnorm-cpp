from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="batchnorm_cuda",
    ext_modules=[
        CUDAExtension(
            "batchnorm_cuda",
            [
                "batchnorm_cuda.cpp",
                "batchnorm_cuda_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
