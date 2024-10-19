from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import include_paths

setup(
    name='cagra',
    ext_modules=[
        CUDAExtension(
            name='cagra',
            sources=['./csrc/cagra_search.cu'],
            include_dirs=include_paths(),
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)