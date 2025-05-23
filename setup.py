import os
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


if __name__ == '__main__':
    # nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
    # assert nvshmem_dir is not None and os.path.exists(nvshmem_dir), 'Failed to find NVSHMEM'
    # print(f'NVSHMEM directory: {nvshmem_dir}')

    # TODO: currently, we only support Hopper architecture, we may add Ampere support later
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable',
                 '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3', '-rdc=true', '--ptxas-options=--register-usage-level=10']
    # include_dirs = ['csrc/', f'{nvshmem_dir}/include']
    include_dirs = ['csrc/']
    sources = ['csrc/reduce.cu']
    # library_dirs = [f'{nvshmem_dir}/lib']

    # Disable aggressive PTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '0')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    # Disable DLTO (default by PyTorch)
    # nvcc_dlink = ['-dlink', f'-L{nvshmem_dir}/lib', '-lnvshmem']
    nvcc_dlink = ['-dlink']
    # extra_link_args = ['-l:libnvshmem.a', '-l:nvshmem_bootstrap_uid.so', f'-Wl,-rpath,{nvshmem_dir}/lib']
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
        'nvcc_dlink': nvcc_dlink
    }

    setuptools.setup(
        name='deep',
        version='1.0.0',
        # packages=setuptools.find_packages(
        #     include=['deep_ep']
        # ),
        ext_modules=[
            CUDAExtension(
                name='deep_reduce',
                include_dirs=include_dirs,
                # library_dirs=library_dirs,
                sources=sources,
                extra_compile_args=extra_compile_args,
                # extra_link_args=extra_link_args
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )

