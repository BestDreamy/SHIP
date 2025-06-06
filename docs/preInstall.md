# 主机端
```
wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.4.tar.gz
tar -zxvf v2.4.4.tar.gz
cd gdrcopy-2.4.4/
make -j$(nproc)
sudo make prefix=/opt/gdrcopy install

# Kernel module installation
cd packages
sudo yum groupinstall 'Development Tools'
sudo yum install dkms rpm-build make
CUDA=/usr/local/cuda ./build-rpm-packages.sh
sudo rpm -Uvh gdrcopy-kmod-2.4.4-1dkms.el3.noarch.rpm
sudo rpm -Uvh gdrcopy-2.4.4-1.el3.x86_64.rpm
sudo rpm -Uvh gdrcopy-devel-2.4.4-1.el3.noarch.rpm

# 需要主机端确保加载模块
cd ..
sudo ./insmod.sh

# lsmod | grep gdrdrv
```

```
gdrcopy_copybw
```

# 容器端

```
${CUDA_DEVICE_DRIVER}=570.124.06

docker run -it --runtime=nvidia --name=wbc_pplx   --privileged --gpus all --shm-size=32G \
-v /usr/src/nvidia-570.124.06:/usr/src/nvidia-570.124.06 \
--device=/dev/gdrdrv \
--network=host \
--device /dev/infiniband \
-v /usr/lib/mlnx-ofed:/usr/lib/mlnx-ofed:ro \
-v ~/work/:/root/work \
nvcr.io/nvidia/pytorch:24.10-py3
```

GDRCopy
```
cd
wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.4.tar.gz
tar -zxvf v2.4.4.tar.gz
cd gdrcopy-2.4.4/

apt update
apt install build-essential devscripts debhelper fakeroot pkg-config dkms -y
pushd packages
CUDA=/usr/local/cuda ./build-deb-packages.sh
dpkg -i gdrdrv-dkms_2.4.4_amd64.Ubuntu22_04.deb
dpkg -i libgdrapi_2.4.4_amd64.Ubuntu22_04.deb
dpkg -i gdrcopy-tests_2.4.4_amd64.Ubuntu22_04+cuda12.6.deb
popd
```
```
gdrcopy_copybw
```
IBGDA
```
echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' >> /etc/modprobe.d/nvidia.conf
apt install initramfs-tools
update-initramfs -u
cat /etc/modprobe.d/nvidia.conf
```
重启容器/裸机： docker restart {CONTAINER}


NVSHMEM
```
# find libmlx5.so
ldconfig -p | grep mlx5
cd /usr/lib/x86_64-linux-gnu
ln -s libmlx5.so.1 libmlx5.so

cd
wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz
mkdir nvshmem_src_3.2.5-1
tar xf nvshmem_src_3.2.5-1.txz -C nvshmem_src_3.2.5-1
cd nvshmem_src_3.2.5-1/nvshmem_src

mkdir -p build
cd build
cmake \
    -DNVSHMEM_PREFIX=/opt/nvshmem-3.2.5 \
    -DCMAKE_CUDA_ARCHITECTURES=90a \
    -DNVSHMEM_MPI_SUPPORT=1 \
    -DNVSHMEM_PMIX_SUPPORT=0 \
    -DNVSHMEM_LIBFABRIC_SUPPORT=0 \
    -DNVSHMEM_IBRC_SUPPORT=1 \
    -DNVSHMEM_IBGDA_SUPPORT=1 \
    -DNVSHMEM_BUILD_HYDRA_LAUNCHER=1 \
    -DNVSHMEM_BUILD_TXZ_PACKAGE=1 \
    -DMPI_HOME=/opt/hpcx/ompi \
    -DGDRCOPY_HOME=/root/gdrdrv-2.4.4 \
    -G Ninja \
    ..
ninja
ninja install
  
# CUDA_HOME=/usr/local/cuda \
# GDRCOPY_HOME=/root/gdrdrv-2.4.4 \
# NVSHMEM_SHMEM_SUPPORT=0 \
# NVSHMEM_UCX_SUPPORT=0 \
# NVSHMEM_USE_NCCL=0 \
# NVSHMEM_MPI_SUPPORT=0 \
# NVSHMEM_IBGDA_SUPPORT=1 \
# NVSHMEM_PMIX_SUPPORT=0 \
# NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
# NVSHMEM_USE_GDRCOPY=1 \
# cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/opt/nvshmem
    
# cd build
# make -j$(nproc)
# make install
```
```
export MPI_HOME=/opt/hpcx/ompi
export NVSHMEM_MPI_SUPPORT=1

export HPCX_HOME=/opt/hpcx
export PATH=$HPCX_HOME/ompi/bin:$HPCX_HOME/ucx/bin:$PATH
export LD_LIBRARY_PATH=$HPCX_HOME/ompi/lib:$HPCX_HOME/ucx/lib:$LD_LIBRARY_PATH
export CPATH=$HPCX_HOME/ompi/include:$HPCX_HOME/ucx/include:$CPATH

export NVSHMEM_HOME=/opt/nvshmem-3.2.5
export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_HOME}/bin:$PATH"

# For one node
export NVSHMEM_REMOTE_TRANSPORT=none

# For nvshmrun
export HYDRA_HOME=/opt/hydra
export PATH="${HYDRA_HOME}/bin:$PATH"
export NVCC_GENCODE="arch=compute_90,code=sm_90a"
```
Hydra
```
cd nvshmem_src_3.2.5-1/nvshmem_src/
sed -i 's/^make/make -j/' scripts/install_hydra.sh
sudo bash scripts/install_hydra.sh hydra-build /opt/hydra
```
