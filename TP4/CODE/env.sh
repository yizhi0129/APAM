#!/bin/bash

module load gcc/8.4.0/gcc-4.8.5
module load libelf/0.8.13/gcc-9.2.0 libffi/3.2.1/intel-19.0.3.199 cuda/10.2.89/intel-19.0.3.199
module load cmake/3.16.2/gcc-9.2.0

export LLVM_PATH=/gpfs/users/roussela/softs/llvm/11.0.0

export PATH=$LLVM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_PATH/libexec:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LLVM_PATH/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LLVM_PATH/libexec:$LIBRARY_PATH
export LIBRARY_PATH=$LLVM_PATH/lib:$LIBRARY_PATH
export MANPATH=$LLVM_PATH/share/man:$MANPATH
export C_INCLUDE_PATH=$LLVM_PATH/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$LLVM_PATH/include:$CPLUS_INCLUDE_PATH
