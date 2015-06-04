################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../main.cpp 

CU_SRCS += \
../computeDistance.cu 

CU_DEPS += \
./cu_computeDistance.d 

OBJS += \
./cu_computeDistance.o \
./main.o 


# Each subdirectory must supply rules for building sources it contributes
cu_%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: CUDA NVCC Compiler'
	/usr/local/cuda/bin/nvcc -c -I/usr/local/cuda/include -use_fast_math -m 64 -O3 -arch sm_13  -Xcompiler -fopenmp -lgomp -o "$@" "$<" && \
	echo -n '$(@:%.o=%.d)' $(dir $@) > '$(@:%.o=%.d)' && \
	/usr/local/cuda/bin/nvcc --host-compilation c++ -M -I/usr/local/cuda/include   "$<" >> '$(@:%.o=%.d)'
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: C++ Compiler'
	g++ -I/home/asa943/boost_1_39_0/ -O2 -g -Wall -c -fmessage-length=0 -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


