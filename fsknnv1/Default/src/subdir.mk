################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/vectorgpu1d.cpp 

OBJS += \
./src/vectorgpu1d.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: C++ Compiler'
	g++ -O2 -g -Wall -c -fmessage-length=0 -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


