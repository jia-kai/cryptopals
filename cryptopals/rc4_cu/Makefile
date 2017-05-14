all: single_byte_bias

single_byte_bias: single_byte_bias.cu *.cuh getcudacap
	nvcc $< -o $@ --std=c++11 -gencode $$(./getcudacap) -lineinfo -g -O2
	nvcc $< -o $@.ptx --std=c++11 -gencode $$(./getcudacap) -lineinfo -ptx

getcudacap: getcudacap.cpp
	nvcc $< -o $@

run: single_byte_bias
	./$<

memcheck: single_byte_bias
	cuda-memcheck ./$<

clean:
	rm -f single_byte_bias getcudacap

.PHONY: all run clean memcheck
