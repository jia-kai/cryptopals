all: single_byte_bias

single_byte_bias: single_byte_bias.cu *.cuh getcudacap
	nvcc $(NVCCFLAGS) $< -o $@ -gencode $$(./getcudacap) -lineinfo \
		--std=c++11 -g -O2 -Xcompiler -Wall,-Wextra
	nvcc $(NVCCFLAGS) $< -o $@.ptx --std=c++11 -gencode \
		$$(./getcudacap) -lineinfo -ptx

getcudacap: getcudacap.cpp
	nvcc $< -o $@

run: single_byte_bias
	./$< $(RUN_ARGS)

memcheck: single_byte_bias
	cuda-memcheck ./$<

clean:
	rm -f single_byte_bias getcudacap

.PHONY: all run clean memcheck
