The procedure needs additional libraries: GSL(http://mirrors.kernel.org/gnu/gsl/) and GMP (https://gmplib.org/). Please install these libraries first in case of uninstalled.

Usage:
	make
	./uncertain-core [filename] [alg] [eta] [threads]
example:
	./uncertain-core youtube.bin 6


If an error 'error while loading shared libraries: libgsl.so.*' occurs, using the following commands to fix it.
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
	export LD_LIBRARY_PATH
