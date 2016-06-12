MarchSAT - OpenCL SAT solver
----------------------------

MarchSAT is a small, incomplete stochastic local search, OpenCL-based SAT solver.
MarchSAT is loosely based on clsat (https://github.com/vegard/clsat) and implements
a highly parallel GPU version of WalkSAT (https://en.wikipedia.org/wiki/WalkSAT).

The algorithm reproduces WalkSAT on GPU and does NOT exploit the full power of GPUs.
Instead it only uses the highly parallel computing capacity and is NOT optimized
with respect to GPU programming. Especially memory usage is not optimized.

However, the algorithm encodes SAT instances as bit fields and thus can handle quite
big instances for which WalkSAT fails.

Compiling
---------

You need to have the AMD APP SDK installed to compile and run the program. See
make.sh how to compile. Boost libraries are also needed.

Miscellaneous
-------------

Tested on an Ubuntu 12.04 64bit and two AMD HD9770 GPUs as well as on an Ubuntu 14.04
64 bit with Intel i7 CPU.

Code contains redundant parts and surely several bugs.

Example for Intel i7 CPU
------------------------

Enter the MarchSAT directory and run (see also examples/example.run)

./a.out --infile examples/random_ksat_2500_10000.dimacs --device 0 --threads 32 --workgroups 32 --flips 0.5 --reuse 0.5 --iterations 2048

Make sure that the device is selected correctly. Also keep in mind that it's all about
having the parameters right.
