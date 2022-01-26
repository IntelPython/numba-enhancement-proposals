Unified GPU Programming
=======================

Meeting Notes Dec 7th 2021
--------------------------

Goal
~~~~

A unified Pythonic language and API for writing vendor-agnostic GPU
code.

Numba can provide a common API for vendors to register their
implementation. The vendor provided implementation will:

-  provide low-level drivers to discover and select hardware units
-  decide on the compilation toolchain
-  provide a runtime system to schedule and launch kernels

Prior arts
~~~~~~~~~~

SYCL is providing similar functionality for unified GPU programming for
C++. There are useful concepts that we can learn from the SYCL spec.

We can start by considering a subset of SYCL that maps well to the
existing GPU support Numba currently has.

What level should the unified language target?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Important usecases include preprocessing for deep-learning, and
   classical machine learning algorithms.
-  These cases will require access to private/shared memory for
   cooperative parallel operations (e.g at the thread block level.)

Explore Plugin interface of DPC++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  [TODO] study the abstraction layer of SYCL concepts to map to actual
   backend. https://intel.github.io/llvm-docs/PluginInterface.html
-  different SYCL compiler does it differently

Compiler-extension/Type-inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  [Stuart] Numba needs to provide a strongly-typed common language such
   that variable types do not change depending on the actual backend.

Adoption and migration from existing code
-----------------------------------------

CUDA code is prevalent for Numba given the longer history of CUDA
support. Compatibility and reuse of existing CUDA code will be important
for the adoption of the new unified language.
