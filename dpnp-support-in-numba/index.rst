EP – Support dpnp.ndarray inside numba.njit decorated functions
===============================================================

Abstract
--------

We propose adding the ability to recognize the dpnp.ndarray data type as
an argument to the numba.jit decorator and the ability to compile
expressions involving dpnp.ndarray inside a numba.njit decorated
function in the nopython compilation mode. The implementation for the
proposal will be inside Numba-dppy and distributed as part of
Numba-dppy.

Rationale
---------

Numba-dppy’s automatic offload compilation for *parfor* regions, *i.e.,*
NumPy operations, array expressions and prange loops, in a numba.jit
function depends on the use of the dpctl.device_context context manager
to call the numba.jit function. The context manager specifies which
queue to use for task execution. The approach has several shortcomings:

1. Automatic offloading works only for NumPy ndarray data types that are
   host allocated or allocated using SYCL’s USM shared memory allocator.

2. Users cannot pass an array data type that was allocated on a device
   to an numba.jit function.

3. All expressions that are offloaded will run on the same device.

4. It prevents the usage of `Python array
   API <https://data-apis.org/array-api/latest/>`__ libraries that
   include the queue or device as an attribute of the array type.

To overcome these challenges, we propose supporting the dpnp.ndarray
data type in numba.jit functions. dpnp.array should support NumPy array
API inside numba.jit. The dpnp.ndarray data type implements the `Python
array API <https://data-apis.org/array-api/latest/>`__ spec and includes
a queue attribute that is used to specify the queue used to launch
operations on the array. A dpnp.ndarray can be allocated on any USM
allocator and supports the *\__sycl_usm_array_interface_\_* protocol,
thus ensuring interoperability with other SYCL USM-based array
libraries.

Use Cases
---------

1) Triggering automatic offload by passing a dpnp array to an existing
   numba.jit function.

   a. Substitute dpnp overloads in place of Numba’s numpy overloads.

   b. Queue is identified using “queue equivalency” rules.

2) Disallow mixing NumPy and dpnp arrays in same array expressions

3) Allow defining JIT functions that take both NumPy and dpnp arrays but
   do not mix their respective uses.

4) Handle scenario where dpnp.ndarray is passed to a NumPy function. The
   behavior should mimic default dpnp behavior.

5) Handle dispatching using numba-dppy pipeline when dpnp.ndarray is
   passed to a numba.jit function.

Requirements
------------

1) Extend Numba to support dpnp.ndarray inside numba.jit

2) Implement a module that provides dpnp overloads. If possible, reuse
   Numba NumPy overloads.

3) Finalize the dispatcher design as we will not use
   dpctl.device_context for dispatch.
