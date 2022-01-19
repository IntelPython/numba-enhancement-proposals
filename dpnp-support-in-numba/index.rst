(11/09/2021) Agenda: Accepted: Design Session: Adding dpnp and Data API
support in Numba (1)

Design Questions

a) A well-defined API for a library like dpnp to develop a Numba
   extension to enable JIT compilation of Python functions that include
   dpnp calls.

   1. Overloads

   2. Types

   3. Programming model

b) Since dpnp (and CuPy and any other NumPy-like library) aims to
   support a NumPy-like API, ideally existing Numba overloads for NumPy
   should be reusable wherever possible.

c) Offload target specification

   1. The programming model for dpnp is evolving towards what we call
      “compute follows data”, *i.e.,* every dpnp ndarray instance will
      have a device attribute that will determine the execution
      placement of any computation on that ndarray.

..

   Corollary: The usage of \`dpctl.device_context\` is deprecated.

2. With regards to c.1, the target is specified not using a decorator
   attribute or context manager, rather it is encoded inside the
   function argument.

Meeting Notes:

1. Combining NumPy and dpnp (what is the expected behavior) [Siu]

2. Existing overloads for some of the transcendental functions use C
   functions (e.g., complex) [Siu]

3. Handling the case: “numpy.sum(a)”, where “a” is a dpnp.ndarray. The
   behavior should be consistent across Python and Numba.

4. Reusing overloads: subclassing types.Array gives all the overloads.
   How can a library override a subset of the Numba overloads?

5.

6. How to infer the USM allocator type for the LHS of an expression when
   the RHS mixes dpnp arrays of different USM types?

..

   # a: usm shared dpnp array

   # b: usm device dpnp array

   def foo (a, b):

   c = a + b # Where is c is allocated? How will Numba handle it?

   TA: We can implement three different types.Array sub-classes and then
   use the \__array_\_ function.

   SA: A single type with a property. [Follow-up]

   TA: Do we have a hook that allows us to do it today?

**11/16/2021: Design Session: Adding dpnp and Data API support in Numba
(2)**

**Requirements identified previously:**

a) What should be the Numba type associated with dpnp.ndarray?

Note: [Todd Anderson] dpnp.ndarray can be represented as a subclass of
numba.types.Array. Doing so will enable dpnp.ndarray to reuse existing
Numba NumPy overloads.

**To be discussed (how will these scenarios be handled by Numba)**

b) Dpnp will not use all existing overloads. We need a feature in Numba
   to help a library like dpnp select the overloaded functions it is
   going to use as-is and override the ones where dpnp provides a SYCL
   specific reimplementation. [follow up]

c) Disallow mixing NumPy and dpnp arrays in same array expressions

def foo(a, b, d):

c = a + b # *Not allowed when “a” is a NumPy array and “b” is a
dpnp.ndarray*

e = a + d

d) Allow defining JIT functions that take both NumPy and dpnp arrays but
   do not mix their respective uses.

import numpy as np

import dpnp

def foo(a,b):

“”” a is a numpy array and b is a dpnp array.

“””

c = np.square(a)

d = dpnp.sum(b)

e) How to handle the scenario where a dpnp.array is passed as an
   argument to a numpy function call?

import numpy as np

import dpnp

def foo(a,b):

“”” a is a dpnp array.

“””

c = np.square(a)

@numba.njit

def foo(a, b):

return a+b

# Current state of the art

with dpctl.device_context():

a =

b =

foo(a, b)

# Future

a =

b =

foo(a, b)

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
