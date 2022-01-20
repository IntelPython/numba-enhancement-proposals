Design Sessions
===============

(11/09/2021) Adding dpnp and Data API support in Numba (1)
----------------------------------------------------------

Design Questions
````````````````

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

      Corollary: The usage of `dpctl.device_context` is deprecated.

   2. With regards to c.1, the target is specified not using a decorator
      attribute or context manager, rather it is encoded inside the
      function argument.

Meeting Notes
`````````````

1. Combining NumPy and dpnp (what is the expected behavior) [Siu]

2. Existing overloads for some of the transcendental functions use C
   functions (e.g., complex) [Siu]

3. Handling the case: “numpy.sum(a)”, where “a” is a dpnp.ndarray. The
   behavior should be consistent across Python and Numba.

4. Reusing overloads: subclassing types.Array gives all the overloads.
   How can a library override a subset of the Numba overloads?

5. How to infer the USM allocator type for the LHS of an expression when
   the RHS mixes dpnp arrays of different USM types?

.. code-block:: python

    # a: usm shared dpnp array
    # b: usm device dpnp array
    def foo (a, b):
        c = a + b # Where is c is allocated? How will Numba handle it?

   TA: We can implement three different types.Array sub-classes and then
   use the __array__ function.
   SA: A single type with a property. [Follow-up]
       TA: Do we have a hook that allows us to do it today?

11/16/2021: Design Session: Adding dpnp and Data API support in Numba (2)
-------------------------------------------------------------------------

Requirements identified previously:
```````````````````````````````````

a) What should be the Numba type associated with dpnp.ndarray?

Note: [Todd Anderson] dpnp.ndarray can be represented as a subclass of
numba.types.Array. Doing so will enable dpnp.ndarray to reuse existing
Numba NumPy overloads.

To be discussed (how will these scenarios be handled by Numba)
``````````````````````````````````````````````````````````````

b) Dpnp will not use all existing overloads. We need a feature in Numba
   to help a library like dpnp select the overloaded functions it is
   going to use as-is and override the ones where dpnp provides a SYCL
   specific reimplementation. [follow up]

c) Disallow mixing NumPy and dpnp arrays in same array expressions

.. code-block:: python

    def foo(a, b, d):
        c = a + b # *Not allowed when “a” is a NumPy array and “b” is a dpnp.ndarray*
        e = a + d

d) Allow defining JIT functions that take both NumPy and dpnp arrays but
   do not mix their respective uses.

.. code-block:: python

    import numpy as np
    import dpnp

    def foo(a,b):
        """ a is a numpy array and b is a dpnp array.
        """
        c = np.square(a)
        d = dpnp.sum(b)

e) How to handle the scenario where a dpnp.array is passed as an
   argument to a numpy function call?

.. code-block:: python

    import numpy as np
    import dpnp

    def foo(a,b):
        """ a is a dpnp array.
        """
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
