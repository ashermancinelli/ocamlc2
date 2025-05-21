## LTO

We do not want to do full separable compilation, but whole-program analysis at the mlir level.

Each translation unit will be at a relatively high level, preserving the closure and function operations in a way that
we can get rid of as many runtime calls for function applications as possible and drop down into direct function calls.

Also circumvents the need for rigid abi, as long as we can merge all the modules together at the mlir level we don't really have to worry about calling anything other than the runtime.
