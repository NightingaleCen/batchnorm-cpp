# CUDA Extensions of Batch Normalization in PyTorch

An example of writing a C++ extension of Batch Normalization for PyTorch. See
[here](http://pytorch.org/tutorials/advanced/cpp_extension.html) for the accompanying tutorial.

- Inspect the CUDA extensions in the `cuda/` folders,
- Build CUDA extensions by going into the `cuda/` folder and executing `python setup.py install`,
- Run gradient checks and output checks on the code by running `python grad_check.py`.

## Acknowledgements

Partly based on the code of [Peter Goldsborough](https://github.com/goldsborough).
