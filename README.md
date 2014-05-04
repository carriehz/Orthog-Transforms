Orthog-Transforms
=================

C++ functions for DCT-II, Haar, Fast Walsh, Parametric Slant Transforms implemented with OpenCV v 2.4.7

C++ .hpp files include Discrete Cosine Transform (DCT-II), Fast Walsh Hadamard Transform (FWT), Haar Transform, Parametric Slant Transform

All functions work with 256 x 256 sized matrices.  

DCT-II function:

  - Accepts any size matrix
  - Implements a Discrete Cosine Transform Type II
  - Includes inverse DCT-II function

Fast Walsh Hadamard Transform

  - Accepts 256 x 256, 16 x 16, or 8 x 8 size matrices
  - Depends on Kronecker Product function (commented out in .hpp file, available in repository as a separate header)
  - Includes Fast Paley implementation
  - Includes inverse functions

Haar Transform

  - Accepts any size square m x m matrix
  - Input variable to determine matrix size is a power of 2, i.e. an 8 x 8 matrix requires a value of 3, a 256 x 256 matrix requires a value of 8
  - Includes inverse Haar
  - Includes Haar wavelet iteration function which decomposes into low and high pass Haar wavelet filters
  - Depends on Kronecker Product function (commented out in .hpp file, available in repository as a separate header)

Parametric Slant Transform

 - Implementation of a Parametric Slant Transform as defined in Agaian, S., Tourshan, K., Noonan, J.  "Parametric Slant-Hadamard Transforms With Applications,"IEEE Signal Processing Letters, Vol 9, No 11, November 2002.
 - Accepts 256 x 256 or 8 x 8 matrices as input
 - Code can be modified for single constant or multiple betas
 - Depends on Kronecker Product function (commented out in .hpp file, available in repository as a separate header)
