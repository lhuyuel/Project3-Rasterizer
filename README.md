-------------------------------------------------------------------------------
CIS565: Project 3: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2012
-------------------------------------------------------------------------------
Due Monday 11/05/2012
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
REQUIREMENTS:
-------------------------------------------------------------------------------
In this project, you are given code for:

* A library for loading/reading standard Alias/Wavefront .obj format mesh files and converting them to OpenGL style VBOs/IBOs
* A suggested order of kernels with which to implement the graphics pipeline
* Working code for CUDA-GL interop

You will need to implement the following stages of the graphics pipeline and features:

* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a scanline or a tiled approach
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A simple lighting/shading scheme, such as Lambert or Blinn-Phong, implemented in the fragment shader

You are also required to implement at least 3 of the following features:

* Additional pipeline stages. Each one of these stages can count as 1 feature:
   * Geometry shader
   * Transformation feedback
   * Back-face culling
   * Scissor test
   * Stencil test
   * Blending

IMPORTANT: For each of these stages implemented, you must also add a section to your README stating what the expected performance impact of that pipeline stage is, and real performance comparisons between your rasterizer with that stage and without.

* Correct color interpretation between points on a primitive
* Texture mapping WITH texture filtering and perspective correct texture coordinates
* Support for additional primitices. Each one of these can count as HALF of a feature.
   * Lines
   * Line strips
   * Triangle fans
   * Triangle strips
   * Points
* Anti-aliasing
* Order-independent translucency using a k-buffer
* MOUSE BASED interactive camera support. Interactive camera support based only on the keyboard is not acceptable for this feature.

-------------------------------------------------------------------------------
BLOG
-------------------------------------------------------------------------------
http://cghuyue.blogspot.com/2012/11/hw3-raterizer.html
-------------------------------------------------------------------------------
About RasterizationCore
-------------------------------------------------------------------------------
still working to the eliminate the bugs.
-------------------------------------------------------------------------------
Optional Features
-------------------------------------------------------------------------------
* Back-face culling
* Interactive Camera
