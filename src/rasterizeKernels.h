// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include <cutil_math.h>
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

struct info
{
	glm::vec3 lightPos;
	glm::vec3 lightClr;
	glm::vec3 objClr;
	float kSpecular;
	glm::vec3 camPos;
	glm::mat4 viewM;
	glm::mat4 projM;
	glm::vec4 viewPort;
	glm::mat4 modelM;
};

void kernelCleanup();

void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize, info &otherInfo);

#endif //RASTERIZEKERNEL_H
