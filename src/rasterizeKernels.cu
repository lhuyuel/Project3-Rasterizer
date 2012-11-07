// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <cutil_math.h>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

/////////////////////////////////////////////////////////////////////
//glm::vec3 camPos;
//glm::mat4 viewM;
//glm::mat4 projM;
//glm::vec4 viewPort;
//glm::mat4 modelM = utilityCore::buildTransformationMatrix(	glm::vec3(0.0, 0.0, 0.0),
//	glm::vec3(0.0, 0.0, 0.0),
//	glm::vec3(1.0, 1.0, 1.0));
////////////////////////////////////////////////////////////////////////////////


glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_nbo;
float* device_cbo;
int* device_ibo;
//bos device;
triangle* primitives;

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
		exit(EXIT_FAILURE); 
	}
} 

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
	a = (a+0x7ed55d16) + (a<<12);
	a = (a^0xc761c23c) ^ (a>>19);
	a = (a+0x165667b1) + (a<<5);
	a = (a+0xd3a2646c) ^ (a<<9);
	a = (a+0xfd7046c5) + (a<<3);
	a = (a^0xb55a4f09) ^ (a>>16);
	return a;
}

//Writes a given fragment to a fragment buffer at a given location
__host__ __device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution){
	if(x<resolution.x && y<resolution.y){
		int index = (y*resolution.x) + x;
		depthbuffer[index] = frag;
	}
}

//Reads a fragment from a given location in a fragment buffer
__host__ __device__ fragment getFromDepthbuffer(int x, int y, fragment* depthbuffer, glm::vec2 resolution){
	if(x<resolution.x && y<resolution.y){
		int index = (y*resolution.x) + x;
		return depthbuffer[index];
	}else{
		fragment f;
		return f;
	}
}

//Writes a given pixel to a pixel buffer at a given location
__host__ __device__ void writeToFramebuffer(int x, int y, glm::vec3 value, glm::vec3* framebuffer, glm::vec2 resolution){
	if(x<resolution.x && y<resolution.y){
		int index = (y*resolution.x) + x;
		framebuffer[index] = value;
	}
}

//Reads a pixel from a pixel buffer at a given location
__host__ __device__ glm::vec3 getFromFramebuffer(int x, int y, glm::vec3* framebuffer, glm::vec2 resolution){
	if(x<resolution.x && y<resolution.y){
		int index = (y*resolution.x) + x;
		return framebuffer[index];
	}else{
		return glm::vec3(0,0,0);
	}
}

//Kernel that clears a given pixel buffer with a given color
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image, glm::vec3 color){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y){
		image[index] = color;
	}
}

//Kernel that clears a given fragment buffer with a given fragment
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y){
		fragment f = frag;
		f.position.x = x;
		f.position.y = y;
		buffer[index] = f;
	}
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x<=resolution.x && y<=resolution.y){

		glm::vec3 color;      
		color.x = image[index].x*255.0;
		color.y = image[index].y*255.0;
		color.z = image[index].z*255.0;

		if(color.x>255){
			color.x = 255;
		}

		if(color.y>255){
			color.y = 255;
		}

		if(color.z>255){
			color.z = 255;
		}

		// Each thread writes one pixel location in the texture (textel)
		PBOpos[index].w = 0;
		PBOpos[index].x = color.x;     
		PBOpos[index].y = color.y;
		PBOpos[index].z = color.z;
	}
}

//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(float* vbo, int vbosize, info otherInfo)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < vbosize/3)
	{	
		int idx = index*3;
		glm::vec3 vert = glm::project(glm::vec3(vbo[idx], vbo[idx + 1], vbo[idx + 2]), otherInfo.viewM * otherInfo.modelM, otherInfo.projM, otherInfo.viewPort);

		vbo[idx] = vert.x;
		vbo[idx + 1] = vert.y;
		vbo[idx + 2] = vert.z;
	}
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives, info otherInfo)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int primitivesCount = ibosize/3;
	if(index<primitivesCount)
	{
		int idx = 3*index;

		primitives[index].p0 = glm::vec3( vbo[3*idx], vbo[3*idx + 1], vbo[3*idx + 2] );
		primitives[index].p1 = glm::vec3( vbo[3*(idx + 1)], vbo[3*(idx + 1) + 1], vbo[3*(idx + 1) + 2] );
		primitives[index].p2 = glm::vec3( vbo[3*(idx + 2)], vbo[3*(idx + 2) + 1], vbo[3*(idx + 2) + 2]);

		primitives[index].n0 = glm::vec3( nbo[3*idx], nbo[3 * idx + 1], nbo[3 * idx + 2]);
		primitives[index].n1 = glm::vec3( nbo[3*(idx + 1)], nbo[3*(idx + 1) + 1], nbo[3*(idx + 1) + 2] );
		primitives[index].n2 = glm::vec3( nbo[3*(idx + 2)], nbo[3*(idx + 2) + 1], nbo[3*(idx + 2) + 2] );

		primitives[index].c0 = glm::vec3( cbo[0], cbo[1], cbo[2] );
		primitives[index].c1 = glm::vec3( cbo[3], cbo[4], cbo[5] );
		primitives[index].c2 = glm::vec3( cbo[6], cbo[7], cbo[8] );

		glm::vec3 normal = glm::normalize( glm::vec3(primitives[index].n0 + primitives[index].n1 + primitives[index].n2) / 3.0f );
		glm::vec3 faceCenter = glm::unProject( (primitives[index].p0 + primitives[index].p1 + primitives[index].p1) / 3.0f, otherInfo.viewM*otherInfo.modelM, otherInfo.projM, otherInfo.viewPort);
		glm::vec3 viewDir = glm::normalize( faceCenter- otherInfo.camPos);

		//back face check
		if(glm::dot(normal, viewDir) < 0.4f )	
			primitives[index].backFaceFlag = false;
		else 
			primitives[index].backFaceFlag = true;
	}
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, info otherInfo)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < primitivesCount)
	{
		triangle curFace = primitives[index];
		if( curFace.backFaceFlag ) return;

		glm::vec3 minPoint(0.0, 0.0, 0.0);
		glm::vec3 maxPoint(0.0, 0.0, 0.0);
		getAABBForTriangle( curFace, minPoint, maxPoint);

	////////////////////////////////////////////////////////////////////////
	}
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, info otherInfo)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y)
	{
		glm::vec3 lightDir = glm::normalize(otherInfo.lightPos - depthbuffer[index].position);
		glm::vec3 cameraDir = glm::normalize(otherInfo.camPos - depthbuffer[index].position);
		glm::vec3 normal = glm::normalize(depthbuffer[index].normal);		
		glm::vec3 refelcDir = glm::normalize( cameraDir - normal*glm::dot(cameraDir, normal)*2.0f );

		float diffuseClr = max( glm::dot(normal, lightDir), .0f );
		float specularClr = max( glm::dot(refelcDir, cameraDir), .0f );
		//diffuse + ambient + specular
		glm::vec3 color = depthbuffer[index].color*diffuseClr*otherInfo.lightClr + depthbuffer[index].color*otherInfo.objClr + depthbuffer[index].color*otherInfo.lightClr*pow(specularClr, otherInfo.kSpecular);
		depthbuffer[index].color = color;
	}
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x<=resolution.x && y<=resolution.y){
		framebuffer[index] = depthbuffer[index].color;
	}
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize, info &otherInfo)
{
	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

	//set up framebuffer
	framebuffer = NULL;
	cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3));

	//set up depthbuffer
	depthbuffer = NULL;
	cudaMalloc((void**)&depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(fragment));

	//kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
	clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));

	fragment frag;
	frag.color = glm::vec3(0,0,0);
	frag.normal = glm::vec3(0,0,0);
	frag.position = glm::vec3(0,0,-10000);
	frag.iniPos = glm::vec3(0, 0, -10000);
	frag.synchroizeLock = 0;
	clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

	//------------------------------
	//memory stuff
	//------------------------------
	primitives = NULL;
	cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));

	device_ibo = NULL;
	cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
	cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

	device_vbo = NULL;
	cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
	cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

	device_nbo = NULL;
	cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
	cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

	device_cbo = NULL;
	cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
	cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

	tileSize = 32;
	int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

	//------------------------------
	//vertex shader
	//------------------------------
	vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, otherInfo);
	cudaDeviceSynchronize();
	//------------------------------
	//primitive assembly
	//------------------------------
	primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
	primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, device_cbo, cbosize, device_ibo, ibosize, primitives, otherInfo);
	cudaDeviceSynchronize();
	//------------------------------
	//rasterization
	//------------------------------
	rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, otherInfo);
	cudaDeviceSynchronize();
	//------------------------------
	//fragment shader
	//------------------------------
	fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, otherInfo);
	cudaDeviceSynchronize();
	//------------------------------
	//write fragments to framebuffer
	//------------------------------
	render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

	cudaDeviceSynchronize();

	kernelCleanup();

	checkCUDAError("Kernel failed!");
}

void kernelCleanup(){
	cudaFree( primitives );
	cudaFree( device_vbo );
	cudaFree( device_cbo );
	cudaFree( device_ibo );
	cudaFree( framebuffer );
	cudaFree( depthbuffer );
	cudaFree( device_nbo );
}