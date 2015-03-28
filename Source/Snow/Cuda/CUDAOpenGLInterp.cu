#include <windows.h>
#include <cuda_gl_interop.h>
#include <Snow/SnowParticle.h>
#include "Functions.h"

void registerVBO( cudaGraphicsResource **resource, GLuint vbo )
{
    checkCudaErrors( cudaGraphicsGLRegisterBuffer(resource, vbo, cudaGraphicsMapFlagsNone) );
}

void unregisterVBO( cudaGraphicsResource *resource )
{
    checkCudaErrors( cudaGraphicsUnregisterResource(resource) );
}