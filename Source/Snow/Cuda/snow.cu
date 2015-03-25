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

//__global__ void snow_kernel( float time, Particle *particles )
//{
//    int index = blockIdx.x*blockDim.x + threadIdx.x;
//    vec3 pn = vec3::normalize( particles[index].position );
//    particles[index].position += 0.05f*sinf(6*time)*pn;
////    particles[index].position += 0.01f*pn;
//}

//void updateParticles( Particle *particles, float time, int particleCount )
//{
//    snow_kernel<<< particleCount/512, 512 >>>( time, particles );
//    checkCudaErrors( cudaDeviceSynchronize() );
//}

