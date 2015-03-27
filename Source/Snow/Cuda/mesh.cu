#include <windows.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "CUDAHelpers.h"
#include "CUDAVector.h"
#include "Functions.h"
#include "Noise.h"
#include "SnowTypes.h"

#include <Snow/SnowParticle.h>
#include <Snow/Grid.h>

// struct Tri {
// 	CUDAVec3 v0, n0;
// 	CUDAVec3 v1, n1;
// 	CUDAVec3 v2, n2;
// };

struct Tri 
{
	CUDAVec3 v0, v1, v2;
};

/*
 * Moller, T, and Trumbore, B. Fast, Minimum Storage Ray/Triangle Intersection.
 */
__device__ int intersectTri(const CUDAVec3 &v1, const CUDAVec3 &v2, const CUDAVec3 &v3,
                            const CUDAVec3 &O, const CUDAVec3 &D, float &t)
{
    CUDAVec3 e1, e2;  //Edge1, Edge2
    CUDAVec3 P, Q, T;
    float det, inv_det, u, v;
    e1 = v2-v1;
    e2 = v3-v1;
    P = CUDAVec3::cross(D,e2);
    det = CUDAVec3::dot(e1,P);
    if(det > -1e-8 && det < 1e-8) return 0;
    inv_det = 1.f / det;
    T = O-v1;
    u = CUDAVec3::dot(T, P) * inv_det;
    if(u < 0.f || u > 1.f) return 0;
    Q = CUDAVec3::cross(T, e1);
    v = CUDAVec3::dot(D,Q)*inv_det;
    if(v < 0.f || u + v  > 1.f) return 0;
    t = CUDAVec3::dot(e2, Q) * inv_det;
    if(t > 1e-8) { //ray intersection
        return 1;
    }
    // No hit, no win
    return 0;
}


__global__ void voxelizeMeshKernel( Tri *tris, int triCount, Grid grid, bool *flags )
{
    const glm::ivec3 &dim = grid.dim;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if ( x >= dim.x ) return;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ( y >= dim.y ) return;

    // Shoot ray in z-direction
    CUDAVec3 origin = grid.pos + grid.h * CUDAVec3( x+0.5f, y+0.5f, 0.f );
    CUDAVec3 direction = CUDAVec3( 0.f, 0.f, 1.f );

    // Flag surface-intersecting voxels
    float t;
    int xyOffset = x*dim.y*dim.z + y*dim.z, z;
    for ( int i = 0; i < triCount; ++i ) {
        const Tri &tri = tris[i];
        if ( intersectTri(tri.v0, tri.v1, tri.v2, origin, direction, t) ) {
            z = (int)(t/grid.h);
            flags[xyOffset+z] = true;
        }
    }

    // Scanline to fill inner voxels
    int end = xyOffset + dim.z, zz;
    for ( int z = xyOffset; z < end; ++z ) {
        if ( flags[z] ) {
            do { z++; } while ( flags[z] && z < end );
            zz = z;
            do { zz++; } while ( !flags[zz] && zz < end );
            if ( zz < end - 1 ) {
                for ( int i = z; i < zz; ++i ) flags[i] = true;
                z = zz;
            } else break;
        }
    }

}

__global__ void initReduction( bool *flags, int voxelCount, int *reduction, int reductionSize )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= reductionSize ) return;
    reduction[tid] = ( tid < voxelCount ) ? flags[tid] : 0;
}

__global__ void reduce( int *reduction, int size )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= size ) return;
    reduction[tid] += reduction[tid+size];
}

__global__ void fillMeshVoxelsKernel( curandState *states, unsigned int seed, Grid grid, bool *flags, SnowParticle *particles, float particleMass, int particleCount )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= particleCount ) return;

    curandState &localState = states[tid];
    curand_init( seed, tid, 0, &localState );

    const glm::ivec3 &dim = grid.dim;

    // Rejection sample
    unsigned int i;
    unsigned int voxelCount = dim.x * dim.y * dim.z;
    do { i = curand(&localState) % voxelCount; } while ( !flags[i] );

    // Get 3D voxel index
    unsigned int x = i / (dim.y*dim.z);
    unsigned int y = (i - x*dim.y*dim.z) / dim.z;
    unsigned int z = i - y*dim.z - x*dim.y*dim.z;

    // Generate random point in voxel cube
    CUDAVec3 r = CUDAVec3( curand_uniform(&localState), curand_uniform(&localState), curand_uniform(&localState) );
    CUDAVec3 min = grid.pos + grid.h * CUDAVec3( x, y, z );
    CUDAVec3 max = min + CUDAVec3( grid.h, grid.h, grid.h );

    SnowParticle particle;
    particle.mass = particleMass;
    particle.position = min + r*(max-min);
    particle.velocity = CUDAVec3(0,-1,0);
    particle.material = SnowMaterial();
    particles[tid] = particle;
}

void fillMesh( cudaGraphicsResource **resource, int triCount, const Grid &grid, SnowParticle *particles, int particleCount, float targetDensity, int materialPreset)
{
    // Get mesh data
    cudaGraphicsMapResources( 1, resource, 0 );
    Tri *devTris;
    size_t size;
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void**)&devTris, &size, *resource) );

    // Voxelize mesh
    int x = grid.dim.x > 16 ? MAX( 1, MIN(16, grid.dim.x/8)) : 1;
    int y = grid.dim.y > 16 ? MAX( 1, MIN(16, grid.dim.y/8)) : 1;
    dim3 blocks( (grid.dim.x+x-1)/x, (grid.dim.y+y-1)/y ), threads( x, y );
    int voxelCount = grid.dim.x * grid.dim.y * grid.dim.z;
    bool *devFlags;
    checkCudaErrors( cudaMalloc((void**)&devFlags, voxelCount*sizeof(bool)) );
    checkCudaErrors( cudaMemset((void*)devFlags, 0, voxelCount*sizeof(bool)) );
    voxelizeMeshKernel<<< blocks, threads >>>( devTris, triCount, grid, devFlags );
    //checkCudaErrors( cudaDeviceSynchronize() );

    int powerOfTwo = (int)(log2f(voxelCount)+1);
    int reductionSize = 1 << powerOfTwo;
    int *devReduction;
    checkCudaErrors( cudaMalloc((void**)&devReduction, reductionSize*sizeof(int)) );
    initReduction<<< (reductionSize+511)/512, 512 >>>( devFlags, voxelCount, devReduction, reductionSize );
    //checkCudaErrors( cudaDeviceSynchronize() );
    for ( int i = 0; i < powerOfTwo-1; ++i ) {
        int size = 1 << (powerOfTwo-i-1);
        reduce<<< (size+511)/512, 512 >>>( devReduction, size );
        checkCudaErrors( cudaDeviceSynchronize() );
    }
    int count;
    checkCudaErrors( cudaMemcpy(&count, devReduction, sizeof(int), cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaFree(devReduction) );
    float volume = count*grid.h*grid.h*grid.h;
    float particleMass = targetDensity * volume / particleCount;
    printf( "Average %.2f particles per grid cell.", float(particleCount)/count );
    printf( "Target Density: %.1f kg/m3 -> Particle Mass: %g kg", targetDensity, particleMass );


    // Randomly fill mesh voxels and copy back resulting particles
    curandState *devStates;
    checkCudaErrors( cudaMalloc(&devStates, particleCount*sizeof(curandState)) );
    SnowParticle *devParticles;
    checkCudaErrors( cudaMalloc((void**)&devParticles, particleCount*sizeof(SnowParticle)) );
    fillMeshVoxelsKernel<<< (particleCount+511)/512, 512 >>>( devStates, time(NULL), grid, devFlags, devParticles, particleMass, particleCount );
    //checkCudaErrors( cudaDeviceSynchronize() );

    switch (materialPreset)
    {
    case 0:
        break;
    case 1:
        LAUNCH( applyChunky<<<(particleCount+511)/512, 512>>>(devParticles,particleCount) ); // TODO - we could use the uisettings materialstiffness here
        printf( "Chunky applied" );
        break;
    default:
        break;
    }

    checkCudaErrors( cudaMemcpy(particles, devParticles, particleCount*sizeof(SnowParticle), cudaMemcpyDeviceToHost) );

    checkCudaErrors( cudaFree(devFlags) );
    checkCudaErrors( cudaFree(devStates) );
    checkCudaErrors( cudaGraphicsUnmapResources(1, resource, 0) );
}
