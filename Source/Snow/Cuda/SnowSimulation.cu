#include <Snow/Caches.h>
#include <Snow/ImplicitCollider.h>
#include <Snow/SnowParticle.h>
#include <Snow/GridNode.h>

#include "CUDAHelpers.h"
#include "Atomic.h"
#include "Collider.h"
#include "Decomposition.h"
#include "Implicit.h"
#include "Weighting.h"
#include "Functions.h"

#define ALPHA 0.05f

#define GRAVITY CUDAVec3(0.f,-9.8f,0.f)

// Chain to compute the volume of the particle
/**
 * Part of one time operation to compute particle volumes. First rasterize particle masses to grid
 *
 * Operation done over Particles over grid node particle affects
 */
__global__ void computeNodeMasses( const SnowParticle *particles, int numParticles, const Grid *grid, float *nodeMasses )
{
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    if ( particleIdx >= numParticles ) return;

    const SnowParticle &particle = particles[particleIdx];

    glm::ivec3 currIJK;
    Grid::gridIndexToIJK( threadIdx.y, glm::ivec3(4,4,4), currIJK );
    CUDAVec3 particleGridPos = (particle.position - grid->pos) / grid->h;
    currIJK += glm::ivec3(particleGridPos-1);

    if ( Grid::withinBoundsInclusive(currIJK, glm::ivec3(0,0,0), grid->dim) ) {
        CUDAVec3 nodePosition(currIJK);
        CUDAVec3 dx = CUDAVec3::abs( particleGridPos - nodePosition );
        float w = weight( dx );
        atomicAdd( &nodeMasses[Grid::getGridIndex(currIJK, grid->dim+1)], particle.mass*w );
     }
}

/**
 * Computes the particle's density * grid's volume. This needs to be separate from computeCellMasses(...) because
 * we need to wait for ALL threads to sync before computing the density
 *
 * Operation done over Particles over grid node particle affects
 */
__global__ void computeParticleDensity( SnowParticle *particles, int numParticles, const Grid *grid, const float *cellMasses )
{
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    if ( particleIdx >= numParticles ) return;

    SnowParticle &particle = particles[particleIdx];

    glm::ivec3 currIJK;
    Grid::gridIndexToIJK( threadIdx.y, glm::ivec3(4,4,4), currIJK );
    CUDAVec3 particleGridPos = ( particle.position - grid->pos ) / grid->h;
    currIJK += glm::ivec3(particleGridPos-1);

    if ( Grid::withinBoundsInclusive(currIJK, glm::ivec3(0,0,0), grid->dim) ) {
        CUDAVec3 nodePosition(currIJK);
        CUDAVec3 dx = CUDAVec3::abs( particleGridPos - nodePosition );
        float w = weight( dx );
        float gridVolume = grid->h * grid->h * grid->h;
        atomicAdd( &particle.volume, cellMasses[Grid::getGridIndex(currIJK, grid->dim+1)] * w / gridVolume ); //fill volume with particle density. Then in final step, compute volume
     }
}


/**
 * Computes the particle's volume. Assumes computeParticleDensity(...) has just been called.
 *
 * Operation done over particles
 */
__global__ void computeParticleVolume( SnowParticle *particleData, int numParticles )
{
    int particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( particleIdx >= numParticles ) return;
    SnowParticle &particle = particleData[particleIdx];
    particle.volume = particle.mass / particle.volume; // Note: particle.volume is assumed to be the (particle's density ) before we compute it correctly
}

__host__ void initializeParticleVolumes( SnowParticle *particles, int numParticles, const Grid *grid, int numNodes )
{
    float *devNodeMasses;
    checkCudaErrors( cudaMalloc( (void**)&devNodeMasses, numNodes*sizeof(float) ) );
    cudaMemset( devNodeMasses, 0, numNodes*sizeof(float) );

    const dim3 blocks( (numParticles+THREAD_COUNT-1)/THREAD_COUNT, 64 );
    static const dim3 threads( THREAD_COUNT / 64, 64 );

    LAUNCH( computeNodeMasses<<<blocks,threads>>>(particles,numParticles,grid,devNodeMasses) );

    LAUNCH( computeParticleDensity<<<blocks,threads>>>(particles,numParticles,grid,devNodeMasses) );

    LAUNCH( computeParticleVolume<<<(numParticles+THREAD_COUNT-1)/THREAD_COUNT,THREAD_COUNT>>>(particles,numParticles) );

    checkCudaErrors( cudaFree(devNodeMasses) );
}

__global__ void computeSigma( const SnowParticle *particles, SnowParticleCache *particleCache, int numParticles, const Grid *grid )
{
    int particleIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if ( particleIdx >= numParticles ) return;

    const SnowParticle &particle = particles[particleIdx];

    const CUDAMat3 &Fp = particle.plasticF; //for the sake of making the code look like the math
    const CUDAMat3 &Fe = particle.elasticF;

    float Jpp = CUDAMat3::determinant(Fp);
    float Jep = CUDAMat3::determinant(Fe);

    CUDAMat3 Re;
    computePD( Fe, Re );

    const SnowMaterial material = particle.material;

    float muFp = material.mu*expf(material.xi*(1-Jpp));
    float lambdaFp = material.lambda*expf(material.xi*(1-Jpp));

    particleCache->sigmas[particleIdx] = (2*muFp*CUDAMat3::multiplyABt(Fe-Re, Fe) + CUDAMat3(lambdaFp*(Jep-1)*Jep)) * -particle.volume;
}

/**
 * Called on each particle.
 *
 * Each particle adds it's mass, velocity and force contribution to the grid nodes within 2h of itself.
 *
 * In:
 * particleData -- list of particles
 * grid -- Stores grid paramters
 * worldParams -- Global parameters dealing with the physics of the world
 *
 * Out:
 * nodes -- list of every node in grid ((dim.x+1)*(dim.y+1)*(dim.z+1))
 *
 */
__global__ void computeCellMassVelocityAndForceFast( const SnowParticle *particleData, const SnowParticleCache *particleCache, int numParticles, const Grid *grid, Node *nodes )
{
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    if ( particleIdx >= numParticles ) return;

    const SnowParticle &particle = particleData[particleIdx];

    glm::ivec3 currIJK;
    Grid::gridIndexToIJK(threadIdx.y, glm::ivec3(4,4,4), currIJK);
    CUDAVec3 particleGridPos = (particle.position-grid->pos)/grid->h;
    currIJK += glm::ivec3( particleGridPos-1 );

    if ( Grid::withinBoundsInclusive(currIJK, glm::ivec3(0,0,0), grid->dim) ) {
        Node &node = nodes[Grid::getGridIndex(currIJK, grid->dim+1)];

        float w;
        CUDAVec3 wg;
        CUDAVec3 nodePosition(currIJK.x, currIJK.y, currIJK.z);
        weightAndGradient( particleGridPos - nodePosition, w, wg );

        atomicAdd( &node.mass, particle.mass*w );
        atomicAdd( &node.velocity, particle.velocity*particle.mass*w );
        atomicAdd( &node.force, particleCache->sigmas[particleIdx]*wg );
     }
}

/**
 * Called on each grid node.
 *
 * Updates the velocities of each grid node based on forces and collisions
 *
 * In:
 * nodes -- list of all nodes in the grid.
 * dt -- delta time, time step of simulation
 * colliders -- array of colliders in the scene.
 * numColliders -- number of colliders in the scene
 * worldParams -- Global parameters dealing with the physics of the world
 * grid -- parameters defining the grid
 *
 * Out:
 * nodes -- updated velocity and velocityChange
 *
 */
__global__ void updateNodeVelocities( Node *nodes, int numNodes, float dt, const ImplicitCollider* colliders, int numColliders, const Grid *grid, bool updateVelocityChange )
{
    int nodeIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if ( nodeIdx >= numNodes ) return;

    Node &node = nodes[nodeIdx];

    if ( node.mass > 0.f ) {

        // Have to normalize velocity by mass to conserve momentum
        float scale = 1.f / node.mass;
        node.velocity *= scale;

        // Initialize velocityChange with pre-update velocity
        node.velocityChange = node.velocity;

        // Gravity for node forces
        node.force += node.mass * GRAVITY;

        // Update velocity with node force
        node.velocity += dt * scale * node.force;

        // Handle collisions
        int gridI, gridJ, gridK;
        Grid::gridIndexToIJK( nodeIdx, gridI, gridJ, gridK, grid->dim+1 );
        CUDAVec3 nodePosition = CUDAVec3(gridI, gridJ, gridK)*grid->h + grid->pos;
        checkForAndHandleCollisions( colliders, numColliders, nodePosition, node.velocity );

        if ( updateVelocityChange ) node.velocityChange = node.velocity - node.velocityChange;

    }
}

// Use weighting functions to compute particle velocity gradient and update particle velocity
__device__ void processGridVelocities( SnowParticle &particle, const Grid *grid, const Node *nodes, CUDAMat3 &velocityGradient )
{
    const CUDAVec3 &pos = particle.position;
    const glm::ivec3 &dim = grid->dim;
    const float h = grid->h;

    // Compute neighborhood of particle in grid
    CUDAVec3 particleGridPos = (pos - grid->pos) / h,
         gridMax = CUDAVec3::floor( particleGridPos + CUDAVec3(2,2,2) ),
         gridMin = CUDAVec3::ceil( particleGridPos - CUDAVec3(2,2,2) );
    glm::ivec3 maxIndex = glm::clamp( glm::ivec3(gridMax), glm::ivec3(0,0,0), dim ),
               minIndex = glm::clamp( glm::ivec3(gridMin), glm::ivec3(0,0,0), dim );

    // For computing particle velocity gradient:
    //      grad(v_p) = sum( v_i * transpose(grad(w_ip)) ) = [3x3 matrix]
    // For updating particle velocity:
    //      v_PIC = sum( v_i * w_ip )
    //      v_FLIP = v_p + sum( dv_i * w_ip )
    //      v = (1-alpha)*v_PIC _ alpha*v_FLIP
    CUDAVec3 v_PIC(0,0,0), dv_FLIP(0,0,0);
    int rowSize = dim.z+1;
    int pageSize = (dim.y+1)*rowSize;
    for ( int i = minIndex.x; i <= maxIndex.x; ++i ) {
        CUDAVec3 d, s;
        d.x = particleGridPos.x - i;
        d.x *= ( s.x = ( d.x < 0 ) ? -1.f : 1.f );
        int pageOffset = i*pageSize;
        for ( int j = minIndex.y; j <= maxIndex.y; ++j ) {
            d.y = particleGridPos.y - j;
            d.y *= ( s.y = ( d.y < 0 ) ? -1.f : 1.f );
            int rowOffset = pageOffset + j*rowSize;
            for ( int k = minIndex.z; k <= maxIndex.z; ++k ) {
                d.z = particleGridPos.z - k;
                d.z *= ( s.z = ( d.z < 0 ) ? -1.f : 1.f );
                const Node &node = nodes[rowOffset+k];
                float w;
                CUDAVec3 wg;
                weightAndGradient( s, d, w, wg );
                velocityGradient += CUDAMat3::outerProduct( node.velocity, wg );
                // Particle velocities
                v_PIC += node.velocity * w;
                dv_FLIP += node.velocityChange * w;
            }
        }
    }
    particle.velocity = (1.f-ALPHA)*v_PIC + ALPHA*(particle.velocity+dv_FLIP);
}

__device__ void updateParticleDeformationGradients( SnowParticle &particle, const CUDAMat3 &velocityGradient, float timeStep )
{
    // Temporarily assign all deformation to elastic portion
    particle.elasticF = CUDAMat3::addIdentity( timeStep*velocityGradient ) * particle.elasticF;
    const SnowMaterial &material = particle.material;
    // Clamp the singular values
    CUDAMat3 W, S, Sinv, V;
    computeSVD( particle.elasticF, W, S, V );

    // FAST COMPUTATION:
    S = CUDAMat3( CLAMP( S[0], material.criticalCompressionRatio, material.criticalStretchRatio ), 0.f, 0.f,
              0.f, CLAMP( S[4], material.criticalCompressionRatio, material.criticalStretchRatio ), 0.f,
              0.f, 0.f, CLAMP( S[8], material.criticalCompressionRatio, material.criticalStretchRatio ) );
    Sinv = CUDAMat3( 1.f/S[0], 0.f, 0.f,
                 0.f, 1.f/S[4], 0.f,
                 0.f, 0.f, 1.f/S[8] );
    particle.plasticF = CUDAMat3::multiplyADBt( V, Sinv, W ) * particle.elasticF * particle.plasticF;
    particle.elasticF = CUDAMat3::multiplyADBt( W, S, V );

//     // MORE ACCURATE COMPUTATION:
//    S[0] = CLAMP( S[0], material->criticalCompressionRatio, material->criticalStretchRatio );
//    S[4] = CLAMP( S[4], material->criticalCompressionRatio, material->criticalStretchRatio );
//    S[8] = CLAMP( S[8], material->criticalCompressionRatio, material->criticalStretchRatio );
//    particle.plasticF = V * CUDAMat3::inverse( S ) * CUDAMat3::transpose( W ) * particle.elasticF * particle.plasticF;
//    particle.elasticF = W * S * CUDAMat3::transpose( V );

}

__global__ void updateParticlesFromGrid( SnowParticle *particles, int numParticles, const Grid *grid, const Node *nodes, float timeStep, const ImplicitCollider *colliders, int numColliders )
{
    int particleIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( particleIdx >= numParticles ) return;

    SnowParticle &particle = particles[particleIdx];

    // Update particle velocities and fill in velocity gradient for deformation gradient computation
    CUDAMat3 velocityGradient = CUDAMat3( 0.f );
    processGridVelocities( particle, grid, nodes, velocityGradient );

    updateParticleDeformationGradients( particle, velocityGradient, timeStep );

    checkForAndHandleCollisions( colliders, numColliders, particle.position, particle.velocity );

    particle.position += timeStep * ( particle.velocity );
}

__global__ void updateColliderPositions(ImplicitCollider *colliders, int numColliders,float timestep)
{
    int colliderIdx = blockDim.x*blockIdx.x + threadIdx.x;
    colliders[colliderIdx].center += colliders[colliderIdx].velocity*timestep;
}

__host__ void updateParticles( SnowParticle *particles, SnowParticleCache *devParticleCache, SnowParticleCache *hostParticleCache, int numParticles,
                               Grid *grid, Node *nodes, NodeCache *nodeCaches, int numNodes,
                               ImplicitCollider *colliders, int numColliders,
                               float timeStep, bool implicitUpdate )
{
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

    // Clear data before update
    checkCudaErrors( cudaMemset(nodes, 0, numNodes*sizeof(Node)) );
    checkCudaErrors( cudaMemset(nodeCaches, 0, numNodes*sizeof(NodeCache)) );

    // All dat SnowParticleCache data
    cudaMemset( hostParticleCache->sigmas, 0, numParticles*sizeof(CUDAMat3) );
    cudaMemset( hostParticleCache->Aps, 0, numParticles*sizeof(CUDAMat3) );
    cudaMemset( hostParticleCache->FeHats, 0, numParticles*sizeof(CUDAMat3) );
    cudaMemset( hostParticleCache->ReHats, 0, numParticles*sizeof(CUDAMat3) );
    cudaMemset( hostParticleCache->SeHats, 0, numParticles*sizeof(CUDAMat3) );
    cudaMemset( hostParticleCache->dFs, 0, numParticles*sizeof(CUDAMat3) );

    const dim3 pBlocks1D( (numParticles+THREAD_COUNT-1)/THREAD_COUNT );
    const dim3 nBlocks1D( (numNodes+THREAD_COUNT-1)/THREAD_COUNT );
    const dim3 threads1D( THREAD_COUNT );
    const dim3 pBlocks2D( (numParticles+THREAD_COUNT-1)/THREAD_COUNT, 64 );
    const dim3 threads2D( THREAD_COUNT/64, 64 );

    LAUNCH( updateColliderPositions<<<numColliders,1>>>(colliders,numColliders,timeStep) );

    LAUNCH( computeSigma<<<pBlocks1D,threads1D>>>(particles,devParticleCache,numParticles,grid) );

    LAUNCH( computeCellMassVelocityAndForceFast<<<pBlocks2D,threads2D>>>(particles,devParticleCache,numParticles,grid,nodes) );

    LAUNCH( updateNodeVelocities<<<nBlocks1D,threads1D>>>(nodes,numNodes,timeStep,colliders,numColliders,grid,!implicitUpdate) );

    if ( implicitUpdate ) integrateNodeForces( particles, devParticleCache, numParticles, grid, nodes, nodeCaches, numNodes, timeStep );

    LAUNCH( updateParticlesFromGrid<<<pBlocks1D,threads1D>>>(particles,numParticles,grid,nodes,timeStep,colliders,numColliders) );
}
