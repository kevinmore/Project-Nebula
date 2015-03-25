#pragma once

#include "Noise.h"
#include <Snow/SnowParticle.h>

/*
 * theta_c, theta_s -> determine when snow starts breaking.
 *          larger = chunky, wet. smaller = powdery, dry
 *
 * low xi, E0 = muddy. high xi, E0 = Icy
 * low xi = ductile, high xi = brittle
 *
 */

__global__ void applyChunky(SnowParticle *particles, int particleCount)
{
    // spatially varying constitutive parameters
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= particleCount ) return;
    SnowParticle &particle = particles[tid];
    CUDAVec3 pos = particle.position;
    float fbm = fbm3( pos * 30.f ); // adjust the .5 to get desired frequency of chunks within fbm
    SnowMaterial mat;
    mat.setYoungsAndPoissons( MIN_E0 + fbm*(MAX_E0-MIN_E0), POISSONS_RATIO );
    mat.xi = MIN_XI + fbm*(MAX_XI-MIN_XI);
    mat.setCriticalStrains( 5e-4, 1e-4 );
    particle.material = mat;
}

// hardening on the outside should be achieved with shells, so I guess this is the only spatially varying