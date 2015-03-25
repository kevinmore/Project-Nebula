#pragma once
#include "CUDAVector.h"

/*
 * 1D B-spline falloff
 * d is the distance from the point to the node center,
 * normalized by h such that particles <1 grid cell away
 * will have 0<d<1, particles >1 and <2 grid cells away will
 * still get some weight, and any particles further than that get
 * weight =0
 */
static __host__ __device__ __forceinline__ float funcN(float d)
{
	return  ((0 <= d && d < 1) * (.5*d*d*d - d*d + 2.f / 3.f) + 
		(1 <= d && d < 2) * (-1.f / 6.f*d*d*d + d*d - 2 * d + 4.f / 3.f));
}

/*
 * sets w = interpolation weights (w_ip)
 * input is dx because we'd rather pre-compute abs outside so we can re-use again
 * in the weightGradient function.
 * by paper notation, w_ip = N_{i}^{h}(p) = N((xp-ih)/h)N((yp-jh)/h)N((zp-kh)/h)
 */
__host__ __device__ __forceinline__ void weight( CUDAVec3 &dx, float h, float &w )
{
	w = funcN(dx.x / h) * funcN(dx.y / h) * funcN(dx.z / h);
}

// Same as above, but dx is already normalized with h
__host__ __device__ __forceinline__ void weight( CUDAVec3 &dx, float &w )
{
	w = funcN(dx.x) * funcN(dx.y) * funcN(dx.z);
}

__host__ __device__ __forceinline__ float weight( CUDAVec3 &dx)
{
	return funcN(dx.x) * funcN(dx.y) * funcN(dx.z);
}

/*
 * derivative of N with respect to d
 */
static __host__ __device__ __forceinline__ float funcNd(float _d)
{
	return ((0 <= _d && _d < 1) * (1.5f*_d*_d - 2 * _d) +
		(1 <= _d && _d < 2) * (-.5*_d*_d + 2 * _d - 2));
}

/*
 * returns gradient of interpolation weights  \grad{w_ip}
 * xp = sign( distance from grid node to particle )
 * dx = abs( distance from grid node to particle )
 */
__host__ __device__ __forceinline__ void weightGradient( const CUDAVec3 &sdx, const CUDAVec3 &dx, float h, CUDAVec3 &wg )
{
    const CUDAVec3 dx_h = dx / h;
	const CUDAVec3 N = CUDAVec3(funcN(dx_h.x), funcN(dx_h.y), funcN(dx_h.z));
	const CUDAVec3 Nx = sdx * CUDAVec3(funcNd(dx_h.x), funcNd(dx_h.y), funcNd(dx_h.z));
    wg.x = Nx.x * N.y * N.z;
    wg.y = N.x  * Nx.y* N.z;
    wg.z = N.x  * N.y * Nx.z;
}

// Same as above, but dx is already normalized with h
__host__ __device__ __forceinline__ void weightGradient( const CUDAVec3 &sdx, const CUDAVec3 &dx, CUDAVec3 &wg )
{
	const CUDAVec3 N = CUDAVec3(funcN(dx.x), funcN(dx.y), funcN(dx.z));
	const CUDAVec3 Nx = sdx * CUDAVec3(funcNd(dx.x), funcNd(dx.y), funcNd(dx.z));
    wg.x = Nx.x * N.y * N.z;
    wg.y = N.x  * Nx.y* N.z;
    wg.z = N.x  * N.y * Nx.z;
}

// Same as above, but dx is not already absolute-valued
__host__ __device__ __forceinline__ void weightGradient( const CUDAVec3 &dx, CUDAVec3 &wg )
{
    const CUDAVec3 sdx = CUDAVec3::sign( dx );
    const CUDAVec3 adx = CUDAVec3::abs( dx );
	const CUDAVec3 N = CUDAVec3(funcN(adx.x), funcN(adx.y), funcN(adx.z));
	const CUDAVec3 Nx = sdx * CUDAVec3(funcNd(adx.x), funcNd(adx.y), funcNd(adx.z));
    wg.x = Nx.x * N.y * N.z;
    wg.y = N.x  * Nx.y* N.z;
    wg.z = N.x  * N.y * Nx.z;
}

/*
 * returns weight and gradient of weight, avoiding duplicate computations if applicable
 */
__host__ __device__ __forceinline__ void weightAndGradient( const CUDAVec3 &sdx, const CUDAVec3 &dx, float h, float &w, CUDAVec3 &wg )
{
    const CUDAVec3 dx_h = dx / h;
	const CUDAVec3 N = CUDAVec3(funcN(dx_h.x), funcN(dx_h.y), funcN(dx_h.z));
    w = N.x * N.y * N.z;
	const CUDAVec3 Nx = sdx * CUDAVec3(funcNd(dx_h.x), funcNd(dx_h.y), funcNd(dx_h.z));
    wg.x = Nx.x * N.y * N.z;
    wg.y = N.x  * Nx.y* N.z;
    wg.z = N.x  * N.y * Nx.z;
}

// Same as above, but dx is already normalized with h
__host__ __device__ __forceinline__ void weightAndGradient( const CUDAVec3 &sdx, const CUDAVec3 &dx, float &w, CUDAVec3 &wg )
{
	const CUDAVec3 N = CUDAVec3(funcN(dx.x), funcN(dx.y), funcN(dx.z));
    w = N.x * N.y * N.z;
	const CUDAVec3 Nx = sdx * CUDAVec3(funcNd(dx.x), funcNd(dx.y), funcNd(dx.z));
    wg.x = Nx.x * N.y * N.z;
    wg.y = N.x  * Nx.y* N.z;
    wg.z = N.x  * N.y * Nx.z;
}

// Same as above, but dx is not already absolute-valued
__host__ __device__ __forceinline__ void weightAndGradient( const CUDAVec3 &dx, float &w, CUDAVec3 &wg )
{
    const CUDAVec3 sdx = CUDAVec3::sign( dx );
    const CUDAVec3 adx = CUDAVec3::abs( dx );
	const CUDAVec3 N = CUDAVec3(funcN(adx.x), funcN(adx.y), funcN(adx.z));
    w = N.x * N.y * N.z;
	const CUDAVec3 Nx = sdx * CUDAVec3(funcNd(adx.x), funcNd(adx.y), funcNd(adx.z));
    wg.x = Nx.x * N.y * N.z;
    wg.y = N.x  * Nx.y* N.z;
    wg.z = N.x  * N.y * Nx.z;
}