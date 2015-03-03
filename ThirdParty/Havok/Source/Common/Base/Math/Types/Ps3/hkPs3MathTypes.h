/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_MATH_PS3_MATH_TYPES_H
#define HK_MATH_PS3_MATH_TYPES_H

#include <vec_types.h>
#if defined(__PPU__)
#include <altivec.h>
#include <ppu_intrinsics.h>
#include <fastmath.h>
#include <spu2vmx.h>
#else
#include <spu_intrinsics.h>
#include <vmx2spu.h>
#endif

// Share double types from Fpu.
#include <Common/Base/Math/Types/Fpu/hkFpuDoubleMathTypes.h>

enum hkVector4Perm2
{
	HK_VECTOR4_PERM2_X = 0x00010203,
	HK_VECTOR4_PERM2_Y = 0x04050607,
	HK_VECTOR4_PERM2_Z = 0x08090a0b,
	HK_VECTOR4_PERM2_W = 0x0c0d0e0f,
	HK_VECTOR4_PERM2_A = 0x10111213,
	HK_VECTOR4_PERM2_B = 0x14151617,
	HK_VECTOR4_PERM2_C = 0x18191a1b,
	HK_VECTOR4_PERM2_D = 0x1c1d1e1f,
	HK_VECTOR4_PERM2__ = 0x00010203  // don't care (= X to avoid NaNs)
};

#define VPERMWI_CONST(_a,_b,_c,_d) ((vector unsigned char)(vector unsigned int){HK_VECTOR4_PERM2_##_a, HK_VECTOR4_PERM2_##_b, HK_VECTOR4_PERM2_##_c,HK_VECTOR4_PERM2_##_d })

// transpose 4 hkVector4
#define HK_TRANSPOSE4f(v0,v1,v2,v3) { \
	const hkQuadFloat32 tmp0 = vec_mergeh( v0.m_quad, v2.m_quad ); \
	const hkQuadFloat32 tmp1 = vec_mergeh( v1.m_quad, v3.m_quad ); \
	const hkQuadFloat32 tmp2 = vec_mergel( v0.m_quad, v2.m_quad ); \
	const hkQuadFloat32 tmp3 = vec_mergel( v1.m_quad, v3.m_quad ); \
	v0.m_quad = vec_mergeh( tmp0, tmp1 ); \
	v1.m_quad = vec_mergel( tmp0, tmp1 ); \
	v2.m_quad = vec_mergeh( tmp2, tmp3 ); \
	v3.m_quad = vec_mergel( tmp2, tmp3 ); }

// transpose 3 hkVector4: w component is undefined
#define HK_TRANSPOSE3f(v0,v1,v2) { \
	const hkQuadFloat32 tmp0 = vec_mergeh( v0.m_quad, v2.m_quad ); \
	const hkQuadFloat32 tmp1 = vec_mergel( v0.m_quad, v2.m_quad ); \
	v0.m_quad = vec_mergeh( tmp0, v1.m_quad ); \
	v2.m_quad = vec_perm( tmp1, v1.m_quad, VPERMWI_CONST(X,C,Y,_) ); \
	v1.m_quad = vec_perm( tmp0, v1.m_quad, VPERMWI_CONST(Z,B,W,_) ); }


// storage type for hkVector4 (and thus hkQuaternion)
typedef vector float hkQuadFloat32;

// storage type for hkSimdReal
typedef vector float hkSingleFloat32;

// storage type for hkVector4Comparison
typedef vector unsigned int hkVector4fMask;

// storage type for hkIntVector
typedef vector unsigned int hkQuadUint;

struct hkSingleInt128
{
	public:

#if ( HK_ENDIAN_LITTLE )
		// Unsigned int access
		template <int I> HK_FORCE_INLINE hkUint32 getU32() const	{	return u.u32[I];	}
		HK_FORCE_INLINE hkUint32 getU32(int I) const				{	return u.u32[I];	}
		template <int I> HK_FORCE_INLINE void setU32(hkUint32 x)	{	u.u32[I] = x;		}
		HK_FORCE_INLINE void setU32(int I, hkUint32 x)				{	u.u32[I] = x;		}

		// Signed int access
		template <int I> HK_FORCE_INLINE hkInt32 getS32() const		{	return u.i32[I];	}
		HK_FORCE_INLINE hkInt32 getS32(int I) const					{	return u.i32[I];	}
		template <int I> HK_FORCE_INLINE void setS32(hkInt32 x)		{	u.i32[I] = x;		}
		HK_FORCE_INLINE void setS32(int I, hkInt32 x)				{	u.i32[I] = x;		}

		// Unsigned long access
		template <int I> HK_FORCE_INLINE hkUint64 getU64() const	{	return u.u64[I];	}
		HK_FORCE_INLINE hkUint64 getU64(int I) const				{	return u.u64[I];	}
		template <int I> HK_FORCE_INLINE void setU64(hkUint64 x)	{	u.u64[I] = x;		}
		HK_FORCE_INLINE void setU64(int I, hkUint64 x)				{	u.u64[I] = x;		}

		// Signed long access
		template <int I> HK_FORCE_INLINE hkInt64 getS64() const		{	return u.i64[I];	}
		HK_FORCE_INLINE hkInt64 getS64(int I) const					{	return u.i64[I];	}
		template <int I> HK_FORCE_INLINE void setS64(hkInt64 x)		{	u.i64[I] = x;		}
		HK_FORCE_INLINE void setS64(int I, hkInt64 x)				{	u.i64[I] = x;		}
#endif

#if ( HK_ENDIAN_BIG )
		// Unsigned int access
		template <int I> HK_FORCE_INLINE hkUint32 getU32() const	{	return u.u32[3 - I];	}
		HK_FORCE_INLINE hkUint32 getU32(int I) const				{	return u.u32[3 - I];	}
		template <int I> HK_FORCE_INLINE void setU32(hkUint32 x)	{	u.u32[3 - I] = x;		}
		HK_FORCE_INLINE void setU32(int I, hkUint32 x)				{	u.u32[3 - I] = x;		}

		// Signed int access
		template <int I> HK_FORCE_INLINE hkInt32 getS32() const		{	return u.i32[3 - I];	}
		HK_FORCE_INLINE hkInt32 getS32(int I) const					{	return u.i32[3 - I];	}
		template <int I> HK_FORCE_INLINE void setS32(hkInt32 x)		{	u.i32[3 - I] = x;		}
		HK_FORCE_INLINE void setS32(int I, hkInt32 x)				{	u.i32[3 - I] = x;		}

		// Unsigned long access
		template <int I> HK_FORCE_INLINE hkUint64 getU64() const	{	return u.u64[1 - I];	}
		HK_FORCE_INLINE hkUint64 getU64(int I) const				{	return u.u64[1 - I];	}
		template <int I> HK_FORCE_INLINE void setU64(hkUint64 x)	{	u.u64[1 - I] = x;		}
		HK_FORCE_INLINE void setU64(int I, hkUint64 x)				{	u.u64[1 - I] = x;		}

		// Signed long access
		template <int I> HK_FORCE_INLINE hkInt64 getS64() const		{	return u.i64[1 - I];	}
		HK_FORCE_INLINE hkInt64 getS64(int I) const					{	return u.i64[1 - I];	}
		template <int I> HK_FORCE_INLINE void setS64(hkInt64 x)		{	u.i64[1 - I] = x;		}
		HK_FORCE_INLINE void setS64(int I, hkInt64 x)				{	u.i64[1 - I] = x;		}
#endif

	protected:
		
		union
		{
			HK_ALIGN_REAL(hkUint32 u32[4]);
			HK_ALIGN_REAL(hkInt32 i32[4]);
			HK_ALIGN_REAL(hkUint64 u64[2]);
			HK_ALIGN_REAL(hkInt64 i64[2]);
		} u;
};

struct hkQuadUlong
{
	//+hk.MemoryTracker(ignore = True)
	hkSingleInt128 xy;
	hkSingleInt128 zw;
};

// argument types
class hkVector4f;
class hkVector4fComparison;
class hkSimdFloat32;
class hkQuaternionf;
class hkIntVector;

#ifdef HK_PLATFORM_SPU
typedef const hkVector4f hkVector4fParameter;
typedef const hkVector4fComparison hkVector4fComparisonParameter;
typedef const hkSimdFloat32 hkSimdFloat32Parameter;
typedef const hkQuaternionf hkQuaternionfParameter;
typedef const hkIntVector hkIntVectorParameter;
#else
typedef const hkVector4f& hkVector4fParameter;
typedef const hkVector4fComparison hkVector4fComparisonParameter;
typedef const hkSimdFloat32& hkSimdFloat32Parameter;
typedef const hkQuaternionf& hkQuaternionfParameter;
typedef const hkIntVector& hkIntVectorParameter;
#endif


// this causes problems for the optimizer, use for debug checks only
#define HK_QUADFLOAT_CONSTANT(a,b,c,d) {a,b,c,d}
#define HK_QUADINT_CONSTANT(a,b,c,d) {a,b,c,d}



// Some abstractions for often uses vector calls
#if defined(HK_PLATFORM_PS3_PPU)
#	define vec_mul(_a,_b) ((vec_float4)(si_fm((qword)(_a), (qword)(_b))))
#	define vec_div HK_STD_NAMESPACE::divf4
#	define vec_rotl(_a,_n) vec_sld(_a,_a,_n)
#else
#	define vec_mul	spu_mul
#	define vec_div  HK_STD_NAMESPACE::divf4
#	define vec_rotl	spu_rlqwbyte
#endif

#endif // HK_MATH_PS3_MATH_TYPES_H

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
