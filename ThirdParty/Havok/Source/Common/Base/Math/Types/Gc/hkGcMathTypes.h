/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_MATH_GC_MATH_TYPES_H
#define HK_MATH_GC_MATH_TYPES_H

// Share double types from Fpu.
#include <Common/Base/Math/Types/Fpu/hkFpuDoubleMathTypes.h>

// transpose 4 hkVector4
#define HK_TRANSPOSE4f_SWAP(a,b) __tmp = a; a = b; b = __tmp
#define HK_TRANSPOSE4f(v0,v1,v2,v3) { hkFloat32 __tmp; \
	HK_TRANSPOSE4f_SWAP( v0(1), v1(0) ); \
	HK_TRANSPOSE4f_SWAP( v0(2), v2(0) ); \
	HK_TRANSPOSE4f_SWAP( v0(3), v3(0) ); \
	HK_TRANSPOSE4f_SWAP( v1(2), v2(1) ); \
	HK_TRANSPOSE4f_SWAP( v1(3), v3(1) ); \
	HK_TRANSPOSE4f_SWAP( v2(3), v3(2) ); }

// transpose 3 hkVector4: w component is undefined
#define HK_TRANSPOSE3f_SWAP(a,b) __tmp = a; a = b; b = __tmp
#define HK_TRANSPOSE3f(v0,v1,v2) { hkDouble64 __tmp; \
	HK_TRANSPOSE3f_SWAP( v0(1), v1(0) ); \
	HK_TRANSPOSE3f_SWAP( v0(2), v2(0) ); \
	HK_TRANSPOSE3f_SWAP( v1(2), v2(1) ); }

// hardware implemented vector of two single precision floats
typedef __vec2x32float__ hkPairedSingleFloat;

// storage type for hkVector4 (and thus hkQuaternion)
struct hkQuadFloat32
{	
	HK_ALIGN16(hkPairedSingleFloat m_a);
	hkPairedSingleFloat m_b;	
};


// storage type for hkSimdReal
typedef hkFloat32 hkSingleFloat32;

// storage type for hkVector4Comparison
typedef hkInt32 hkVector4fMask;

// storage type for hkIntVector
union hkQuadUint 
{
	//+hk.MemoryTracker(ignore = True)
	HK_ALIGN_REAL(hkUint32 u32[4]);
	HK_ALIGN_REAL(hkInt32 i32[4]);
};

struct hkSingleInt128
{
public:

	// always HK_ENDIAN_BIG 
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
typedef const hkVector4f& hkVector4fParameter;

class hkSimdFloat32;
typedef const hkSimdFloat32 hkSimdFloat32Parameter;

class hkVector4fComparison;
typedef const hkVector4fComparison hkVector4fComparisonParameter;

class hkQuaternionf;
typedef const hkQuaternionf& hkQuaternionfParameter;

class hkIntVector;
typedef const hkIntVector& hkIntVectorParameter;

// this causes problems for the optimizer, use for debug checks only
#define HK_QUADFLOAT_CONSTANT(a,b,c,d) ((hkQuadFloat32Union){{a,b,c,d}}).q

#define HK_QUADINT_CONSTANT(a, b, c, d)		{(a), (b), (c), (d)}

#endif // HK_MATH_GC_MATH_TYPES_H

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
