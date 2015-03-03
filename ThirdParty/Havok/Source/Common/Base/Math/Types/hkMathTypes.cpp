/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

#if defined(HK_PLATFORM_XBOX360) || defined(HK_PLATFORM_PS3_PPU) || defined(HK_PLATFORM_PS3_SPU) || (defined(HK_COMPILER_HAS_INTRINSICS_IA32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)) || (defined(HK_COMPILER_HAS_INTRINSICS_NEON) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED))


#define X unsigned(-1)
#define Y unsigned(-1)
#define Z unsigned(-1)
#define W unsigned(-1)

#	if defined(HK_PLATFORM_XBOX360)
typedef __vector4 MaskType;
inline __vector4 QUADf(unsigned a, unsigned b, unsigned c, unsigned d)
{
	__vector4 v;
	v.u[0] = a;
	v.u[1] = b;
	v.u[2] = c;
	v.u[3] = d;
	return v;
}

inline __vector4 INV_QUADf(unsigned a, unsigned b, unsigned c, unsigned d)
{
	__vector4 v;
	v.u[0] = ~a;
	v.u[1] = ~b;
	v.u[2] = ~c;
	v.u[3] = ~d;
	return v;
}
#	elif defined(HK_PLATFORM_PS3_PPU) || defined(HK_PLATFORM_PS3_SPU)

typedef vector unsigned int MaskType;
#	define QUADf(a,b,c,d) { a, b, c, d}
#	define INV_QUADf(a,b,c,d) {~a,~b,~c,~d}

#	elif (defined(HK_COMPILER_HAS_INTRINSICS_NEON) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED))

typedef uint32x4_t MaskType;
#	define QUADf(a,b,c,d) { a, b, c, d}
#	define INV_QUADf(a,b,c,d) {~a,~b,~c,~d}

#	elif (defined(HK_COMPILER_HAS_INTRINSICS_IA32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED))

typedef __m128 MaskType;

union hkQuadUint32Union
{
	hkUint32 u[4];
	hkQuadFloat32 q;
};

inline hkQuadFloat32 QUADf(unsigned a, unsigned b, unsigned c, unsigned d)
{
	hkQuadUint32Union v;
	v.u[0] = d;
	v.u[1] = c;
	v.u[2] = b;
	v.u[3] = a;
	return v.q;
}

#	endif

// const MaskType hkVector4Comparison::s_maskFromBits[hkVector4ComparisonMask::MASK_XYZW+1] =
// {
// 	QUAD(0,0,0,0),
// 	QUAD(0,0,0,W),
// 	QUAD(0,0,Z,0),
// 	QUAD(0,0,Z,W),
// 
// 	QUAD(0,Y,0,0),
// 	QUAD(0,Y,0,W),
// 	QUAD(0,Y,Z,0),
// 	QUAD(0,Y,Z,W),
// 
// 	QUAD(X,0,0,0),
// 	QUAD(X,0,0,W),
// 	QUAD(X,0,Z,0),
// 	QUAD(X,0,Z,W),
// 
// 	QUAD(X,Y,0,0),
// 	QUAD(X,Y,0,W),
// 	QUAD(X,Y,Z,0),
// 	QUAD(X,Y,Z,W)
// };
// #	if !defined(HK_PLATFORM_PS3_SPU) && !(defined(HK_COMPILER_HAS_INTRINSICS_IA32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED))
// const MaskType hkVector4Comparison::s_invMaskFromBits[hkVector4ComparisonMask::MASK_XYZW+1] =
// {
// 	INV_QUAD(0,0,0,0),
// 	INV_QUAD(0,0,0,W),
// 	INV_QUAD(0,0,Z,0),
// 	INV_QUAD(0,0,Z,W),
// 
// 	INV_QUAD(0,Y,0,0),
// 	INV_QUAD(0,Y,0,W),
// 	INV_QUAD(0,Y,Z,0),
// 	INV_QUAD(0,Y,Z,W),
// 
// 	INV_QUAD(X,0,0,0),
// 	INV_QUAD(X,0,0,W),
// 	INV_QUAD(X,0,Z,0),
// 	INV_QUAD(X,0,Z,W),
// 
// 	INV_QUAD(X,Y,0,0),
// 	INV_QUAD(X,Y,0,W),
// 	INV_QUAD(X,Y,Z,0),
// 	INV_QUAD(X,Y,Z,W)
// };
// #	endif //!defined(HK_PLATFORM_PS3_SPU) && !(defined(HK_COMPILER_HAS_INTRINSICS_IA32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED))
#endif // altivec

// const hkVector4Comparison::Mask hkVector4Comparison::s_components[4] = {	hkVector4ComparisonMask::MASK_X, 
// hkVector4ComparisonMask::MASK_Y, 
// hkVector4ComparisonMask::MASK_Z, 
// hkVector4ComparisonMask::MASK_W		};

#if !defined(HK_PLATFORM_PS3_SPU)
// sanity check masks
HK_COMPILE_TIME_ASSERT( hkVector4ComparisonMask::MASK_NONE == 0);
#define SAME2(A,B) ((hkVector4ComparisonMask::MASK_##A | hkVector4ComparisonMask::MASK_##B) == hkVector4ComparisonMask::MASK_##A##B)
HK_COMPILE_TIME_ASSERT( SAME2(X,Y) );
HK_COMPILE_TIME_ASSERT( SAME2(X,Z) );
HK_COMPILE_TIME_ASSERT( SAME2(X,W) );
HK_COMPILE_TIME_ASSERT( SAME2(Y,Z) );
HK_COMPILE_TIME_ASSERT( SAME2(Y,W) );
HK_COMPILE_TIME_ASSERT( SAME2(Z,W) );

#endif // !defined HK_PLATFORM_PS3_SPU

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
