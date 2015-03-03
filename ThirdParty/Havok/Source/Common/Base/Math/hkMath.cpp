/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/Physics/ContactPoint/hkContactPoint.h> // move to collide lib


#define P16 30000.0
#define P31 (P16 * 0x10000)

#define HK_QUADFLOAT_SINGLE(X) HK_QUADFLOAT_CONSTANT(X,X,X,X)
#define HK_QUADDOUBLE_SINGLE(X) HK_QUADDOUBLE_CONSTANT(X,X,X,X)


HK_ALIGN_REAL(const hkQuadFloat32 g_vectorfConstants[HK_QUADREAL_END]) = 
{
	HK_QUADFLOAT_SINGLE(-1),
	HK_QUADFLOAT_SINGLE(0),
	HK_QUADFLOAT_SINGLE(1),
	HK_QUADFLOAT_SINGLE(2),
	HK_QUADFLOAT_SINGLE(3),
	HK_QUADFLOAT_SINGLE(4),
	HK_QUADFLOAT_SINGLE(5),
	HK_QUADFLOAT_SINGLE(6),
	HK_QUADFLOAT_SINGLE(7),
	HK_QUADFLOAT_SINGLE(8),
	HK_QUADFLOAT_SINGLE(15),
	HK_QUADFLOAT_SINGLE(16),
	HK_QUADFLOAT_SINGLE(255),
	HK_QUADFLOAT_SINGLE(256),
	HK_QUADFLOAT_SINGLE(1<<23),

	HK_QUADFLOAT_SINGLE(0),
	HK_QUADFLOAT_SINGLE(1.0f/1),
	HK_QUADFLOAT_SINGLE(1.0f/2),
	HK_QUADFLOAT_SINGLE(1.0f/3),
	HK_QUADFLOAT_SINGLE(1.0f/4),
	HK_QUADFLOAT_SINGLE(1.0f/5),
	HK_QUADFLOAT_SINGLE(1.0f/6),
	HK_QUADFLOAT_SINGLE(1.0f/7),
	HK_QUADFLOAT_SINGLE(1.0f/8),
	HK_QUADFLOAT_SINGLE(1.0f/15),
	HK_QUADFLOAT_SINGLE(1.0f/127),
	HK_QUADFLOAT_SINGLE(1.0f/226),
	HK_QUADFLOAT_SINGLE(1.0f/255),

	HK_QUADFLOAT_CONSTANT(1,0,0,0),
	HK_QUADFLOAT_CONSTANT(0,1,0,0),
	HK_QUADFLOAT_CONSTANT(0,0,1,0),
	HK_QUADFLOAT_CONSTANT(0,0,0,1),
	HK_QUADFLOAT_SINGLE(HK_FLOAT_MAX),
	HK_QUADFLOAT_SINGLE(HK_FLOAT_HIGH),
	HK_QUADFLOAT_SINGLE(HK_FLOAT_EPSILON),
	HK_QUADFLOAT_SINGLE(HK_FLOAT_EPSILON * HK_FLOAT_EPSILON),
	HK_QUADFLOAT_SINGLE(HK_FLOAT_MIN),
	HK_QUADFLOAT_SINGLE(-HK_FLOAT_MAX),
	HK_QUADFLOAT_SINGLE(-HK_FLOAT_MIN),

	HK_QUADFLOAT_CONSTANT(-1,1,-1,1),
	HK_QUADFLOAT_CONSTANT(1,-1,1,-1),
	HK_QUADFLOAT_CONSTANT( 1,0, 1,0),
	HK_QUADFLOAT_CONSTANT( 1,1, 0,0),
	HK_QUADFLOAT_CONSTANT( 0,0, 1,1),
	HK_QUADFLOAT_CONSTANT( 1,2, 4,8), 
	HK_QUADFLOAT_CONSTANT( 8,4, 2,1), 
	HK_QUADFLOAT_SINGLE( 1.0f + 1.0f/256 ), // HK_QUADREAL_PACK_HALF
	HK_QUADFLOAT_SINGLE( P31 ),			   // HK_QUADREAL_PACK16_UNIT_VEC
	HK_QUADFLOAT_SINGLE( 1.0f/P31 ),		   // HK_QUADREAL_UNPACK16_UNIT_VEC

	HK_QUADFLOAT_SINGLE(HK_FLOAT_PI),
	HK_QUADFLOAT_SINGLE(HK_FLOAT_PI*0.5f),
	HK_QUADFLOAT_SINGLE(HK_FLOAT_PI*0.25f),
	HK_QUADFLOAT_SINGLE(HK_FLOAT_PI*4.0f/3.0f),
	HK_QUADFLOAT_SINGLE(HK_FLOAT_PI*2.0f),
	HK_QUADFLOAT_SINGLE(HK_FLOAT_PI*4.0f),
};

HK_ALIGN_REAL(const hkQuadDouble64 g_vectordConstants[HK_QUADREAL_END]) = 
{
	HK_QUADDOUBLE_SINGLE(-1),
	HK_QUADDOUBLE_SINGLE(0),
	HK_QUADDOUBLE_SINGLE(1),
	HK_QUADDOUBLE_SINGLE(2),
	HK_QUADDOUBLE_SINGLE(3),
	HK_QUADDOUBLE_SINGLE(4),
	HK_QUADDOUBLE_SINGLE(5),
	HK_QUADDOUBLE_SINGLE(6),
	HK_QUADDOUBLE_SINGLE(7),
	HK_QUADDOUBLE_SINGLE(8),
	HK_QUADDOUBLE_SINGLE(15),
	HK_QUADDOUBLE_SINGLE(16),
	HK_QUADDOUBLE_SINGLE(255),
	HK_QUADDOUBLE_SINGLE(256),
	HK_QUADDOUBLE_SINGLE(1<<23),

	HK_QUADDOUBLE_SINGLE(0),
	HK_QUADDOUBLE_SINGLE(1.0/1),
	HK_QUADDOUBLE_SINGLE(1.0/2),
	HK_QUADDOUBLE_SINGLE(1.0/3),
	HK_QUADDOUBLE_SINGLE(1.0/4),
	HK_QUADDOUBLE_SINGLE(1.0/5),
	HK_QUADDOUBLE_SINGLE(1.0/6),
	HK_QUADDOUBLE_SINGLE(1.0/7),
	HK_QUADDOUBLE_SINGLE(1.0/8),
	HK_QUADDOUBLE_SINGLE(1.0/15),
	HK_QUADDOUBLE_SINGLE(1.0/127),
	HK_QUADDOUBLE_SINGLE(1.0/226),
	HK_QUADDOUBLE_SINGLE(1.0/255),

	HK_QUADDOUBLE_CONSTANT(1,0,0,0),
	HK_QUADDOUBLE_CONSTANT(0,1,0,0),
	HK_QUADDOUBLE_CONSTANT(0,0,1,0),
	HK_QUADDOUBLE_CONSTANT(0,0,0,1),
	HK_QUADDOUBLE_SINGLE(HK_DOUBLE_MAX),
	HK_QUADDOUBLE_SINGLE(HK_DOUBLE_HIGH),
	HK_QUADDOUBLE_SINGLE(HK_DOUBLE_EPSILON),
	HK_QUADDOUBLE_SINGLE(HK_DOUBLE_EPSILON * HK_DOUBLE_EPSILON),
	HK_QUADDOUBLE_SINGLE(HK_DOUBLE_MIN),
	HK_QUADDOUBLE_SINGLE(-HK_DOUBLE_MAX),
	HK_QUADDOUBLE_SINGLE(-HK_DOUBLE_MIN),

	HK_QUADDOUBLE_CONSTANT(-1,1,-1,1),
	HK_QUADDOUBLE_CONSTANT(1,-1,1,-1),
	HK_QUADDOUBLE_CONSTANT( 1,0, 1,0),
	HK_QUADDOUBLE_CONSTANT( 1,1, 0,0),
	HK_QUADDOUBLE_CONSTANT( 0,0, 1,1),
	HK_QUADDOUBLE_CONSTANT( 1,2, 4,8), 
	HK_QUADDOUBLE_CONSTANT( 8,4, 2,1), 
	HK_QUADDOUBLE_SINGLE( 1.0 + 1.0/256 ), // HK_QUADDOUBLE_PACK_HALF
	HK_QUADDOUBLE_SINGLE( P31 ),			   // HK_QUADDOUBLE_PACK16_UNIT_VEC
	HK_QUADDOUBLE_SINGLE( 1.0/P31 ),		   // HK_QUADDOUBLE_UNPACK16_UNIT_VEC

	HK_QUADDOUBLE_SINGLE(HK_DOUBLE_PI),
	HK_QUADDOUBLE_SINGLE(HK_DOUBLE_PI*0.5),
	HK_QUADDOUBLE_SINGLE(HK_DOUBLE_PI*0.25),
	HK_QUADDOUBLE_SINGLE(HK_DOUBLE_PI*4.0/3.0),
	HK_QUADDOUBLE_SINGLE(HK_DOUBLE_PI*2.0),
	HK_QUADDOUBLE_SINGLE(HK_DOUBLE_PI*4.0),
};


void HK_CALL hkCheckFlushDenormals()
{
#if defined(HK_COMPILER_HAS_INTRINSICS_IA32) && HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED
	// _MM_GET_FLUSH_ZERO_MODE takes an argument but doesn't use it in MSVC, but not GCC
#if defined(HK_COMPILER_MSVC) 
#	define DUMMY_ARG 1
#else
#	define DUMMY_ARG 
#endif
#if defined(_MSC_VER) && (_MSC_VER >= 1700) || defined(HK_PLATFORM_PS4)
  // does not even like dummarg as nothing since __MM_GET is a macro too
	HK_WARN_ONCE_ON_DEBUG_IF(_MM_GET_FLUSH_ZERO_MODE( ) != _MM_FLUSH_ZERO_ON, 0xDE404A11, "Flushing denormals is required inside Havok code. Please call _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); before stepping Havok. See the \"Using the Math Library\" section in the User Guide for more information." );
#else
	HK_WARN_ONCE_ON_DEBUG_IF(_MM_GET_FLUSH_ZERO_MODE( DUMMY_ARG ) != _MM_FLUSH_ZERO_ON, 0xDE404A11, "Flushing denormals is required inside Havok code. Please call _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); before stepping Havok. See the \"Using the Math Library\" section in the User Guide for more information." );
#endif
#endif
}


#if defined(HK_PLATFORM_GC) || defined(HK_PLATFORM_RVL) || defined(HK_PLATFORM_WIIU)
extern hkFloat32  hkInfinityf;
extern hkDouble64 hkInfinityd;
hkFloat32  hkInfinityf = 1.0f/0.0f;
hkDouble64 hkInfinityd = 1.0/0.0;
#endif

// TODO: is this ok?
// get rid of this here, not in math please!
#if !defined(HK_PLATFORM_LRB) && !defined(HK_REAL_IS_DOUBLE)
HK_COMPILE_TIME_ASSERT( sizeof( hkContactPoint ) == 32 );
#endif

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
