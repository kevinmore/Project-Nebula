/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>


#include <Common/Base/Types/Geometry/Aabb20/hkAabb20.h>

int aabbUnitTest_main()
{
	hkAabb aabb;

	aabb.m_min.setZero();
	aabb.m_max.set( 0x7ffff0, 0x7ff0, 0x7ffff0 );

	hkAabb24_16_24_Codec codec;
	codec.set( aabb );

	hkAabb a0;
	a0.m_min.set ( 1,2,3 );
	a0.m_max.set ( 2,3,4 );

	hkAabb24_16_24 a0i; codec.packAabb( a0, &a0i );

	{
		hkAabb t;			codec.unpackAabbUnscaled( a0i, &t );
		HK_TEST( t.m_min.allEqual<3>(a0.m_min, hkSimdReal_Inv2 ));
	}

	{	// tests if less really takes minX
		hkAabb a1;
		a1.m_min.set ( 0,100,100 );
		a1.m_max.set ( 100,100,100 );
		hkAabb24_16_24 a1i; 
		codec.packAabb( a1, &a1i );
		bool a0Less = a0i < a1i;	/// test minX
		HK_TEST( !a0Less );
	}

	{
		hkAabb a;
		a.m_min.set ( 4,2,3 );
		a.m_max.set ( 5,3,4 );

		HK_ALIGN16(hkAabb24_16_24) ai; 
		codec.packAabb( a, &ai );
		hkBoolLL t = a0i.disjoint( ai );
		HK_TEST( t != hkFalse32 );
	}
	{
		hkAabb a;
		a.m_min.set ( 1,5,3 );
		a.m_max.set ( 2,6,4 );

		HK_ALIGN16(hkAabb24_16_24) ai; 
		codec.packAabb( a, &ai );
		hkBoolLL t = a0i.disjoint( ai );
		HK_TEST( t != hkFalse32 );
	}
	{
		hkAabb a;
		a.m_min.set ( 1,2,6 );
		a.m_max.set ( 2,3,7 );

		HK_ALIGN16(hkAabb24_16_24) ai; 
		codec.packAabb( a, &ai );
		hkBoolLL t = a0i.disjoint( ai );
		HK_TEST( t != hkFalse32 );
	}

		
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

//HK_TEST_REGISTER(aabbUnitTest_main,     "Fast", "Test/Test/UnitTest/UnitTest/UnitTest/Base/",     __FILE__    );

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
