/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

int aabb_main()
{
	hkVector4 zero; zero.setZero();
	hkVector4 v0, v1, v2;
	v0.set(100,200,300);
	v1.set(200,300,400);
	v2.set(50,100,350);

	hkAabb a0; a0.m_min = zero; a0.m_max =  v0;
	hkAabb a1; a1.m_min = zero; a1.m_max =  v1;
	hkAabb a2; a2.m_min = v2; a2.m_max =  v1;

	HK_TEST( a0.isValid() );
	HK_TEST( a1.isValid() );
	HK_TEST( a2.isValid() );
	HK_TEST( a0.isEmpty() == false);

	hkAabb a3;
	a3.m_min = v2;
	a3.m_max = v0;
	HK_TEST( a3.isValid() == false);
	HK_TEST( a3.isEmpty() );

	HK_TEST( a1.contains(a0) );
	HK_TEST( a0.contains(a1) == hkFalse32);

	HK_TEST( a1.contains(a2) );
	HK_TEST( a2.contains(a1) == hkFalse32);
	HK_TEST( a0.contains(a2) == hkFalse32);
	HK_TEST( a2.contains(a0) == hkFalse32);

	// Test setEmpty/isEmpty
	{
		hkAabb a4; 
		a4.setEmpty();
		HK_TEST( a3.isEmpty() );
	}

	// Test expandBy/containsPoint
	{
		hkAabb a5; a5.m_min = v2; a5.m_max =  v1;
		hkVector4 testPoint;
		testPoint.set( v2(0) - 1.0f, v1(1) + 1.0f, .5f * (v2(2) + v1(2)) );
		HK_TEST( !a5.containsPoint(testPoint) );

		a5.expandBy(hkSimdReal::fromFloat(1.5f));
		HK_TEST( a5.containsPoint( testPoint ) );
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(aabb_main, "Fast", "Physics2012/Test/UnitTest/Collide/", __FILE__     );

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
