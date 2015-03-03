/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Internal/hkcdInternal.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Geometry/Internal/Algorithms/ClosestPoint/hkcdClosestPointLineLine.h>
#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastCapsule.h>
#include <Geometry/Internal/Types/hkcdRay.h>


static void rayCapsule()
{
	hkVector4 ls;
	hkVector4 le;
	
	hkVector4 v0;
	hkVector4 v1;

	ls.set(-7.873028f, -7.814732f, 8.701378f);
	le.set(-5.637238f, 0.887411f, 1.918097f);
	v0.set(6.479851f, -1.458720f, 9.268354f);
	v1.set(-6.231992f, -1.820987f, 6.866690f);
	hkSimdReal radius; radius.setFromFloat(2.526909f);
	
	hkcdRay ray;
	ray.setEndPoints(ls, le);
	hkSimdReal fraction = hkSimdReal::getConstant<HK_QUADREAL_1>();
	hkVector4 normal; normal.setZero();
	hkcdRayCastCapsuleHitType hitInfo;

	hkcdRayCastCapsule(ray, v0, v1, radius, &fraction, &normal, &hitInfo, hkcdRayQueryFlags::NO_FLAGS);

	HK_TEST(hkMath::equal(fraction.getReal(), 0.442245f));
	hkVector4 testNormal; testNormal.set(-0.258129f, -0.848968f, -0.461111f);
	hkSimdReal eps; eps.setFromFloat(1e-5f);
	HK_TEST(testNormal.allEqual<3>(normal, eps));
}

static void rayCapsuleInsideHits()
{
	hkVector4 ls;
	hkVector4 le;

	hkVector4 v0;
	hkVector4 v1;

	ls.set(0, 0, 0);
	le.set(6.0f, 0, 0);
	v0.set(-2.0f, 0, 0);
	v1.set(2.0f, 0, 0);
	hkSimdReal radius; radius.setFromFloat(1.0f);

	hkcdRay ray;
	ray.setEndPoints(ls, le);
	hkSimdReal fraction = hkSimdReal_1;
	hkVector4 normal; normal.setZero();
	hkcdRayCastCapsuleHitType hitInfo;

	fraction = hkSimdReal_1;
	HK_TEST( hkcdRayCastCapsule(ray, v0, v1, radius, &fraction, &normal, &hitInfo, hkcdRayQueryFlags::NO_FLAGS) == hkFalse32);

	fraction = hkSimdReal_1;
	HK_TEST( hkcdRayCastCapsule(ray, v0, v1, radius, &fraction, &normal, &hitInfo, hkcdRayQueryFlags::ENABLE_INSIDE_HITS) != hkFalse32);

	HK_TEST(hkMath::equal(fraction.getReal(), 0.5f));
	hkVector4 testNormal; testNormal.set(1.0f, 0, 0);
	hkSimdReal eps; eps.setFromFloat(1e-5f);
	HK_TEST(testNormal.allEqual<3>(normal, eps));
}

static void rayCapsuleBundle()
{
	hkVector4 ls;
	hkVector4 le;

	hkVector4 v0;
	hkVector4 v1;

	ls.set(-7.873028f, -7.814732f, 8.701378f);
	le.set(-5.637238f, 0.887411f, 1.918097f);
	v0.set(6.479851f, -1.458720f, 9.268354f);
	v1.set(-6.231992f, -1.820987f, 6.866690f);
	hkSimdReal radius; radius.setFromFloat(2.526909f);

	hkcdRayBundle ray;
	ray.m_start.setAll(ls);
	ray.m_end.setAll(le);
	ray.m_activeRays.set<hkVector4ComparisonMask::MASK_XYZW>();
	hkVector4 fraction = hkVector4::getConstant<HK_QUADREAL_1>();
	hkFourTransposedPoints fourNormals;

	/*hkVector4Comparison mask =*/ hkcdRayBundleCapsuleIntersect(ray, v0, v1, radius, fraction, fourNormals);

	HK_TEST(hkMath::equal(fraction(0), 0.442245f));

	hkVector4 normal[4];
	fourNormals.extract(normal[0], normal[1], normal[2], normal[3]);
	hkVector4 testNormal; testNormal.set(-0.258129f, -0.848968f, -0.461111f);
	hkSimdReal eps; eps.setFromFloat(1e-5f);
	HK_TEST(testNormal.allEqual<3>(normal[0], eps));
}

int RayCapsule_main()
{
	rayCapsule();
	rayCapsuleInsideHits();

	rayCapsuleBundle();

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER(RayCapsule_main, "Fast", "Geometry/Test/UnitTest/Internal/", __FILE__ );

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
