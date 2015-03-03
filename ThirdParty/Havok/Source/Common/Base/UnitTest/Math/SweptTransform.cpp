/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

static void testDifference(hkQuaternion &a, hkQuaternion &b, hkReal eps)
{
	// We don't just check whether they are componentwise equal, we also check whether a = -b,
	// since (as rotations), the quaternions are also "the same".

	hkVector4 va = a.m_vec;
	hkVector4 vb = b.m_vec;

	hkReal r = va.dot<4>(vb).getReal();

	hkBool is1 = ( hkMath::fabs(1.0f - r ) < eps );
	hkBool isMinus1 = ( hkMath::fabs(-1.0f - r ) < eps );
	HK_TEST ( is1 || isMinus1 );

}

static void sweptTransformTest()
{
	{
		hkMotionState ms0;
		ms0.initMotionState( hkVector4::getZero(), hkQuaternion::getIdentity() );

		hkVector4 mc; mc.set( 1, 2, 3 );
		hkSweptTransformUtil::setCentreOfRotationLocal( mc, ms0 );

		hkQuaternion q; q.setAxisAngle( hkTransform::getIdentity().getColumn(1), 1.0f );

		hkSweptTransformUtil::warpToRotation( q, ms0 );

		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( ms0.getTransform().getTranslation().allEqual<3>( hkVector4::getZero(),eps ) );
	}

	// Checking _approxTransformAt() and approxTransformAt().
	// approxTransformAt just calls _approxTransformAt - why do we even test this?
	{
		hkPseudoRandomGenerator rand(1337);

		hkTime time = 0.07f;
		hkQuaternion q0, q1;
		hkVector4 com0, com1, comLocal;

		rand.getRandomRotation(q0);
		rand.getRandomRotation(q1);
		rand.getRandomVector11(com0);
		rand.getRandomVector11(com1);
		rand.getRandomVector11(comLocal);
		
		hkSweptTransform str;
		str.m_rotation0 = q0;
		str.m_rotation1 = q1;
		str.m_centerOfMass0 = com0;
		str.m_centerOfMass1 = com1;
		str.m_centerOfMassLocal = comLocal;
		hkTransform ts;
		str._approxTransformAt(time , ts );

		hkSweptTransform str_test;
		str_test.m_rotation0 = q0;
		str_test.m_rotation1 = q1;
		str_test.m_centerOfMass0 = com0;
		str_test.m_centerOfMass1 = com1;
		str_test.m_centerOfMassLocal = comLocal;
		hkTransform ts_test;
		str_test.approxTransformAt( time, ts_test );

		HK_TEST( ts.isOk() );
		HK_TEST( ts_test.isOk() );
		HK_TEST( ts_test.getRotation().isApproximatelyEqual(ts.getRotation()) );
	}

	// Checking getInterpolationValue().
	{
		hkQuadReal BaseTimeVal = HK_QUADREAL_CONSTANT( 1, 2, 3, 0.77f );
		hkVector4 BaseTime; BaseTime.m_quad = BaseTimeVal;
		hkQuadReal DeltaTimeVal = HK_QUADREAL_CONSTANT( 4, 5, 6, 0.124f );
		hkVector4 DeltaTime; DeltaTime.m_quad = DeltaTimeVal;
		hkSweptTransform str;
		str.m_centerOfMass0 = BaseTime;
		str.m_centerOfMass1 = DeltaTime;
		hkReal r = str.getInterpolationValue( 0.788f );
		HK_TEST( hkMath::fabs(r - 0.002232f) < 1e-3f );
		hkReal testipv = ( 0.788f - str.getBaseTime()) * str.getInvDeltaTime();
		HK_TEST( hkMath::equal(r,testipv) );

		BaseTime.set( 1, -2, 3, 0.788f );
		DeltaTime.set( 4, 5, -6, 0.124f );
		str.m_centerOfMass0 = BaseTime;
		str.m_centerOfMass1 = DeltaTime;
		r = str.getInterpolationValue( 0.799f );
		HK_TEST( hkMath::fabs(r - 0.001364f) < 1e-3f );
		testipv = ( 0.799f - str.getBaseTime()) * str.getInvDeltaTime();
		HK_TEST( hkMath::equal(r,testipv) );
	}

	// Checking getInterpolationValueHiAccuracy().
	{
		hkQuadReal BaseTimeVal = HK_QUADREAL_CONSTANT( 1, 2, 3, 0.788f );
		hkVector4 BaseTime; BaseTime.m_quad = BaseTimeVal;
		hkQuadReal DeltaTimeVal = HK_QUADREAL_CONSTANT( 4, 5, 6, 0.0124f );
		hkVector4 DeltaTime; DeltaTime.m_quad = DeltaTimeVal;
		hkSweptTransform str;
		str.m_centerOfMass0 = BaseTime;
		str.m_centerOfMass1 = DeltaTime;
		hkReal r = str.getInterpolationValueHiAccuracy( 0.789f, 0.00999f );
		HK_TEST( hkMath::fabs(r - 0.000136276f) < 1e-3f );

		hkReal dt = 0.789f - str.getBaseTime();
		dt += 0.00999f;
		hkReal ddt_test = dt * str.getInvDeltaTime();
		HK_TEST( hkMath::equal(r,ddt_test) );

		BaseTime.set( 1, -2, 3, 0.12222f );
		DeltaTime.set( 4, 5, -6, 0.0f );

		str.m_centerOfMass0 = BaseTime;
		str.m_centerOfMass1 = DeltaTime;
		r = str.getInterpolationValueHiAccuracy( 0.123f, 0.00123f );
		HK_TEST( hkMath::fabs(r - 0.0f) < 1e-3f );
		dt = 0.123f - str.getBaseTime();
		dt += 0.00999f;
		ddt_test = dt * str.getInvDeltaTime();
		HK_TEST( hkMath::equal(r,ddt_test) );
	}

	// Checking getBaseTime().
	{
		hkQuadReal BaseTimeVal = HK_QUADREAL_CONSTANT( 1, 2, 3, 0.0788f );
		hkVector4 BaseTime; BaseTime.m_quad = BaseTimeVal;
		hkSweptTransform str;
		str.m_centerOfMass0 = BaseTime;
		hkReal r = str.getBaseTime();
		HK_TEST( hkMath::fabs(r - 0.0788f) < 1e-3f );
		HK_TEST( hkMath::equal(r,0.0788f) );

		BaseTime.set( 1, 2, 3, 0.999f );
		str.m_centerOfMass0 = BaseTime;
		r = str.getBaseTime();
		HK_TEST( hkMath::fabs(r - 0.999f) < 1e-3f );
		HK_TEST( hkMath::equal(r,0.999f) );

		BaseTime.set( 1, 2, 3, 0.0001f );
		str.m_centerOfMass0 = BaseTime;
		r = str.getBaseTime();
		HK_TEST( hkMath::fabs(r - 0.0001f) < 1e-3f );
		HK_TEST( hkMath::equal(r,0.0001f) );
	}

	// Checking getInvDeltaTime().
	{
		hkQuadReal BaseTimeVal = HK_QUADREAL_CONSTANT( 1, 2, 3, 0.0788f );
		hkVector4 BaseTime; BaseTime.m_quad = BaseTimeVal;
		hkSweptTransform str;
		str.m_centerOfMass1 = BaseTime;
		hkReal r = str.getInvDeltaTime();
		HK_TEST( hkMath::fabs(r - 0.0788f) < 1e-3f );
		HK_TEST( hkMath::equal(r,0.0788f) );

		BaseTime.set( 1, 2, 3, 0.999f );
		str.m_centerOfMass1 = BaseTime;
		r = str.getInvDeltaTime();
		HK_TEST( hkMath::fabs(r - 0.999f) < 1e-3f );
		HK_TEST( hkMath::equal(r,0.999f) );

		BaseTime.set( 1, 2, 3, 0.0001f );
		str.m_centerOfMass1 = BaseTime;
		r = str.getInvDeltaTime();
		HK_TEST( hkMath::fabs(r - 0.0001f) < 1e-3f );
		HK_TEST( hkMath::equal(r,0.0001f) );
	}

	// Checking initSweptTransform().
	{
		hkVector4 a;
		a.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );

		a.normalize<3>();
		hkVector4 b;
		b.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
		hkVector4 c;
		c.setCross( a, b );
		c.normalize<3>();
		b.setCross( c, a );
		b.normalize<3>();

		hkRotation r;
		r.setCols( a, b, c );
		hkQuaternion q;
		q.setAndNormalize(r);
		hkSweptTransform str;

		hkQuadReal positionVal = HK_QUADREAL_CONSTANT( 1, 2, 3, 0.05f );
		hkVector4 position; position.m_quad = positionVal;
		str.initSweptTransform( position, q );
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( str.m_centerOfMass0.allEqual<3>(position,eps));
		HK_TEST( str.m_centerOfMass1.allEqual<3>(position,eps));

		testDifference( str.m_rotation0, q, 1e-3f );
		testDifference( str.m_rotation1, q, 1e-3f );

		hkVector4 t; t = hkVector4::getConstant<HK_QUADREAL_0>();
		HK_TEST( str.m_centerOfMassLocal.allEqual<3>(t,eps) );
	}
}

int swepttransform_main()
{
	sweptTransformTest();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(swepttransform_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
