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
#include <Common/Base/Math/Quaternion/hkQuaternionUtil.h>
#include <Common/Base/Math/Matrix/hkMatrix4Util.h>

static void testDifference(hkQuaternion &a, hkQuaternion &b, hkReal eps)
{
	// We don't just check whether they are componentwise equal, we also check whether a = -b,
	// since (as rotations), the quaternions are also "the same".
	hkVector4 va = a.m_vec;
	hkVector4 vb = b.m_vec;

	hkReal r = va.dot<4>(vb).getReal();

	hkBool is1 = (hkMath::fabs(1.0f - r ) < eps);
	hkBool isMinus1 = (hkMath::fabs(-1.0f - r ) < eps);
	HK_TEST ( is1 || isMinus1);

}

static void checkMulMulInv()
{
	hkVector4 axis0; axis0.set(1,2,3);
	hkVector4 axis1; axis1.set(-3,1,-0.5f);

	axis0.normalize<3>();
	axis1.normalize<3>();

	hkQuaternion q0; q0.setAxisAngle(axis0, 0.7f);
	hkQuaternion q1; q1.setAxisAngle(axis1, -1.3f);

	hkQuaternion q0q1InvTest;
	q0q1InvTest.setMulInverse(q0, q1);

	hkQuaternion q0q1Inv;
	hkQuaternion q1Inv;
	q1Inv.setInverse(q1);
	q0q1Inv.setMul(q0, q1Inv);

	testDifference(q0q1InvTest, q0q1Inv, 1e-3f);

	// Check Aliasing
	hkQuaternion q0Copy = q0;
	q0Copy.setMulInverse( q0Copy, q1 );
	testDifference(q0Copy, q0q1Inv, 1e-3f);

}

static void checkMulInvMul()
{
	hkVector4 axis0; axis0.set(1,2,3);
	hkVector4 axis1; axis1.set(-3,1,-0.5f);

	axis0.normalize<3>();
	axis1.normalize<3>();

	hkQuaternion q0; q0.setAxisAngle(axis0, 0.7f);
	hkQuaternion q1; q1.setAxisAngle(axis1, -1.3f);

	hkQuaternion q0Invq1Test;
	q0Invq1Test.setInverseMul(q0, q1);

	hkQuaternion q0Invq1, q0Inv;
	q0Inv.setInverse(q0);
	q0Invq1.setMul(q0Inv, q1);

	testDifference(q0Invq1Test, q0Invq1, 1e-3f);

	//Check Aliasing
	hkQuaternion q0Copy = q0;
	q0Copy.setInverseMul( q0Copy, q1 );
	testDifference(q0Copy, q0Invq1, 1e-3f);

}

static void checkMul()
{
	hkVector4 axis0; axis0.set(1,2,3);
	hkVector4 axis1; axis1.set(-3,1,-0.5f);

	axis0.normalize<3>();
	axis1.normalize<3>();

	hkQuaternion q0; q0.setAxisAngle(axis0, 0.7f);
	hkQuaternion q1; q1.setAxisAngle(axis1, -1.3f);

	hkQuaternion q0q1;
	{

		// The naive way:
		// pq.m_real = p.m_real*q.m_real - p.m_imag.Dot(q.m_imag)
		// pq.m_imag = p.m_real*q.m_imag + q.m_real*p.m_imag + p.m_imag.Cross(q.m_imag)
		// uses  16 multiplications and 12 adds


		// A better way uses only 9 mults but 27 adds.
		// This way is implemented below:


		const hkReal temp_1 = (q0.getImag()(2)-q0.getImag()(1))	*	(q1.getImag()(1)-q1.getImag()(2));
		const hkReal temp_2 = (q0.getReal()+q0.getImag()(0))		*	(q1.getReal()+q1.getImag()(0));
		const hkReal temp_3 = (q0.getReal()-q0.getImag()(0))		*	(q1.getImag()(1)+q1.getImag()(2));
		const hkReal temp_4 = (q0.getImag()(2)+q0.getImag()(1))	*	(q1.getReal()-q1.getImag()(0));
		const hkReal temp_5 = (q0.getImag()(2)-q0.getImag()(0))	*	(q1.getImag()(0)-q1.getImag()(1));
		const hkReal temp_6 = (q0.getImag()(2)+q0.getImag()(0))	*	(q1.getImag()(0)+q1.getImag()(1));
		const hkReal temp_7 = (q0.getReal()+q0.getImag()(1))		*	(q1.getReal()-q1.getImag()(2));
		const hkReal temp_8 = (q0.getReal()-q0.getImag()(1))		*	(q1.getReal()+q1.getImag()(2));

		const hkReal temp_9 = temp_6 + temp_7 + temp_8;
		const hkReal temp_10 = (temp_5 + temp_9)*0.5f;

		hkQuadRealUnion v;
		v.r[0] = temp_2+temp_10-temp_9;
		v.r[1] = temp_3+temp_10-temp_8;
		v.r[2] = temp_4+temp_10-temp_7;
		v.r[3] = temp_1+temp_10-temp_6;
		q0q1.m_vec.m_quad = v.q;
	}


	hkQuaternion q0q1Test;
	q0q1Test.setMul(q0, q1);

	testDifference(q0q1Test, q0q1, 1e-3f);

	// test hkQuaternion::mul too
	hkQuaternion q0q1Test2;
	q0q1Test2 = q0;
	q0q1Test2.mul(q1);

	testDifference(q0q1Test2, q0q1, 1e-3f);

	//Check Aliasing
	hkQuaternion q0Copy = q0;
	q0Copy.setMul( q0Copy, q1 );
	testDifference(q0Copy, q0q1Test, 1e-3f);
}


static void checkGetAxisStability(const hkVector4& axis, const hkReal angle)
{

//	if(angle >= 1e-6f)
	{
		hkQuaternion q; q.setAxisAngle(axis, angle);

	//	hkVector4 a = q.m_vec;
	//	a.normalize_23BitAccurate<3>();

		hkVector4 axisTest;
		q.getAxis(axisTest);

			// First check axis is normalized
		HK_TEST(axisTest.dot<3>(axis).getReal() > 1 - 1e-3f);

			// Then check it is "equal" to the original
		HK_TEST( hkMath::fabs(axisTest(0) - axis(0)) < 1e-3f);
		HK_TEST( hkMath::fabs(axisTest(1) - axis(1)) < 1e-3f);
		HK_TEST( hkMath::fabs(axisTest(2) - axis(2)) < 1e-3f);
	}
}

	// See Mantis 746
	// http://havok2bugs.telekinesys/view_bug_page.php?f_id=0000746
static void checkAngleAxis()
{
    {
    hkVector4 axis; axis.set(1,-2,3);
    axis.normalize<3>();

		// Test with small (+ve) angle. We expect to get the same angle/axis pair back.
	{
		hkReal angle = 0.7f;
		hkQuaternion q0; q0.setAxisAngle(axis, angle);

		hkReal angleTest = q0.getAngle();

		hkVector4 axisTest;
		q0.getAxis(axisTest);


		HK_TEST( hkMath::fabs(angle - angleTest) < 1e-3f);
		HK_TEST( hkMath::fabs(axisTest(0) - axis(0)) < 1e-3f);
		HK_TEST( hkMath::fabs(axisTest(1) - axis(1)) < 1e-3f);
		HK_TEST( hkMath::fabs(axisTest(2) - axis(2)) < 1e-3f);
	}

		// Test with large (-ve) angle. We expect to get the negative angle/axis pair back.
		// Angle will always be in range 0-PI. If we pass in angle > PI, then angle will be returned as (-angle), ie. 2PI-angle
		// and axis will be flipped.
		// eg. Rot(1.5 PI, (1,0,0)) will return:
		// 0.5 PI as angle
		// (-1,0,0) as axis.
	{
		hkReal angle = 3.7f;
		hkQuaternion q0; q0.setAxisAngle(axis, angle);

		hkReal angleTest = q0.getAngle();

		hkVector4 axisTest;
		q0.getAxis(axisTest);

		angle = 2 * HK_REAL_PI - angle;
		axis.setNeg<4>(axis);


		HK_TEST( hkMath::fabs(angle - angleTest) < 1e-3f);
		HK_TEST( hkMath::fabs(axisTest(0) - axis(0)) < 1e-3f);
		HK_TEST( hkMath::fabs(axisTest(1) - axis(1)) < 1e-3f);
		HK_TEST( hkMath::fabs(axisTest(2) - axis(2)) < 1e-3f);
	}
	}


	// Check with very small angle: getAxis() should avoid numerically unstable calculations
	{
			// This is a "known" breaking cases for the previous algorithm:
			// Basically since a quaternion stores
			// (Cos(theta/2), Sin(theta/2) * axis)
			// the computation used to be:
			// 1. getSinAngleOver2 as sqrt(1-CosAngleOver2)
			// 2. Divide to get axis
			// but when theta ~ 0, we're dividing numbers close to 0, and one of these is computed
			// as sqrt(1-csqrd), hence already has error, so we get inaccurate results.
			// The new algorithm is to take the Sin(theta/2) * axis part and just normalize it,
		{
			hkVector4 axis; axis.set(1,0,0);
			hkReal angle = 5e-4f;

			checkGetAxisStability(axis, angle);

		}
		// Worst angle would be where sin(theta/2) ~ HK_REAL_EPSILON because we'd try and normalize
		// a vector of length ~ HK_REAL_EPSILON in that case. Such an angle would be:
		HK_ON_DEBUG( const hkReal worstAngle = hkMath::asin(HK_REAL_EPSILON) * 1.999f );
			// Comment in the next line to confirm an ASSERT will occur in DEBUG.
		//checkGetAxisStability(axis, worstAngle);

		// Check 100000 "random" (axis, angle) pairs, each angle chosen to be in range 1e-2 to 1e-6, by
		// fist picking "exponent" in range 2 to 6, then picking "mantissa"
		{
			hkPseudoRandomGenerator psrng(23527);

					// Get "exponent" in range 2-6

			for(int i = 0; i < 100000; i++)
			{

				hkUint32 exponent =  2 + psrng.getRandChar(5);
				hkReal scale = 1.0f;
				while(exponent--)
				{
					scale *= 0.1f;
				}


				hkReal angle = psrng.getRandReal01() * scale;
				while(angle < scale)
				{
					angle *= 10.0f;
				}

					// Sanity check
				HK_ASSERT(0x68500ee6, angle > worstAngle);

				hkVector4 axis; axis.set(psrng.getRandReal11(), psrng.getRandReal11(), psrng.getRandReal11());
				axis.normalize<3>();

				checkGetAxisStability(axis, angle);
			}
		}
	}

}

void checkShortestRotation()
{
	hkPseudoRandomGenerator psrng(23527);

	int i;
	for (i = 0 ; i < 100; i++)
	{
		hkVector4 from, to;
		psrng.getRandomVector11( from );
		from.normalize<3>();
		psrng.getRandomVector11( to );
		to.normalize<3>();

		hkQuaternion q;	hkQuaternionUtil::_computeShortestRotation( from, to, q );

		hkVector4 result;
		result.setRotatedDir( q, from );

		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( result.allEqual<3>( to,eps ) );
	}

	// Test boundaries : to = from
	{
		hkVector4 from, to;
		psrng.getRandomVector11( from );
		from.normalize<3>();
		to = from;

		hkQuaternion q;
		hkQuaternionUtil::_computeShortestRotation( from, to, q );

		hkVector4 result;
		result.setRotatedDir( q, from );

		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( result.allEqual<3>( to,eps ) );
	}

	// Test boundaries : to = -from
	{
		hkVector4 from, to;
		psrng.getRandomVector11( from );
		from.normalize<3>();
		to.setNeg<4>( from ) ;

		hkQuaternion q;
		hkQuaternionUtil::_computeShortestRotation( from, to, q );

		hkVector4 result;
		result.setRotatedDir( q, from );

		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( result.allEqual<3>( to,eps ) );
	}

	// Try damped versions
	for (i=0 ; i < 100; i++)
	{
		hkVector4 from, to;
		psrng.getRandomVector11( from );
		from.normalize<3>();
		psrng.getRandomVector11( to );
		to.normalize<3>();

		hkVector4 result;
		hkQuaternion q;
		hkQuaternionUtil::_computeShortestRotationDamped(from, to, hkSimdReal::getConstant<HK_QUADREAL_1>(), q );
		result.setRotatedDir( q, from );
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( result.allEqual<3>( to,eps ) );

		hkQuaternionUtil::_computeShortestRotationDamped(from, to, hkSimdReal::getConstant<HK_QUADREAL_0>(), q );
		result.setRotatedDir( q, from );
		HK_TEST( result.allEqual<3>( from,eps ) );
	}
}

static void checkNormalize()
{
	if(0)
	for( int i = 0; i < 100; ++i)
	{
		hkRotation r;
		{
			// get 3 perp vectors
			hkVector4 a; a.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
			a.normalize<3>();
			hkVector4 b; b.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
			hkVector4 c; c.setCross(a,b);
			c.normalize<3>();
			b.setCross(c,a);
			b.normalize<3>();
			// add some error
			hkSimdReal ii; ii.setFromFloat(i/10000.0f);
			b.addMul( ii, c);
			b.addMul( ii, a);
			a.addMul( ii, b);
			r.setCols(a,b,c);

			HK_TEST2( r.isOrthonormal(1e-3f), "failed at" << i );
		}
		hkQuaternion q;
		q.set(r); // will assert if not orthonormal
	}
}

static void checkConversiontoRotation()
{
	hkPseudoRandomGenerator random(22);

	// Create a large number of quaternions
#if defined(HK_PLATFORM_CTR) || defined(HK_PLATFORM_RVL)
#define NUM_QUATS 100
#else
#define NUM_QUATS 10000
#endif
	hkRotation r;

	// Run fpu tests
	{
		for (int i = 0; i < NUM_QUATS; i++)
		{
			hkQuaternion refQuat; random.getRandomRotation(refQuat);
			const hkQuaternion& qRef = refQuat;
			if (!qRef.isOk()) continue;
			r.set(qRef);
			if (!r.isOrthonormal() || !r.isOk()) continue;

			hkQuaternion q;
			q.set(r);
			q.setClosest(q, qRef);

			hkSimdReal eps; eps.setFromFloat(1e-3f);
			HK_TEST(q.m_vec.allEqual<4>(qRef.m_vec,eps));
		}
	}
#undef NUM_QUATS 
}

static void checkMisc()
{
	{
		hkVector4 axis; axis.set(-3,1,-0.5f);
		axis.normalize<3>();
		hkQuaternion q0; q0.setAxisAngle(axis, -1.3f);
		hkVector4 x = q0.m_vec;
		hkQuaternion q1; q1.set(x(0), x(1), x(2), x(3));
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( q0.m_vec.allEqual<4>(q1.m_vec,eps) );

		hkQuaternion id0;
		id0.setIdentity();
		HK_TEST( id0.m_vec.allEqual<4>( hkQuaternion::getIdentity().m_vec,eps ) );
	}

	// Checking setInverseMul().
	{
		hkVector4 axis0; axis0.set( 1, 2, 3 );
		hkVector4 axis1; axis1.set(-3, 1, -0.5f );

		axis0.normalize<3>();
		axis1.normalize<3>();

		hkQuaternion q0; q0.setAxisAngle(axis0, 0.7f);
		hkQuaternion q1; q1.setAxisAngle(axis1, -1.3f);

		hkQuaternion q0InvTestq1;
		q0InvTestq1.setInverseMul(q0, q1);

		hkQuaternion q0Invq1;
		hkQuaternion q0Inv;

		q0Inv.setInverse(q0);
		q0Invq1.setMul(q0Inv, q1);

		testDifference(q0InvTestq1, q0Invq1, 1e-3f);
	}

	// Checking setAxisAngle().
	{
		hkReal angle = 0.7f;
		hkVector4 axis;
		axis.set(1,2,3);
		axis.normalize<4>();

		hkQuaternion q0;
		q0.setAxisAngle(axis,angle);

		hkVector4 axisTest;
		q0.getAxis(axisTest);

		hkReal angleTest = q0.getAngle();

		HK_TEST( hkMath::fabs(angle - angleTest) < 1e-3f );
		HK_TEST( hkMath::fabs(axisTest(0) - axis(0)) < 1e-3f );
		HK_TEST( hkMath::fabs(axisTest(1) - axis(1)) < 1e-3f );
		HK_TEST( hkMath::fabs(axisTest(2) - axis(2)) < 1e-3f );
	}

	// Checking setAndNormalize().
	{
		// get 3 perp vectors
		hkVector4 a;
		a.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );

		a.normalize<3>();
		hkVector4 b;
		b.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
		hkVector4 c;
		c.setCross(a, b);
		c.normalize<3>();
		b.setCross(c, a);
		b.normalize<3>();

		hkRotation r;
		r.setCols(a, b, c);
		hkQuaternion q;
		q.setAndNormalize(r);
		HK_TEST( r.isOrthonormal(1e-3f) );
	}

	// Checking setSlerp().
	{
		hkVector4 axis0;
		axis0.set(1, 2, 3);
		hkVector4 axis1;
		axis1.set(1, 2, 4);

		axis0.normalize<3>();
		axis1.normalize<3>();

		hkQuaternion q0; q0.setAxisAngle(axis0, 0.7f);
		hkQuaternion q1; q1.setAxisAngle(axis1, 0.3f);

		hkSimdReal t= hkSimdReal::getConstant<HK_QUADREAL_0>();
		hkQuaternion q3;
		q3.setSlerp(q0,q1,t);
		testDifference(q0, q3, 1e-3f);
	}

	// Checking setReal().
	{
		hkVector4 axis0;
		axis0.set(1,2,3);
		axis0.normalize<3>();
		hkQuaternion q0; q0.setAxisAngle(axis0, 0.9f);
		hkReal r = q0.getReal();
		HK_TEST( hkMath::fabs(r - 0.9f) < 1e-3f );
		r = 0.67f;
		q0.setReal(r);
		r = q0.getReal();
		HK_TEST( hkMath::fabs(r - 0.67f) < 1e-3f );
	}

	// Checking setImag().
	{
		hkVector4 axis0;
		axis0.set(1,2,3);
		axis0.normalize<3>();
		hkQuaternion q0; q0.setAxisAngle(axis0, 0.9f);

		axis0.set(4,5,6);
		q0.setImag(axis0);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( q0.m_vec.allEqual<3>(axis0,eps) );
	}

	// Checking isok().
	{
		hkVector4 axis0;
		axis0.set(1, 2, 3);
		axis0.normalize<3>();
		hkQuaternion q0; q0.setAxisAngle(axis0, 0.9f);
		HK_TEST(q0.isOk() );
	}

	// Checking removeAxisComponent().
	{
		hkVector4 axis0;
		axis0.set(1, 2, 3);
		axis0.normalize<3>();
		hkQuaternion q0; q0.setAxisAngle(axis0, 0.5f);

		q0.removeAxisComponent(axis0);
		hkReal r = q0.getAngle();
		HK_TEST(r == 0);
	}

	// Checking decomposeRestAxis().
	{
		hkVector4 axis0;
		axis0.set(1, 2, 3);
		axis0.normalize<3>();
		hkQuaternion q0; q0.setAxisAngle(axis0, 0.5f);
		hkQuaternion q1;
		hkSimdReal r;
		q0.decomposeRestAxis(axis0, q1, r);
		HK_TEST( hkMath::fabs( r.getReal() - 0.5f ) < 1.0e-3f );
	}
}

static void checkOperators()
{
	hkPseudoRandomGenerator rng(13);

	const hkSimdReal eps = hkSimdReal::fromFloat(1.0e-5f);
	const int numTests = 10000;
	for (int k = 0; k < numTests; k++)
	{
		hkQuaternion q1;	rng.getRandomRotation(q1);
		hkQuaternion q2;	rng.getRandomRotation(q2);
		hkQuaternion iq1;	iq1.setInverse(q1);
		hkQuaternion iq2;	iq2.setInverse(q2);

		hkQuaternion q12;	q12.setMul(q1, q2);
		hkQuaternion q21;	q21.setMul(q2, q1);
		hkQuaternion qi12;	qi12.setMul(iq1, q2);
		hkQuaternion q1i2;	q1i2.setMul(q1, iq2);
		hkQuaternion q2i1;	q2i1.setMul(q2, iq1);
		hkQuaternion qi21;	qi21.setMul(iq2, q1);

		hkMatrix4 preOp1;		hkQuaternionUtil::_computePreMultiplyOperator(q1,preOp1);
		hkMatrix4 preOp2;		hkQuaternionUtil::_computePreMultiplyOperator(q2,preOp2);
		hkMatrix4 postOp1;		hkQuaternionUtil::_computePostMultiplyOperator(q1,postOp1);
		hkMatrix4 postOp2;		hkQuaternionUtil::_computePostMultiplyOperator(q2,postOp2);
		hkMatrix4 invPreOp1;	hkQuaternionUtil::_computeInversePreMultiplyOperator(q1,invPreOp1);
		hkMatrix4 invPostOp1;	hkQuaternionUtil::_computeInversePostMultiplyOperator(q1,invPostOp1);
		hkMatrix4 invPreOp2;	hkQuaternionUtil::_computeInversePreMultiplyOperator(q2,invPreOp2);
		hkMatrix4 invPostOp2;	hkQuaternionUtil::_computeInversePostMultiplyOperator(q2,invPostOp2);

		hkQuaternion q12Pre;	preOp2.multiplyVector(q1.m_vec, q12Pre.m_vec);
		hkQuaternion q21Pre;	preOp1.multiplyVector(q2.m_vec, q21Pre.m_vec);
		hkQuaternion q1i2Pre;	invPreOp2.multiplyVector(q1.m_vec, q1i2Pre.m_vec);
		hkQuaternion q2i1Pre;	invPreOp1.multiplyVector(q2.m_vec, q2i1Pre.m_vec);

		hkQuaternion q12Post;	postOp1.multiplyVector(q2.m_vec, q12Post.m_vec);
		hkQuaternion q21Post;	postOp2.multiplyVector(q1.m_vec, q21Post.m_vec);
		hkQuaternion qi12Post;	invPostOp1.multiplyVector(q2.m_vec, qi12Post.m_vec);
		hkQuaternion qi21Post;	invPostOp2.multiplyVector(q1.m_vec, qi21Post.m_vec);

		HK_TEST(q12Pre.m_vec.allEqual<4>(q12.m_vec, eps));
		HK_TEST(q21Pre.m_vec.allEqual<4>(q21.m_vec, eps));
		HK_TEST(q1i2Pre.m_vec.allEqual<4>(q1i2.m_vec, eps));
		HK_TEST(q2i1Pre.m_vec.allEqual<4>(q2i1.m_vec, eps));

		HK_TEST(q12Post.m_vec.allEqual<4>(q12.m_vec, eps));
		HK_TEST(q21Post.m_vec.allEqual<4>(q21.m_vec, eps));
		HK_TEST(qi12Post.m_vec.allEqual<4>(qi12.m_vec, eps));
		HK_TEST(qi21Post.m_vec.allEqual<4>(qi21.m_vec, eps));
	}

	// Test relative quaternion operators
	for (int k = 0; k < numTests; k++)
	{
		//	Let body quaternions be qA, qB and local pivot quaternions in body space lA, lB. The world quaternions for
		//	the pivots will be (qA * lA) and (qB * lB) respectively, and therefore, the relative quaternion will be:
		//		qPivotRel	= Inverse(qA * lA) * (qB * lB)

		hkQuaternion qA;	rng.getRandomRotation(qA);
		hkQuaternion qB;	rng.getRandomRotation(qB);
		hkQuaternion lA;	rng.getRandomRotation(lA);
		hkQuaternion lB;	rng.getRandomRotation(lB);
		
		hkQuaternion pA;	pA.setMul(qA, lA);
		hkQuaternion pB;	pB.setMul(qB, lB);
		hkQuaternion pRel;	pRel.setInverseMul(pA, pB);	// Relative rotation between pivots
		hkQuaternion qRel;	qRel.setInverseMul(qA, qB);	// Relative rotation between bodies

		// Get operators
		hkMatrix4 TA;	hkQuaternionUtil::_computeInversePostMultiplyOperator(lA, TA);
		hkMatrix4 TB;	hkQuaternionUtil::_computeInversePreMultiplyOperator(lB, TB);

		// Multiply them and extract the first 3 columns
		hkMatrix4 H;		hkMatrix4Util::_computeTransposeMul(TA, TB, H);
		hkQuaternion pr;	hkQuaternionUtil::_computeTransposeMul(H, qRel, pr);
		HK_TEST(pr.m_vec.allEqual<4>(pRel.m_vec, eps));
	}
}

int quaternion_main()
{
	checkMul();
	checkMulMulInv();
	checkMulInvMul();
	checkAngleAxis();
	checkShortestRotation();
	checkNormalize();
	checkConversiontoRotation();
	checkMisc();
	checkOperators();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER(quaternion_main, "Fast", "Common/Test/UnitTest/Base/", "UnitTest/Math/Linear/quaternion.cpp"     );

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
