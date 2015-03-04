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
#include <Common/Base/Math/Vector/hkVector4Util.h>

int vector3util_main()
{
	// Testing invert2x2Matrix functionality
	{
		hkVector4 m; m.set(1, 2, 2, 3);
		hkVector4 out; out.setZero();
		HK_TEST(hkVector4Util::invert2x2Matrix(m, hkSimdReal_Eps, out) == HK_SUCCESS);
		HK_TEST(hkMath::equal(out(0),-3.0f));
		HK_TEST(hkMath::equal(out(1), 2.0f));
		HK_TEST(hkMath::equal(out(2), 2.0f));
		HK_TEST(hkMath::equal(out(3),-1.0f));
	}

	hkPseudoRandomGenerator random(1337);

	// test hkVector4Util::dot4_4vs4 and dot3_4vs4
	{	
		hkVector4 a1; a1.set(random.getRandReal11(), random.getRandReal11(), random.getRandReal11(), random.getRandReal11());
		hkVector4 a2; a2.set(random.getRandReal11(), random.getRandReal11(), random.getRandReal11(), random.getRandReal11());
		hkVector4 a3; a3.set(random.getRandReal11(), random.getRandReal11(), random.getRandReal11(), random.getRandReal11());
		hkVector4 a4; a4.set(random.getRandReal11(), random.getRandReal11(), random.getRandReal11(), random.getRandReal11());

		hkVector4 b1; b1.set(random.getRandReal11(), random.getRandReal11(), random.getRandReal11(), random.getRandReal11());
		hkVector4 b2; b2.set(random.getRandReal11(), random.getRandReal11(), random.getRandReal11(), random.getRandReal11());
		hkVector4 b3; b3.set(random.getRandReal11(), random.getRandReal11(), random.getRandReal11(), random.getRandReal11());
		hkVector4 b4; b4.set(random.getRandReal11(), random.getRandReal11(), random.getRandReal11(), random.getRandReal11());

		hkVector4 dots3, dots4;
		hkVector4Util::dot4_4vs4(a1, b1, a2, b2, a3, b3, a4, b4, dots4);
		hkVector4Util::dot3_4vs4(a1, b1, a2, b2, a3, b3, a4, b4, dots3);

		HK_TEST( hkMath::equal(dots4(0), a1.dot<4>(b1).getReal()));
		HK_TEST( hkMath::equal(dots4(1), a2.dot<4>(b2).getReal()));
		HK_TEST( hkMath::equal(dots4(2), a3.dot<4>(b3).getReal()));
		HK_TEST( hkMath::equal(dots4(3), a4.dot<4>(b4).getReal()));

		HK_TEST( hkMath::equal(dots3(0), a1.dot<3>(b1).getReal()));
		HK_TEST( hkMath::equal(dots3(1), a2.dot<3>(b2).getReal()));
		HK_TEST( hkMath::equal(dots3(2), a3.dot<3>(b3).getReal()));
		HK_TEST( hkMath::equal(dots3(3), a4.dot<3>(b4).getReal()));
	}

	// Test dot4_1vs4
	{
		hkVector4 vN;	random.getRandomVector11(vN);
		hkVector4 vA;	random.getRandomVector11(vA);
		hkVector4 vB;	random.getRandomVector11(vB);
		hkVector4 vC;	random.getRandomVector11(vC);
		hkVector4 vD;	random.getRandomVector11(vD);

		hkVector4 d0;
		hkVector4Util::dot4_1vs4(vN, vA, vB, vC, vD, d0);

		hkVector4 d1;
		d1.set(vN.dot<4>(vA), vN.dot<4>(vB), vN.dot<4>(vC), vN.dot<4>(vD));

		hkSimdReal tol;	tol.setFromFloat(1.0e-4f);
		HK_TEST(d0.allEqual<4>(d1, tol));
	}

	// Test dot4xyz1_1vs4
	{
		hkVector4 vN;	random.getRandomVector11(vN);
		hkVector4 vA;	random.getRandomVector11(vA);
		hkVector4 vB;	random.getRandomVector11(vB);
		hkVector4 vC;	random.getRandomVector11(vC);
		hkVector4 vD;	random.getRandomVector11(vD);

		hkVector4 d0;
		hkVector4Util::dot4xyz1_1vs4(vN, vA, vB, vC, vD, d0);

		hkVector4 d1;
		d1.set(vN.dot4xyz1(vA), vN.dot4xyz1(vB), vN.dot4xyz1(vC), vN.dot4xyz1(vD));

		hkSimdReal tol;	tol.setFromFloat(1.0e-4f);
		HK_TEST(d0.allEqual<4>(d1, tol));
	}
	// Test cross_3vs1 / cross_4vs1
	{
		hkVector4 vN;	random.getRandomVector11(vN);
		hkVector4 vA;	random.getRandomVector11(vA);
		hkVector4 vB;	random.getRandomVector11(vB);
		hkVector4 vC;	random.getRandomVector11(vC);
		hkVector4 vD;	random.getRandomVector11(vD);

		hkVector4 vAN0, vBN0, vCN0, vDN0;
		hkVector4Util::cross_3vs1(vA, vB, vC, vN, vAN0, vBN0, vCN0);

		hkVector4 vAN1;	vAN1.setCross(vA, vN);
		hkVector4 vBN1;	vBN1.setCross(vB, vN);
		hkVector4 vCN1;	vCN1.setCross(vC, vN);
		hkVector4 vDN1;	vDN1.setCross(vD, vN);

		hkSimdReal tol;	tol.setFromFloat(1.0e-4f);
		HK_TEST(vAN0.allEqual<3>(vAN1, tol));
		HK_TEST(vBN0.allEqual<3>(vBN1, tol));
		HK_TEST(vCN0.allEqual<3>(vCN1, tol));

		hkVector4Util::cross_4vs1(vA, vB, vC, vD, vN, vAN0, vBN0, vCN0, vDN0);
		HK_TEST(vAN0.allEqual<3>(vAN1, tol));
		HK_TEST(vBN0.allEqual<3>(vBN1, tol));
		HK_TEST(vCN0.allEqual<3>(vCN1, tol));
		HK_TEST(vDN0.allEqual<3>(vDN1, tol));
	}

	// Test cyclic cross product
	{
		hkVector4 vA;	random.getRandomVector11(vA);
		hkVector4 vB;	random.getRandomVector11(vB);
		hkVector4 vC;	random.getRandomVector11(vC);

		hkVector4 vAB0, vBC0, vCA0;
		hkVector4Util::computeCyclicCrossProducts(vA, vB, vC, vAB0, vBC0, vCA0);

		hkVector4 vAB1;	vAB1.setCross(vA, vB);
		hkVector4 vBC1;	vBC1.setCross(vB, vC);
		hkVector4 vCA1;	vCA1.setCross(vC, vA);

		hkSimdReal tol;	tol.setFromFloat(1.0e-4f);
		HK_TEST(vAB0.allEqual<3>(vAB1, tol));
		HK_TEST(vBC0.allEqual<3>(vBC1, tol));
		HK_TEST(vCA0.allEqual<3>(vCA1, tol));
	}

	{
		{
			hkVector4 a; a.set( -5, -4, 2.3f, 100);
			hkVector4 b; b.set( -5, +.01f, -99.3f, 9);
			hkVector4 ab;
			ab.setFlipSign(a, b);

			HK_TEST(ab(0) > 0);
			HK_TEST(ab(1) < 0);
			HK_TEST(ab(2) < 0);
			HK_TEST(ab(3) > 0);
		}

		// zero offset, unit scale
		{
			hkVector4 offset;
			offset.setZero();
			hkVector4 scale; scale.set(1,1,1,1);

			hkVector4 a; a.set( 0.99f, 110.49f, 100.89f, 65535.9f);
			HK_ALIGN16(hkIntUnion64 out);
			hkVector4Util::convertToUint16(a, offset, scale, out );

			HK_TEST(out.u16[0] == 0);
			HK_TEST(out.u16[1] == 110);
			HK_TEST(out.u16[2] == 100);
			HK_TEST(out.u16[3] == 65535);
		}

		// nonzero offset, unit scale
		{
			hkVector4 offset; offset.set(-1005, -99, 9901, 32000);
			hkVector4 scale; scale.set(1,1,1,1);

			hkVector4 a; a.set( 1005, 199, 10000, 32000);
			HK_ALIGN16(hkIntUnion64 out);
			hkVector4Util::convertToUint16( a, offset, scale, out );

			HK_TEST(out.u16[0] == 0);
			HK_TEST(out.u16[1] == 100);
			HK_TEST(out.u16[2] == 19901);
			HK_TEST(out.u16[3] == 64000);
		}

		// nonzero offset, nonunit scale
		{
			hkVector4 offset; offset.set(-2000, -10,  199, 32000);
			hkVector4 scale; scale.set(100.0f,    0.1000001f, -200.0f,  1.0f);

			hkVector4 a; a.set( 2000, 110, -399, 32000);
			HK_ALIGN16(hkIntUnion64 out);
			hkVector4Util::convertToUint16( a, offset, scale, out );

			HK_TEST(out.u16[0] == 0);
			HK_TEST(out.u16[1] == 10);
			HK_TEST(out.u16[2] == 40000);
			HK_TEST(out.u16[3] == 64000);
		}
		
		// bad values
		{
			hkVector4 offset;
			offset.setZero();
			hkVector4 scale; scale.set(1,1,1,1);

			hkVector4 a; a.set( -100.0f, -.01f, 65536.0f, 70000.0f);
			HK_ALIGN16(hkIntUnion64 out);
			hkVector4Util::convertToUint16(  a, offset, scale, out );

			//HK_TEST(out.u16[0] == 0); TEST(out.u16[1] == 110); TEST(out.u16[2] == 100); TEST(out.u16[3] == 65535);
		}

		// bad values - clip them
		{ 
			hkVector4 min; min.set(0,0,0,0);
			hkVector4 max; max.set(65535,65535,65535,65535);

			hkVector4 offset;
			offset.setZero();
			hkVector4 scale; scale.set(1,1,1,1);

			hkVector4 a; a.set( -100.0f, -0.10001f, 65536.0f, 70000.0f);
			HK_ALIGN16(hkIntUnion64 out);
			hkVector4Util::convertToUint16WithClip( a, offset, scale, min, max, out );

			HK_TEST(out.u16[0] == 0);
			HK_TEST(out.u16[1] == 0);
			HK_TEST(out.u16[2] == 65535);
			HK_TEST(out.u16[3] == 65535);
		}

		// transform points
		{
			hkPseudoRandomGenerator random2(1);
			for ( int i= 0; i < 100; i++ )
			{
				hkVector4 in;
				random2.getRandomVector11( in );
				
				hkTransform t;
				hkRotation& r = t.getRotation();
				random2.getRandomRotation( r );
				random2.getRandomVector11( t.getTranslation() );

					// do it by hand
				hkVector4 ref3;	// rotation mult
				hkVector4 ref4;	// transform mult
				{
					hkReal v0 = in(0);
					hkReal v1 = in(1);
					hkReal v2 = in(2);
					//hkReal v3 = in(2);

					{
						hkVector4& d = ref3;
						d(0) = t(0,0)*v0 + t(0,1)*v1 + t(0,2)*v2;
						d(1) = t(1,0)*v0 + t(1,1)*v1 + t(1,2)*v2;
						d(2) = t(2,0)*v0 + t(2,1)*v1 + t(2,2)*v2;
						d(3) = t(3,0)*v0 + t(3,1)*v1 + t(3,2)*v2;
					}
					{
						hkVector4& d = ref4;
						d(0) = t(0,0)*v0 + t(0,1)*v1 + t(0,2)*v2 + t(0,3);
						d(1) = t(1,0)*v0 + t(1,1)*v1 + t(1,2)*v2 + t(1,3);
						d(2) = t(2,0)*v0 + t(2,1)*v1 + t(2,2)*v2 + t(2,3);
						d(3) = t(3,0)*v0 + t(3,1)*v1 + t(3,2)*v2 + t(3,3);
					}
				}

				// normal operation
				{
					hkVector4 test3; test3.setRotatedDir( r, in );
					hkVector4 test4; test4.setTransformedPos( t, in );
					hkSimdReal eps; eps.setFromFloat(1e-3f);
					HK_TEST( test3.allEqual<3>( ref3,eps ) );
					HK_TEST( test4.allEqual<3>( ref4,eps ) );

					hkVector4 o3; o3.setRotatedInverseDir( r, test3 );
					hkVector4 o4; o4.setTransformedInversePos( t, test4 );
					HK_TEST( o3.allEqual<3>( in,eps ) );
					HK_TEST( o4.allEqual<3>( in,eps ) );
				}

				// normal inline operation
				{
					hkVector4 test3; test3._setRotatedDir( r, in );
					hkVector4 test4; test4._setTransformedPos( t, in );
					hkSimdReal eps; eps.setFromFloat(1e-3f);
					HK_TEST( test3.allEqual<3>( ref3,eps ) );
					HK_TEST( test4.allEqual<3>( ref4,eps ) );

					hkVector4 o3; o3._setRotatedInverseDir( r, test3 );
					hkVector4 o4; o4._setTransformedInversePos( t, test4 );
					HK_TEST( o3.allEqual<3>( in,eps ) );
					HK_TEST( o4.allEqual<3>( in,eps ) );
				}

				// 
				{
					hkVector4 test3; test3._setRotatedDir( r, in );
					hkVector4 test4; test4._setTransformedPos( t, in );
					hkSimdReal eps; eps.setFromFloat(1e-3f);
					HK_TEST( test3.allEqual<4>( ref3,eps ) );
					HK_TEST( test4.allEqual<4>( ref4,eps ) );
				}

				// 
				{
					hkVector4 test3; test3._setRotatedDir( r, in );
					hkSimdReal eps; eps.setFromFloat(1e-3f);
					HK_TEST( test3.allEqual<3>( ref3,eps ) );
				}


				// normal operation
				{
					hkVector4 test3; hkVector4Util::rotatePoints( r, &in, 1, &test3 );
					hkVector4 test4; hkVector4Util::transformPoints( t, &in, 1, &test4 );
					hkSimdReal eps; eps.setFromFloat(1e-3f);
					HK_TEST( test3.allEqual<3>( ref3,eps ) );
					HK_TEST( test4.allEqual<3>( ref4,eps ) );

					hkVector4 o3; hkVector4Util::rotateInversePoints( r, &test3, 1, &o3 );
					HK_TEST( o3.allEqual<3>( in,eps ) );
				}

				// normal operation
				{
					hkVector4 test3; hkVector4Util::mul4xyz1Points(   t, &in, 1, &test3 );
					hkVector4 test4; hkVector4Util::transformSpheres( t, &in, 1, &test4 );
					hkSimdReal eps; eps.setFromFloat(1e-3f);
					HK_TEST( test3.allEqual<4>( ref4,eps ) );
					HK_TEST( test4.allEqual<3>( ref4,eps ) );
					HK_TEST( hkMath::equal( in(3), test4(3) ) );
				}

				// matrix operation
				{
					hkMatrix3 id; id.setMulInverse( r, r );
					HK_TEST( id.isApproximatelyEqual( hkTransform::getIdentity().getRotation() ) );
				}

				// matrix operation
				{
					hkRotation rr; rr.setMul( r, r );
					hkRotation r3; r3.setMul( r, rr );
					hkRotation rd; rd.setMulInverse( r3, rr );
					HK_TEST( r.isApproximatelyEqual( rd ) );
				}
			}
		}
	}

	// test setPermutation2
	{
		hkVector4 a, b;
		a.set(0,1,2,3);
		b.set(4,5,6,7);

		hkVector4 expect, out;

		
		hkVector4Util::setPermutation2<0,1,2,3>(a, b, out);
		expect.set(0,1,2,3); 
		HK_TEST(out.allExactlyEqual<4>(expect));

		hkVector4Util::setPermutation2<4,5,6,7>(a, b, out);
		expect.set(4,5,6,7); 
		HK_TEST(out.allExactlyEqual<4>(expect));

		hkVector4Util::setPermutation2<0,1,6,7>(a, b, out);
		expect.set(0,1,6,7); 
		HK_TEST(out.allExactlyEqual<4>(expect));

		hkVector4Util::setPermutation2<4,5,2,3>(a, b, out);
		expect.set(4,5,2,3); 
		HK_TEST(out.allExactlyEqual<4>(expect));

		hkVector4Util::setPermutation2<0,2,4,6>(a, b, out);
		expect.set(0,2,4,6); 
		HK_TEST(out.allExactlyEqual<4>(expect));

		hkVector4Util::setPermutation2<7,5,3,1>(a, b, out);
		expect.set(7,5,3,1); 
		HK_TEST(out.allExactlyEqual<4>(expect));

		hkVector4Util::setPermutation2<0,4,0,4>(a, b, out);
		expect.set(0,4,0,4); 
		HK_TEST(out.allExactlyEqual<4>(expect));

		hkVector4Util::setPermutation2<6,6,6,6>(a, b, out);
		expect.set(6,6,6,6); 
		HK_TEST(out.allExactlyEqual<4>(expect));

	}

	{
		// test pack/unpack for quaternion
		hkPseudoRandomGenerator random2(1);
		for (int i=0; i<100; i++)
		{
			hkQuaternion q; 
			random2.getRandomRotation( q );
			hkUint32 i32 = hkVector4Util::packQuaternionIntoInt32( q.m_vec );

			hkQuaternion ref; hkVector4Util::unPackInt32IntoQuaternion(i32, ref.m_vec);
			hkSimdReal eps; eps.setFromFloat(0.04f); // eps see comment on hkVector4Util::unPackInt32IntoQuaternion
			HK_TEST( ref.m_vec.allEqual<4>(q.m_vec, eps));
		}
	}


	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(vector3util_main, "Fast", "Common/Test/UnitTest/Base/", "UnitTest/Math/Linear/vector3util.cpp"     );

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
