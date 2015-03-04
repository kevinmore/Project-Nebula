/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Physics2012/Collide/Util/hkpTriangleUtil.h>
#include <stdio.h>

int tricontainspoint_main()
{

	// Basis vectors
	hkVector4 X; X.set(1.f, 0.f, 0.f);
	hkVector4 Y; Y.set(0.f, 1.f, 0.f);
	hkVector4 Z; Z.set(0.f, 0.f, 1.f);

	for (int iter = 0; iter < 100; ++iter)
	{
		// Choose a random orientation
		hkQuaternion orientation;
		{
			hkVector4 axis;
			hkReal angle;
			axis.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
			axis.normalize<3>();
			angle = hkUnitTest::rand01()*HK_REAL_PI;

			orientation.setAxisAngle( axis, angle );
		}

		// Choose a random translation
		hkVector4 offset; offset.set( hkUnitTest::randRange(-100.f, 100.f),
									  hkUnitTest::randRange(-100.f, 100.f),
									  hkUnitTest::randRange(-100.f, 100.f) );

		// Get rotated basis
		hkVector4 u; u.setRotatedDir( orientation, X );
		hkVector4 v; v.setRotatedDir( orientation, Y );
		hkVector4 w; w.setRotatedDir( orientation, Z );

		// Construct a right triangle (0,0) - (0,1) - (1,0)
		// using the new rotated basis
		hkVector4 a = offset;
		hkVector4 b;
		b.setMul( hkVector4::getConstant(HK_QUADREAL_1), u );
		b.add( offset );

		hkVector4 c;
		c.setMul( hkVector4::getConstant(HK_QUADREAL_1), v );
		c.add( offset );

		// Number of grid points to test
		int n = 25;

		// Tessellate a square encasing the triangle into (n*2+1)^2 grid points
		for (int i = -n; i <= n; ++i)
		{
			for (int j = -n; j <= n; ++j)
			{
				// Construct a 2D point in the square
				const hkReal x = (hkReal)i / n;
				const hkReal y = (hkReal)j / n;

				// Transform the 2D point into the rotated basis
				hkVector4 p;
				p.setMul( hkSimdReal::fromFloat(x), u );
				p.addMul( hkSimdReal::fromFloat(y), v );
				p.add( offset );

				// Trivial to test if 2D point is in the right triangle
				// in the unrotated basis.
				hkBool isInTriangle = false;

				// Don't test edge cases -- machine roundoff can produce
				// inconsistent edge cases.
				if (hkMath::fabs(x) > HK_REAL_EPSILON && hkMath::fabs(y) > HK_REAL_EPSILON)
				{
					if (hkMath::fabs(1 - x - y)>HK_REAL_EPSILON)
					{
						if (x > 0 && y > 0)
						{
							if (y < 1 - x)
							{
								isInTriangle = true;
							}
						}

						// Compare isInTriangle with the triangle util result
						const hkBool containsPoint = hkpTriangleUtil::containsPoint( p, a, b, c );
						HK_TEST( containsPoint == isInTriangle );
					}
				}
			}
		}
	}

	return 0;
}

//void ___1() { }
#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(tricontainspoint_main, "Fast", "Physics2012/Test/UnitTest/Collide/", __FILE__ );

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
