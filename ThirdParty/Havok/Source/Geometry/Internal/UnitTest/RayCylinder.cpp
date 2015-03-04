/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Internal/hkcdInternal.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastCylinder.h>
#include <Geometry/Internal/Types/hkcdRay.h>
#include <Common/Base/Memory/Util/hkMemUtil.h>

static float castHexToFloat(unsigned int hex)
{
	float f; 
	hkMemUtil::memCpy(&f, &hex, sizeof(hex));
	return f;
}

static void setVector4Hex(hkVector4& vec,unsigned int a, unsigned int  b, unsigned int  c, unsigned int d)
{
	vec.set( castHexToFloat(a), castHexToFloat(b), castHexToFloat(c), castHexToFloat(d) );
}


static void testdRayCastCylinder(bool shouldHit, bool shouldInverseHit, const hkcdRay& ray, hkVector4Parameter vP, hkVector4Parameter vQ, hkSimdRealParameter radius, hkSimdReal* HK_RESTRICT fractionInOut, hkVector4* HK_RESTRICT  normalOut)
{
	hkSimdReal frac = *fractionInOut;
	hkBool32 hit;

	*fractionInOut = frac;
	hit = hkcdRayCastCylinder(ray,vP,vQ,radius, fractionInOut, normalOut);
	HK_TEST( (hit != hkFalse32) == shouldHit );

	// Swap the cylinder points and test again.
	*fractionInOut = frac;
	hit = hkcdRayCastCylinder(ray,vQ,vP,radius, fractionInOut, normalOut);
	HK_TEST( (hit != hkFalse32) == shouldHit );

	// Swap the ray's start and end and test again.
	hkcdRay ray2;
	hkVector4 endp; ray.getEndPoint(endp);
	ray2.setEndPoints( endp, ray.m_origin );

	*fractionInOut = frac;
	hit = hkcdRayCastCylinder(ray2,vP,vQ,radius, fractionInOut, normalOut);
	HK_TEST( (hit != hkFalse32) == shouldInverseHit );

	*fractionInOut = frac;
	hit = hkcdRayCastCylinder(ray2,vQ,vP,radius, fractionInOut, normalOut);
	HK_TEST( (hit != hkFalse32) == shouldInverseHit );
}


static void rayCylinder()
{
	// Simple tests using a vertical cylinder.
	{
		hkVector4 vP; vP.set(0.0f, -1.0, 0.0f);
		hkVector4 vQ; vQ.set(0.0f, 1.0, 0.0f);
		hkSimdReal radius = hkSimdReal::fromFloat(10.0);

		hkVector4 testPos[3];
		testPos[0].set( 0.0f, -1.6f, 0.0f );
		testPos[1].set( 0.0f, 0.1f, 0.0f );
		testPos[2].set( 0.0f, 1.1f, 0.0f );
		
		const int testCount = 8;
		hkReal angleInc = (2.0f * HK_REAL_PI) / ( (hkReal) testCount );
		
		hkVector4 normalOut;
		hkSimdReal fractionInOut;

		for (int i = 0; i < 3; ++i)
		{
			float astep = 0.0f;
			for ( int j = 0; j < testCount; ++j, astep+=1.0f )
			{
				hkReal angle = astep * angleInc;
				hkReal x; x = hkMath::sin(angle);
				hkReal y; y = hkMath::cos(angle);
				hkVector4 posDir; posDir.set( x, 0.0f, y );
				
				// test short parallel rays inside the radius
				{
					hkSimdReal dist = radius * hkSimdReal_Half;

					hkVector4 orig;
					orig.setAddMul( testPos[i], posDir, dist );
					hkVector4 dir;
					dir.set( 0.0f, 0.5f, 0.0f );

					hkcdRay ray;
					ray.setOriginDirection( orig, dir );

					fractionInOut = hkSimdReal_1;
					testdRayCastCylinder(false, false, ray,vP,vQ,radius, &fractionInOut, &normalOut);
					// for i 0 or 2, we are outside and do not touch because of direction and fraction.
					// for i 1 we are inside
				}

				// test long parallel rays inside the radius
				{
					hkSimdReal dist = radius * hkSimdReal_Half;

					hkVector4 orig; orig.setAddMul( testPos[i], posDir, dist );
					hkVector4 rayDir; rayDir.set( 0.0f, (i >= 2 ? -1.5f : 1.5f), 0.0f );

					hkcdRay ray;
					ray.setOriginDirection( orig, rayDir );

					fractionInOut = hkSimdReal_1;
					testdRayCastCylinder(i == 0 || i == 2, i == 1, ray,vP,vQ,radius, &fractionInOut, &normalOut);
					// for i 0 or 2, we are outside and do hit.
					// for i 1 we are inside and cannot hit
					// the reverse is true for inverse rays.
				}

				// test short parallel rays outside the radius
				{
					hkSimdReal dist = radius * hkSimdReal_1 +  ( hkSimdReal_Half*hkSimdReal_Half );

					hkVector4 orig;
					orig.setAddMul( testPos[i], posDir, dist );
					hkVector4 dir;
					dir.set( 0.0f, 0.5f, 0.0f );

					hkcdRay ray;
					ray.setOriginDirection( orig, dir );

					fractionInOut = hkSimdReal_1;
					testdRayCastCylinder(false, false, ray,vP,vQ,radius, &fractionInOut, &normalOut);
					//never hit.
				}

				// test long parallel rays outside the radius
				{
					hkSimdReal dist = radius * hkSimdReal_1 + ( hkSimdReal_Half*hkSimdReal_Half );

					hkVector4 orig; orig.setAddMul( testPos[i], posDir, dist );
					hkVector4 rayDir; rayDir.set( 0.0f, (i >= 2 ? -1.5f : 1.5f), 0.0f );

					hkcdRay ray;
					ray.setOriginDirection( orig, rayDir );

					fractionInOut = hkSimdReal_1;
					testdRayCastCylinder(false, false, ray,vP,vQ,radius, &fractionInOut, &normalOut);
					// never hit.
				}

				// test rays on the horizontal plane that only miss because of height (y).
				{
					hkSimdReal dist = radius * hkSimdReal_2;

					hkVector4 start; start.setAddMul( testPos[i], posDir, dist );
					hkVector4 rayDir; rayDir.setMul( posDir, hkSimdReal_0 -  hkSimdReal_2 * dist );
					hkVector4 end; end.setAddMul( start, rayDir, hkSimdReal_1 );

					hkcdRay ray;
					ray.setEndPoints( start, end );

					fractionInOut = hkSimdReal_1;
					testdRayCastCylinder(i == 1, i == 1, ray,vP,vQ,radius, &fractionInOut, &normalOut);
					// for i 0 or 2, we are outside vertically
					// for i 1 we hit
				}
			}
		}
	}


	// Special unit test that is known to fail for some naive implementations where the quadratic (for infinite cylinder versus ray) 
	// is negative (e.g: x87 or Xenon) even though it should theoretically be zero when the ray is parallel to the cylinder.
	{
		hkVector4 vP; 
		hkVector4 vQ; 
		hkSimdReal radius = hkSimdReal::fromFloat(10.0);
		hkcdRay ray;

		hkVector4 normalOut;
		hkSimdReal fractionInOut;

		setVector4Hex(ray.m_origin,  0x3f20cffd, 0x3f645df8, 0x3fef8999, 0xbf7ffffc );
		hkVector4 dir; setVector4Hex(dir, 0x00000000, 0xc043126c, 0x000000000, 0x00000000 ); 
		ray.setDirection(dir);
		setVector4Hex( vP, 0x80000000, 0x3eea161e, 0x80000000, 0x40c3126f );
		setVector4Hex( vQ, 0x00000000, 0xbeea161e, 0x00000000, 0x40c3126f );

		fractionInOut = hkSimdReal_1;
		testdRayCastCylinder(true, true, ray,vP,vQ,radius, &fractionInOut, &normalOut);
	}
}



int RayCylinder_main()
{
	rayCylinder();

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER(RayCylinder_main, "Fast", "Geometry/Test/UnitTest/Internal/", __FILE__ );

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
