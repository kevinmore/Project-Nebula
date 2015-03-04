/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/GeometryUtilities/Mesh/Simplifiers/QemSimplifier/hkQuadricMetric.h>

static int quadricMetric_main()
{
	{
		hkVector4 a; a.set( 1, 2, 3);
		hkVector4 b; b.set( 2, -3, -5);
		hkVector4 c; c.set( 3, -9, 7);

		hkVectorN an(&a(0), 3);
		hkVectorN bn(&b(0), 3);
		hkVectorN cn(&c(0), 3);

		hkQuadricMetric qm;
		qm.setFromPlane(an, bn, cn);

		const hkSimdReal epsilon = hkSimdReal::fromFloat(1e-4f); // dsb below on ARM is not within 1e5, but is close.

		hkSimdReal dsa; dsa.setAbs(qm.calcDistanceSquared(an));
		HK_TEST( dsa < epsilon);

		hkSimdReal dsb; dsb.setAbs(qm.calcDistanceSquared(bn));
		HK_TEST( dsb < epsilon);

		hkSimdReal dsc; dsc.setAbs(qm.calcDistanceSquared(cn));
		HK_TEST( dsc < epsilon);

		HK_TEST( qm.isOk());
		HK_TEST( qm.equals(qm));

		hkQuadricMetric qm2(qm);
		HK_TEST( qm.equals(qm2));

		hkQuadricMetric qm3;
		HK_TEST( !qm.equals(qm3));

		hkArray<hkReal> store;
		store.setSize(qm.getStoreSize());
		qm.store(store.begin());

		qm3.load(qm.getSize(), store.begin());

		HK_TEST( qm.equals(qm3));

		//hkVector4 p(4, 6, 9);
		//hkVectorN pn(&p(0), 3);
	}
	{
		hkQuadricMetric qm1;
		{
			const hkReal v[] = 
			{
				1, 3, 2,
				2, 3, 2,
				2.5, 3, 3,
			};
			qm1.setFromPlane(v, 3);
		}
		hkQuadricMetric qm2;
		{
			const hkReal v[] = 
			{
				3, 1, -2,
				-2, 1, -2,
				1, 2, -2,
			};
			qm2.setFromPlane(v, 3);
		}

		HK_TEST( qm1.getStoreSize() == 10);
		hkVector4 t0[3];
		hkVector4 t1[3];
		hkVector4 r[3];

		qm1.store((hkReal*)t0);
		qm2.store((hkReal*)t1);

		for (int i = 0; i < 3; i++)
		{
			r[i].setAdd(t0[i], t1[i]);
		}

		hkQuadricMetric qm; 
		qm.load(3, (hkReal*)r);

#if 0
		// 
		hkVector4 p0; p0.set(0, 3, -2);
		hkReal dist0 = qm.calcDistanceSquared(p0);

		hkVector4 p1; p1.set(3, 3, -2);
		hkReal dist1 = qm.calcDistanceSquared(p1);


		hkVector4 p2; p2.set(3, 3, 0);
		hkReal dist2 = qm.calcDistanceSquared(p2);


		// 5? or 25?
		hkVector4 p3; p3.set(3, 2, 0);
		hkReal dist3 = qm.calcDistanceSquared(p3);
#endif
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(quadricMetric_main, "Fast", "Common/Test/UnitTest/GeometryUtilities/", __FILE__     );

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
