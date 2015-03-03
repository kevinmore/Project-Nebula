/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Physics2012/Collide/Util/hkpTriangleUtil.h>

int degeneracy_main()
{
	hkVector4 a, b, c;
	a.set(70.832f, 48.553f, 4.297f, 1.f);
	b.set(81.061f, 43.619f, 5.761f, 1.f);
	c.set(70.575f, 48.674f, 4.261f, 1.f);

	HK_TEST( !hkpTriangleUtil::isDegenerate( a, b, c ) );

	// point triangle distance check
	{
		hkVector4 Q; Q.setSub(a, b);      
		hkVector4 R; R.setSub(c, b);

		const hkReal QQ = hkpTriangleUtil::dot3fullAcc(Q, Q);
		const hkReal RR = hkpTriangleUtil::dot3fullAcc(R, R);
		const hkReal QR = hkpTriangleUtil::dot3fullAcc(R, Q);

		volatile hkReal QQRR = QQ * RR;
		volatile hkReal QRQR = QR * QR;
		hkReal Det = (QQRR - QRQR);

#if defined(HK_REAL_IS_DOUBLE)
		HK_TEST( Det == 0.00098646730839391239 );
#else
		HK_TEST( Det == 0.001953125f );
#endif
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(degeneracy_main, "Fast", "Physics2012/Test/UnitTest/Collide/", __FILE__ );

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
