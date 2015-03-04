/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Vector/hkIntVector.h>

#define HK_QUADINT_SINGLE(X) HK_QUADINT_CONSTANT(X, X, X, X)

HK_ALIGN16(const hkUint32 g_intVectorConstants[HK_QUADINT_END][4]) = 
{
	HK_QUADINT_SINGLE(0),
	HK_QUADINT_SINGLE(1),
	HK_QUADINT_SINGLE(2),
	HK_QUADINT_SINGLE(4),
	HK_QUADINT_CONSTANT(0, 1, 2, 3),
	HK_QUADINT_CONSTANT(0x3f000000, 0x3f000001, 0x3f000002, 0x3f000003),
	HK_QUADINT_SINGLE(3),

	// Permutation constants, for Xbox broadcast(i)
	HK_QUADINT_SINGLE(0x00010203),
	HK_QUADINT_SINGLE(0x04050607),
	HK_QUADINT_SINGLE(0x08090A0B),
	HK_QUADINT_SINGLE(0x0C0D0E0F)
};

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
