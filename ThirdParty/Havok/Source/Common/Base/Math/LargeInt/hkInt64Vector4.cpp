/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/LargeInt/hkLargeIntTypes.h>

//
//	Returns the dot product between this and v

template <> const hkSimdInt<128> hkInt64Vector4::dot<3>(hkInt64Vector4Parameter vB) const
{
	// Build a 128Vector by multiplying this and v
	hkInt128Vector4 mulV;
	mulV.setMul<3>(*this, vB);

	// Return the sum of the three components of mulV
	mulV.m_w.setAdd(mulV.m_x, mulV.m_y);
	hkSimdInt<128> res;
	res.setAdd(mulV.m_w, mulV.m_z);

	return res;
}

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
