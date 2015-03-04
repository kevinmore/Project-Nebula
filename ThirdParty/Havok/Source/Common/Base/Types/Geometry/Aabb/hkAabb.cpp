/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>

bool hkAabb::isValid() const
{
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkBool32 minOK = m_min.isOk<3>();
	hkBool32 maxOK = m_max.isOk<3>();
	hkBool32 minLEmax = m_min.lessEqual(m_max).allAreSet<hkVector4ComparisonMask::MASK_XYZ>();
	return minOK && maxOK && minLEmax;
#else
	for(int i = 0; i < 3; ++i)
	{
		if(		hkMath::isFinite(m_min(i)) == false
			||	hkMath::isFinite(m_max(i)) == false
			||	m_min(i) > m_max(i) )
		{
			return false;
		}
	}
	return true;
#endif
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
