/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayBundleCastInput.h>

void HK_CALL hkBundleRays( const hkpShapeRayCastInput rays[4], hkpShapeRayBundleCastInput& bundleOut )
{
	HK_ASSERT(0x5f54833a, rays[0].m_filterInfo == rays[1].m_filterInfo );
	HK_ASSERT(0x5f54833a, rays[2].m_filterInfo == rays[3].m_filterInfo );
	HK_ASSERT(0x5f54833a, rays[0].m_filterInfo == rays[3].m_filterInfo );

	HK_ASSERT(0x258e0c9f, rays[0].m_rayShapeCollectionFilter == rays[1].m_rayShapeCollectionFilter );
	HK_ASSERT(0x258e0c9f, rays[2].m_rayShapeCollectionFilter == rays[3].m_rayShapeCollectionFilter );
	HK_ASSERT(0x258e0c9f, rays[0].m_rayShapeCollectionFilter == rays[3].m_rayShapeCollectionFilter );

	bundleOut.m_from.set(rays[0].m_from, rays[1].m_from, rays[2].m_from, rays[3].m_from);
	bundleOut.m_to.set(rays[0].m_to, rays[1].m_to, rays[2].m_to, rays[3].m_to);

	bundleOut.m_filterInfo = rays[0].m_filterInfo;
	bundleOut.m_rayShapeCollectionFilter = rays[0].m_rayShapeCollectionFilter;
}

void HK_CALL hkUnBundleRays(const hkpShapeRayBundleCastInput& bundleIn, hkpShapeRayCastInput rays[4])
{
	bundleIn.m_from.extract(rays[0].m_from, rays[1].m_from, rays[2].m_from, rays[3].m_from);
	bundleIn.m_to.extract(rays[0].m_to, rays[1].m_to, rays[2].m_to, rays[3].m_to);

	rays[0].m_filterInfo				= bundleIn.m_filterInfo;
	rays[0].m_rayShapeCollectionFilter	= bundleIn.m_rayShapeCollectionFilter;
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
