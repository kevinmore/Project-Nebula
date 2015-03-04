/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/BoxBox/hkpBoxBoxManifold.h>
#include <Physics2012/Collide/Agent/hkpCollisionInput.h>
#include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgr.h>


hkpBoxBoxManifold::hkpBoxBoxManifold()
{
	m_numPoints = 0;
	m_faceVertexFeatureCount = 0;
	m_isComplete = false;
}




int hkpBoxBoxManifold::addPoint( const hkpCdBody& bodyA, const hkpCdBody& bodyB, hkpFeatureContactPoint& fcp )
{

	//!me could have a faster lookup for agent specific manifolds.
	int size = m_numPoints;

	//!me
	if( size > HK_BOXBOX_MANIFOLD_MAX_POINTS )
		return -1;

	if ( 1 )
	{
		if ( findInManifold( fcp ) )
		{
			// this point is already in the manifold
			HK_ASSERT(0x72283b85, 0);
			return -1;
		}
	}

	// ok, we've got a new point
	const int i = m_numPoints;
	if( i < HK_BOXBOX_MANIFOLD_MAX_POINTS )
	{
		m_contactPoints[i] = fcp;

		m_numPoints++;

	}
	else
	{
		// out of manifold points
		HK_ASSERT(0x1eca4c57, 0); 
		return -1;
	}

	return i;

}


void hkpBoxBoxManifold::removePoint( int i )
{

	m_isComplete = false;

	m_contactPoints[i] = m_contactPoints[m_numPoints - 1];

	m_numPoints--;


}

#if !defined(HK_REAL_IS_DOUBLE)
	HK_COMPILE_TIME_ASSERT( sizeof(hkpProcessCdPoint) == 48);
	HK_COMPILE_TIME_ASSERT( sizeof(hkpBoxBoxManifold) <= 64 + 32 );
#endif

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
