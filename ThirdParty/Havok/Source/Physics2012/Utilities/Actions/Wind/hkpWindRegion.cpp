/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Actions/Wind/hkpWindRegion.h>

hkpWindRegion::hkpWindRegion( hkpAabbPhantom* phantom, const hkpWind* wind, hkReal resistanceFactor, hkReal obbFactor )
:	m_phantom( phantom ),
	m_wind( wind ),
	m_resistanceFactor( resistanceFactor ),
	m_obbFactor( obbFactor )
{
	m_phantom->addReference();
	m_wind->addReference();
}

hkpWindRegion::~hkpWindRegion()
{
	m_wind->removeReference();
	m_phantom->removeReference();
}

void hkpWindRegion::postSimulationCallback( hkpWorld* world )
{
	hkArray<hkpCollidable*>& collidables = m_phantom->getOverlappingCollidables();

	const int numCollidables = collidables.getSize();
	for ( int i = 0; i < numCollidables; ++i )
	{
		hkpRigidBody* rb = hkpGetRigidBody( collidables[i] );

		// Ignore other phantoms and fixed rigid bodies.
		if ( (rb != HK_NULL) && !rb->isFixed() )
		{
			m_wind->applyWindAndResistance( rb, world->m_dynamicsStepInfo.m_stepInfo.m_deltaTime, m_resistanceFactor, m_obbFactor );
		}
	}
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
