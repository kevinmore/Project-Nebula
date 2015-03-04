/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Collide/ContactListener/hkpContactPointEvent.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>

hkContactPointId hkpContactPointEvent::getContactPointId() const
{
	if ( isToi() )
	{
		return HK_INVALID_CONTACT_POINT;
	}
	else
	{
		const hkpSimpleContactConstraintData& cc = m_contactMgr->m_contactConstraintData;
		const int indexOfPoint = int( this->m_contactPoint - cc.m_atom->getContactPoints() );
		const int contactPointId = cc.m_idMgrA.indexOf( indexOfPoint );
		return hkContactPointId( contactPointId );
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
