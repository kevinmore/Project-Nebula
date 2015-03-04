/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Collide/Deprecated/hkpCollisionEvents.h>

const class hkpDynamicsContactMgr* hkpContactPointConfirmedEvent::getContactMgr() const
{
	int offset = HK_OFFSET_OF( hkpSimpleConstraintContactMgr, m_contactConstraintData );
	const void* cmgr = hkAddByteOffsetConst( m_contactData, -offset );

	const hkpSimpleConstraintContactMgr* contactMgr = reinterpret_cast<const hkpSimpleConstraintContactMgr*>(cmgr);
	return contactMgr;
}


hkContactPointId hkpContactPointConfirmedEvent::getContactPointId() const
{
	if ( isToi() )
	{
		return HK_INVALID_CONTACT_POINT;
	}

	const hkpSimpleContactConstraintData* cc = this->m_contactData;
	int indexOfPoint   = int(this->m_contactPoint - cc->m_atom->getContactPoints());
	int contactPointId = cc->m_idMgrA.indexOf( indexOfPoint );
	return hkContactPointId(contactPointId);
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
