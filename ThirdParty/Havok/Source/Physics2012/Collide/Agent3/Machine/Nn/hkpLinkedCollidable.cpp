/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpLinkedCollidable.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnTrack.h>


#if ! defined (HK_PLATFORM_SPU)
static HK_FORCE_INLINE bool cmpLessCollisionEntries(const hkpLinkedCollidable::CollisionEntry& entry0, const hkpLinkedCollidable::CollisionEntry& entry1)
{
	return entry0.m_partner->getBroadPhaseHandle()->m_id < entry1.m_partner->getBroadPhaseHandle()->m_id;
}


void hkpLinkedCollidable::getCollisionEntriesSorted(hkArray<struct hkpLinkedCollidable::CollisionEntry>& entries) const 
{
	entries = m_collisionEntries;

#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
	// ensure deterministic order for fixed entities
	hkpEntity* entity = (hkpEntity*)getOwner();
	if ( entity->isFixed() ) // sort only if fixed
	{
		hkSort(entries.begin(), entries.getSize(), cmpLessCollisionEntries);	
	}
#endif
}

const hkArray<struct hkpLinkedCollidable::CollisionEntry>& hkpLinkedCollidable::getCollisionEntriesDeterministicUnchecked() const
{
#if defined (HK_ENABLE_DETERMINISM_CHECKS)
	for (int i =0; i < m_collisionEntries.getSize(); i++)
	{
		int partnerId = m_collisionEntries[i].m_partner->getBroadPhaseHandle()->m_id;
		hkCheckDeterminismUtil::checkMt( 0xf00001b8, partnerId);
	}
#endif
	return m_collisionEntries;
}

hkArray<struct hkpLinkedCollidable::CollisionEntry>& hkpLinkedCollidable::getCollisionEntriesDeterministicUnchecked()
{
#if defined (HK_ENABLE_DETERMINISM_CHECKS)
	for (int i =0; i < m_collisionEntries.getSize(); i++)
	{
		int partnerId = m_collisionEntries[i].m_partner->getBroadPhaseHandle()->m_id;
		hkCheckDeterminismUtil::checkMt( 0xf00001b8, partnerId);
	}
#endif
	return m_collisionEntries;
}

void hkpLinkedCollidable::sortEntries()
{
	HK_ASSERT(0XAD234666, hkpGetRigidBody(this) != HK_NULL);
	HK_ACCESS_CHECK_OBJECT( hkpGetRigidBody(this)->getWorld(), HK_ACCESS_RW );

	hkSort(m_collisionEntries.begin(), m_collisionEntries.getSize(), cmpLessCollisionEntries);

	for (int ei = 0; ei < m_collisionEntries.getSize(); ei++)
	{
		CollisionEntry& e = m_collisionEntries[ei];
		int indexOnAgent = (this == e.m_agentEntry->getCollidableB());
		e.m_agentEntry->m_agentIndexOnCollidable[indexOnAgent] = hkObjectIndex(ei);
	}
}
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
