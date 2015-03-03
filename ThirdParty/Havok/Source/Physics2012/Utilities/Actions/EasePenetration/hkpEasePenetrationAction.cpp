/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Actions/EasePenetration/hkpEasePenetrationAction.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Collide/hkpSimpleConstraintContactMgr.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnTrack.h>


	// This allows you to specify duration of the action. Other values can be set directly after the action is constructed.
hkpEasePenetrationAction::hkpEasePenetrationAction(hkpEntity* entity, hkReal duration) : hkpUnaryAction(entity)
{
	m_duration = duration;
	m_timePassed = 0.0f;
	m_originalAllowedPenetrationDepth = entity->getCollidableRw()->m_allowedPenetrationDepth;

	m_initialAllowedPenetrationDepthMultiplier = 6.0f;
	m_initialAdditionalAllowedPenetrationDepth = 0.0f;

	m_reducePenetrationDistance = true;
	m_initialContactDepthMultiplier = 0.2f;
}

	// Resets m_entity's m_allowedPenetrationDepth to m_originalAllowedPenetrationDepth to handle case where entity is removed from world before m_duration has elapsed
hkpEasePenetrationAction::~hkpEasePenetrationAction()
{
	if (m_entity != HK_NULL)
	{
		m_entity->getCollidableRw()->m_allowedPenetrationDepth = m_originalAllowedPenetrationDepth;
	}
}

	// hkpAction implementation.
void hkpEasePenetrationAction::applyAction( const hkStepInfo& stepInfo )
{
	// a fix for unwanted TOI's
	m_entity->getCollidableRw()->m_allowedPenetrationDepth 
		= hkMath::interpolate2d(m_timePassed, hkReal(0.0f), m_duration, m_originalAllowedPenetrationDepth * m_initialAllowedPenetrationDepthMultiplier + m_initialAdditionalAllowedPenetrationDepth, m_originalAllowedPenetrationDepth);

	// an attempt to fix unwanted jitter
	if (m_reducePenetrationDistance)
	{
		const hkReal multiplier = hkMath::interpolate2d(m_timePassed, hkReal(0), m_duration, m_initialContactDepthMultiplier, hkReal(1));
		hkSimdReal multiplierSr; multiplierSr.load<1>(&multiplier);

		const hkArray<hkpLinkedCollidable::CollisionEntry>& collisions = m_entity->getLinkedCollidable()->getCollisionEntriesNonDeterministic();
		for (int i = 0; i < collisions.getSize(); i++)
		{
			hkpSimpleConstraintContactMgr* mgr = static_cast<hkpSimpleConstraintContactMgr*>(collisions[i].m_agentEntry->m_contactMgr);
			if (mgr->getType() == hkpContactMgr::TYPE_SIMPLE_CONSTRAINT_CONTACT_MGR)
			{
				hkArray<hkContactPointId> ids;
				mgr->getAllContactPointIds(ids);
				for (int j = 0; j < ids.getSize(); j++)
				{
					hkContactPointId id = ids[j];
					hkContactPoint* cp = mgr->getContactPoint(id);
					const hkSimdReal dist = cp->getDistanceSimdReal();
					if (dist.isLessZero())
					{
						cp->setDistanceSimdReal(dist * multiplierSr);
					}
				}	
			}
			else
			{
				HK_WARN_ONCE(0xad343242, "hkpEasePenetrationAction used for an unsupported hkpContactMgr type.");
			}
		}
	}

	// remove action and restore original body properties
	m_timePassed += stepInfo.m_deltaTime;
	if (m_timePassed >= m_duration)
	{
		m_entity->getCollidableRw()->m_allowedPenetrationDepth = m_originalAllowedPenetrationDepth;
		m_entity->getWorld()->removeAction(this);
	}

}

hkpAction* hkpEasePenetrationAction::clone( const hkArray<hkpEntity*>& newEntities, const hkArray<hkpPhantom*>& newPhantoms ) const 
{
	HK_ASSERT2(0x3aff4bcf, false, "This action does not support cloning().");
	return HK_NULL;
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
