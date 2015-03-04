/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Collide/ContactListener/Util/hkpCollisionCallbackUtil.h>

#include <Physics2012/Dynamics/Entity/Util/hkpEntityCallbackUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>

void hkpCollisionCallbackUtil::performAttachments( hkpWorld* world )
{
	world->addConstraintListener( this );
	world->addWorldPostSimulationListener( &m_endOfStepCallbackUtil );
}

void hkpCollisionCallbackUtil::performDetachments( hkpWorld* world )
{
	world->removeWorldPostSimulationListener( &m_endOfStepCallbackUtil );
	world->removeConstraintListener( this );
}

void hkpCollisionCallbackUtil::constraintAddedCallback( hkpConstraintInstance* constraint )
{
	HK_TIMER_BEGIN("CollLtUtil", HK_NULL);
	const hkpConstraintData *const data = constraint->getData();
	if ( data->getType() == hkpConstraintData::CONSTRAINT_TYPE_CONTACT )
	{
		const hkpSimpleContactConstraintData *const contactData = static_cast<const hkpSimpleContactConstraintData*>( data );
		hkpSimpleConstraintContactMgr *const mgr = contactData->getSimpleConstraintContactMgr();
		hkpRigidBody *const bodyA = constraint->getRigidBodyA();
		hkpRigidBody *const bodyB = constraint->getRigidBodyB();
		hkpCollisionEvent event( hkpCollisionEvent::SOURCE_WORLD, bodyA, bodyB, mgr );
		hkpWorld *const world = bodyA->getWorld();
		hkpWorldCallbackUtil::fireContactConstraintAddedCallback( world, event );

		event.m_source = hkpCollisionEvent::SOURCE_A;
		hkpEntityCallbackUtil::fireContactConstraintAddedCallback( bodyA, event );

		event.m_source = hkpCollisionEvent::SOURCE_B;
		hkpEntityCallbackUtil::fireContactConstraintAddedCallback( bodyB, event );
	}
	HK_TIMER_END();
}

void hkpCollisionCallbackUtil::constraintRemovedCallback( hkpConstraintInstance* constraint )
{
	HK_TIMER_BEGIN("CollLfUtil", HK_NULL);
	const hkpConstraintData *const data = constraint->getData();
	if ( data->getType() == hkpConstraintData::CONSTRAINT_TYPE_CONTACT )
	{
		const hkpSimpleContactConstraintData *const contactData = static_cast<const hkpSimpleContactConstraintData*>( data );
		hkpSimpleConstraintContactMgr *const mgr = contactData->getSimpleConstraintContactMgr();
		hkpRigidBody *const bodyA = constraint->getRigidBodyA();
		hkpRigidBody *const bodyB = constraint->getRigidBodyB();
		hkpCollisionEvent event( hkpCollisionEvent::SOURCE_WORLD, bodyA, bodyB, mgr );
		hkpWorld *const world = bodyA->getWorld();
		hkpWorldCallbackUtil::fireContactConstraintRemovedCallback( world, event );

		event.m_source = hkpCollisionEvent::SOURCE_A;
		hkpEntityCallbackUtil::fireContactConstraintRemovedCallback( bodyA, event );

		event.m_source = hkpCollisionEvent::SOURCE_B;
		hkpEntityCallbackUtil::fireContactConstraintRemovedCallback( bodyB, event );
	}
	HK_TIMER_END();
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
