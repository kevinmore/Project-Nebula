/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Physics2012/Dynamics/Entity/hkpEntity.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldAgentUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationQueue.h>

#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>

#include <Physics2012/Dynamics/Collide/Filter/Constraint/hkpConstraintCollisionFilter.h>


#if !defined(HK_PLATFORM_SPU)

hkpConstraintCollisionFilter::hkpConstraintCollisionFilter(const hkpCollisionFilter* childFilter) : hkpPairCollisionFilter(childFilter) 
{
	m_type = HK_FILTER_CONSTRAINT;
}

hkpConstraintCollisionFilter::~hkpConstraintCollisionFilter()
{
}

#endif


#if !defined(HK_PLATFORM_SPU)

void hkpConstraintCollisionFilter::init( hkpWorld* world )
{
	
	if ( world->m_constraintListeners.indexOf( this ) < 0 )
	{
		world->addConstraintListener( this );
	}
}

void hkpConstraintCollisionFilter::updateFromWorld(hkpWorld* world)
{
	hkpPairCollisionFilter::clearAll();

	{
		// The first iteration will process all active islands.
		const hkArray<hkpSimulationIsland*>* islands = &world->getActiveSimulationIslands();

		for (int j = 0; j < 2; j++)
		{
			for (int i = 0; i < islands->getSize(); i++ )
			{
				hkpSimulationIsland* island = (*islands)[i];
				for (int b = 0; b < island->getEntities().getSize(); b++ )
				{
					hkpEntity* body =  island->getEntities()[b];

					int numConstraints = body->getNumConstraints();
					{
						for (int c = 0; c < numConstraints; c++)
						{
							constraintAddedCallback(body->getConstraint(c));
						}
					}
				}
			}

			// The second iteration will process all inactive islands.
			islands = &world->getInactiveSimulationIslands();
		}
	}
}


void hkpConstraintCollisionFilter::constraintAddedCallback( hkpConstraintInstance* constraint )
{
	
	// Collisions do not need to be disabled if one of the bodies does not have a shape
	if ( constraint && (constraint->getData()->getType() != hkpConstraintData::CONSTRAINT_TYPE_CONTACT) &&
		( constraint->getEntityA()->getCollidable()->getShape() ) && ( constraint->getEntityB()->getCollidable()->getShape() ) )
	{
		hkpEntity* entityA = constraint->getEntityA();
		hkpEntity* entityB = constraint->getEntityB();

		int count = disableCollisionsBetween(entityA, entityB);

		// Only remove the agent if this was the first constraint.
		if ( count == 1 )
		{
			// Check if there is an agent connecting the two bodies, if so remove the agent
			hkpAgentNnEntry* entry = hkAgentNnMachine_FindAgent( constraint->getEntityA()->getLinkedCollidable(), constraint->getEntityB()->getLinkedCollidable() );

			if (entry)
			{
				hkpWorld* world = entityA->getWorld();
				if ( !world )
				{
					return;
				}
				world->lockCriticalOperations();
				hkpWorldAgentUtil::removeAgentAndItsToiEvents(entry);
				world->unlockAndAttemptToExecutePendingOperations();
			}
		}
	}
}


void hkpConstraintCollisionFilter::constraintRemovedCallback( hkpConstraintInstance* constraint )
{
	
	// Collisions should not be re-enabled if one of the bodies does not have a shape
	if ( constraint && ( constraint->getData()->getType() != hkpConstraintData::CONSTRAINT_TYPE_CONTACT ) &&
		( constraint->getEntityA()->getCollidable()->getShape() ) && ( constraint->getEntityB()->getCollidable()->getShape() ) )
	{
		hkpEntity* entityA = constraint->getEntityA();
		hkpEntity* entityB = constraint->getEntityB();
		int count = enableCollisionsBetween(entityA, entityB);
		if ( count > 0 )
		{
			return;
		}

		if ( m_childFilter && !m_childFilter->isCollisionEnabled( *entityA->getCollidableMtUnchecked(), *entityB->getCollidableMtUnchecked()  ) )
		{
			return;
		}
		// re enable collision if broadphase overlaps
		hkpWorld* world = entityA->getWorld();
		if ( !world )
		{
			return;
		}
		world->lockCriticalOperations();
		world->reenableCollisionBetweenEntityPair( entityA, entityB );
		world->unlockAndAttemptToExecutePendingOperations();
	}
}

void hkpConstraintCollisionFilter::_constraintBreakingCallbackImmediate( hkpConstraintInstance* instance, hkBool constraintBroken )
{
	if ( constraintBroken )
	{
		constraintRemovedCallback( instance );
	}
	else
	{
		constraintAddedCallback( instance );
	}

}


void hkpConstraintCollisionFilter::constraintBreakingCallback( const hkpConstraintBrokenEvent& event )
{
	if ( event.m_world->areCriticalOperationsLocked() )
	{
		hkWorldOperation::ConstraintCollisionFilterConstraintBroken op;
		op.m_filter = this;
		op.m_constraintInstance = event.m_constraintInstance;
		op.m_constraintBroken = true;
		event.m_world->queueOperation( op );
		return;
	}
	_constraintBreakingCallbackImmediate( event.m_constraintInstance, true );
}

void hkpConstraintCollisionFilter::constraintRepairedCallback( const hkpConstraintRepairedEvent& event )
{
	if ( event.m_world->areCriticalOperationsLocked() )
	{
		hkWorldOperation::ConstraintCollisionFilterConstraintBroken op;
		op.m_filter = this;
		op.m_constraintInstance = event.m_constraintInstance;
		op.m_constraintBroken = false;
		event.m_world->queueOperation( op );
		return;
	}
	_constraintBreakingCallbackImmediate( event.m_constraintInstance, false );
}


#endif

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
