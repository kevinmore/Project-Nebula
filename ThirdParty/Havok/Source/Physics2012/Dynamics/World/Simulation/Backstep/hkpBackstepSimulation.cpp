/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Physics2012/Dynamics/World/Simulation/Backstep/hkpBackstepSimulation.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Motion/hkpMotion.h>
#include <Common/Base/Types/Physics/MotionState/hkMotionState.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransform.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldAgentUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>

#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>


hkpBackstepSimulation::hkpBackstepSimulation( hkpWorld* world, BackstepMode backstepMode ) 
:	hkpContinuousSimulation( world ), 
	m_backsteppingMode(backstepMode) 
{
}

void hkpBackstepSimulation::simulateToi( hkpWorld* world, hkpToiEvent& event, hkReal physicsDeltaTime, hkReal rotateNormal )
{
	m_world->lockCriticalOperations();

	HK_ASSERT2( 0xf03234fe, world->areCriticalOperationsLocked(), "World is not locked for critical operations");
	{
		for (int e = 0; e < 2; e++)
		{
			hkpRigidBody* body = static_cast<hkpRigidBody*>(event.m_entities[e]);
			hkMotionState& motionState = body->getRigidMotion()->m_motionState;

			if (!body->isFixedOrKeyframed() && motionState.getSweptTransform().getInvDeltaTimeSr().isNotEqualZero())
			{
				// Backstep
				hkSweptTransformUtil::backStepMotionState( event.m_time, motionState );
				
				// Freeze: Set initial and final positions/orientations in hkSweptTransform to present pos (pos1)
				motionState.getSweptTransform().m_centerOfMass0 = motionState.getSweptTransform().m_centerOfMass1;
				motionState.getSweptTransform().m_rotation0 = motionState.getSweptTransform().m_rotation1;

				motionState.getSweptTransform().m_centerOfMass0(3) = event.m_time;
				motionState.getSweptTransform().m_centerOfMass1.zeroComponent<3>(); //1.0f / (world->m_timeOfNextPsi - event.m_time);

				hkpEntity* entity = body;
				resetCollisionInformationForEntities(&entity, 1, world);

				// Skip broadphase collision detection.

				// Collide body inplace
				if (m_backsteppingMode == NON_PENETRATING)
				{
					// Generate new TOI events
					collideEntitiesNarrowPhaseContinuous(&entity, 1, *world->m_collisionInput);
				}
				else if (m_backsteppingMode == SIMPLE)
				{
					// Collide bodies in discrete mode to properly build contact information.
					collideEntitiesNarrowPhaseDiscrete(&entity, 1, *world->m_collisionInput, FIND_CONTACTS_DEFAULT);
				}
			}
		}
 		if ( m_backsteppingMode == SIMPLE  && !(event.m_entities[0]->isFixedOrKeyframed() ^ event.m_entities[1]->isFixedOrKeyframed()) )
		{
			HK_WARN(0xad2345a, "Continuous collision detection performed for dynamic-dynamic (or fixedOrKeyframed-fixedOrKeyframed) pairs of bodies. This simulation (SIMULATION_TYPE_BACKSTEPPING_SIMPLE) does not prevent pairs of dynamic bodies from penetrating.");
		}
	}
	m_world->unlockAndAttemptToExecutePendingOperations();
}

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
