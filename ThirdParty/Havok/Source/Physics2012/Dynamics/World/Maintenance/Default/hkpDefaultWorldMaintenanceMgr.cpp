/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Common/Base/Monitor/hkMonitorStream.h>


#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldAgentUtil.h>
#include <Physics2012/Dynamics/World/Maintenance/Default/hkpDefaultWorldMaintenanceMgr.h>
#include <Physics2012/Dynamics/World/Simulation/Continuous/hkpContinuousSimulation.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

#if defined(HK_PLATFORM_HAS_SPU) && defined(HK_DEBUG)
#	include <Physics2012/Collide/Filter/hkpCollisionFilter.h>
#endif

hkpDefaultWorldMaintenanceMgr::hkpDefaultWorldMaintenanceMgr()
{
	m_minAllowedTimeValue = 32.0f + 1.0f;
	m_maxAllowedTimeValue = 64.0f - 1.0f;
}

void hkpDefaultWorldMaintenanceMgr::init( hkpWorld* world )
{
	world->m_simulation->setCurrentTime( hkTime(m_minAllowedTimeValue) );
	world->m_simulation->setCurrentPsiTime( hkTime(m_minAllowedTimeValue) );
}



void hkpDefaultWorldMaintenanceMgr::resetWorldTime( hkpWorld* world, hkStepInfo& stepInfo)
{
	HK_ACCESS_CHECK_OBJECT( world, HK_ACCESS_RW );

	//
	// Reset time information for the entire world
	//
	hkReal stepDeltaTime  = stepInfo.m_endTime  - stepInfo.m_startTime;

	// reset stepInfo
	hkStepInfo& newStepInfo = world->m_dynamicsStepInfo.m_stepInfo;

	HK_ASSERT(0xcaa5e0d, newStepInfo.m_startTime == world->m_simulation->getCurrentPsiTime());

	{
		newStepInfo.m_startTime = hkTime(m_minAllowedTimeValue);
		newStepInfo.m_endTime   = hkTime(m_minAllowedTimeValue + stepDeltaTime);
		newStepInfo.m_deltaTime = stepDeltaTime;
		newStepInfo.m_invDeltaTime = 1.0f / stepDeltaTime;
	}

	hkReal warpDeltaTime = newStepInfo.m_startTime - stepInfo.m_startTime;

		// reset time variables in hkpWorld
	{
		world->m_simulation->setCurrentTime( world->m_simulation->getCurrentTime() + warpDeltaTime );
		world->m_simulation->setCurrentPsiTime( newStepInfo.m_startTime );
	}

	if (world->m_simulation->getSimulateUntilTime() != -1)
	{
		world->m_simulation->setSimulateUntilTime( world->m_simulation->getSimulateUntilTime() + warpDeltaTime );
	}


		// reset time in all swept transforms and agents
	const hkArray<hkpSimulationIsland*>& islands = world->getActiveSimulationIslands();
	{
		for (int i = 0; i < islands.getSize(); i++)
		{
			hkpSimulationIsland* island = islands[i];

			for (int e = 0; e < island->m_entities.getSize(); e++)
			{
				hkpRigidBody* body = static_cast<hkpRigidBody*>(island->m_entities[e]);
				hkMotionState* ms = body->getRigidMotion()->getMotionState();
				ms->getSweptTransform().m_centerOfMass0(3) += warpDeltaTime;
			}

			// reset time in all agents
			hkpWorldAgentUtil::warpTime(island, stepInfo.m_endTime, newStepInfo.m_endTime, *world->m_collisionInput);
		}
	}

	//
	// Call hkpSimulation::warpTime() to update whatever variables needed.
	//
	world->m_simulation->warpTime( warpDeltaTime );


	stepInfo = newStepInfo;
	world->m_collisionInput->m_stepInfo = newStepInfo;
}

#if 0
	// checks deactivators and sets m_active (status_to_be) status for those islands
void hkpDefaultWorldMaintenanceMgr::markIslandsForDeactivationDeprecated( hkpWorld* world, hkStepInfo& stepInfo)
{
	HK_ACCESS_CHECK_OBJECT( world, HK_ACCESS_RW );

	if ( world->m_minDesiredIslandSize != 0 )
	{
		HK_WARN( 0xf0323454, "Requesting old style deactivation, this will also disable the world->m_minDesiredIslandSize optimization."
			" As a result the engine will run slower in multithreaded mode if the physics scene contains lots of small unconnected objects");
	}
	world->m_minDesiredIslandSize = 0;

	const hkArray<hkpSimulationIsland*>& islands = world->getActiveSimulationIslands();

	{
		for (int i = islands.getSize()-1; i>=0; i--)
		{
			hkpSimulationIsland* activeIsland = islands[i];
			HK_ASSERT(0x3b3ca726,  activeIsland->m_storageIndex == i );
			if ( activeIsland->shouldDeactivateDeprecated( stepInfo ) )
			{
				// the island has requested deactivation
				hkpWorldOperationUtil::markIslandInactive(world, activeIsland);
			}
		}
	}
}
#endif

void hkpDefaultWorldMaintenanceMgr::performMaintenance( hkpWorld* world, hkStepInfo& stepInfo )
{
	HK_ACCESS_CHECK_OBJECT( world, HK_ACCESS_RW );

	HK_TIMER_BEGIN_LIST("Maintenance", "Split");

	hkpWorldOperationUtil::splitSimulationIslands(world);

	if (stepInfo.m_startTime >= m_maxAllowedTimeValue)
	{
		HK_TIMER_SPLIT_LIST("ResetTime");
		resetWorldTime(world, stepInfo);
	}

// 	if (world->m_wantOldStyleDeactivation)
// 	{
// 		HK_TIMER_SPLIT_LIST("CheckDeactOld");
// 		markIslandsForDeactivationDeprecated(world, stepInfo);
// 	}

	HK_TIMER_END_LIST();
}

void hkpDefaultWorldMaintenanceMgr::performMaintenanceNoSplit( hkpWorld* world, hkStepInfo& stepInfo )
{
	HK_ACCESS_CHECK_OBJECT( world, HK_ACCESS_RW );

	HK_TIMER_BEGIN_LIST("Maintenance", "ResetTime");

#if defined(HK_PLATFORM_HAS_SPU) && defined(HK_DEBUG)
	if ( (world->getCollisionFilter()->m_type != hkpCollisionFilter::HK_FILTER_GROUP) &&
		 (world->getCollisionFilter()->m_type != hkpCollisionFilter::HK_FILTER_NULL ))
	{
		HK_WARN_ONCE(0xaf351ee1, "You are using a collision filter other than the hkpGroupFilter. Only the hkpGroupFilter is supported on SPU by default; other filter logic will not be used. You still need to make sure that your filter is 16-byte aligned.");
	}
#endif

	if (stepInfo.m_startTime >= m_maxAllowedTimeValue)
	{
		resetWorldTime(world, stepInfo);
	}

// 	if (world->m_wantOldStyleDeactivation)
// 	{
// 		HK_TIMER_SPLIT_LIST("CheckDeact");
// 		markIslandsForDeactivationDeprecated(world, stepInfo);
// 	}

	HK_TIMER_END_LIST();
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
