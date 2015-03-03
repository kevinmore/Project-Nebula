/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/Simulation/hkpSimulation.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>

#include <Physics2012/Dynamics/World/Util/hkpWorldMemoryUtil.h>




void hkpWorldMemoryUtil::watchHeapMemory( hkpWorld* world )
{
	
	if ( world->getMemoryWatchDog() ) 
	{
		int freeHeapRequested = world->getMemoryWatchDog()->getAmountOfFreeHeapMemoryRequested();
		hkMemorySystem& sys = hkMemorySystem::getInstance();
		if( sys.heapCanAllocTotal(freeHeapRequested) == false )
		{
			HK_TIMER_BEGIN("WatchDog:FreeMem", HK_NULL);
			world->getMemoryWatchDog()->freeHeapMemoryTillRequestedAmountIsAvailable( world );
			HK_TIMER_END();
		}
		if ( hkGetOutOfMemoryState() != hkMemoryAllocator::MEMORY_STATE_OK )
		{
			hkSetOutOfMemoryState( hkMemoryAllocator::MEMORY_STATE_OK );
		}
	}
}

void hkpWorldMemoryUtil::checkMemoryForIntegration( hkpWorld* world)
{
	// Code taken directly from LimitedRuntimeBlocksDemo::stepDemo()
	world->lock();
	HK_ASSERT2(0xad907073, world->getMemoryWatchDog(), "Memory watchdog required.");
	int origMinDesiredIslandSize	= world->m_minDesiredIslandSize;

	int numIterations = 0;

	hkWorldMemoryAvailableWatchDog::MemUsageInfo memInfo;
	world->calcRequiredSolverBufferSize( memInfo );
	while ( !hkMemorySystem::getInstance().solverCanAllocSingleBlock( memInfo.m_maxRuntimeBlockSize ) )
	{
		numIterations++;

		while ( memInfo.m_largestSimulationIsland->m_isSparse )
		{
			hkpSimulationIsland* origIsland = memInfo.m_largestSimulationIsland;
			int origRuntimeBlockSize		= memInfo.m_maxRuntimeBlockSize;
			while((memInfo.m_largestSimulationIsland->m_isSparse) && (world->m_minDesiredIslandSize > 0))
			{
				hkpWorldOperationUtil::splitSimulationIsland( world, origIsland );
				world->calcRequiredSolverBufferSize( memInfo );
				if ( memInfo.m_largestSimulationIsland != origIsland || memInfo.m_maxRuntimeBlockSize != origRuntimeBlockSize )
				{
					// split successful, continue
					break;
				}
				HK_WARN( 0xf03465fd, "Your hkpWorld::m_minDesiredIslandSize is bigger than supported by the largest runtime block" );
				world->m_minDesiredIslandSize >>= 1;
			}
			
			// splitSimulationIsland may not be able to split an island, even if it
			// is listed as sparse, so we must still check that we are not looping forever
			if(world->m_minDesiredIslandSize == 0)
			{
				break;
			}
		}

		// reduce the biggest island
		if ( !hkMemorySystem::getInstance().solverCanAllocSingleBlock( memInfo.m_maxRuntimeBlockSize ) )
		{
			// now we have to remove objects from the island
			hkpSimulationIsland* island = memInfo.m_largestSimulationIsland;

			// If minDesiredIslandSize has already been set to 0, run reduce constraints anyway
			// as repeating the above will not split the island further. A not fully connected
			// island is not necessarily splittable
			if ( (! island->isFullyConnected( )) && (world->m_minDesiredIslandSize > 0))
			{	// force above loop, m_isSparse might have been set too late
				island->m_isSparse = true;
				continue;
			}

			world->getMemoryWatchDog()->reduceConstraintsInIsland(memInfo, numIterations);

			world->calcRequiredSolverBufferSize( memInfo );
		}
	}
	world->m_minDesiredIslandSize = origMinDesiredIslandSize;
	world->unlock();
}

void hkpWorldMemoryUtil::tryToRecoverFromMemoryErrors( hkpWorld* world )
{
	if (world->getMemoryWatchDog())
	{
		// Run the world in a single threaded way.
		// This is called recursively upon failures.
		int attemptsLeft = 10;
		while ( world->m_simulation->m_previousStepResult != HK_STEP_RESULT_SUCCESS && attemptsLeft-- > 0)
		{
#		if defined (HK_ENABLE_MEMORY_EXCEPTION_UTIL)
			hkMemoryExceptionTestingUtil::allowMemoryExceptions(false);
#		endif

			world->getMemoryWatchDog()->freeHeapMemoryTillRequestedAmountIsAvailable(world);
			hkSetOutOfMemoryState( hkMemoryAllocator::MEMORY_STATE_OK );
			repeatCollideAndToiHandling(world);
		}

#	if defined (HK_ENABLE_MEMORY_EXCEPTION_UTIL)
		hkSetOutOfMemoryState( hkMemoryAllocator::MEMORY_STATE_OK );
		hkMemoryExceptionTestingUtil::allowMemoryExceptions(true);
#	endif

		HK_ASSERT2(0xad907041, world->m_simulation->m_previousStepResult == HK_STEP_RESULT_SUCCESS, "Critical error: Failed to recover simluation after 10 attempts. Simulation will stall, and you must correct the problem by reducing the hkpWorld's memory.");
	}
}


void hkpWorldMemoryUtil::repeatCollideAndToiHandling( hkpWorld* world )
{
	if (	( world->m_simulation->m_previousStepResult == HK_STEP_RESULT_SUCCESS ) || 
			( world->m_simulation->m_previousStepResult == HK_STEP_RESULT_MEMORY_FAILURE_DURING_COLLIDE ))
	{
		world->m_simulation->collide();
	}

	if (	( world->m_simulation->m_previousStepResult == HK_STEP_RESULT_SUCCESS ) || 
			( world->m_simulation->m_previousStepResult == HK_STEP_RESULT_MEMORY_FAILURE_DURING_TOI_SOLVE ))
	{
		world->m_simulation->advanceTime();
	}
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
