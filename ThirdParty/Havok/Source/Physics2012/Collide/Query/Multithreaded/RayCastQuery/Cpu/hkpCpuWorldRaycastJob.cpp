/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/BroadPhase/3AxisSweep/hkp3AxisSweep.h>
#include <Physics2012/Collide/Filter/Group/hkpGroupFilter.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/hkpRayCastQueryJobs.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/Util/hkpCollisionQueryUtil.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/Cpu/hkpCpuWorldRaycastJob.h>
#include <Physics2012/Collide/Query/Collector/RayCollector/hkpFixedBufferRayHitCollector.h>
#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastOutput.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>


hkReal hkCpuWorldRayCastCollector::addBroadPhaseHandle( const hkpBroadPhaseHandle* broadPhaseHandle, int castIndex )
{
	const hkpCollidable*	collidable	= static_cast<hkpCollidable*>( static_cast<const hkpTypedBroadPhaseHandle*>(broadPhaseHandle)->getOwner() );
	const hkpShape*			shape		= collidable->getShape();

	if( shape )
	{
		if ( m_filter->isCollisionEnabled( *m_originalInput, *collidable ) )
		{
			const hkTransform& transform = collidable->getTransform();

			m_workInput.m_from._setTransformedInversePos( transform, m_originalInput->m_from );
			m_workInput.m_to  ._setTransformedInversePos( transform, m_originalInput->m_to   );
			m_workInput.m_collidable = collidable;
			m_workInput.m_userData = m_originalInput->m_userData; 

			hkpShapeRayCastOutput output;
			output.m_hitFraction = m_hitFraction;

			if (!m_fixedBufferRayHitCollector)
			{
				// RayCast.
				hkBool hit = shape->castRay( m_workInput, output);

				// If we have a hit, we try to insert the result into the array. This might drop the furthest hit from the array.
				if ( hit )
				{
					hkReal maxHitFraction = 0.0f;
					hkpWorldRayCastOutput* insertAt = //hkpWorldRayCastJobUtil::getNextFreeResult(m_command, m_result, m_nextFreeResult, output.m_hitFraction);
						hkpRayCastJobUtil<hkpWorldRayCastCommand, hkpWorldRayCastOutputPpu>::getNextFreeResult(m_command, m_result, m_nextFreeResult, 
						output.m_hitFraction, maxHitFraction);

					// Insert if there's still room left in the array OR our current hit is closer than the furthest hit in the array.
					if ( insertAt )
					{
						// Transform results back to world space
						output.m_normal._setRotatedDir( transform.getRotation(), output.m_normal );

						// transfer results into result array;
						hkString::memCpy16NonEmpty(insertAt, &output, sizeof(hkpWorldRayCastOutput) >> 4);

						insertAt->m_rootCollidable = collidable;

						output.m_hitFraction = (m_command->m_stopAfterFirstHit)? 0.0f : output.m_hitFraction;
						m_hitFraction = (m_command->m_resultsCapacity == 1)? output.m_hitFraction : maxHitFraction;
						m_hit = true;
						m_earlyOutHitFraction.setAll( m_hitFraction );
					}
				}
			}
			else
			{
				shape->castRayWithCollector(m_workInput, *collidable, *m_fixedBufferRayHitCollector);
				m_command->m_numResultsOut = m_fixedBufferRayHitCollector->m_numOutputs;
				m_earlyOutHitFraction.setAll( m_fixedBufferRayHitCollector->m_earlyOutHitFraction );
			}
		}
	}

	return m_hitFraction;
}


void HK_CALL castRayBroadPhase(const hkpBroadPhase* broadphase, hkpWorldRayCastCommand* command, hkCpuWorldRayCastCollector* collector, hkpFixedBufferRayHitCollector* fixedBufferHitCollector)
{
	// Create cast ray input
	hkpBroadPhase::hkpCastRayInput rayInput;
	{
		rayInput.m_numCasts	= 1;
		rayInput.m_from		=  command->m_rayInput.m_from;
		rayInput.m_toBase	= &command->m_rayInput.m_to;
	}

	// Set the command-dependent values that are needed by the collector during the addBroadPhaseHandle() callback
	{
		collector->m_originalInput			= &command->m_rayInput;
		collector->m_workInput.m_filterInfo	=  command->m_rayInput.m_filterInfo;
		collector->m_result					=  command->m_results;
		collector->m_nextFreeResult			=  command->m_results;
		collector->m_command					= command;

		if ( command->m_rayInput.m_enableShapeCollectionFilter )
		{
			collector->m_workInput.m_rayShapeCollectionFilter = collector->m_filter;
		}
		else
		{
			collector->m_workInput.m_rayShapeCollectionFilter = HK_NULL;
		}

		if ( command->m_useCollector )
		{
			collector->m_fixedBufferRayHitCollector = new (fixedBufferHitCollector) hkpFixedBufferRayHitCollector(command->m_results, command->m_resultsCapacity);
			registerFixedBufferRayHitCollectorAddRayHitCallbackFunction(hkpFixedBufferRayHitCollector::addRayHitImplementation);
		}
		else
		{
			collector->m_fixedBufferRayHitCollector = HK_NULL;
		}
	}

	command->m_results->reset();

	// query the broadphase
	broadphase->markForRead();
	broadphase->castRay( rayInput, collector, 0 );
	broadphase->unmarkForRead();
}

hkJobQueue::JobStatus HK_CALL hkCpuWorldRayCastJob( hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& nextJobOut )
{
	HK_CHECK_FLUSH_DENORMALS();
	HK_TIMER_BEGIN("CollQueryWorldRayCast", HK_NULL);

	const hkpWorldRayCastJob& worldRayCastJob = reinterpret_cast<hkpWorldRayCastJob&>(nextJobOut);

	HK_ASSERT( 0x5c2fc2ed, worldRayCastJob.m_broadphase );

	// create collector and init some persistent and/or command-independent values
	hkCpuWorldRayCastCollector collector;
	{
		collector.m_filter = static_cast<const hkpCollisionFilter*>( worldRayCastJob.m_collisionInput->m_filter.val() );
	}

	hkpFixedBufferRayHitCollector fixedBufferRayHitCollector(HK_NULL, 1); // passing size of 1 to skip an assert
	{
		hkpWorldRayCastCommand* commandArray = const_cast<hkpWorldRayCastCommand*>( worldRayCastJob.m_commandArray );
		for (int i = 0; i < worldRayCastJob.m_numCommands; i++)
		{
			collector.resetHitInfo();
			castRayBroadPhase(worldRayCastJob.m_broadphase, commandArray, &collector, &fixedBufferRayHitCollector);
			commandArray++;
		}

		int numBRcCommands = worldRayCastJob.m_numBundleCommands;
		hkpWorldRayCastBundleCommand* bundledCommandArray = const_cast<hkpWorldRayCastBundleCommand*>(worldRayCastJob.m_bundleCommandArray);

		for(int i=0; i<numBRcCommands; i++)
		{
			for(int iray=0; iray<bundledCommandArray->m_numActiveRays; iray++)
			{
				hkpWorldRayCastCommand singleCommand;
				singleCommand.m_rayInput = bundledCommandArray->m_rayInput[ iray ];
				singleCommand.m_results = bundledCommandArray->m_results[iray];
				singleCommand.m_resultsCapacity = 1;
				singleCommand.m_numResultsOut = 0;
				singleCommand.m_useCollector = bundledCommandArray->m_useCollector;
				
				collector.resetHitInfo();
				castRayBroadPhase(worldRayCastJob.m_broadphase, &singleCommand, &collector, &fixedBufferRayHitCollector);

				bundledCommandArray->m_numResultsOut[iray] = singleCommand.m_numResultsOut;
			}
			bundledCommandArray++;
		}
	}

	HK_TIMER_END();

	return jobQueue.finishJobAndGetNextJob( &nextJobOut, nextJobOut, hkJobQueue::WAIT_FOR_NEXT_JOB );
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
