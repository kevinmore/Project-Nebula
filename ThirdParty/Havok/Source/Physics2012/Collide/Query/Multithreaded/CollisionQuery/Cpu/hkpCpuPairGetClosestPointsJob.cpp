/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Query/Collector/PointCollector/hkpFixedBufferCdPointCollector.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobs.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuPairGetClosestPointsJob.h>


void HK_CALL hkCpuPairGetClosestPointsImplementation(	const hkpPairGetClosestPointsJob&		pairGetClosestPointsJob,
														const hkpProcessCollisionInput&			collisionInput,
														hkpPairGetClosestPointsCommand*	commandsBase,
														int								numCommands)
{
	//
	// Init collector.
	// Note: we will properly initialize the collector's capacity individually for each command right before calling getClosestPoints().
	//
	hkpRootCdPoint resultsArray[hkpPairGetClosestPointsCommand::MAXIMUM_RESULTS_CAPACITY];
	hkpFixedBufferCdPointCollector collector(&resultsArray[0], 1);

	//
	// create local collision input
	//
	hkpCollisionInput input;
	{
		input.m_tolerance	= pairGetClosestPointsJob.m_tolerance;
		input.m_dispatcher	= collisionInput.m_dispatcher;
		input.m_filter		= collisionInput.m_filter;
	}

	{
		hkpPairGetClosestPointsCommand* command = commandsBase;
		for (int i = 0; i < numCommands; i++ )
		{
			const hkpShape* shapeA = command->m_collidableA->getShape();
			const hkpShape* shapeB = command->m_collidableB->getShape();

			// properly initialize the collector's buffer and capacity
			new (&collector) hkpFixedBufferCdPointCollector(&resultsArray[0], command->m_resultsCapacity);

			//
			// call getClosestPoints function
			//
			{
				hkpCollisionDispatcher::GetClosestPointsFunc getClosestPoints = collisionInput.m_dispatcher->getGetClosestPointsFunc(shapeA->getType(), shapeB->getType());
				getClosestPoints( *command->m_collidableA, *command->m_collidableB, input, collector );
			}

			//
			// write back result array and # of hits
			//
			{
				//
				// Find out where on PPU to write our results to:
				// - if this is a user-built hkpPairGetClosestPointsCommand we can directly write into m_results.
				// - if this hkpPairGetClosestPointsCommand has been automatically built by another job we need to find out where in the shared results array (if at all)
				//   we are allowed to write our results to. See below for details.
				//
				hkpRootCdPoint* destination;
				{
					if ( command->m_indexIntoSharedResults == HK_NULL )
					{
						destination = command->m_results;
					}
					else
					{
						int numHits = collector.getNumHits();

						//
						// If we share m_results between multiple commands we do this:
						// - atomically increase the index pointer. This returns the index where to write our data to.
						// - if our data fits into the remaining array we can safely write.
						// - if it won't fit we need to atomically revert our modification to the index, so that other (smaller) commands are able to write their results to the array.
						//
						int indexIntoResultsArray = hkCriticalSection::atomicExchangeAdd(command->m_indexIntoSharedResults, numHits);
						if ( (indexIntoResultsArray+numHits) <= command->m_resultsCapacity )
						{
							destination = command->m_results + indexIntoResultsArray;
						}
						else
						{
							hkCriticalSection::atomicExchangeAdd(command->m_indexIntoSharedResults, -numHits);
							destination = HK_NULL;
							HK_WARN_ONCE(0xaf131e10, "Shared result array's remaining capacity too small. Current results will be dropped.");
						}
					}
				}

				if ( destination )
				{
					// write back the number of actual results
					command->m_numResultsOut = hkUint16(collector.getNumHits());

					// copy results from intermediate buffer into command's results array
					if ( command->m_numResultsOut > 0 )
					{
						hkString::memCpy16NonEmpty(destination, &resultsArray, command->m_numResultsOut * (sizeof(hkpRootCdPoint)>>4));
					}

				}
			}

			command++;
		}
	}
}


hkJobQueue::JobStatus HK_CALL hkCpuPairGetClosestPointsJob( hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& nextJobOut )
{
	HK_CHECK_FLUSH_DENORMALS();
	HK_TIMER_BEGIN("CollQueryPairGetClosestPoints", HK_NULL);

	const hkpPairGetClosestPointsJob& pairGetClosestPointsJob = reinterpret_cast<hkpPairGetClosestPointsJob&>(nextJobOut);

	hkCpuPairGetClosestPointsImplementation(pairGetClosestPointsJob, *pairGetClosestPointsJob.m_collisionInput, const_cast<hkpPairGetClosestPointsCommand*>(pairGetClosestPointsJob.m_commandArray), pairGetClosestPointsJob.m_numCommands);

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
