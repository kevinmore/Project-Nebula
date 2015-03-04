/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Agent/hkpCollisionAgentConfig.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpFixedBufferCdPointCollector.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobs.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuPairLinearCastJob.h>


void HK_CALL hkCpuPairLinearCastImplementation(	const hkpPairLinearCastJob&		pairLinearCastJob,
												const hkpProcessCollisionInput&	collisionInput,
													  hkpPairLinearCastCommand*	commandsBase,
													  int						numCommands)
{
	//
	// Init collector.
	// Note: we will properly initialize the collector's capacity individually for each command right before calling linearCast().
	//
	hkpRootCdPoint resultsArray[hkpPairLinearCastCommand::MAXIMUM_RESULTS_CAPACITY];
	hkpRootCdPoint startPointResultsArray[hkpPairLinearCastCommand::MAXIMUM_RESULTS_CAPACITY];

	hkpFixedBufferCdPointCollector collector(&resultsArray[0], hkpPairLinearCastCommand::MAXIMUM_RESULTS_CAPACITY);
	hkpFixedBufferCdPointCollector startPointCollector(&startPointResultsArray[0], hkpPairLinearCastCommand::MAXIMUM_RESULTS_CAPACITY);

	//
	// create local collision agent config
	//
	hkpCollisionAgentConfig config;
	{
		config.m_iterativeLinearCastEarlyOutDistance = pairLinearCastJob.m_iterativeLinearCastEarlyOutDistance;
		config.m_iterativeLinearCastMaxIterations	 = pairLinearCastJob.m_iterativeLinearCastMaxIterations;
	}

	//
	// create persistent part of local collision input
	//
	hkpLinearCastCollisionInput input;
	{
		input.m_tolerance	= pairLinearCastJob.m_tolerance;
		input.m_dispatcher	= collisionInput.m_dispatcher;
		input.m_filter		= collisionInput.m_filter;
		input.m_config		= &config;
	}

	{
		hkpPairLinearCastCommand* command = commandsBase;
		for (int i = 0; i < numCommands; i++ )
		{
			const hkpShape* shapeA = command->m_collidableA->getShape();
			const hkpShape* shapeB = command->m_collidableB->getShape();

			//
			// init command-dependent part of local collision input
			//
			{
				hkVector4 dif;
				dif.setSub(command->m_to, command->m_from);
				input.setPathAndTolerance(dif, pairLinearCastJob.m_tolerance);
				input.m_maxExtraPenetration = pairLinearCastJob.m_maxExtraPenetration;
			}

			// properly initialize the collector's buffer and capacity
			new (&collector) hkpFixedBufferCdPointCollector(&resultsArray[0], command->m_resultsCapacity);

			if( command->m_startPointResults )
			{
				// properly initialize the collector's buffer and capacity
				new (&startPointCollector) hkpFixedBufferCdPointCollector(&startPointResultsArray[0], command->m_startPointResultsCapacity);
			}
			

			//
			// call linearCast function
			//
			{
				hkpCollisionDispatcher::LinearCastFunc linearCast = collisionInput.m_dispatcher->getLinearCastFunc(shapeA->getType(), shapeB->getType());
				linearCast( *command->m_collidableA, *command->m_collidableB, input, collector, command->m_startPointResults ? &startPointCollector : HK_NULL );
			}

			//
			// write back result array and # of hits
			//
			{
				// write back the number of actual results
				command->m_numResultsOut = hkUint16(collector.getNumHits());
				command->m_startPointNumResultsOut = hkUint16(startPointCollector.getNumHits());

				// copy results from intermediate buffer into command's results array
				if ( command->m_numResultsOut > 0 )
				{
					hkString::memCpy16NonEmpty(command->m_results, &resultsArray, command->m_numResultsOut * (sizeof(hkpRootCdPoint)>>4));
				}

				if ( command->m_startPointResults && command->m_startPointNumResultsOut > 0 )
				{
					hkString::memCpy16NonEmpty(command->m_startPointResults, &startPointResultsArray, command->m_startPointNumResultsOut * (sizeof(hkpRootCdPoint)>>4));
				}
			}

			command++;
		}
	}
}


hkJobQueue::JobStatus HK_CALL hkCpuPairLinearCastJob( hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& nextJobOut )
{
	HK_CHECK_FLUSH_DENORMALS();
	HK_TIMER_BEGIN("CollQueryPairLinearCast", HK_NULL);

	const hkpPairLinearCastJob& pairLinearCastJob = reinterpret_cast<hkpPairLinearCastJob&>(nextJobOut);

	hkCpuPairLinearCastImplementation(pairLinearCastJob, *pairLinearCastJob.m_collisionInput, const_cast<hkpPairLinearCastCommand*>(pairLinearCastJob.m_commandArray), pairLinearCastJob.m_numCommands);

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
