/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuPostCollideJob.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>


hkJobQueue::JobStatus HK_CALL hkCpuPostCollideJob(	hkpMtThreadStructure&		tl,
													hkJobQueue*					jobQueue,
													hkJobQueue::JobQueueEntry&	nextJobOut )
{
	hkpPostCollideJob& job = reinterpret_cast<hkpPostCollideJob&>(nextJobOut);

	HK_TIMER_BEGIN_LIST("NarrowPhase", "PostCollide");

	hkpSimulationIsland* island = job.m_island;

	HK_ON_DEBUG_MULTI_THREADING( island->m_allowIslandLocking = true );
	tl.m_world->lockIslandForConstraintUpdate( island );
	{
		int numTotalTasks = job.m_header->m_numTotalTasks;
		for (int task = 0 ; task < numTotalTasks; task++ )
		{
			if ( task < numTotalTasks-4 ) { hkMath::forcePrefetch<64>( job.m_header->getJobInfo(task+4) ); }

			hkpAgentSectorHeader::JobInfo* jobInfo = job.m_header->getJobInfo(task);
			island->m_constraintInfo.merge( jobInfo->m_constraintInfo );

			if ( jobInfo->m_commandQueue.m_size )
			{
				HK_MONITOR_ADD_VALUE( "numCmds", float(jobInfo->m_commandQueue.m_size >> 4), HK_MONITOR_TYPE_INT);
				hkPhysicsCommandMachineProcess( tl.m_world, jobInfo->m_commandQueue.m_start, jobInfo->m_commandQueue.m_size );
			}
		}
#	ifdef HK_ENABLE_EXTENSIVE_WORLD_CHECKING
		if (jobQueue)
		{
			island->markAllEntitiesReadOnly();
			island->isValid();
			island->unmarkAllEntitiesReadOnly();
		}
#	endif
	}

#if 0 && defined(HK_DEBUG)
	{
		//
		// check whether every agent is consistent
		//
		HK_FOR_ALL_AGENT_ENTRIES_BEGIN( island->m_agentTrack, entry)
		{
			hkpDynamicsContactMgr* mgr = (hkpDynamicsContactMgr*)entry->m_contactMgr;
			hkpConstraintInstance* ci = mgr->getConstraintInstance();
			if ( ci )
			{
				hkArray<hkContactPointId> ids;
				mgr->getAllContactPointIds( ids );
				HK_ASSERT(0xa0cb1fa, (ids.getSize()!=0) == (ci->getOwner() != HK_NULL) );
			}
		}
		HK_FOR_ALL_AGENT_ENTRIES_END;
	}
#endif

	// Deallocate any shape key tracks.
	const int numShapeKeyTracks = job.m_header->m_numShapeKeyTracks;
	if ( numShapeKeyTracks )
	{
		hkpShapeKeyTrack *const shapeKeyTracks = job.m_header->m_shapeKeyTracks;
#if defined(HK_DEBUG)
		for ( int i = 0; i < job.m_header->m_numShapeKeyTracks; ++i )
		{
			shapeKeyTracks[i].checkEmpty();
		}
#endif
		hkDeallocateChunk( shapeKeyTracks, numShapeKeyTracks, HK_MEMORY_CLASS_COLLIDE );
	}

	job.m_header->deallocate();

	tl.m_world->unlockIslandForConstraintUpdate( island );
	HK_ON_DEBUG_MULTI_THREADING( island->m_allowIslandLocking = false );
	HK_TIMER_END_LIST();
	if ( jobQueue )
	{
		return jobQueue->finishJobAndGetNextJob( (const hkJobQueue::JobQueueEntry*)&job, nextJobOut );
	}
	else
	{
		return hkJobQueue::ALL_JOBS_FINISHED;
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
