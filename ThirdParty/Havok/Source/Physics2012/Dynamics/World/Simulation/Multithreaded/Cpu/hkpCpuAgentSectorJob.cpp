/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>
#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>

#include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgr.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuAgentSectorJob.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDeferredConstraintOwner.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpMultithreadedSimulation.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/Entity/hkpEntity.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Collide/hkpSimpleConstraintContactMgr.h>

#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>


hkJobQueue::JobStatus HK_CALL hkCpuAgentSectorJob(	hkpMtThreadStructure&		tl,
													hkJobQueue&					jobQueue,
													hkJobQueue::JobQueueEntry&	nextJobOut )
{
	hkpAgentBaseJob& job = reinterpret_cast<hkpAgentBaseJob&>(nextJobOut);

	hkpProcessCollisionInput inputCopy = tl.m_collisionInput;
	inputCopy.m_stepInfo = job.m_stepInfo;

	hkpDeferredConstraintOwner constraintOwner;
	HK_ALIGN16( hkpProcessCollisionOutput ) processOutput( &constraintOwner );

	hkCheckDeterminismUtil::checkMt(0xf0000060, job.m_numElements);

	if ( job.m_jobSubType == hkpDynamicsJob::DYNAMICS_JOB_AGENT_SECTOR )
	{
		hkpAgentSectorBaseJob& sectorJob = reinterpret_cast<hkpAgentSectorBaseJob&>(job);

	    // Just do the narrowphase for one island here, for now
	    hkpSimulationIsland* island = job.m_island;
    
	    HK_TIMER_BEGIN("NarrowPhase", HK_NULL);
    
	    island->markAllEntitiesReadOnly();
	    island->getMultiThreadCheck().markForRead();
	    {
    	    int sectorIndex = 0;
    
			HK_ON_DETERMINISM_CHECKS_ENABLED( hkCheckDeterminismUtil::checkMt( 0xf0000070, island->m_uTag); )
			hkCheckDeterminismUtil::checkMt( 0xf0000071, sectorJob.m_numElements );
			//hkCheckDeterminismUtil::checkMt( 0xf0000072, sectorJob.m_sectorIndex);
    
			int numSectors = job.m_numElements - 1;
			hkpAgentNnSector *const * sectorPtr = reinterpret_cast<hkpAgentNnSector *const *>( job.m_elements ) + sectorIndex;
		    hkpAgentNnEntry* entry = (*sectorPtr)->getBegin();
			// Obtain the track's agent size from the first entry.
			const int agentSize = hkpAgentNnTrack::getAgentSize( entry->m_nnTrackType );
			int sectorSize = ( job.m_numElements > 1 ) ? HK_AGENT3_SECTOR_SIZE : sectorJob.m_bytesUsedInLastSector;
		    hkpAgentNnEntry* endEntry = hkAddByteOffset(entry, sectorSize );

		    hkpAgentNnEntry* nextEntry  = hkAddByteOffset( entry, agentSize );
		    hkpAgentNnEntry* next2Entry = hkAddByteOffset( nextEntry, agentSize );
		    if ( nextEntry >= endEntry ) { nextEntry = HK_NULL; next2Entry = HK_NULL; }
		    else if ( next2Entry >= endEntry ){ next2Entry = HK_NULL; }
    
		    do
		    {
				hkCheckDeterminismUtil::checkMt( 0xad000405, hkpGetRigidBody(entry->getCollidableA())->getUid() );
				hkCheckDeterminismUtil::checkMt( 0xad000406, hkpGetRigidBody(entry->getCollidableB())->getUid() );
				hkCheckDeterminismUtil::checkMt( 0xad000407, hkpGetRigidBody(entry->getCollidableA())->m_storageIndex );
				hkCheckDeterminismUtil::checkMt( 0xad000408, hkpGetRigidBody(entry->getCollidableB())->m_storageIndex );

			    // perform collide
			    {
					hkCpuProcessAgentHelperFunc(entry, inputCopy, processOutput, tl.m_simulation );
					
				    if ( constraintOwner.m_constraintAddRemoveCounter == 0 )
				    {
					    constraintOwner.m_callbackRequestForAddConstraint = 0;
				    }
				    else if ( constraintOwner.m_constraintAddRemoveCounter > 0 )
				    {
					    constraintOwner.m_commandQueue.addCommand( hkpAddConstraintToCriticalLockedIslandPhysicsCommand( constraintOwner.m_constraintForCommand, constraintOwner.m_callbackRequestForAddConstraint ) );
					    constraintOwner.m_callbackRequestForAddConstraint = 0;
				    }
				    else
				    {
					    HK_ASSERT( 0xf0235456, constraintOwner.m_callbackRequestForAddConstraint == 0);
					    constraintOwner.m_commandQueue.addCommand( hkpRemoveConstraintFromCriticalLockedIslandPhysicsCommand( constraintOwner.m_constraintForCommand ) );
				    }
				    constraintOwner.m_constraintAddRemoveCounter = 0;
			    }
    
			    // advance pointers
			    entry = nextEntry;
			    nextEntry = next2Entry;
    
				if (hkMemoryStateIsOutOfMemory(4) )
				{
					break;
				}

			    // advance next2entry
			    if ( next2Entry)
			    {
				    next2Entry = hkAddByteOffset( next2Entry, agentSize );
				    if ( next2Entry >= endEntry )
				    {
					    // sector completely consumed, jump to next sector
					    ++sectorIndex;
						++sectorPtr;
					    if ( --numSectors >= 0)
					    {
						    hkpAgentNnSector* nsector = *sectorPtr;
						    next2Entry = nsector->getBegin();
						    if ( numSectors > 0 )
						    {
							    hkMath::forcePrefetch<HK_AGENT3_SECTOR_SIZE>( *( sectorPtr + 1 ) );
						    }
							sectorSize = ( numSectors > 0 ) ? HK_AGENT3_SECTOR_SIZE : sectorJob.m_bytesUsedInLastSector;
						    endEntry   = hkAddByteOffset(next2Entry, sectorSize );
					    }
					    else
					    {
						    // we reached the end
						    next2Entry = HK_NULL;
						    continue;
					    }
				    }
    #if (defined(HK_PLATFORM_PS3_PPU) || defined(HK_PLATFORM_XBOX360))
				    const int colidableOffset = HK_OFFSET_OF( hkpEntity, m_collidable );
				    const int motionOffset    = HK_OFFSET_OF( hkpEntity, m_motion );
				    const int offset = motionOffset - colidableOffset;
				    hkMath::prefetch128 ( next2Entry->m_contactMgr );
				    hkMath::prefetch128 ( next2Entry->getCollidableA() );
				    hkMath::prefetch128 ( hkAddByteOffsetConst(next2Entry->getCollidableA(), offset) );
				    hkMath::prefetch128 ( next2Entry->getCollidableB() );
				    hkMath::prefetch128 ( hkAddByteOffsetConst(next2Entry->getCollidableB(), offset) );
    #endif
			    }
		    }
		    while (entry);
    
		    if ( !job.m_header)
		    {
				// now we are the only job for this island, apply the changes immediately
				const hkpConstraintInfoSpu2& ci = constraintOwner.m_constraintInfo;
				int t = ci.m_maxSizeOfSchema | ci.m_numSolverElemTemps | ci.m_numSolverResults | ci.m_sizeOfSchemas;
				if ( t || constraintOwner.m_commandQueue.m_size )
				{
					HK_ON_DEBUG_MULTI_THREADING( island->m_allowIslandLocking = true; );
					island->getMultiThreadCheck().unmarkForRead();
					tl.m_world->lockIslandForConstraintUpdate( island );
					island->m_constraintInfo.merge( ci );
					if ( constraintOwner.m_commandQueue.m_size )
					{
						hkPhysicsCommandMachineProcess( tl.m_world, constraintOwner.m_commandQueue.m_start, constraintOwner.m_commandQueue.m_size );
					}
					tl.m_world->unlockIslandForConstraintUpdate( island );
					island->getMultiThreadCheck().markForRead();
					
					HK_ON_DEBUG_MULTI_THREADING( island->m_allowIslandLocking = false; );
				}
		    }
		    else
		    {
			    // copy the command queue to the JobInfo, so it can be processed later
			    hkpAgentSectorHeader::JobInfo* sh = job.m_header->getJobInfo(job.m_taskIndex);
			    sh->m_constraintInfo = constraintOwner.m_constraintInfo;
    
			    int size = constraintOwner.m_commandQueue.m_size;
			    if ( size )
			    {
				    hkString::memCpy16NonEmpty( sh->m_commandQueue.m_start, constraintOwner.m_commandQueue.m_start, size>>4);
				    sh->m_commandQueue.m_size = size;
			    }
		    }
	    }
	    island->getMultiThreadCheck().unmarkForRead();
	    island->unmarkAllEntitiesReadOnly();
    
	    HK_TIMER_NAMED_END("NarrowPhase");
	}
	else
	{
		HK_ASSERT( 0xa34bc8a9, job.m_jobSubType == hkpDynamicsJob::DYNAMICS_JOB_AGENT_NN_ENTRY );

		HK_TIMER_BEGIN("NarrowPhaseTOI", HK_NULL);

		hkpAgentNnEntryBaseJob& entryJob = reinterpret_cast<hkpAgentNnEntryBaseJob&>(job);

		int elementBaseIndex = 0;

		{
			for (int e = 0; e < job.m_numElements; e++)
			{
				hkpAgentNnEntry* entry = reinterpret_cast<hkpAgentNnEntry*>( job.m_elements[elementBaseIndex + e] );

				hkCheckDeterminismUtil::checkMt( 0xad000401, hkpGetRigidBody(entry->getCollidableA())->getUid() );
				hkCheckDeterminismUtil::checkMt( 0xad000402, hkpGetRigidBody(entry->getCollidableB())->getUid() );
				hkCheckDeterminismUtil::checkMt( 0xad000403, hkpGetRigidBody(entry->getCollidableA())->m_storageIndex );
				hkCheckDeterminismUtil::checkMt( 0xad000404, hkpGetRigidBody(entry->getCollidableB())->m_storageIndex );

				switch(entryJob.m_collisionQualityOverride)
				{
				case hkpContinuousSimulation::PROCESS_NORMALLY: 
					inputCopy.m_collisionQualityInfo = inputCopy.m_dispatcher->getCollisionQualityInfo( entry->m_collisionQualityIndex );
					break;
				case hkpContinuousSimulation::DISABLE_TOIS: 
					inputCopy.m_collisionQualityInfo = inputCopy.m_dispatcher->getCollisionQualityInfo( hkpCollisionDispatcher::COLLISION_QUALITY_PSI );
					break;
				case hkpContinuousSimulation::DISABLE_TOIS_AND_EXPAND_MANIFOLD: 
					inputCopy.m_collisionQualityInfo = inputCopy.m_dispatcher->getCollisionQualityInfo( hkpCollisionDispatcher::COLLISION_QUALITY_TMP_EXPAND_MANIFOLD );
					if (entry->m_collisionQualityIndex == hkpCollisionDispatcher::COLLISION_QUALITY_CHARACTER )
					{
						inputCopy.m_collisionQualityInfo = inputCopy.m_dispatcher->getCollisionQualityInfo( hkpCollisionDispatcher::COLLISION_QUALITY_CHARACTER );
					}

					break;
				default:
					HK_ASSERT2(0xad239412, false, "Undefined override value.");
				}

				hkpCollisionQualityInfo* origInfo = inputCopy.m_dispatcher->getCollisionQualityInfo( entry->m_collisionQualityIndex );
				inputCopy.m_createPredictiveAgents = origInfo->m_useContinuousPhysics;

				// perform collide
				{
					{
						processOutput.reset();

						hkAgentNnMachine_ProcessAgent( entry, inputCopy, processOutput, entry->m_contactMgr );

						if ( !processOutput.isEmpty() )
						{
							hkpCollidable* collA = entry->getCollidableA();
							hkpCollidable* collB = entry->getCollidableB();

							entry->m_contactMgr->processContact( *collA, *collB, inputCopy, processOutput );
						}

						if ( processOutput.hasToi() )
						{
							HK_ASSERT2(0xad234353, entryJob.m_collisionQualityOverride == hkpContinuousSimulation::PROCESS_NORMALLY, "No TOIs expected.");
							HK_ASSERT( 0xf0324f54, inputCopy.m_stepInfo.m_startTime <= processOutput.m_toi.m_time );
							HK_ASSERT2(0xad876fdd, processOutput.m_toi.m_time >= tl.m_simulation->getCurrentTime(), "Generating a TOI event before hkpWorld->m_currentTime.");

							tl.m_simulation->addToiEventWithCriticalSectionLock(processOutput, *entry, &tl.m_simulation->m_toiQueueCriticalSection );
						}
					}

					if ( constraintOwner.m_constraintAddRemoveCounter == 0 )
					{
						constraintOwner.m_callbackRequestForAddConstraint = 0;
					}
					else if ( constraintOwner.m_constraintAddRemoveCounter > 0 )
					{
						constraintOwner.m_commandQueue.addCommand( hkpAddConstraintToCriticalLockedIslandPhysicsCommand( constraintOwner.m_constraintForCommand, constraintOwner.m_callbackRequestForAddConstraint ) );
						constraintOwner.m_callbackRequestForAddConstraint = 0;
					}
					else
					{
						HK_ASSERT( 0xf0235456, constraintOwner.m_callbackRequestForAddConstraint == 0);
						constraintOwner.m_commandQueue.addCommand( hkpRemoveConstraintFromCriticalLockedIslandPhysicsCommand( constraintOwner.m_constraintForCommand ) );
					}
					constraintOwner.m_constraintAddRemoveCounter = 0;

					if (hkMemoryStateIsOutOfMemory(5 ) )
					{
						break;
					}
				}
			}
		}

		// copy the command queue to the JobInfo, so it can be processed later
		{
			hkpAgentSectorHeader::JobInfo* sh = job.m_header->getJobInfo(job.m_taskIndex);
			sh->m_constraintInfo = constraintOwner.m_constraintInfo;

			int size = constraintOwner.m_commandQueue.m_size;
			if ( size )
			{
				hkString::memCpy16NonEmpty( sh->m_commandQueue.m_start, constraintOwner.m_commandQueue.m_start, size>>4);
				sh->m_commandQueue.m_size = size;
			}
		}

		HK_TIMER_NAMED_END("NarrowPhaseTOI");
	}

	hkCheckDeterminismUtil::checkMt(0xf0000076, 0xbdbdbdbdul);

	return jobQueue.finishJobAndGetNextJob( (const hkJobQueue::JobQueueEntry*)&job, nextJobOut );
}

HK_COMPILE_TIME_ASSERT( sizeof( hkpAddConstraintToCriticalLockedIslandPhysicsCommand ) <= hkpPhysicsCommandQueue::BYTES_PER_COMMAND );
HK_COMPILE_TIME_ASSERT( sizeof( hkpRemoveConstraintFromCriticalLockedIslandPhysicsCommand ) <= hkpPhysicsCommandQueue::BYTES_PER_COMMAND );

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
