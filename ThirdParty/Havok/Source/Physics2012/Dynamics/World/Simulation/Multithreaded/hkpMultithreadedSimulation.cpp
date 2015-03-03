/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/DebugUtil/TraceStream/hkTraceStream.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>
#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>

#include <Common/Base/Thread/Job/ThreadPool/hkJobThreadPool.h>

#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>

#include <Physics2012/Collide/BroadPhase/hkpBroadPhase.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhaseHandle.h>

#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseHandlePair.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseDispatcher.h>

#include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgr.h>
#include <Physics2012/Collide/Agent/hkpCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>
#include <Physics2012/Collide/Util/Welding/hkpWeldingUtility.h>

#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>

#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>
#include <Physics/ConstraintSolver/Constraint/Bilateral/hkp1dBilateralConstraintInfo.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Action/hkpAction.h>

#include <Physics2012/Dynamics/Phantom/hkpPhantom.h>
#include <Physics2012/Dynamics/Motion/Util/hkpRigidMotionUtil.h>

#include <Physics2012/Dynamics/Constraint/Setup/hkpConstraintSolverSetup.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldConstraintUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldAgentUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationQueue.h>
#include <Physics2012/Dynamics/World/Maintenance/hkpWorldMaintenanceMgr.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldMemoryUtil.h>
#include <Physics2012/Dynamics/World/CommandQueue/hkpPhysicsCommandQueue.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDeferredConstraintOwner.h>


#include <Physics/ConstraintSolver/Solve/hkpSolve.h>
#include <Physics2012/Internal/Dynamics/World/Simulation/Continuous/ToiResourceMgr/hkpToiResourceMgr.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpMultithreadedSimulation.h>

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuIntegrateJob.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuSingleThreadedJobsOnIsland.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuBuildAccumulatorsJob.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuBuildJacobiansJob.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuSolveConstraintsJobs.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuIntegrateMotionJob.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuSplitSimulationIslandJob.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuAgentSectorJob.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuPostCollideJob.h>

#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnTrack.h>

#include <Common/Base/Config/hkOptionalComponent.h>

////////////////////////////////////////////////////////////////////////
//
// Collision Agent helper functions
//
////////////////////////////////////////////////////////////////////////


inline hkpAgentNnEntry* addAgentHelperFunc( hkpLinkedCollidable* collA, hkpLinkedCollidable* collB, const hkpProcessCollisionInput* input )
{
	hkpCollidableQualityType qt0 = collA->getQualityType();
	hkpCollidableQualityType qt1 = collB->getQualityType();
	hkChar collisionQuality = input->m_dispatcher->getCollisionQualityIndex( qt0, qt1 );
	if ( collisionQuality == hkpCollisionDispatcher::COLLISION_QUALITY_INVALID )
	{
		return HK_NULL;
	}
	hkpCollisionQualityInfo* origInfo = input->m_dispatcher->getCollisionQualityInfo( collisionQuality );
	input->m_createPredictiveAgents = origInfo->m_useContinuousPhysics;

	return hkpWorldAgentUtil::addAgent(collA, collB, *input);
}


////////////////////////////////////////////////////////////////////////
//
// Job Dispatch loop
//
////////////////////////////////////////////////////////////////////////

hkJobQueue::JobStatus HK_CALL hkpMultiThreadedSimulation::processNextJob( hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& job )
{
	hkpDynamicsJob& dynamicsJob = reinterpret_cast<hkpDynamicsJob&>(job);

	HK_ON_DETERMINISM_CHECKS_ENABLED(hkCheckDeterminismUtil::Fuid jobFuid = dynamicsJob.getFuid());
	HK_ON_DETERMINISM_CHECKS_ENABLED(hkCheckDeterminismUtil::registerAndStartJob(jobFuid));

	hkBool jobWasCancelled = false;
	hkJobQueue::JobStatus jobStatus;
	switch ( dynamicsJob.m_jobSubType )
	{
		// Note: Each job is responsible for getting the next job from the job queue.
		// The 'job' parameter passed into each job function gets overwritten inside
		// the function with the 'next job to be processed'.
		case hkpDynamicsJob::DYNAMICS_JOB_INTEGRATE:
		{
			jobStatus = integrateJob( *dynamicsJob.m_mtThreadStructure, jobQueue, job, jobWasCancelled );
			break;
		}

		case hkpDynamicsJob::DYNAMICS_JOB_BUILD_ACCUMULATORS:
		{
			jobStatus = hkCpuBuildAccumulatorsJob( *dynamicsJob.m_mtThreadStructure, jobQueue, job );
			break;
		}

		case hkpDynamicsJob::DYNAMICS_JOB_FIRE_JACOBIAN_SETUP_CALLBACK:
		{
			jobStatus = hkpSingleThreadedJobsOnIsland::cpuFireJacobianSetupCallbackJob( *dynamicsJob.m_mtThreadStructure, jobQueue, job );
			break;
		}

		case hkpDynamicsJob::DYNAMICS_JOB_BUILD_JACOBIANS:
		{
			jobStatus =  hkCpuBuildJacobiansJob( *dynamicsJob.m_mtThreadStructure, jobQueue, job );
			break;
		}

		case hkpDynamicsJob::DYNAMICS_JOB_SOLVE_CONSTRAINTS:
		{
			jobStatus = hkCpuSolveConstraintsJob( *dynamicsJob.m_mtThreadStructure, jobQueue, job );
			break;
		}

		case hkpDynamicsJob::DYNAMICS_JOB_SOLVE_APPLY_GRAVITY:
		{
			jobStatus = hkCpuSolveApplyGravityJob( *dynamicsJob.m_mtThreadStructure, jobQueue, job );
			break;
		}

		case hkpDynamicsJob::DYNAMICS_JOB_SOLVE_CONSTRAINT_BATCH:
		{
			jobStatus = hkCpuSolveConstraintBatchJob( *dynamicsJob.m_mtThreadStructure, jobQueue, job );
			break;
		}

		case hkpDynamicsJob::DYNAMICS_JOB_SOLVE_INTEGRATE_VELOCITIES:
		{
			jobStatus = hkCpuSolveIntegrateVelocitiesJob( *dynamicsJob.m_mtThreadStructure, jobQueue, job );
			break;
		}

		case hkpDynamicsJob::DYNAMICS_JOB_SOLVE_EXPORT_RESULTS:
		{
			jobStatus = hkCpuSolveExportResultsJob( *dynamicsJob.m_mtThreadStructure, jobQueue, job );
			break;
		}

		case hkpDynamicsJob::DYNAMICS_JOB_INTEGRATE_MOTION:
		{
			jobStatus = hkCpuIntegrateMotionJob( *dynamicsJob.m_mtThreadStructure, jobQueue, job );
			break;
		}

		case hkpDynamicsJob::DYNAMICS_JOB_SPLIT_ISLAND:
		{
			jobStatus = hkCpuSplitSimulationIslandJob( *dynamicsJob.m_mtThreadStructure, jobQueue, job );
			break;
		}

		case hkpDynamicsJob::DYNAMICS_JOB_BROADPHASE:
		{
			// Broad phase can add an agent sector OR a NOP job (if there's no narrow phase resulting
			// from the broad phase).
			jobStatus = hkpSingleThreadedJobsOnIsland::cpuBroadPhaseJob( *dynamicsJob.m_mtThreadStructure, jobQueue, job );
			break;
		}

		case hkpDynamicsJob::DYNAMICS_JOB_AGENT_SECTOR:
		case hkpDynamicsJob::DYNAMICS_JOB_AGENT_NN_ENTRY:
		{
			// Agent sector always adds a NOP job, however we don't actually add one;
			// we just fall through to the NOP case below.
			jobStatus = hkCpuAgentSectorJob( *dynamicsJob.m_mtThreadStructure, jobQueue, job );
			break;
		}

		case hkpDynamicsJob::DYNAMICS_JOB_POST_COLLIDE:
		{
			jobStatus = hkCpuPostCollideJob( *dynamicsJob.m_mtThreadStructure, &jobQueue, job );
			break;
		}

		default:
		{
			HK_ASSERT2(0xafe1a256, 0, "Internal error - unknown job type");
			jobStatus = hkJobQueue::NO_JOBS_AVAILABLE;
			break;
		}
	}
	HK_ON_DETERMINISM_CHECKS_ENABLED(if(!jobWasCancelled){ hkCheckDeterminismUtil::finishJob(jobFuid, false); });

	return jobStatus;
}

////////////////////////////////////////////////////////////////////////
//
// Special broadphase listener to lock phantoms
//
////////////////////////////////////////////////////////////////////////





void hkpMultiThreadedSimulation::MtPhantomBroadPhaseListener::addCollisionPair( hkpTypedBroadPhaseHandlePair& pair )
{
	//hkCriticalSectionLock lock( m_criticalSection ); updateAabb+dispatching now is atomic
	if ( pair.getElementA()->getType() == hkpWorldObject::BROAD_PHASE_PHANTOM )
	{
		hkpCollidable* collA = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_a)->getOwner() );
		hkpCollidable* collB = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_b)->getOwner() );
		hkpPhantom* p = static_cast<hkpPhantom*>( collA->getOwner() );
		p->markForWrite();
		p->addOverlappingCollidable( collB );
		p->unmarkForWrite();
	}

	if ( pair.getElementB()->getType() == hkpWorldObject::BROAD_PHASE_PHANTOM )
	{
		hkpCollidable* collA = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_a)->getOwner() );
		hkpCollidable* collB = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_b)->getOwner() );
		hkpPhantom* p = static_cast<hkpPhantom*>( collB->getOwner() );
		p->markForWrite();
		p->addOverlappingCollidable( collA );
		p->unmarkForWrite();
	}
}

void hkpMultiThreadedSimulation::MtPhantomBroadPhaseListener::removeCollisionPair( hkpTypedBroadPhaseHandlePair& pair )
{
	//hkCriticalSectionLock lock( m_criticalSection ); updateAabb+dispatching now is atomic
	if ( pair.getElementA()->getType() == hkpWorldObject::BROAD_PHASE_PHANTOM )
	{
		hkpCollidable* collA = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_a)->getOwner() );
		hkpCollidable* collB = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_b)->getOwner() );
		hkpPhantom* p = static_cast<hkpPhantom*>( collA->getOwner() );
		p->markForWrite();
		p->removeOverlappingCollidable( collB );
		p->unmarkForWrite();
	}

	if ( pair.getElementB()->getType() == hkpWorldObject::BROAD_PHASE_PHANTOM )
	{
		hkpCollidable* collA = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_a)->getOwner() );
		hkpCollidable* collB = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_b)->getOwner() );
		hkpPhantom* p = static_cast<hkpPhantom*>( collB->getOwner() );
		p->markForWrite();
		p->removeOverlappingCollidable( collA );
		p->unmarkForWrite();
	}
}




////////////////////////////////////////////////////////////////////////
//
// Special broadphase listener to delay adding of new pairs
//
////////////////////////////////////////////////////////////////////////




void hkpMultiThreadedSimulation::MtEntityEntityBroadPhaseListener::addCollisionPair( hkpTypedBroadPhaseHandlePair& pair )
{
	hkpLinkedCollidable* collA = static_cast<hkpLinkedCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_a)->getOwner() );
	hkpLinkedCollidable* collB = static_cast<hkpLinkedCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_b)->getOwner() );

	hkpEntity* entityA = static_cast<hkpEntity*>( collA->getOwner() );
	hkpEntity* entityB = static_cast<hkpEntity*>( collB->getOwner() );

	if ( m_simulation->m_crossIslandPairsCollectingActive && 
		!entityA->isFixed() && !entityB->isFixed() &&
		entityA->getSimulationIsland() != entityB->getSimulationIsland() )
	{
		//m_simulation->m_addCrossIslandPairCriticalSection.enter(); (whole broadphase+dispatch is locked)
		m_simulation->m_addedCrossIslandPairs.pushBack(pair);
		//m_simulation->m_addCrossIslandPairCriticalSection.leave();
	}
	else
	{
		hkCheckDeterminismUtil::checkMt( 0xf00000a0, hkUint32(0xad765433) );

		hkCheckDeterminismUtil::checkMt(0xf00000a1, entityA->m_storageIndex);
		hkCheckDeterminismUtil::checkMt(0xf00000a2, entityB->m_storageIndex);
		hkCheckDeterminismUtil::checkMt(0xf00000a3, entityA->getUid());
		hkCheckDeterminismUtil::checkMt(0xf00000a4, entityB->getUid());
		addAgentHelperFunc(collA, collB, m_simulation->getWorld()->getCollisionInput() );
	}
}


void hkpMultiThreadedSimulation::MtEntityEntityBroadPhaseListener::removeCollisionPair( hkpTypedBroadPhaseHandlePair& pair )
{	
	hkpLinkedCollidable* collA = static_cast<hkpLinkedCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_a)->getOwner() );
	hkpLinkedCollidable* collB = static_cast<hkpLinkedCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_b)->getOwner() );

	hkpEntity* entityA = static_cast<hkpEntity*>( collA->getOwner() );
	hkpEntity* entityB = static_cast<hkpEntity*>( collB->getOwner() );

	if ( m_simulation->m_crossIslandPairsCollectingActive && 
		!entityA->isFixed() && !entityB->isFixed() &&
		entityA->getSimulationIsland() != entityB->getSimulationIsland() )
	{
		//m_simulation->m_removeCrossIslandPairCriticalSection.enter();
		m_simulation->m_removedCrossIslandPairs.pushBack(pair);
		//m_simulation->m_removeCrossIslandPairCriticalSection.leave();
	}
	else
	{
		hkCheckDeterminismUtil::checkMt( 0xf00000a5, hkUint32(0xad765434) );

		hkCheckDeterminismUtil::checkMt(0xf00000a6, entityA->m_storageIndex); 
		hkCheckDeterminismUtil::checkMt(0xf00000a7, entityB->m_storageIndex);
		hkCheckDeterminismUtil::checkMt(0xf00000a8, entityA->getUid());
		hkCheckDeterminismUtil::checkMt(0xf00000a9, entityB->getUid());

		hkpAgentNnEntry* entry = hkAgentNnMachine_FindAgent(collA, collB);

		if (entry)
		{
			hkpWorldAgentUtil::removeAgent(entry);
		}
	}
}

////////////////////////////////////////////////////////////////////////
//
// Special broadphase listener to lock broadphase border
//
////////////////////////////////////////////////////////////////////////



void hkpMultiThreadedSimulation::MtBroadPhaseBorderListener::addCollisionPair( hkpTypedBroadPhaseHandlePair& pair )
{
	//hkCriticalSectionLock lock( m_criticalSection ); updateAabb+dispatching now is atomic
	if (   pair.getElementA()->getType() == hkpWorldObject::BROAD_PHASE_BORDER
		&& pair.getElementB()->getType() == hkpWorldObject::BROAD_PHASE_BORDER )
	{
		return;
	}
	if ( pair.getElementA()->getType() == hkpWorldObject::BROAD_PHASE_BORDER )
	{
		hkpCollidable* collA = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_a)->getOwner() );
		hkpCollidable* collB = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_b)->getOwner() );
		hkpPhantom* p = static_cast<hkpPhantom*>( collA->getOwner() );
		p->markForWrite();
		p->addOverlappingCollidable( collB );
		p->unmarkForWrite();
	}

	if ( pair.getElementB()->getType() == hkpWorldObject::BROAD_PHASE_BORDER )
	{
		hkpCollidable* collA = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_a)->getOwner() );
		hkpCollidable* collB = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_b)->getOwner() );
		hkpPhantom* p = static_cast<hkpPhantom*>( collB->getOwner() );
		p->markForWrite();
		p->addOverlappingCollidable( collA );
		p->unmarkForWrite();
	}
}


void hkpMultiThreadedSimulation::MtBroadPhaseBorderListener::removeCollisionPair( hkpTypedBroadPhaseHandlePair& pair )
{
	//hkCriticalSectionLock lock( m_criticalSection ); updateAabb+dispatching now is atomic
	if (   pair.getElementA()->getType() == hkpWorldObject::BROAD_PHASE_BORDER
		&& pair.getElementB()->getType() == hkpWorldObject::BROAD_PHASE_BORDER )
	{
		return;
	}

	if ( pair.getElementA()->getType() == hkpWorldObject::BROAD_PHASE_BORDER )
	{
		hkpCollidable* collA = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_a)->getOwner() );
		hkpCollidable* collB = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_b)->getOwner() );
		hkpPhantom* p = static_cast<hkpPhantom*>( collA->getOwner() );
		p->markForWrite();
		p->removeOverlappingCollidable( collB );
		p->unmarkForWrite();
	}

	if ( pair.getElementB()->getType() == hkpWorldObject::BROAD_PHASE_BORDER )
	{
		hkpCollidable* collA = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_a)->getOwner() );
		hkpCollidable* collB = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_b)->getOwner() );
		hkpPhantom* p = static_cast<hkpPhantom*>( collB->getOwner() );
		p->markForWrite();
		p->removeOverlappingCollidable( collA );
		p->unmarkForWrite();
	}
}


////////////////////////////////////////////////////////////////////////
//
// Multi Threaded Simulation Implementation
//
////////////////////////////////////////////////////////////////////////

hkpMultiThreadedSimulation::hkpMultiThreadedSimulation( hkpWorld* world )
:
	hkpContinuousSimulation( world ),
	m_addCrossIslandPairCriticalSection( 4000), 
	m_removeCrossIslandPairCriticalSection( 4000 ),
	m_toiQueueCriticalSection(4000),
	m_phantomCriticalSection(4000)
{
	m_jobQueueHandleForToiSolve = HK_NULL;
	m_crossIslandPairsCollectingActive = false;
	HK_ASSERT(0x1aa46876, m_world->m_broadPhaseDispatcher != HK_NULL );

	m_world->getBroadPhase()->enableMultiThreading(10000);

	m_entityEntityBroadPhaseListener.m_simulation = this;
	m_phantomBroadPhaseListener.m_criticalSection = & m_phantomCriticalSection;
	world->m_broadPhaseDispatcher->setBroadPhaseListener( &m_entityEntityBroadPhaseListener, hkpWorldObject::BROAD_PHASE_ENTITY, hkpWorldObject::BROAD_PHASE_ENTITY);

	world->m_broadPhaseDispatcher->setBroadPhaseListener( &m_phantomBroadPhaseListener, hkpWorldObject::BROAD_PHASE_PHANTOM, hkpWorldObject::BROAD_PHASE_ENTITY);
	world->m_broadPhaseDispatcher->setBroadPhaseListener( &m_phantomBroadPhaseListener, hkpWorldObject::BROAD_PHASE_ENTITY,  hkpWorldObject::BROAD_PHASE_PHANTOM);
	world->m_broadPhaseDispatcher->setBroadPhaseListener( &m_phantomBroadPhaseListener, hkpWorldObject::BROAD_PHASE_PHANTOM, hkpWorldObject::BROAD_PHASE_PHANTOM);

	m_broadPhaseBorderListener.m_criticalSection = & m_phantomCriticalSection;
	// Extra five records for the broadphase borders
	world->m_broadPhaseDispatcher->setBroadPhaseListener(&m_broadPhaseBorderListener, hkpWorldObject::BROAD_PHASE_ENTITY, hkpWorldObject::BROAD_PHASE_BORDER);
	world->m_broadPhaseDispatcher->setBroadPhaseListener(&m_broadPhaseBorderListener, hkpWorldObject::BROAD_PHASE_BORDER, hkpWorldObject::BROAD_PHASE_ENTITY);

	// Use border listeners to handle border-phantom overlaps.
	world->m_broadPhaseDispatcher->setBroadPhaseListener(&m_broadPhaseBorderListener, hkpWorldObject::BROAD_PHASE_PHANTOM, hkpWorldObject::BROAD_PHASE_BORDER);
	world->m_broadPhaseDispatcher->setBroadPhaseListener(&m_broadPhaseBorderListener, hkpWorldObject::BROAD_PHASE_BORDER, hkpWorldObject::BROAD_PHASE_PHANTOM);
	world->m_broadPhaseDispatcher->setBroadPhaseListener(&m_broadPhaseBorderListener, hkpWorldObject::BROAD_PHASE_BORDER, hkpWorldObject::BROAD_PHASE_BORDER);

}

hkpMultiThreadedSimulation::~hkpMultiThreadedSimulation()
{
}

hkpSimulation* HK_CALL hkpMultiThreadedSimulation::create( hkpWorld* world )
{
	HK_OPTIONAL_COMPONENT_MARK_USED(hkpMultiThreadedSimulation);

// Disable warning on SNC
#if (defined HK_COMPILER_SNC)
#	pragma diag_push
#	pragma diag_suppress=1646
#endif
	return new hkpMultiThreadedSimulation( world );
#if (defined HK_COMPILER_SNC)
#	pragma diag_pop
#endif
}

static void HK_CALL hkpMultiThreadedSimulation_registerSelf()
{
	hkpSimulation::createContinuous = hkpContinuousSimulation::create;
	hkpSimulation::createMultithreaded = hkpMultiThreadedSimulation::create;
}
HK_OPTIONAL_COMPONENT_DEFINE_MANUAL(hkpMultiThreadedSimulation, hkpMultiThreadedSimulation_registerSelf);


static bool pairLessThan( const hkpTypedBroadPhaseHandlePair& A, const hkpTypedBroadPhaseHandlePair& B )
{
	hkpCollidable* collA1 = static_cast<hkpCollidable*>( A.getElementA()->getOwner() );
	hkpCollidable* collB1 = static_cast<hkpCollidable*>( B.getElementA()->getOwner() );

	int A1 = hkpGetRigidBodyUnchecked( collA1 )->getUid();
	int B1 = hkpGetRigidBodyUnchecked( collB1 )->getUid();

	if ( A1 < B1 )
	{
		return true;
	}
	if ( A1 == B1)
	{
		hkpCollidable* collA2 = static_cast<hkpCollidable*>( A.getElementB()->getOwner() );
		hkpCollidable* collB2 = static_cast<hkpCollidable*>( B.getElementB()->getOwner() );
		int A2 = hkpGetRigidBodyUnchecked( collA2 )->getUid();
		int B2 = hkpGetRigidBodyUnchecked( collB2 )->getUid();
		return A2 < B2;
	}
	return false;
}

void sortPairList( hkArray<hkpTypedBroadPhaseHandlePair>& pairs )
{

	// sort lower uid as object 0
	{
		hkpTypedBroadPhaseHandlePair* pair = pairs.begin();
		const hkpTypedBroadPhaseHandlePair* pairsEnd = pairs.end();
		while(pair < pairsEnd)
		{
			hkpCollidable* collA1 = static_cast<hkpCollidable*>( pair->getElementA()->getOwner() );
			hkpCollidable* collA2 = static_cast<hkpCollidable*>( pair->getElementB()->getOwner() );

			int A1 = collA1->getBroadPhaseHandle()->m_id;
			int A2 = collA2->getBroadPhaseHandle()->m_id;

			if ( A1 > A2 )
			{
				hkAlgorithm::swap( pair->m_a, pair->m_b );
			}
			pair++;
		}
	}

	hkAlgorithm::quickSort(pairs.begin(), pairs.getSize(), pairLessThan);


//#	if defined (HK_ENABLE_DETERMINISM_CHECKS)
//	{
//		hkLocalBuffer<hkUint32> uids( pairs.getSize()*2);
//		int d = 0;
//		{
//			for (int i=0; i < pairs.getSize();i++)
//			{ 
//				hkpCollidable* collA1 = static_cast<hkpCollidable*>( pairs[i].getElementA()->getOwner() );
//				hkpCollidable* collA2 = static_cast<hkpCollidable*>( pairs[i].getElementB()->getOwner() );
//				int A1 = hkpGetRigidBody( collA1 )->getUid();
//				int A2 = hkpGetRigidBody( collA2 )->getUid();
//				HK_ASSERT2(0XAD877D, A1 <= A2, "");
//				uids[d++] = A1;
//				uids[d++] = A2;
//			}
//		}
//
//		
//		hkCheckDeterminismUtil::checkMt(0xf00000aa, pairs.getSize() );
//		hkCheckDeterminismUtil::checkMt(0xf00000ab, uids.begin(), pairs.getSize()*2 );
//	}
//#	endif

}

hkpStepResult hkpMultiThreadedSimulation::stepBeginSt( hkJobQueue* jobQueue, hkReal physicsDeltaTime )
{
	m_world->markForWrite();

	HK_TIMER_BEGIN_LIST("Physics 2012", "Init");


	HK_ASSERT2(0xad000070, !m_world->areCriticalOperationsLocked(), "The world cannot be locked when calling stepWorldInitST()");
	HK_ASSERT2(0xadef876d, !m_world->m_pendingOperationsCount, "No operations may be pending on the hkpWorld::m_pendingOperations queue when calling stepWorldInitST");
	HK_ASSERT2(0xadfefed7, hkGetOutOfMemoryState() == hkMemoryAllocator::MEMORY_STATE_OK, "All memory exceptions must be handled and the memory state flag must be set back to hkMemoryRouter::MEMORY_STATE_OK");
	HK_ASSERT2(0xa0750079, isSimulationAtPsi(), "You may only call stepBeginSt when the simulation is at a PSI. Use isSimulationAtPsi() to check for this.");

	m_crossIslandPairsCollectingActive = true;
	m_world->lockCriticalOperations();

	HK_ON_DEBUG(checkDeltaTimeIsOk( physicsDeltaTime ));

	m_physicsDeltaTime = physicsDeltaTime;


	hkStepInfo physicsStepInfo( m_currentPsiTime, m_currentPsiTime + m_physicsDeltaTime );
	
	m_world->m_dynamicsStepInfo.m_stepInfo = physicsStepInfo;
	m_world->m_collisionInput->m_stepInfo = physicsStepInfo;

	// perform maintenance
	{
		m_world->m_maintenanceMgr->performMaintenanceNoSplit( m_world, physicsStepInfo );
	}

	// Perform memory checks prior to integration but after the dirty list and actions are processed.
	hkWorldMemoryAvailableWatchDog::MemUsageInfo memInfo;
	m_world->calcRequiredSolverBufferSize(memInfo);
	
	if( !hkMemorySystem::getInstance().solverCanAllocSingleBlock( memInfo.m_maxRuntimeBlockSize ) )
	{
		if (m_world->getMemoryWatchDog())
		{
			m_world->unlockAndAttemptToExecutePendingOperations();
			m_world->unmarkForWrite();
			hkpWorldMemoryUtil::checkMemoryForIntegration(m_world);	// hkMemory::getInstance().getRuntimeMemorySize()
			m_world->markForWrite();
			m_world->lockCriticalOperations();

			HK_ON_DEBUG( m_world->calcRequiredSolverBufferSize( memInfo ) );
			HK_ASSERT2(0xad907071, hkMemorySystem::getInstance().solverCanAllocSingleBlock(memInfo.m_maxRuntimeBlockSize), "Still not enough memory for integration.");
		}
		else
		{
			m_world->unmarkForWrite();

			if ( m_world->m_assertOnRunningOutOfSolverMemory )
			{
				HK_ASSERT2( 0xf0324565, false, "Your solver memory is too small, increase it or use a hkWorldMemoryAvailableWatchDog().");
				HK_BREAKPOINT( 0xf0324565 );
			}
			m_previousStepResult = HK_STEP_RESULT_MEMORY_FAILURE_BEFORE_INTEGRATION;
			m_world->unlockAndAttemptToExecutePendingOperations();
			return HK_STEP_RESULT_MEMORY_FAILURE_BEFORE_INTEGRATION;
		}
	}


	//
	// Initialize all parameters of the dynamics step into that depend on the stepInfo
	//
	{
		// Step Info
		m_world->m_dynamicsStepInfo.m_stepInfo = physicsStepInfo;

		// Solver Info
		hkpSolverInfo& solverInfo  = m_world->m_dynamicsStepInfo.m_solverInfo;
		solverInfo.m_deltaTime	  = physicsStepInfo.m_deltaTime    * solverInfo.m_invNumSteps;
		solverInfo.m_invDeltaTime = physicsStepInfo.m_invDeltaTime * solverInfo.m_numSteps;
		solverInfo.m_globalAccelerationPerSubStep.setMul( hkSimdReal::fromFloat(solverInfo.m_deltaTime), m_world->m_gravity );

		hkSimdReal deltaTime; deltaTime.setFromFloat( physicsStepInfo.m_deltaTime );
		solverInfo.m_globalAccelerationPerStep.setMul( deltaTime, m_world->m_gravity );

		m_world->m_violatedConstraintArray->reset();	// ignore TOI violated constraints
	}

	//
	// Initialize persistent job data for the next simulation step
	//
	{
		hkpMtThreadStructure* tl	= m_world->m_multithreadedSimulationJobData;

		tl->m_dynamicsStepInfo		= &m_world->m_dynamicsStepInfo;
		tl->m_collisionInput		= *m_world->m_collisionInput;
		tl->m_constraintQueryIn.set( m_world->m_dynamicsStepInfo.m_solverInfo, m_world->m_dynamicsStepInfo.m_stepInfo, m_world->m_violatedConstraintArray);
		tl->m_simulation			= this;
		tl->m_tolerance				= m_world->m_collisionInput->getTolerance();
		tl->m_weldingTable			= hkpWeldingUtility::m_sinCosTable;
		tl->m_world					= m_world;
	}

	int numIslands = m_world->getActiveSimulationIslands().getSize();
	if (numIslands > 0)
	{
		// Process actions
		if (m_world->m_processActionsInSingleThread)
		{
			HK_TIMER_SPLIT_LIST("Actions");
			m_world->unlockCriticalOperationsForPhantoms();

			applyActions();

			m_world->lockCriticalOperationsForPhantoms();
		}

		// Create the first job - this will spawn other jobs that will eventually complete the step
		hkpIntegrateJob job(numIslands);
		job.m_mtThreadStructure =  m_world->m_multithreadedSimulationJobData;
		jobQueue->setQueueCapacityForJobType(HK_JOB_TYPE_DYNAMICS, numIslands);
		jobQueue->setQueueCapacityForJobType(HK_JOB_TYPE_COLLIDE, numIslands);
		jobQueue->addJob(job, hkJobQueue::JOB_LOW_PRIORITY );

	}

	m_world->m_dynamicsStepInfo.m_solverInfo.incrementDeactivationFlags();

	m_numActiveIslandsAtBeginningOfStep = m_world->getActiveSimulationIslands().getSize();
	m_numInactiveIslandsAtBeginningOfStep = m_world->getInactiveSimulationIslands().getSize();

	m_world->checkDeterminism();
	m_world->unmarkForWrite();
	m_world->m_multiThreadCheck.markForRead( hkMultiThreadCheck::THIS_OBJECT_ONLY );
	m_world->getFixedIsland()->markAllEntitiesReadOnly();
	HK_TIMER_END_LIST();

	return HK_STEP_RESULT_SUCCESS;
}



static HK_FORCE_INLINE hkBool less_hkSimulationIslandPtr( const hkpSimulationIsland* a, const hkpSimulationIsland* b )
{
	return ( a->m_entities[0]->getUid() < b->m_entities[0]->getUid() );
}

hkpStepResult hkpMultiThreadedSimulation::finishMtStep( hkJobQueue* jobQueue, hkJobThreadPool* threadPool )
{
	hkCheckDeterminismUtil::checkMt( 0xf00000b0, hkUint32(0xad983361) );
	HK_TIMER_BEGIN("Physics 2012", HK_NULL );
	{
		m_world->getFixedIsland()->unmarkAllEntitiesReadOnly();
		m_world->m_multiThreadCheck.unmarkForRead();
		m_world->lock();
	}


	int origFinishingFlags = 0;
#ifdef HK_PLATFORM_PS3_PPU
	int maxNumSpus = 0;
	hkJobQueueHwSetup::SpuSchedulePolicy origSpuPolicy = hkJobQueueHwSetup::SEMAPHORE_WAIT_OR_SWITCH_ELF;
#endif
	HK_ASSERT2(0x67f55435, threadPool == HK_NULL || jobQueue != HK_NULL, "If you specify HK_NULL for the jobQueue you must specify HK_NULL for the threadPool" );
	if (jobQueue != HK_NULL )
	{
		if ( threadPool != HK_NULL )
		{
			origFinishingFlags = jobQueue->getMasterThreadFinishingFlags();
			HK_ASSERT2(0x09873e45, threadPool->isProcessing() == false, "If you specify a thread pool you must not be currently running any jobs with it" );

#if defined(HK_PLATFORM_PS3_PPU)
			if ( !jobQueue->m_hwSetup.m_noSpu )
			{
				jobQueue->m_queueSemaphores[ jobQueue->m_cpuThreadIndexToSemaphoreIndex[0] ]->setBusyWaitOnPpu( true );
				origSpuPolicy = jobQueue->setSpuSchedulingPolicy( hkJobQueueHwSetup::SEMAPHORE_WAIT_OR_SWITCH_ELF );
				maxNumSpus = threadPool->getNumThreads();
				
				threadPool->setNumThreads( 1 );
			}
#endif

			jobQueue->setWaitPolicy( hkJobQueue::WAIT_INDEFINITELY );
			jobQueue->setMasterThreadFinishingFlags( ( 1 << HK_JOB_TYPE_COLLIDE ) | ( 1 << HK_JOB_TYPE_COLLIDE_STATIC_COMPOUND ) );
			threadPool->processAllJobs( jobQueue );
		}
		else
		{
			HK_ASSERT2(0x67f55434, jobQueue->m_data->m_waitPolicy == hkJobQueue::WAIT_INDEFINITELY, "If you are managing the processing of jobs yourself (i.e. no threadPool specified) you should set the queue to not release threads");
		}
	}

	{

		// sort new islands:
		{
			const int numNewActiveIslands = m_world->getActiveSimulationIslands().getSize() - m_numActiveIslandsAtBeginningOfStep;
			HK_ASSERT2(0xad78644d, m_world->getInactiveSimulationIslands().getSize() == m_numInactiveIslandsAtBeginningOfStep, "Num of inactive islands changed ?!?");
			//const int numNewInactiveIslands = m_world->getInactiveSimulationIslands().getSize() - m_numInactiveIslandsAtBeginningOfStep;

			hkSort(m_world->m_activeSimulationIslands.begin() + m_numActiveIslandsAtBeginningOfStep, numNewActiveIslands, less_hkSimulationIslandPtr );
			//hkSort(m_world->m_inactiveSimulationIslands.begin() + m_numInactiveIslandsAtBeginningOfStep, numNewInactiveIslands, less_hkSimulationIslandPtr );

			for (int i = m_numActiveIslandsAtBeginningOfStep; i < m_world->m_activeSimulationIslands.getSize(); i++)
			{
				m_world->m_activeSimulationIslands[i]->m_storageIndex = hkObjectIndex(i);
			}

#		if defined (HK_ENABLE_DETERMINISM_CHECKS)
			for (int i = 0; i < m_world->m_activeSimulationIslands.getSize(); i++)
			{
				m_world->m_activeSimulationIslands[i]->m_uTag = m_world->m_activeSimulationIslands[i]->m_entities[0]->m_uid;
				hkCheckDeterminismUtil::checkMt(0xf00000c0, m_world->m_activeSimulationIslands[i]->m_uTag);
				hkCheckDeterminismUtil::checkMt(0xf00000c1, m_world->m_activeSimulationIslands[i]->m_entities.getSize());
				hkCheckDeterminismUtil::checkMt(0xf00000c2, m_world->m_activeSimulationIslands[i]->m_splitCheckFrameCounter);
			}
			for (int i = 0; i < m_world->m_inactiveSimulationIslands.getSize(); i++)
			{
				m_world->m_inactiveSimulationIslands[i]->m_uTag = m_world->m_inactiveSimulationIslands[i]->m_entities[0]->m_uid;
				hkCheckDeterminismUtil::checkMt(0xf00000c4, m_world->m_inactiveSimulationIslands[i]->m_uTag);
				hkCheckDeterminismUtil::checkMt(0xf00000c5, m_world->m_inactiveSimulationIslands[i]->m_entities.getSize());
				hkCheckDeterminismUtil::checkMt(0xf00000c6, m_world->m_inactiveSimulationIslands[i]->m_splitCheckFrameCounter);
			}
#		endif
		}

		m_world->checkDeterminism();

#	if ! defined (HK_ENABLE_DETERMINISM_CHECKS)
		if ( m_addedCrossIslandPairs.getSize() + m_removedCrossIslandPairs.getSize())
#	endif
		{
			// duplicates can be introduced if 2 islands are chasing each other
			HK_TIMER_BEGIN_LIST("InterIsland", "duplicates");
			hkpTypedBroadPhaseDispatcher::removeDuplicates(  reinterpret_cast<hkArray<hkpBroadPhaseHandlePair>&>(m_addedCrossIslandPairs),
				reinterpret_cast<hkArray<hkpBroadPhaseHandlePair>&>(m_removedCrossIslandPairs) );

			// info: removed pairs are non-deterministic when using filtering, that doesn't cause engine's non-deterministic behavior.
			hkCheckDeterminismUtil::checkMt(0xf00000cc, m_addedCrossIslandPairs.getSize());


#		if defined (HK_DEBUG)
			for (int i = 0; i < m_addedCrossIslandPairs.getSize(); i++)
			{
				for (int j = i+1; j < m_addedCrossIslandPairs.getSize(); j++)
				{
					hkpTypedBroadPhaseHandlePair& p0 = m_addedCrossIslandPairs[i];
					hkpTypedBroadPhaseHandlePair& p1 = m_addedCrossIslandPairs[j];
					HK_ASSERT2(0xad8355dd, p0.m_a != p1.m_a || p0.m_b != p1.m_b, "Duplicated entries in m_addedCrossIslandPairs."); 
				}
			}
			for (int i = 0; i < m_removedCrossIslandPairs.getSize(); i++)
			{
				for (int j = i+1; j < m_removedCrossIslandPairs.getSize(); j++)
				{
					hkpTypedBroadPhaseHandlePair& p0 = m_removedCrossIslandPairs[i];
					hkpTypedBroadPhaseHandlePair& p1 = m_removedCrossIslandPairs[j];
					HK_ASSERT2(0xad8255dd, p0.m_a != p1.m_a || p0.m_b != p1.m_b, "Duplicated entries in m_removedCrossIslandPairs.");
				}
			}
#		endif


				// we need to sort to keep determinism
			HK_TIMER_SPLIT_LIST("sortPairs");
			sortPairList( m_addedCrossIslandPairs   );
			sortPairList( m_removedCrossIslandPairs );

			{
				HK_TIMER_SPLIT_LIST("addAgt");
				HK_ALIGN16( hkpProcessCollisionOutput ) processOutput(HK_NULL);
				hkCheckDeterminismUtil::checkMt(0xf00000d0, m_addedCrossIslandPairs.getSize());
				for ( int i = 0; i < m_addedCrossIslandPairs.getSize(); ++i )
				{
					hkpLinkedCollidable* collA = static_cast<hkpLinkedCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(m_addedCrossIslandPairs[i].m_a)->getOwner() );
					hkpLinkedCollidable* collB = static_cast<hkpLinkedCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(m_addedCrossIslandPairs[i].m_b)->getOwner() );

						// merge the island immediately
					{
						hkpEntity* entityA = static_cast<hkpEntity*>(collA->getOwner());
						hkpEntity* entityB = static_cast<hkpEntity*>(collB->getOwner());
						hkCheckDeterminismUtil::checkMt( 0xf00000d1, hkUint32(0xad983361) );
						hkCheckDeterminismUtil::checkMt( 0xf00000d2, entityA->m_storageIndex);
						hkCheckDeterminismUtil::checkMt( 0xf00000d3, entityB->m_storageIndex);

						hkpSimulationIsland* islandA = entityA->getSimulationIsland();
						hkpSimulationIsland* islandB = entityB->getSimulationIsland();
						hkCheckDeterminismUtil::checkMt( 0xf00000d4, islandA->m_storageIndex);
						hkCheckDeterminismUtil::checkMt( 0xf00000d5, islandB->m_storageIndex);

						if ( islandA != islandB)
						{
							hkpWorldOperationUtil::internalMergeTwoIslands( m_world, islandA, islandB );
							islandA = entityA->getSimulationIsland();
							HK_ASSERT2(0xad903251, entityB->getSimulationIsland() == islandA, "Internal error: Merged island lost !?");
						}
						processOutput.m_constraintOwner = islandA;
					}
					hkpAgentNnEntry* entry = addAgentHelperFunc( collA, collB, m_world->getCollisionInput() );

					if ( entry != HK_NULL )
					{
						hkCpuProcessAgentHelperFunc( entry, *m_world->getCollisionInput(), processOutput, this );
					}
					if (hkMemoryStateIsOutOfMemory(6) )
					{
						break;
					}
				}
				m_addedCrossIslandPairs.clear();
			}
			HK_TIMER_SPLIT_LIST("removeAgt");

			{
				// This (removedCrossIslandPairs) should happen very seldomly. 
				// For this reason we're just okay to call
				//   hkpWorldAgentUtil::removeAgentAndToiEvents(entry);
				// Also no extra locking is necessary, as this code section is single threaded / critical-section protected already.
				HK_ACCESS_CHECK_OBJECT( m_world, HK_ACCESS_RW );
				for ( int i = 0; i < m_removedCrossIslandPairs.getSize(); ++i )
				{
					hkpLinkedCollidable* collA = static_cast<hkpLinkedCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(m_removedCrossIslandPairs[i].m_a)->getOwner() );
					hkpLinkedCollidable* collB = static_cast<hkpLinkedCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(m_removedCrossIslandPairs[i].m_b)->getOwner() );
					hkpAgentNnEntry* entry = hkAgentNnMachine_FindAgent(collA, collB);
					if (entry)
					{
						{
							// determinism
							hkpEntity* entityA = static_cast<hkpEntity*>(collA->getOwner());
							hkpEntity* entityB = static_cast<hkpEntity*>(collB->getOwner());
							hkCheckDeterminismUtil::checkMt(0xf00000e0, entityA->m_storageIndex); // we only can check a removedPair once we know there's an agentEntry for it.
							hkCheckDeterminismUtil::checkMt(0xf00000e1, entityB->m_storageIndex);

							hkpSimulationIsland* islandA = entityA->getSimulationIsland();
							hkpSimulationIsland* islandB = entityB->getSimulationIsland();
							hkCheckDeterminismUtil::checkMt(0xf00000e2, islandA->m_storageIndex);
							hkCheckDeterminismUtil::checkMt(0xf00000e3, islandB->m_storageIndex);

						}

						// this is single threaded -- no locking
						hkpWorldAgentUtil::removeAgentAndItsToiEvents(entry);
					}
				}
				m_removedCrossIslandPairs.clear();
			}
			HK_TIMER_END_LIST();
		}

		m_crossIslandPairsCollectingActive = false;
		if (m_world->m_pendingOperations->m_pending.getSize() > 1)
		{
			HK_WARN_ONCE(0x20096aae, "Critical operations generated during simulation's step. They're not deterministically ordered, and therefore the world may behave nondeterministically from here on.");
		}

		hkCheckDeterminismUtil::checkMt( 0xf00000e4, hkUint32(0xad983361) );

		m_world->checkConstraintsViolated();
		m_currentPsiTime = m_currentPsiTime + m_physicsDeltaTime; 

		m_world->unlockAndAttemptToExecutePendingOperations();
		HK_ON_DETERMINISM_CHECKS_ENABLED(m_determinismCheckFrameCounter++);
		hkCheckDeterminismUtil::checkMt( 0xf00000e5, hkUint32(0xad983361) );
	}

	hkpStepResult stepResult =  HK_STEP_RESULT_SUCCESS;

	if (hkMemoryStateIsOutOfMemory(7) )
	{
		stepResult = HK_STEP_RESULT_MEMORY_FAILURE_DURING_COLLIDE;
		m_currentPsiTime = m_currentPsiTime - m_physicsDeltaTime;
		m_world->unlock();
		HK_TIMER_END();
		goto END;
	}

	if ( m_world->m_worldPostCollideListeners.getSize() )
	{
		HK_TIMER_BEGIN("PostCollideCB", HK_NULL);
		hkStepInfo physicsStepInfo( m_currentPsiTime - m_physicsDeltaTime, m_currentPsiTime );
		hkpWorldCallbackUtil::firePostCollideCallback( m_world, physicsStepInfo );
		HK_TIMER_END();
	}

	hkCheckDeterminismUtil::checkMt( 0xf00000e6, hkUint32(0xad983361) );
	m_world->checkDeterminism();

	HK_TIMER_END();


	//
	//	Process TOIs
	//
	{
		// The job queue handle is needed for multithreading TOIs, so is stored in a member variable temporarily
		HK_ASSERT( 0x098dd745, m_jobQueueHandleForToiSolve == HK_NULL );
		if (m_world->m_processToisMultithreaded)
		{
			m_jobQueueHandleForToiSolve = jobQueue;
		}

		m_world->unlock();
		m_world->markForWrite();

		// Function used from hkpContinuousSimulation
		advanceTime();
		m_jobQueueHandleForToiSolve = HK_NULL;
		m_world->unmarkForWrite();

		if (hkMemoryStateIsOutOfMemory(8) )
		{
			stepResult = HK_STEP_RESULT_MEMORY_FAILURE_DURING_TOI_SOLVE;
		}
	}

END:
	// from here on the SPUs are allowed to switch .ELFs again as we are done with TOI processing
	if ( threadPool != HK_NULL )
	{
		HK_TIMER_BEGIN("WaitForWorkerThreads", HK_NULL);
		jobQueue->setWaitPolicy( hkJobQueue::WAIT_UNTIL_ALL_WORK_COMPLETE );
		jobQueue->setMasterThreadFinishingFlags(origFinishingFlags);
		threadPool->waitForCompletion();
		HK_TIMER_END();

#if defined(HK_PLATFORM_PS3_PPU)
		if ( !jobQueue->m_hwSetup.m_noSpu )
		{
			jobQueue->setSpuSchedulingPolicy( origSpuPolicy );
			threadPool->setNumThreads( maxNumSpus );
			jobQueue->m_queueSemaphores[ jobQueue->m_cpuThreadIndexToSemaphoreIndex[0] ]->setBusyWaitOnPpu( false );
		}
#endif
	}
	m_previousStepResult = stepResult;

	// Attempt recovering world simulation after memory failure.
	m_world->markForWrite();
	if (m_previousStepResult != HK_STEP_RESULT_SUCCESS && m_world->getMemoryWatchDog())
	{
		hkpWorldMemoryUtil::tryToRecoverFromMemoryErrors(m_world);
	}
	m_world->unmarkForWrite();

	return (hkpStepResult)m_previousStepResult;
}


void hkpMultiThreadedSimulation::getMultithreadConfig( hkpMultithreadConfig& config )
{
	config = m_multithreadConfig;
}

void hkpMultiThreadedSimulation::setMultithreadConfig( const hkpMultithreadConfig& config, hkJobQueue* queue )
{
	m_multithreadConfig = config;
}


void hkpMultiThreadedSimulation::assertThereIsNoCollisionInformationForAgent( hkpAgentNnEntry* agent )
{
#if defined HK_DEBUG
	hkCriticalSectionLock lock( &m_toiQueueCriticalSection );

	hkpContinuousSimulation::assertThereIsNoCollisionInformationForAgent( agent );
#endif // if defined HK_DEBUG
}

void hkpMultiThreadedSimulation::assertThereIsNoCollisionInformationForEntities( hkpEntity** entities, int numEntities, hkpWorld* world )
{
#if defined HK_DEBUG
	HK_ASSERT(0xad44bb3e, numEntities && entities[0]->getWorld() );

	hkCriticalSectionLock lock( &m_toiQueueCriticalSection );

	hkpContinuousSimulation::assertThereIsNoCollisionInformationForEntities( entities, numEntities, world );
#endif // if defined HK_DEBUG
}

void hkpMultiThreadedSimulation::addToiEventWithCriticalSectionLock(const hkpProcessCollisionOutput& processOutput, const hkpAgentNnEntry& entry, hkCriticalSection* criticalSection )
{
	HK_TIMER_BEGIN("AgentJob.addToi", HK_NULL);
	hkpToiEvent* event;
	{
		hkCriticalSectionLock lock( criticalSection );
		if( m_toiEvents.getSize() >= m_toiEvents.getCapacity())
		{
			HK_WARN_ALWAYS(0xf0323454, "TOI event queue full, consider using HK_COLLIDABLE_QUALITY_DEBRIS for some objects or increase hkpWorldCinfo::m_sizeOfToiEventQueue" );
			return;
		}
		event = m_toiEvents.expandByUnchecked(1);
		HK_ON_DEBUG( event->m_entities[0] = HK_NULL );
		HK_ON_DEBUG( event->m_entities[1] = HK_NULL );
	}
	HK_TIMER_END();

	event->m_time = processOutput.m_toi.m_time;
	event->m_useSimpleHandling = m_world->getCollisionDispatcher()->getCollisionQualityInfo(entry.m_collisionQualityIndex)->m_useSimpleToiHandling;
	event->m_seperatingVelocity = processOutput.m_toi.m_seperatingVelocity;
	event->m_contactPoint = processOutput.m_toi.m_contactPoint;
	event->m_entities[0] = reinterpret_cast<hkpEntity*>(entry.m_collidable[0]->getOwner());
	event->m_entities[1] = reinterpret_cast<hkpEntity*>(entry.m_collidable[1]->getOwner());
	event->m_properties = static_cast<const hkpContactPointProperties&>(processOutput.m_toi.m_properties);
	event->m_contactMgr = static_cast<hkpDynamicsContactMgr*>(entry.m_contactMgr);

	hkString::memCpy4(event->m_extendedUserDatas, processOutput.m_toi.m_properties.m_extendedUserDatas, sizeof(event->m_extendedUserDatas) >> 2);

	if ( hkDebugToi && HK_IS_TRACE_ENABLED( event->m_entities[0], event->m_entities[1] ) )
	{
		hkToiPrintf("add.toi.event", "#    add  TOI     @ %2.7f: %-6s %-6s \n", event->m_time, event->m_entities[0]->getName(), event->m_entities[1]->getName());
	}
}

struct hkpIslandsAgentEntriesInfo
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DYNAMICS, hkpIslandsAgentEntriesInfo);

	enum AgentCategory
	{
		MIDPHASE_CPU,
		NARROWPHASE_CPU,
#if defined(HK_PLATFORM_HAS_SPU)
		NUM_AGENT_CPU_CATEORIES,

		MIDPHASE_SPU = NUM_AGENT_CPU_CATEORIES,
		NARROWPHASE_SPU,
#endif
		NUM_AGENT_CATEGORIES,
	};

	hkpIslandsAgentEntriesInfo() : m_island(HK_NULL)
	{
		for ( int i = 0; i < NUM_AGENT_CATEGORIES; ++i )
		{
			m_numEntries[i] = 0;
			m_firstEntryIdx[i] = 0;
			m_nextEntryIdx[i] = 0;
		}
	}

	HK_FORCE_INLINE int getNumEntries()
	{
		int num = 0;
		for ( int i = 0; i < NUM_AGENT_CATEGORIES; ++i )
		{
			num += m_numEntries[i];
		}
		return num;
	}

	static HK_FORCE_INLINE AgentCategory getAgentCategory( const hkpAgentNnEntry& entry )
	{
		// -1 because the first nnTrackType is 1.
		int category = -1;
		category += entry.m_nnTrackType;
#if defined(HK_PLATFORM_HAS_SPU)
		if ( entry.m_forceCollideOntoPpu == 0 )
		{
			category += hkpIslandsAgentEntriesInfo::NUM_AGENT_CPU_CATEORIES;
		}
#endif
		return hkpIslandsAgentEntriesInfo::AgentCategory( category );
	}


		// The corresponding simulation island.
	hkpSimulationIsland* m_island;
		// Number of agent entries owned by the referencing hkpSimulationIsland.
	int m_numEntries[NUM_AGENT_CATEGORIES];
		// Index of the first agent entry pointer for this island -- this is an index to one concatenated array.
	int m_firstEntryIdx[NUM_AGENT_CATEGORIES];
		// Index of the next agent entry pointer for this island. Used when filling in the array.
	int m_nextEntryIdx[NUM_AGENT_CATEGORIES];
};


void hkpMultiThreadedSimulation::collideEntitiesOfOneIslandNarrowPhaseContinuous_toiOnly( hkpEntity** entities, int numEntities, const hkpProcessCollisionInput& input, hkPointerMap<hkUint32, hkpEntity*>& entitiesNeedingPsiCollisionDetection )
{
	if ( m_jobQueueHandleForToiSolve != HK_NULL )
	{
		HK_ASSERT(0xf0ff0044, numEntities > 0);
		HK_ASSERT(0xad63ee33, entities[0]->getWorld()->areCriticalOperationsLocked());

		// Get the common island for the entities
		hkpSimulationIsland* commonIsland = entities[0]->getSimulationIsland();
#if defined(HK_DEBUG)
		{ for (int i = 0; i < numEntities; i++){	HK_ASSERT(0xf0ff0045, entities[i]->getSimulationIsland() == commonIsland); } }
#endif

		// Initialize a local look-up table to determine whether a given agent should be fired
		const hkInt32 totalNumEntities = commonIsland->m_entities.getSize();
		hkLocalArray<hkBool> isProcessed(totalNumEntities); isProcessed.setSizeUnchecked(totalNumEntities);
		hkString::memSet(isProcessed.begin(), 0, totalNumEntities);

		hkLocalArray<hkpAgentNnEntry*> entries(1000);
		hkArray<struct hkpLinkedCollidable::CollisionEntry> collisionEntriesTmp;

		// Use this to remember the counts of agent categories.
		hkpIslandsAgentEntriesInfo info;
		info.m_island = commonIsland;

		// Create a job for each entity
		for( hkInt32 i = 0; i < numEntities; ++i )
		{			
			// The current entity we are processing.  We will only touch the collision agents of this entity once.
			hkpEntity& entity = *entities[i]; HK_ASSERT(0xad0987fe, entity.getSimulationIsland() == commonIsland );

			// The linked collidable of the entity we are processing.  We will update all agents connected with this collidable		
			hkpLinkedCollidable& linkedCollidable = *entity.getLinkedCollidable();

			// Mark the entity as processed
			isProcessed[entity.m_storageIndex] = true;

			// Process all agents associated with the entity
			linkedCollidable.getCollisionEntriesSorted(collisionEntriesTmp);
			const hkArray<struct hkpLinkedCollidable::CollisionEntry>& collisionEntries = collisionEntriesTmp;

			for( hkInt32 j = 0; j < collisionEntries.getSize(); ++j ) // order matters
			{
				// The current collision entry we are processing
				const hkpLinkedCollidable::CollisionEntry& collisionEntry = collisionEntries[j]; // order matters

				// Get the other entity the collision agent is tracking
				hkpEntity* partner = static_cast<hkpEntity*>(collisionEntry.m_partner->getOwner());  HK_ASSERT( 0xf0321245, hkpGetRigidBody(collisionEntry.m_partner) );

				// Don't process this entry if we have already done so (note that the partner's simulation
				// island can be different here - which is ok since the world's simulation islands are locked
				// during TOI processing)
				if( partner->getSimulationIsland() == commonIsland && isProcessed[partner->m_storageIndex] )
				{
					continue;					
				}

				const hkBool32 useContinuousPhysics = m_world->getCollisionDispatcher()->getCollisionQualityInfo(collisionEntry.m_agentEntry->m_collisionQualityIndex)->m_useContinuousPhysics;
				const hkBool useSimplifiedToi = m_world->getCollisionDispatcher()->getCollisionQualityInfo( collisionEntry.m_agentEntry->m_collisionQualityIndex )->m_useSimpleToiHandling;
				if (useContinuousPhysics && !useSimplifiedToi)
				{
					++info.m_numEntries[ hkpIslandsAgentEntriesInfo::getAgentCategory( *collisionEntry.m_agentEntry ) ];
					entries.pushBack(collisionEntry.m_agentEntry);
				}
				else if (HK_NULL == entitiesNeedingPsiCollisionDetection.getWithDefault(entity.getUid(), HK_NULL))
				{
					entity.addReference();
					entitiesNeedingPsiCollisionDetection.insert(entity.getUid(), &entity);
				}
			}
		}

		if ( entries.getSize()  )
		{
			processAgentNnEntries(entries.begin(), entries.getSize(), input, info, PROCESS_NORMALLY);
		}
	}
	else
	{
		hkpContinuousSimulation::collideEntitiesOfOneIslandNarrowPhaseContinuous_toiOnly( entities, numEntities, input, m_entitiesNeedingPsiCollisionDetection );
	}
}

static HK_FORCE_INLINE hkpIslandsAgentEntriesInfo& getIslandEntriesInfo(hkpSimulationIsland* island, hkArray<hkpIslandsAgentEntriesInfo>& islandEntriesInfos)
{
	for (int i = 0; i < islandEntriesInfos.getSize(); i++)
	{
		if (island == islandEntriesInfos[i].m_island)
		{
			return islandEntriesInfos[i];
		}
	}
	
	// no entry found -- create a new one
	new (&islandEntriesInfos.expandOne()) hkpIslandsAgentEntriesInfo;
	islandEntriesInfos.back().m_island = island;

	return islandEntriesInfos.back();
}

namespace {
	struct EntryAndIsland
	{
		hkpAgentNnEntry* m_entry;
		hkpSimulationIsland* m_island;
		hkpIslandsAgentEntriesInfo::AgentCategory m_category;
	};

}
void hkpMultiThreadedSimulation::collideEntitiesNeedingPsiCollisionDetectionNarrowPhase_toiOnly( const hkpProcessCollisionInput& input, hkPointerMap<hkUint32, hkpEntity*>& entitiesNeedingPsiCollisionDetection )
{
	HK_TIMER_BEGIN_LIST("Physics 2012", "Recollide PSI");

	m_world->lockCriticalOperations();

	hkPointerMap<hkpAgentNnEntry*, int> entriesToCollideMap; 
	hkArray<EntryAndIsland> entriesToCollide;
	//hkPointerMap<hkpSimulationIsland*, int> entryInfoIdxPerSimulation; xxx kill & directly scan through the entryInfos
	hkArray<hkpIslandsAgentEntriesInfo> islandEntriesInfos;
	int numAllEntries = 0;

	// Collect all agentNnEntries to collide AND group them by the owning island.
	// Iterate through all entities, and collect non-continuous collision entries.
	//
	{
		hkPointerMap<hkpEntity*, int>::Iterator it = entitiesNeedingPsiCollisionDetection.getIterator();
		while (entitiesNeedingPsiCollisionDetection.isValid(it))
		{
			const hkpEntity* entity = entitiesNeedingPsiCollisionDetection.getValue(it);
			hkCheckDeterminismUtil::checkMt( 0xad000006, entity->getUid() );
			hkCheckDeterminismUtil::checkMt( 0xad000007, entity->m_storageIndex );

			hkArray<hkpLinkedCollidable::CollisionEntry> collisionEntriesTmp; 
			entity->getLinkedCollidable()->getCollisionEntriesSorted(collisionEntriesTmp);
			const hkArray<hkpLinkedCollidable::CollisionEntry>& collisionEntries = collisionEntriesTmp; 

			const int numCollisionEntries = collisionEntries.getSize();
			for (int e = 0; e < numCollisionEntries; e++)
			{
				hkpAgentNnEntry* entry = collisionEntries[e].m_agentEntry;

				if (!entriesToCollideMap.getWithDefault(entry, HK_NULL))
				{
					const hkBool32 isContinuous = m_world->getCollisionDispatcher()->getCollisionQualityInfo(entry->m_collisionQualityIndex)->m_useContinuousPhysics;
					const hkBool isSimplified = m_world->getCollisionDispatcher()->getCollisionQualityInfo(entry->m_collisionQualityIndex)->m_useSimpleToiHandling;
					hkBool32 motionIsFrozen = static_cast<const hkpRigidBody*>(entity)->getRigidMotion()->getMotionState()->getSweptTransform().getInvDeltaTimeSr().isEqualZero();
					if ((!isContinuous || (isSimplified && motionIsFrozen)))
					{
						hkpSimulationIsland* island = entity->getSimulationIsland();
						if ( HK_VERY_UNLIKELY(entity->isFixed()) )
						{
							// get other entity connected to the entry
							const hkpEntity* entityA = hkpGetRigidBodyUnchecked( entry->m_collidable[0] );
							const hkpEntity* entityB = hkpGetRigidBodyUnchecked( entry->m_collidable[1] );
							const hkpEntity* otherEntity = hkSelectOther(entity, entityA, entityB);
							island = otherEntity->getSimulationIsland();

							HK_ASSERT2(0xad903062, ( ( entry->m_nnTrackType == HK_AGENT3_NARROWPHASE_TRACK ) ? hkAgentNnMachine_IsEntryOnTrack(island->m_narrowphaseAgentTrack, entry) : hkAgentNnMachine_IsEntryOnTrack(island->m_midphaseAgentTrack, entry) ), "Agent entry not found in the island, as expected.");
						}
						entriesToCollideMap.insert(entry, 1);
						EntryAndIsland& ei = entriesToCollide.expandOne();
						ei.m_entry = entry;
						ei.m_island = island;
						ei.m_category = hkpIslandsAgentEntriesInfo::getAgentCategory( *entry );
						hkpIslandsAgentEntriesInfo& info = getIslandEntriesInfo(island, islandEntriesInfos);
						++info.m_numEntries[ei.m_category];
						++numAllEntries;
					}
				}
			}
			entity->removeReference();
			it = entitiesNeedingPsiCollisionDetection.getNext(it);
		}

		// References are removed. Now clear the entity list.
		entitiesNeedingPsiCollisionDetection.clear();
	}

	{
		// Prepare each info's firstEntryIdx and nextFirstIdx members to allow sorting in the next loop.
		int nextFirstIdx = 0;
		for (int i = 0; i < islandEntriesInfos.getSize(); i++)
		{
			hkpIslandsAgentEntriesInfo& info = islandEntriesInfos[i];
			for ( int c = 0; c < hkpIslandsAgentEntriesInfo::NUM_AGENT_CATEGORIES; ++c )
			{
				info.m_firstEntryIdx[c] = nextFirstIdx;
				info.m_nextEntryIdx[c] = nextFirstIdx;
				nextFirstIdx += info.m_numEntries[c];
			}
		}
	}

	// Create a sorted array of entry pointers (the entry pointers are grouped by info/category).
	hkLocalBuffer<hkpAgentNnEntry*> entries(numAllEntries);
	{
		for (int i = 0; i < entriesToCollide.getSize(); i++)
		{
			EntryAndIsland& infoInfo = entriesToCollide[i];
			hkpAgentNnEntry* entry = infoInfo.m_entry;
			hkpSimulationIsland* island = infoInfo.m_island;
			hkpIslandsAgentEntriesInfo::AgentCategory category = infoInfo.m_category;

			HK_ON_DEBUG(int size = islandEntriesInfos.getSize());
			hkpIslandsAgentEntriesInfo& info = getIslandEntriesInfo(island, islandEntriesInfos);
			HK_ASSERT(0xad810211, islandEntriesInfos.getSize() == size);

			int entryIdx = info.m_nextEntryIdx[category]++;
			entries[entryIdx] = entry;
		}

		for (int i = 0; i < islandEntriesInfos.getSize()-1; i++)
		{
			HK_ASSERT2(0xad810168, islandEntriesInfos[i].m_nextEntryIdx[hkpIslandsAgentEntriesInfo::NUM_AGENT_CATEGORIES - 1] == islandEntriesInfos[i+1].m_firstEntryIdx[0], "Not all agent entries were filled in.");
		}
		HK_ASSERT2(0xad810168, islandEntriesInfos.isEmpty() || islandEntriesInfos.back().m_nextEntryIdx[hkpIslandsAgentEntriesInfo::NUM_AGENT_CATEGORIES - 1] == entries.getSizeDebug(), "Not all agent entries were filled in.");
	}

	HK_ON_DETERMINISM_CHECKS_ENABLED(m_determinismCheckFrameCounter++);

	// For each simulation island collide its entries
	processAgentNnEntriesFromMultipleIslands(entries.begin(), numAllEntries, islandEntriesInfos.begin(), islandEntriesInfos.getSize(), *m_world->getCollisionInput(), hkpContinuousSimulation::DISABLE_TOIS);

	m_world->unlockAndAttemptToExecutePendingOperations();

	HK_TIMER_END_LIST();
}

// These confirm the use of newJobsBuffer in the following method.
HK_COMPILE_TIME_ASSERT( sizeof( hkpAgentNnEntryJob ) <= sizeof( hkpAgentNnEntryBaseJob ) );

void hkpMultiThreadedSimulation::processAgentNnEntries_oneInfo( hkpAgentNnEntry** allEntries, const hkpProcessCollisionInput& collisionInput, struct hkpIslandsAgentEntriesInfo& info, hkpContinuousSimulation::CollisionQualityOverride collisionQualityOverride, hkpPostCollideJob* postCollideJobOut )
{
	// We may create a job for each category.
	// The compile-time asserts confirm that this is safe.
	// Note that we want a local buffer to ensure correct alignment on the jobs
	
	hkLocalBuffer<hkpAgentNnEntryBaseJob> newJobsBuffer( hkpIslandsAgentEntriesInfo::NUM_AGENT_CATEGORIES );
	hkpAgentNnEntryBaseJob* newJobsThisInfo = newJobsBuffer.begin();

	int numTasksThisInfo = 0;
	int numJobsThisInfo = 0;
	int numMidphaseTasksThisInfo = 0;

	HK_ASSERT3(0xafe14231, m_world->m_maxEntriesPerToiMidphaseCollideTask <= hkpAgentBaseJob::MAX_AGENT_NN_ENTRIES_PER_TASK, "hkpWorld::m_maxElementsPerAgentNnEntryTask may not exceed " << hkpAgentBaseJob::MAX_AGENT_NN_ENTRIES_PER_TASK << ".");
	HK_ASSERT3(0xafe14231, m_world->m_maxEntriesPerToiNarrowphaseCollideTask <= hkpAgentBaseJob::MAX_AGENT_NN_ENTRIES_PER_TASK, "hkpWorld::m_maxElementsPerAgentNnEntryTask may not exceed " << hkpAgentBaseJob::MAX_AGENT_NN_ENTRIES_PER_TASK << ".");

#if defined(HK_PLATFORM_HAS_SPU)
	{
		const int numSpuMidphaseEntries = info.m_numEntries[hkpIslandsAgentEntriesInfo::MIDPHASE_SPU];
		const int numSpuNarrowphaseEntries = info.m_numEntries[hkpIslandsAgentEntriesInfo::NARROWPHASE_SPU];

		{
			if ( numSpuMidphaseEntries )
			{
				const int maxEntriesPerAgentNnEntryTask = m_world->m_maxEntriesPerToiMidphaseCollideTask;
				hkpAgentNnEntryBaseJob *const aNnJob = new ( &newJobsThisInfo[numJobsThisInfo] ) hkpAgentNnEntryJob( collisionInput.m_stepInfo, &allEntries[info.m_firstEntryIdx[hkpIslandsAgentEntriesInfo::MIDPHASE_SPU]], numSpuMidphaseEntries, maxEntriesPerAgentNnEntryTask, collisionQualityOverride, m_world->m_useCompoundSpuElf, HK_AGENT3_MIDPHASE_TRACK );
				HK_ON_DETERMINISM_CHECKS_ENABLED(aNnJob->m_jobSid = 0x3000);
				aNnJob->m_taskIndex = hkUint16( numTasksThisInfo );
				aNnJob->m_shapeKeyTrack = (hkpShapeKeyTrack*) numMidphaseTasksThisInfo;
				++numJobsThisInfo;
				numTasksThisInfo += ( (numSpuMidphaseEntries-1) / maxEntriesPerAgentNnEntryTask ) + 1;
			}
			if ( numSpuNarrowphaseEntries )
			{
				const int maxEntriesPerAgentNnEntryTask = m_world->m_maxEntriesPerToiNarrowphaseCollideTask;
				hkpAgentNnEntryBaseJob *const aNnJob = new ( &newJobsThisInfo[numJobsThisInfo] ) hkpAgentNnEntryJob( collisionInput.m_stepInfo, &allEntries[info.m_firstEntryIdx[hkpIslandsAgentEntriesInfo::NARROWPHASE_SPU]], numSpuNarrowphaseEntries, maxEntriesPerAgentNnEntryTask, collisionQualityOverride, m_world->m_useCompoundSpuElf, HK_AGENT3_NARROWPHASE_TRACK );
				HK_ON_DETERMINISM_CHECKS_ENABLED(aNnJob->m_jobSid = 0x4000);
				aNnJob->m_taskIndex = hkUint16( numTasksThisInfo );
				++numJobsThisInfo;
				numTasksThisInfo += ( (numSpuNarrowphaseEntries-1) / maxEntriesPerAgentNnEntryTask ) + 1;
			}
		}
	}
#endif
	{
		const int numCpuMidphaseEntries = info.m_numEntries[hkpIslandsAgentEntriesInfo::MIDPHASE_CPU];
		const int numCpuNarrowphaseEntries = info.m_numEntries[hkpIslandsAgentEntriesInfo::NARROWPHASE_CPU];

		{
			const int numCpuEntries = numCpuMidphaseEntries + numCpuNarrowphaseEntries;
			if ( numCpuEntries )
			{
				const int maxEntriesPerAgentNnEntryTask = m_world->m_maxEntriesPerToiMidphaseCollideTask;
				hkpAgentNnEntryBaseJob *const aNnJob = new ( &newJobsThisInfo[numJobsThisInfo] ) hkpAgentNnEntryJob( collisionInput.m_stepInfo, &allEntries[info.m_firstEntryIdx[0]], numCpuEntries, maxEntriesPerAgentNnEntryTask, collisionQualityOverride, HK_AGENT3_INVALID_TRACK );
				HK_ON_DETERMINISM_CHECKS_ENABLED(aNnJob->m_jobSid = 0x7000);
				aNnJob->m_taskIndex = hkUint16( numTasksThisInfo );
#if defined(HK_PLATFORM_HAS_SPU)
				aNnJob->m_jobSpuType = HK_JOB_SPU_TYPE_DISABLED;
#endif
				++numJobsThisInfo;
				numTasksThisInfo += ( (numCpuEntries-1) / maxEntriesPerAgentNnEntryTask ) + 1;
			}
		}
	}
	// For each of the jobs for this info, set some values and add it to the job queue.
	{
		hkpShapeKeyTrack* shapeKeyTracks = HK_NULL;
		if ( numMidphaseTasksThisInfo )
		{
			shapeKeyTracks = hkAllocateChunk<hkpShapeKeyTrack>( numMidphaseTasksThisInfo, HK_MEMORY_CLASS_COLLIDE );
			for ( int i = 0; i < numMidphaseTasksThisInfo; ++i )
			{
				new ( &shapeKeyTracks[i] ) hkpShapeKeyTrack;
			}
		}
		hkpAgentSectorHeader *const header = hkpAgentSectorHeader::allocate( numTasksThisInfo, hkMath::max2( m_world->m_maxEntriesPerToiMidphaseCollideTask, m_world->m_maxEntriesPerToiNarrowphaseCollideTask ) );
		header->m_shapeKeyTracks = shapeKeyTracks;
		header->m_numShapeKeyTracks = numMidphaseTasksThisInfo;
		for ( int j = 0; j < numJobsThisInfo; ++j )
		{
			hkpAgentNnEntryBaseJob& aNnJob = newJobsThisInfo[j];
			aNnJob.m_header = header;
			aNnJob.m_island = info.m_island;
			aNnJob.m_islandIndex = info.m_island->m_storageIndex;
			aNnJob.m_mtThreadStructure = m_world->m_multithreadedSimulationJobData;
			m_jobQueueHandleForToiSolve->addJob( aNnJob, hkJobQueue::JOB_LOW_PRIORITY ); 
		}
	}
	HK_ASSERT2( 0x23eb78a2, numJobsThisInfo > 0, "Didn't create any jobs for an island info." );
	{
		new ( postCollideJobOut ) hkpPostCollideJob( newJobsThisInfo[0] );
		postCollideJobOut->m_island = newJobsThisInfo[0].m_island;
	}
}

void hkpMultiThreadedSimulation::processAgentNnEntries( hkpAgentNnEntry** entries, int numEntries, const hkpProcessCollisionInput& collisionInput, hkpIslandsAgentEntriesInfo& info, hkpContinuousSimulation::CollisionQualityOverride collisionQualityOverride )
{
	hkpSimulationIsland *const island = info.m_island;

	
	
	if ( ( m_jobQueueHandleForToiSolve != HK_NULL ) && ( numEntries > m_world->m_maxNumToiCollisionPairsSinglethreaded ) ) // set it to pc, xbox: 3-4, spu: 1-2
	{
		//HK_ASSERT(0x09621463, m_jobQueueHandleForToiSolve->m_data->m_waitPolicy == hkJobQueue::WAIT_INDEFINITELY );

		// We need to switch the currently read-write marked world to read-only as multithreaded TOI solving expects this. This will be reverted once we are back.
		//
		// Note that we do that before m_jobQueueHandleForToiSolve->addJob() below, because other threds will pick up that job immediately.
		//
		HK_ACCESS_CHECK_OBJECT( m_world, HK_ACCESS_RW );
		m_world->unmarkForWrite();
		m_world->markForRead();

		char postCollideJobBuffer[sizeof(hkpPostCollideJob)];
		hkpPostCollideJob* postCollideJob = reinterpret_cast<hkpPostCollideJob*>( postCollideJobBuffer );

		// We usually sort the entities array into categories.
		hkLocalBuffer<hkpAgentNnEntry*> sortedEntries( numEntries );

		// But we don't need to if there's only one category.
#if !defined( HK_PLATFORM_HAS_SPU )
		processAgentNnEntries_oneInfo( entries, collisionInput, info, collisionQualityOverride, postCollideJob );
#else
		{
			// Prepare each info's firstEntryIdx and nextFirstIdx members to allow sorting in the next loop.
			{
				int nextFirstIdx = 0;
				for ( int c = 0; c < hkpIslandsAgentEntriesInfo::NUM_AGENT_CATEGORIES; ++c )
				{
					info.m_firstEntryIdx[c] = nextFirstIdx;
					info.m_nextEntryIdx[c] = nextFirstIdx;
					nextFirstIdx += info.m_numEntries[c];
				}
			}
			for ( int i = numEntries-1; i >=0; --i )
			{
				hkpIslandsAgentEntriesInfo::AgentCategory c = hkpIslandsAgentEntriesInfo::getAgentCategory( *entries[i] );
				sortedEntries[info.m_nextEntryIdx[c]] = entries[i];
				++info.m_nextEntryIdx[c];
			}

			processAgentNnEntries_oneInfo( sortedEntries.begin(), collisionInput, info, collisionQualityOverride, postCollideJob );
		}
#endif

		// Process all jobs on the queue and wait until all have been finished.
		m_jobQueueHandleForToiSolve->processAllJobs( );

		m_world->unmarkForRead();
		m_world->markForWrite();

		postCollideJob->m_island = island;

		hkpMtThreadStructure tl;
		tl.m_dynamicsStepInfo		= &m_world->m_dynamicsStepInfo;
		tl.m_collisionInput		= *m_world->m_collisionInput;
		tl.m_constraintQueryIn.set( m_world->m_dynamicsStepInfo.m_solverInfo, m_world->m_dynamicsStepInfo.m_stepInfo, m_world->m_violatedConstraintArray);
		tl.m_simulation			= this;
		tl.m_tolerance			= m_world->m_collisionInput->getTolerance();
		tl.m_weldingTable         = hkpWeldingUtility::m_sinCosTable;
		tl.m_world				= m_world;

		hkCpuPostCollideJob(tl, HK_NULL, *reinterpret_cast<hkJobQueue::JobQueueEntry*>( postCollideJob ));

#if defined(HK_ENABLE_EXTENSIVE_WORLD_CHECKING)
		island->markAllEntitiesReadOnly();
		island->isValid();
		island->unmarkAllEntitiesReadOnly();
#endif

	}
	else
	{
		hkpContinuousSimulation::processAgentNnEntries(entries, numEntries, collisionInput, island, collisionQualityOverride);
	}
}

void hkpMultiThreadedSimulation::processAgentNnEntriesFromMultipleIslands( hkpAgentNnEntry** allEntries, int numAllEntries, struct hkpIslandsAgentEntriesInfo* islandEntriesInfos, int numInfos, const hkpProcessCollisionInput& collisionInput, hkpContinuousSimulation::CollisionQualityOverride collisionQualityOverride )
{
	// (0.1) Make sure we can process islands in multi-threadedly. Otherwise, revert to single threaded version.
	//
	if ( m_jobQueueHandleForToiSolve == HK_NULL || numAllEntries <= m_world->m_maxNumToiCollisionPairsSinglethreaded)
	{
		for (int ii = 0; ii < numInfos; ii++)
		{
			hkpIslandsAgentEntriesInfo& info = islandEntriesInfos[ii];

#			if defined (HK_ENABLE_DETERMINISM_CHECKS)
			for(int category = 0; category < hkpIslandsAgentEntriesInfo::NUM_AGENT_CATEGORIES; category++)
			{
				for (hkpAgentNnEntry** e = allEntries+info.m_firstEntryIdx[category]; e < allEntries+info.m_firstEntryIdx[category]+info.m_numEntries[category]; e++)
				{
					hkCheckDeterminismUtil::checkMt( 0xad000008, hkpGetRigidBody((*e)->getCollidableA())->getUid() );
					hkCheckDeterminismUtil::checkMt( 0xad000009, hkpGetRigidBody((*e)->getCollidableB())->getUid() );
					hkCheckDeterminismUtil::checkMt( 0xad00000a, hkpGetRigidBody((*e)->getCollidableA())->m_storageIndex );
					hkCheckDeterminismUtil::checkMt( 0xad00000b, hkpGetRigidBody((*e)->getCollidableB())->m_storageIndex );
				}
			}
#			endif

			hkpContinuousSimulation::processAgentNnEntries(allEntries+info.m_firstEntryIdx[0], info.getNumEntries(), collisionInput, info.m_island, collisionQualityOverride);
		}

		return;
	}

	// (0.2) Checks copied from hkpMultiThreadedSimulation::processAgentNnEntries().
	//

	//HK_ASSERT(0x09621463, m_jobQueueHandleForToiSolve->m_data->m_waitPolicy == hkJobQueue::WAIT_INDEFINITELY );

	// (1) For each island info create an agent job or two (potentially, if we have SPUs). Add the jobs to the queues. 
	//     Also, if we find a fixed island, we postpone its processing till the end of this function. (And we process it in a single thread, simply, because the multi-threaded code doesn't support operations on the fixed island.)
	//
	hkpIslandsAgentEntriesInfo* fixedIslandInfo = HK_NULL;

	// We create postCollideJobs for each info as we go, but run them single-threaded at the end.
	hkLocalBuffer<hkpPostCollideJob> postCollideJobs(numInfos);

	// We need to switch the currently read-write marked world to read-only as multithreaded TOI solving expects this. This will be reverted once we are back.
	//
	// Note that we do that before m_jobQueueHandleForToiSolve->addJob() below, because other threds will pick up that job immediately.
	//
	HK_ACCESS_CHECK_OBJECT( m_world, HK_ACCESS_RW );
	m_world->unmarkForWrite();
	m_world->markForRead();

	// Iterate through the island infos
	for (int ii = 0; ii < numInfos; ii++)
	{
		hkpIslandsAgentEntriesInfo& info = islandEntriesInfos[ii];

#		if defined (HK_ENABLE_DETERMINISM_CHECKS)
		for(int category = 0; category < hkpIslandsAgentEntriesInfo::NUM_AGENT_CATEGORIES; category++)
		{
			{
				for (hkpAgentNnEntry** e = allEntries+info.m_firstEntryIdx[category]; e < allEntries+info.m_firstEntryIdx[category]+info.m_numEntries[category]; e++)
				{
					hkCheckDeterminismUtil::checkMt( 0xad000008, hkpGetRigidBody((*e)->getCollidableA())->getUid() );
					hkCheckDeterminismUtil::checkMt( 0xad000009, hkpGetRigidBody((*e)->getCollidableB())->getUid() );
					hkCheckDeterminismUtil::checkMt( 0xad00000a, hkpGetRigidBody((*e)->getCollidableA())->m_storageIndex );
					hkCheckDeterminismUtil::checkMt( 0xad00000b, hkpGetRigidBody((*e)->getCollidableB())->m_storageIndex );
				}
			}
		}
#		endif

		if ( HK_VERY_UNLIKELY(info.m_island == m_world->getFixedIsland()) )
		{
			// Postpone processing of the fixed island till the end of this function.
			HK_ASSERT2(0x908131, fixedIslandInfo == HK_NULL, "Multiple infos pointing to a fixed island. There can be only one!");
			fixedIslandInfo = &info;
			continue;
		}

		processAgentNnEntries_oneInfo( allEntries, collisionInput, info, collisionQualityOverride, &postCollideJobs[ii] );
	}

	// (2) Take part in processing of the agent entry jobs
	//

	// Process all jobs on the queue and wait until all have been finished.
	m_jobQueueHandleForToiSolve->processAllJobs( );

	m_world->unmarkForRead();
	m_world->markForWrite();

	// (3) Perform hkpPostCollideJobs for all hkpAgentEntryJobs done.
	//
	{
		hkpMtThreadStructure tl;
		{
			tl.m_dynamicsStepInfo		= &m_world->m_dynamicsStepInfo;
			tl.m_collisionInput		= *m_world->m_collisionInput;
			tl.m_constraintQueryIn.set( m_world->m_dynamicsStepInfo.m_solverInfo, m_world->m_dynamicsStepInfo.m_stepInfo, m_world->m_violatedConstraintArray);
			tl.m_simulation			= this;
			tl.m_tolerance			= m_world->m_collisionInput->getTolerance();
			tl.m_weldingTable         = hkpWeldingUtility::m_sinCosTable;
			tl.m_world				= m_world;
		}

		for (int i = 0; i < numInfos; i++)
		{
			hkCpuPostCollideJob(tl, HK_NULL, reinterpret_cast<hkJobQueue::JobQueueEntry&>( postCollideJobs[i] ) ); 
		}
	}
#if defined(HK_ENABLE_EXTENSIVE_WORLD_CHECKING)
	info.m_island->markAllEntitiesReadOnly();
	info.m_island->isValid();
	info.m_island->unmarkAllEntitiesReadOnly();
#endif

	// (4) Process the fixed island in a single thread.
	//
	if ( HK_VERY_UNLIKELY(fixedIslandInfo) )
	{
		// Process the fixed island.
		// Use single threaded processing in the unlikely case of processing entries between two fixed object (mt version cannot handle the fixed island).
		hkpContinuousSimulation::processAgentNnEntries(allEntries+fixedIslandInfo->m_firstEntryIdx[0], fixedIslandInfo->getNumEntries(), collisionInput, fixedIslandInfo->m_island, collisionQualityOverride );
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
