/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryOut.h>
#include <Physics/ConstraintSolver/Jacobian/hkpJacobianHeaderSchema.h>

#include <Physics2012/Internal/Solver/Atom/hkpBuildJacobianFromAtoms.h>

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuBuildJacobiansJob.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>


namespace {

	// this is nearly a duplication of hkpBuildJacobianTask
	struct hkBuildJacobiansFromTaskInput
	{
		hkpBuildJacobianTask::AtomInfo*	m_atomInfos;
		int								m_numAtomInfos;

		//output:
		const hkpVelocityAccumulator*	m_accumulators;
		hkpJacobianSchema*				m_schemas;

		hkpJacobianSchema*				m_schemasOfNextTask;
		void*							m_nextTask;

		hkBool							m_finishSchemasWithGoto;
	};

}


static HK_FORCE_INLINE void HK_CALL hkCpuBuildJacobiansFromTask(const hkBuildJacobiansFromTaskInput& input, hkpConstraintQueryIn &queryIn)
{
	hkpConstraintQueryOut queryOut;
	queryOut.m_jacobianSchemas      = input.m_schemas;


	const hkpBuildJacobianTask::AtomInfo* atomInfos = input.m_atomInfos;
	int numAtomInfos = input.m_numAtomInfos;

	for (int a = 0; a < numAtomInfos; )
	{
		const hkpBuildJacobianTask::AtomInfo& atomInfo = atomInfos[a];
		a++;
#if (defined(HK_PLATFORM_PS3_PPU) || defined(HK_PLATFORM_XBOX360))
		if ( a < numAtomInfos )
		{
			// prefetch
			const hkpBuildJacobianTask::AtomInfo& next = atomInfos[a];
			char* p = (char*)next.m_atoms;
			hkMath::forcePrefetch<256>(p);
			hkMath::prefetch128( next.m_runtime );
			hkMath::prefetch128( hkAddByteOffset(queryOut.m_jacobianSchemas.val(),512) );
		}
#endif
		// prepare queryIn
		{
			queryIn.m_transformA                    = atomInfo.m_transformA;
			queryIn.m_transformB                    = atomInfo.m_transformB;

			queryIn.m_bodyA                         = input.m_accumulators + atomInfo.m_accumulatorIndexA;
			queryIn.m_bodyB                         = input.m_accumulators + atomInfo.m_accumulatorIndexB;
#		if defined(HK_PLATFORM_HAS_SPU)
			queryIn.m_accumulatorAIndex             = atomInfo.m_accumulatorInterIndexA;
			queryIn.m_accumulatorBIndex             = atomInfo.m_accumulatorInterIndexB;
#		else
			queryIn.m_accumulatorAIndex             = atomInfo.m_accumulatorIndexA;
			queryIn.m_accumulatorBIndex             = atomInfo.m_accumulatorIndexB;
#		endif

			queryIn.m_constraintInstance            = atomInfo.m_instance;
			queryOut.m_constraintRuntime             = atomInfo.m_runtime;
			queryOut.m_constraintRuntimeInMainMemory = atomInfo.m_runtime;

#if defined (HK_PLATFORM_HAS_SPU)
			queryIn.m_atomInMainMemory              = atomInfo.m_atoms;
#endif
		}

		{
			hkSolverBuildJacobianFromAtoms( atomInfo.m_atoms, atomInfo.m_atomsSize, queryIn, queryOut );
		}
	}

	//
	// Add "End" or "Goto" schema
	//
	{
		if ( !input.m_finishSchemasWithGoto )
		{
			// no 'next task' -> write end schema
			hkpJacobianSchema* endSchema = static_cast<hkpJacobianSchema*>(queryOut.m_jacobianSchemas);
			*(reinterpret_cast<hkInt32*>(endSchema)) = 0;
			queryOut.m_jacobianSchemas = hkAddByteOffset(endSchema, hkpJacobianSchemaInfo::End::Sizeof);
		}
		else
		{
			hkLong branchOffset = hkGetByteOffset( queryOut.m_jacobianSchemas, input.m_schemasOfNextTask );
			if ( branchOffset > 0 )
			{
				// add Goto schema
				hkpJacobianGotoSchema* gotoSchema = reinterpret_cast<hkpJacobianGotoSchema*>(queryOut.m_jacobianSchemas.val() );
				gotoSchema->initOffset(branchOffset);
				queryOut.m_jacobianSchemas = HK_AS_JACOBIAN_SCHEMA(gotoSchema+1);
			}

		}
#ifdef HK_DEBUG
		HK_ASSERT(0xaf6451ed, queryOut.m_jacobianSchemas.val() <= input.m_schemasOfNextTask);
#endif
	}
}


hkJobQueue::JobStatus HK_CALL hkCpuBuildJacobiansJob(	hkpMtThreadStructure&		tl,
														hkJobQueue&					jobQueue,
														hkJobQueue::JobQueueEntry&	jobInOut)
{
	HK_TIMER_BEGIN_LIST("Integrate", "BuildJacobians" );

	const hkpBuildJacobiansJob& job = reinterpret_cast<hkpBuildJacobiansJob&>(jobInOut);

	hkpConstraintQueryIn queryIn = tl.m_constraintQueryIn;

	hkBuildJacobiansFromTaskInput input;
	{
		hkpBuildJacobianTask* task = job.m_buildJacobianTask;

		input.m_atomInfos			= task->m_atomInfos;
		input.m_numAtomInfos		= task->m_numAtomInfos;
		input.m_accumulators		= task->m_accumulators;
		input.m_schemas				= task->m_schemas;
		input.m_schemasOfNextTask	= task->m_schemasOfNextTask;
		input.m_nextTask            = task->m_next;
#if defined (HK_PLATFORM_HAS_SPU)
		input.m_finishSchemasWithGoto = job.m_finishSchemasWithGoto && (task->m_next || (task->m_onPpuOnly && task->m_taskHeader->m_tasks.m_numBuildJacobianTasks) );
#else
		input.m_finishSchemasWithGoto = job.m_finishSchemasWithGoto && task->m_next;
#endif
	}

	hkpSimulationIsland* island = job.m_island;
	island->markAllEntitiesReadOnly();
	hkCpuBuildJacobiansFromTask(input, queryIn);
	island->unmarkAllEntitiesReadOnly();

	HK_TIMER_END_LIST();

	return jobQueue.finishJobAndGetNextJob( &jobInOut, jobInOut, hkJobQueue::WAIT_FOR_NEXT_JOB );
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
