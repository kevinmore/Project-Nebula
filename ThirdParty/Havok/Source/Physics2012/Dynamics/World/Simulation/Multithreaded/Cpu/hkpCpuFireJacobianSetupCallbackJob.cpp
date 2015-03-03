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

#include <Physics2012/Internal/Solver/Atom/hkpBuildJacobianFromAtoms.h>

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuSingleThreadedJobsOnIsland.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldConstraintUtil.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/Collide/ContactListener/hkpContactPointEvent.h>


extern void hkSimpleContactConstraintData_fireCallbacks(class hkpSimpleContactConstraintData* constraintData, const hkpConstraintQueryIn* in, hkpSimpleContactConstraintAtom* atom, hkpContactPointEvent::Type type );


hkJobQueue::JobStatus hkpSingleThreadedJobsOnIsland::cpuFireJacobianSetupCallbackJob(hkpMtThreadStructure& tl,
																					hkJobQueue& jobQueue,
																					hkJobQueue::JobQueueEntry& jobInOut)
{
	const hkpFireJacobianSetupCallback& job = reinterpret_cast<hkpFireJacobianSetupCallback&>(jobInOut);

	hkpBuildJacobianTaskHeader* taskHeader = job.m_taskHeader;
	hkpConstraintQueryIn in = tl.m_constraintQueryIn;
	hkpConstraintQueryOut out;
	out.m_jacobianSchemas = 0;

	if ( taskHeader->m_tasks.m_numCallbackConstraints > 0 )
	{
		HK_TIMER_BEGIN_LIST("Integrate", "ConstraintCallbacks" );

		for ( int i = 0; i < taskHeader->m_tasks.m_numCallbackConstraints; i++ )
		{
			const hkConstraintInternal* ci = taskHeader->m_tasks.m_callbackConstraints[i].m_callbackConstraints;
			
			in.m_constraintInstance = ci->m_constraint;
			out.m_constraintRuntime = ci->m_runtime;
			in.m_bodyA = hkAddByteOffset(taskHeader->m_accumulatorsBase, ci->m_entities[0]->m_solverData);
			in.m_bodyB = hkAddByteOffset(taskHeader->m_accumulatorsBase, ci->m_entities[1]->m_solverData);

			// the entity pointers below are only needed in debug.
			HK_ON_DEBUG_MULTI_THREADING( hkpEntity* eA = ci->m_entities[0] );
			HK_ON_DEBUG_MULTI_THREADING( hkpEntity* eB = ci->m_entities[1] );

				// we can markforWrite because this job is called single threadedly for each island
			HK_ON_DEBUG_MULTI_THREADING( if ( !eA->isFixed() ) { eA->markForWrite(); } );
			HK_ON_DEBUG_MULTI_THREADING( if ( !eB->isFixed() ) { eB->markForWrite(); } );
			in.m_transformA = &ci->m_constraint->getEntityA()->getCollidable()->getTransform();
			in.m_transformB = &ci->m_constraint->getEntityB()->getCollidable()->getTransform();

			if ( ci->m_callbackRequest & ( hkpConstraintAtom::CALLBACK_REQUEST_NEW_CONTACT_POINT | hkpConstraintAtom::CALLBACK_REQUEST_CONTACT_POINT_CALLBACK ) )
			{
				HK_ASSERT2( 0x451ace7, ( ci->m_constraint->getData()->getType() == hkpConstraintData::CONSTRAINT_TYPE_CONTACT ), "Only contact constraints can have these callback requests" );
				hkpSimpleContactConstraintData* constraintData = reinterpret_cast<hkpSimpleContactConstraintData*>( ci->m_constraint->getDataRw() );
				hkpModifierConstraintAtom *const firstModifierBefore = ci->m_constraint->getConstraintModifiers();
				hkpConstraintAtom* terminalAtom = hkpWorldConstraintUtil::getTerminalAtom(ci);
				hkpSimpleContactConstraintAtom* atom = reinterpret_cast<hkpSimpleContactConstraintAtom*>( terminalAtom );
				hkSimpleContactConstraintData_fireCallbacks( constraintData, &in, atom, hkpContactPointEvent::TYPE_MANIFOLD );
				hkpModifierConstraintAtom *const firstModifierAfter = ci->m_constraint->getConstraintModifiers();
				if ( firstModifierBefore != firstModifierAfter )
				{
					hkpBuildJacobianTask::AtomInfo *const atomInfo = taskHeader->m_tasks.m_callbackConstraints[i].m_atomInfo;
					atomInfo->m_atoms = firstModifierAfter;
					atomInfo->m_atomsSize = firstModifierAfter->m_modifierAtomSize;
				}
			}

			if ( ci->m_callbackRequest & hkpConstraintAtom::CALLBACK_REQUEST_SETUP_CALLBACK)
			{
				ci->m_constraint->getDataRw()->buildJacobianCallback(in, out);
			}

			HK_ON_DEBUG_MULTI_THREADING( if ( !eB->isFixed() ) { eB->unmarkForWrite(); } );
			HK_ON_DEBUG_MULTI_THREADING( if ( !eA->isFixed() ) { eA->unmarkForWrite(); } );
		}

		taskHeader->m_tasks.m_callbackConstraints[0].m_callbackConstraints->getMasterEntity()->getSimulationIsland()->getMultiThreadCheck().unmarkForRead();
		hkDeallocateChunk( taskHeader->m_tasks.m_callbackConstraints, taskHeader->m_tasks.m_numCallbackConstraints, HK_MEMORY_CLASS_CONSTRAINT_SOLVER );
		taskHeader->m_tasks.m_callbackConstraints = HK_NULL;
		taskHeader->m_tasks.m_numCallbackConstraints = 0;

		HK_TIMER_END_LIST();
	}

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
