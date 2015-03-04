/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKP_CPU_SOLVE_CONSTRAINTS_JOBS_H
#define HKP_CPU_SOLVE_CONSTRAINTS_JOBS_H

#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryOut.h>
#include <Physics/ConstraintSolver/Jacobian/hkpJacobianHeaderSchema.h>
#include <Physics2012/Internal/Solver/SimpleConstraints/hkpSimpleConstraintUtil.h>
#include <Physics2012/Internal/Solver/Atom/hkpBuildJacobianFromAtoms.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobQueueUtils.h>


hkJobQueue::JobStatus HK_CALL hkCpuSolveConstraintsJob(hkpMtThreadStructure& tl, hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& jobInOut);

hkJobQueue::JobStatus HK_CALL hkCpuSolveApplyGravityJob(hkpMtThreadStructure& tl, hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& jobInOut);

hkJobQueue::JobStatus HK_CALL hkCpuSolveConstraintBatchJob(hkpMtThreadStructure& tl, hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& jobInOut);

hkJobQueue::JobStatus HK_CALL hkCpuSolveIntegrateVelocitiesJob(hkpMtThreadStructure& tl, hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& jobInOut);

hkJobQueue::JobStatus HK_CALL hkCpuSolveExportResultsJob(hkpMtThreadStructure& tl, hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& jobInOut);


#endif // HKP_CPU_SOLVE_CONSTRAINTS_JOBS_H

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
