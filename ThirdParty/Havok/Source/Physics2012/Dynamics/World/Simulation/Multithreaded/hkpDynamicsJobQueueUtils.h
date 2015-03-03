/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_DYNAMICS_JOB_QUEUE_UTILS
#define HK_DYNAMICS_JOB_QUEUE_UTILS

#include <Common/Base/Thread/JobQueue/hkJobQueue.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryIn.h>
#include <Physics2012/Dynamics/Constraint/Setup/hkpConstraintSolverSetup.h>
#include <Physics2012/Internal/Solver/Atom/hkpBuildJacobianFromAtoms.h>
#if defined (HK_PLATFORM_HAS_SPU)
#	include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#	include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShapeGetAabbSpuPipelines.h>
#	include <Physics2012/Dynamics/Entity/Util/hkpEntityAabbUtil.h>
#	include <Physics2012/Dynamics/World/Simulation/Multithreaded/Spu/hkpSpuIntegrateMotionsJob.h>
#	include <Physics2012/Dynamics/World/Simulation/Multithreaded/Spu/hkpSpuConfig.h>
#endif

class hkpWorld;
class hkpMultiThreadedSimulation;
struct hkpWorldDynamicsStepInfo;


struct hkpMtThreadStructure
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DYNAMICS, hkpMtThreadStructure);

	hkpMtThreadStructure() : m_constraintQueryIn(HK_ON_CPU(&hkpBeginConstraints)) {}

	hkpWorld*					m_world;
	hkpConstraintQueryIn			m_constraintQueryIn;
	hkpProcessCollisionInput		m_collisionInput;

	hkPadSpu<hkpMultiThreadedSimulation*>	m_simulation;
	hkPadSpu<hkpWorldDynamicsStepInfo*>		m_dynamicsStepInfo;
	hkPadSpu<hkReal>						m_tolerance;
	hkPadSpu<void*>							m_weldingTable; // see hkpWeldingUtility::initWeldingTable

};

#if defined (HK_PLATFORM_HAS_SPU)
int HK_FORCE_INLINE calcStackMemNeededForSolveOnSpu(const hkpBuildJacobianTaskHeader& taskHeader)
{
		// Synchronize these values with all allocateStack() calls within hkSpuSolveConstraintsJob() and above!

		// Stack memory requirements for the solver buffer. See 0xaf5241e4.
	const int solverBufferSize = HK_NEXT_MULTIPLE_OF(128, taskHeader.m_bufferSize);

		// Stack memory requirements for the inlined 'integrate' job.
	int integrateJobBufferSize;
	{
		const int entitiesArraySize			= HK_NEXT_MULTIPLE_OF(128, (taskHeader.m_numAllEntities * sizeof(hkpEntity*))); // See 0xaf5241e2.
		const int pipelineToolSize			= HK_NEXT_MULTIPLE_OF(128, sizeof(hkpSpuIntegrateMotionPipelineTool)); // See 0xaf5241e3.

		
		const int calcAabbShapeCacheSize	= hkSpu4WayCache::getBufferSize( HK_SPU_MAXIMUM_SHAPE_SIZE,		 HK_SPU_AGENT_SECTOR_JOB_ROOT_SHAPE_NUM_CACHE_ROWS ); // See 0xaf5241e5.
		const int calcAabbUntypedCacheSize	= hkSpu4WayCache::getBufferSize( HK_SPU_UNTYPED_CACHE_LINE_SIZE, HK_SPU_AGENT_SECTOR_JOB_UNTYPED_NUM_CACHE_ROWS	   ) + HK_SPU_UNTYPED_CACHE_LINE_SIZE; // See 0xaf5241e6.

		const int sizeOfChildShapeAabbs           = hkpListShape::MAX_CHILDREN_FOR_SPU_MIDPHASE * sizeof(hkAabbUint32); // See 0xaf5241e9.
		const int sizeOfLocalBufferValueIndexPair = hkpListShape::MAX_CHILDREN_FOR_SPU_MIDPHASE * sizeof(hkValueIndexPair); // See 0xaf5241ea.

		const int listShapeGetAabbWithChildShapesPipelineSize = int(hkMath::max2(HK_NEXT_MULTIPLE_OF(128, sizeof(hkListShapeGetAabbWithChildShapes::Pipeline)), HK_NEXT_MULTIPLE_OF(128, sizeof(hkListShapeGetAabbWithChildShapesForAgent::Pipeline)))); // See 0xaf5241e7 and 0xaf5241e8.

		integrateJobBufferSize = entitiesArraySize + pipelineToolSize + calcAabbShapeCacheSize + calcAabbUntypedCacheSize + sizeOfChildShapeAabbs + sizeOfLocalBufferValueIndexPair + listShapeGetAabbWithChildShapesPipelineSize;
	}

		// Stack memory requirements for the solver export. See 0xaf5241e1.
	const int exportBufferSize = HK_NEXT_MULTIPLE_OF(128, 2 * (HK_SPU_SOLVE_RESULTS_WRITER_BASE_BUFFER_SIZE + HK_SPU_SOLVE_RESULTS_WRITER_OVERFLOW_BUFFER_SIZE ) );

	return solverBufferSize + hkMath::max2( exportBufferSize, integrateJobBufferSize );
}
#endif

struct hkpJobQueueUtils
{
	static hkJobQueue::JobPopFuncResult  HK_CALL popIntegrateJob   ( hkJobQueue& queue, hkJobQueue::DynamicData* data,       hkJobQueue::JobQueueEntry& jobIn, hkJobQueue::JobQueueEntry& jobOut );
	static hkJobQueue::JobCreationStatus HK_CALL finishIntegrateJob( hkJobQueue& queue, hkJobQueue::DynamicData* data, const hkJobQueue::JobQueueEntry& jobIn, hkJobQueue::JobQueueEntryInput& newJobCreated );

	static hkJobQueue::JobPopFuncResult  HK_CALL popCollideJob   ( hkJobQueue& queue, hkJobQueue::DynamicData* data,       hkJobQueue::JobQueueEntry& jobIn, hkJobQueue::JobQueueEntry& jobOut );
	static hkJobQueue::JobCreationStatus HK_CALL finishCollideJob( hkJobQueue& queue, hkJobQueue::DynamicData* data, const hkJobQueue::JobQueueEntry& jobIn, hkJobQueue::JobQueueEntryInput& newJobCreated );
};



#endif // HK_DYNAMICS_JOB_QUEUE_UTILS

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
