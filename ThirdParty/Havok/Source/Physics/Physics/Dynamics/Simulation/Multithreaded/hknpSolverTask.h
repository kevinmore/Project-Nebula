/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SOLVER_TASK_H
#define HKNP_SOLVER_TASK_H

#include <Common/Base/Thread/Task/hkTask.h>
#include <Physics/Physics/Dynamics/Solver/hknpConstraintSolver.h>
#include <Physics/Internal/Dynamics/Solver/Scheduler/hknpSolverScheduler.h>
#include <Common/Base/Thread/Task/hkTaskQueue.h>

class hkSimpleSchedulerTaskBuilder;
class hknpSolverData;
class hknpDeactivationStepInfo;
struct hknpSpaceSplitterData;

class hknpSolverTask : public hkTask
{
	public:

		enum
		{
			/// Maximum number of solver subtasks that may be available at any given time
			MAX_AVAILABLE_TASKS = 32
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS);

	#if !defined(HK_PLATFORM_SPU)

		hknpSolverTask( hknpSimulationContext& simulationContext, hknpSolverData* solverData, hkTaskQueue* taskQueue );

		// hknpTask implementation
		virtual void process();

		~hknpSolverTask();

		virtual void* getElf() HK_OVERRIDE;

	#endif

	public:

		hkUint32 m_numThreadsToRun;
		hkUint32 m_orgNumThreadsToRun;	// for sync
		hkUint32* m_syncBuffer;			// for sync
		int m_syncBufferSize;			// (numSolverIterations * numMicroSteps)

		hknpIdxRangeGrid m_cellIdxToGlobalSolverId;
		hknpDeactivationStepInfo* m_deactivationStepInfo;
		hknpSpaceSplitterData* m_spaceSplitterData;
		hknpConstraintSolverJacobianGrid* m_jacobianGrids[hknpJacobianGridType::NUM_TYPES];
		hknpConstraintSolverSchedulerGridInfo m_jacobianGridInfos[hknpJacobianGridType::NUM_TYPES];
		hkSimpleSchedulerTaskBuilder* m_taskBuilder;
		hkDefaultTaskGraph m_taskGraph;
		hkTaskQueue::GraphId m_graphId;
		hkArray<hknpSolverSchedulerTask> m_subTasks;
		hknpSolverStepInfo* m_solverStepInfo;

	#if defined(HK_PLATFORM_HAS_SPU)

		/// PPU v-table pointers of each shape type
		void* m_shapeVTablesPpu[hknpShapeType::NUM_SHAPE_TYPES];

	#endif

		/// Task queue used to schedule solver subtasks.
		/// Not used in SPU, where we use the lockless solver scheduler instead.
		hkTaskQueue* m_taskQueue;
};

#endif // HKNP_SOLVER_TASK_H

/*
 * Havok SDK - Base file, BUILD(#20130912)
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
