/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SOLVER_SCHEDULER_H
#define HKNP_SOLVER_SCHEDULER_H

#include <Common/Base/Thread/SimpleScheduler/hkSimpleScheduler.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverStepInfo.h>
#include <Physics/Physics/Dynamics/Solver/hknpConstraintSolver.h>
#include <Physics/Physics/Dynamics/World/Grid/hknpGrid.h>
#include <Common/Base/Thread/Task/hkTaskGraph.h>

class hknpSimulationThreadContext;
struct hknpLiveJacobianInfo;
class hknpSimulationContext;
class hknpConstraintSolverSchedulerGridInfo;
class hknpSolverTask;
class hknpLiveJacobianInfoGrid;


/// A hknpSolverScheduler task.
struct hknpSolverSchedulerTask
{
	//+hk.MemoryTracker(ignore=True)

	/// Type of the process this task executes.
	enum ProcessType
	{
		PROCESS_STEP_JACOBIAN_CONSTRAINT,	///< Solve jacobian constraints.
		PROCESS_STEP_SUBINTEGRATE,			///< Sub-integrate solver velocities.
#if HKNP_ENABLE_SOLVER_PARALLEL_TASKS == 1
		PROCESS_GC_INACTIVE_CACHES,			///< Non-solver job, grabage collect inactive caches.
		PROCESS_ADD_ACTIVE_BODY_PAIRS,		///< Non-solver job, add active body pairs to union find.
#endif
		PROCESS_CLOSE_QUEUE
	};

	hknpSolverSchedulerTask() {}

	/// Constructor.
	hknpSolverSchedulerTask( ProcessType pt, hkUint8 grid, hkUint8 gridEntryIndex, hknpCellIndex cellA, hknpCellIndex cellB)
		: m_processType(pt), m_grid(grid), m_gridEntryIndex(gridEntryIndex), m_cellA(cellA), m_cellB(cellB), m_launchAB(0) {}

	/// Process type.
	hkEnum<ProcessType, hkUint8> m_processType;

	/// The grid this task processes.
	hkUint8 m_grid;

	/// The grid entry this task processes.
	hkUint8 m_gridEntryIndex;

	/// The grid entry's first cell in case of a link grid, or the cell being processed in case of a cell array grid.
	hknpCellIndex m_cellA;

	/// The grid entry's second cell in case of a link grid.
	hknpCellIndex m_cellB;

	/// Flag to launch a dependent sub-integrate task.
	hkUint8 m_launchAB; // bit 0 is A, bit 1 is B

	hkUint16 m_pad;
};


/// Scheduler for the deterministic multi-threaded constraint solving.
class hknpSolverScheduler
{
	public:
		//+hk.MemoryTracker(ignore=True)

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpSolverScheduler );

		enum ScheduleType
		{
			SCHEDULE_STATIC,
			SCHEDULE_DYNAMIC
		};

#if defined(HKNP_ENABLE_LOCKLESS_SOLVER)

		// Context for adding tasks.
		struct TaskAddContext
		{
			TaskAddContext( int numCells );

			typedef hkArray<hkSimpleSchedulerTaskBuilder::Object> CellLastTask;
			CellLastTask m_cellLastTask;
		#if defined (HK_ENABLE_DETERMINISM_CHECKS)
			hkUint16 m_detJobId;
		#endif
		};

		/// Adds constraint solving and velocity sub-integration tasks and their dependencies.
		static void addSolverTasks(
			TaskAddContext& context, hknpSpaceSplitter* splitter,
			const hknpConstraintSolverSchedulerGridInfo* jacGridInfos,
			hknpConstraintSolverJacobianGrid** jacGrids, hkUint8 jacGridCount,
			hknpLiveJacobianInfoGrid* livejacInfoGrid, hknpIdxRangeGrid* motionGrid,
			hkSimpleSchedulerTaskBuilder* taskBuilder, ScheduleType scheduleType );

	#if HKNP_ENABLE_SOLVER_PARALLEL_TASKS == 1

		/// Adds non-solver tasks and their dependencies.
		static void addNonSolverTasks(
			TaskAddContext& context, hkSimpleSchedulerTaskBuilder* taskBuilder,
			int taskCount, const hknpSolverSchedulerTask::ProcessType* tasks );

	#endif

		/// The entry function of a single thread.
		static void solve(
			hknpSolverStepInfo& solverManager, const hknpSimulationThreadContext& tl, int threadIdx, hknpSolverTask& task );

#else

		/// Structure used to store solver task information in a hkTask pointer
		/// 16 bits for task index
		/// 4 bits for sub step number
		/// 10 bits for micro step number
		struct TaskJob
		{
			// We need to make sure that no TaskJob produces a null pointer and that the bottom bit is always 0,
			// otherwise the job will be considered inactive when in a job group, thus the 0x2 at the end
			TaskJob(int taskIndex, int subStep, int microStep)
				: m_data((taskIndex << 16) | (subStep << 12) | (microStep << 2) | 0x2)
			{
				HK_ASSERT(0x11148199, taskIndex <= 0xFFFF && subStep <= 0xF && microStep <= 0x3FF);
			}

			TaskJob(const hkTask* job) { m_data = (hkUlong) job; }

			hkTask* asJob() { return (hkTask*) m_data; }

			int getTaskIndex()  const { return (int) m_data >> 16; }
			int getSubStep() const { return ((int) m_data >> 12) & 0xF; }
			int getMicroStep() const { return ((int) m_data >> 2) & 0x3FF; }

			hkUlong m_data;
		};

		/// Adds constraint solving and velocity sub-integration tasks and their dependencies.
		static void addSolverTasks(
			hkArray<hkDefaultTaskGraph::TaskId>& lastTaskInCell, hkDefaultTaskGraph* taskGraph,
			hkArray<hknpSolverSchedulerTask>& tasks, int numSteps, int numMicroSteps,
			hknpSpaceSplitter* splitter,
			const hknpConstraintSolverSchedulerGridInfo* jacGridInfos,
			hknpConstraintSolverJacobianGrid** jacGrids, hkUint8 jacGridCount,
			hknpLiveJacobianInfoGrid* livejacInfoGrid,
			hknpIdxRangeGrid* motionGrid, ScheduleType scheduleType );

	#if HKNP_ENABLE_SOLVER_PARALLEL_TASKS == 1

		/// Adds non-solver tasks and their dependencies.
		static void addNonSolverTasks(
			hkDefaultTaskGraph* jobGroup,
			hkArray<hknpSolverSchedulerTask>& tasks, int numTaskTypes,
			const hknpSolverSchedulerTask::ProcessType* taskTypes );

	#endif

#endif

		/// Executes a single task.
		static bool executeTask(
			const hknpSimulationThreadContext& tl, hknpSolverStepInfo* HK_RESTRICT solverInfo,
			int iStep, int microStep, int threadIdx, const hknpSolverSchedulerTask& task );
};


#endif // HKNP_SOLVER_SCHEDULER_H

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
