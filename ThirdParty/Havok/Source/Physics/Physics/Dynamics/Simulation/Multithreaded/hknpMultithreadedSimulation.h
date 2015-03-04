/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MULTITHREADED_SIMULATION_H
#define HKNP_MULTITHREADED_SIMULATION_H

#include <Physics/Physics/Dynamics/Simulation/hknpSimulation.h>

#include <Common/Base/Thread/Task/hkTaskQueue.h>

#include <Physics/Physics/Collide/NarrowPhase/hknpCollidePipeline.h>
#include <Physics/Physics/Dynamics/Simulation/Multithreaded/hknpNarrowPhaseTask.h>
#include <Physics/Physics/Dynamics/Simulation/Multithreaded/hknpConstraintSetupTask.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverStepInfo.h>

#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactSolver.h>
#include <Physics/Internal/Dynamics/World/SpaceSplitter/hknpSpaceSplitter.h>

class hknpSolverData;
class hknpSolverTask;


/// Multi threaded simulation.
class hknpMultithreadedSimulation : public hknpSimulation
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpMultithreadedSimulation );

		hknpMultithreadedSimulation();

		//
		// hknpSimulation implementation
		//

		virtual void collide( hknpSimulationContext& simulationContext, hknpSolverData*& solverDataOut ) HK_OVERRIDE;

		virtual void solve( hknpSimulationContext& simulationContext, hknpSolverData* solverData ) HK_OVERRIDE;

		virtual void checkConsistency( hknpWorld* world, hkInt32 checkFlags ) HK_OVERRIDE;

	protected:

		enum Stage
		{
			STAGE_COLLIDE_1,
			STAGE_COLLIDE_2,
			STAGE_COLLIDE_3,
			STAGE_SOLVE_1,
			STAGE_SOLVE_2
		};

	protected:

		hknpSolverData* collideStage1(
			const hknpSimulationThreadContext& threadContext, hknpSimulationContext& simulationContext );

		hknpSolverData* collideStage2(
			const hknpSimulationThreadContext& threadContext, hknpSimulationContext& simulationContext );

		hknpSolverData* collideStage3(
			const hknpSimulationThreadContext& threadContext, hknpSimulationContext& simulationContext );

		void createNarrowPhaseTask(
			hknpSimulationContext& simulationContext, hknpCollisionCacheManager& cdCacheManager,
			hknpSolverData& solverData, bool processOnlyNew );

		void processNarrowPhaseResults(
			hknpSimulationContext& simulationContext, hknpCollisionCacheManager& cdCacheManager,
			hknpSolverData& solverData, bool processOnlyNew );

		void addTasksForActivatedConstraints(
			const hknpSimulationThreadContext& threadContext, hknpSimulationContext& simulationContext );

		void solveStage1(
			hknpSimulationThreadContext* threadContext, hknpSimulationContext& stepInfo, hknpSolverData* solverData );

		void solveStage2(
			hknpSimulationThreadContext* threadContext, hknpSimulationContext& stepInfo, hknpSolverData* solverData );

		static void HK_CALL printGridSizes(
			const hknpSimulationThreadContext& threadContext, hknpCdCacheGrid* cacheGrid );

		static void HK_CALL checkConsistencyOfCollisionCaches(
			hknpWorld* world, hkInt32 checkFlags, bool allowNewGridsToBeInvalid );

	protected:

		hknpCdCacheGrid m_inactiveCdCacheGrid;
		hknpCdCacheGrid m_crossGridCdCacheGrid;

		/// Stream of new body pairs. Only in use during the collide steps.
		hkBlockStream<hknpBodyIdPair> m_newPairsStream;

		hkRefPtr<hknpNarrowPhaseTask> m_narrowPhaseTask;
		hkRefPtr<hknpGatherConstraintsTask> m_gatherConstraintsTask;
		hkRefPtr<hknpConstraintSetupTask> m_constraintSetupTask;
		hkRefPtr<hknpSolverTask> m_solverTask;

		hknpSolverData* m_solverData;

		// Next simulation stage to be performed on a call to collide() or solve()
		Stage m_nextStage;

		// A task queue used by the solver task to schedule its subtasks
		hkTaskQueue m_solverTaskQueue;

	#if defined(HK_PLATFORM_HAS_SPU)
		hknpCdCacheGrid m_inactiveCdCachePpuGrid;
		hknpCdCacheGrid m_crossGridCdCachePpuGrid;
		hkRefPtr<hknpNarrowPhaseTask> m_narrowPhaseTaskPpu;
	#endif
};


#endif	// HKNP_MULTITHREADED_SIMULATION_H

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
