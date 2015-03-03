/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Simulation/Multithreaded/hknpConstraintSetupTask.h>

#include <Physics/Physics/Dynamics/Simulation/hknpSimulation.h>
#include <Physics/Physics/Dynamics/World/hknpWorld.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverData.h>
#include <Physics/Internal/Dynamics/Solver/ConstraintAtom/hknpConstraintAtomSolver.h>

namespace
{
	// Helper struct to push/pop appropriate timers based on the thread index
	struct TimerScope
	{
		TimerScope( hkMonitorStream& monitorSteam, int threadIndex )
		:	m_monitorSteam( monitorSteam ),
			m_threadIndex( threadIndex )
		{
			// If this is being processed in a worker thread, set up the same timer path as the master thread
			if( m_threadIndex > 0 )
			{
				HK_TIMER_BEGIN2( m_monitorSteam, "Physics", HK_NULL );
				HK_TIMER_BEGIN2( m_monitorSteam, "Collide", HK_NULL );
			}
		}

		~TimerScope()
		{
			if( m_threadIndex > 0 )
			{
				HK_TIMER_END2( m_monitorSteam );
				HK_TIMER_END2( m_monitorSteam );
			}
		}

		hkMonitorStream& m_monitorSteam;
		int m_threadIndex;
	};
}


hknpGatherConstraintsTask::hknpGatherConstraintsTask( hknpSimulationContext& simulationContext )
	: m_simulationContext( simulationContext )
{

}

void hknpGatherConstraintsTask::process()
{
	HK_CHECK_FLUSH_DENORMALS();

	const int threadIndex = m_simulationContext.getCurrentThreadNumber();

	// Get thread monitor stream just once as access to TLS may be expensive
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
	TimerScope timerScope( timerStream, threadIndex );
#endif

	HK_TIMER_BEGIN_LIST2( timerStream, "GatherConstraintsTask", "Gather" );

	hknpWorld* world = m_simulationContext.getThreadContext( threadIndex )->m_world;

	// Gather all constraints and sort them into groups
	m_constraintStates.gatherConstraints(
		world, world->m_constraintAtomSolver->getConstraints(), world->m_constraintAtomSolver->getNumConstraints() );

	HK_TIMER_SPLIT_LIST2( timerStream, "CreateSubTasks" );

	// Create subtasks for the active groups
	m_subTasks.create(
		world->m_constraintAtomSolver->getConstraints(), m_constraintStates, 0, m_constraintStates.m_numActive - 1 );

	HK_TIMER_END_LIST2( timerStream );
}


hknpConstraintSetupTask::hknpConstraintSetupTask( hknpSimulationContext& simulationContext, hknpSolverData& solverData )
:	m_simulationContext( simulationContext ),
	m_constraintStates( HK_NULL ),
	m_subTasks( HK_NULL ),
	m_currentSubTaskIndex( 0 ),
	m_solverData( solverData )
{

}

void hknpConstraintSetupTask::process()
{
	HK_CHECK_FLUSH_DENORMALS();
	HK_ASSERT( 0x63dea183, m_constraintStates && m_subTasks );

	const int threadIndex = m_simulationContext.getCurrentThreadNumber();

	// Get thread monitor stream just once as access to TLS may be expensive
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
	TimerScope timerScope( timerStream, threadIndex );
#endif

	HK_TIMER_BEGIN2( timerStream, "ConstraintSetupTask", HK_NULL );

	hknpSimulationThreadContext* threadContext = m_simulationContext.getThreadContext( threadIndex );
	hknpSolverData::ThreadData& threadData = m_solverData.m_threadData[threadIndex];
	hknpWorld* world = threadContext->m_world;

	while (1)
	{
		// Get a subtask
		const hkUint32 subTaskIndex = hkDmaManager::atomicExchangeAdd( &m_currentSubTaskIndex, 1 );
		if( int(subTaskIndex) < m_subTasks->getSize() )
		{
			HK_TIMER_BEGIN2( timerStream, "SubTask", HK_NULL );

			const hknpConstraintAtomSolverSetup::SubTask& subtask = (*m_subTasks)[subTaskIndex];

			// Process the subtask's constraints
			hknpConstraintAtomSolverSetup::setupConstraintsMt(
				*threadContext, world, world->m_constraintAtomSolver->getConstraints(),
				*m_constraintStates, subtask.m_firstStateIndex, subtask.m_lastStateIndex,
				m_solverData.m_jacConstraintsGrid, &threadData.m_jacConstraintsStream,
				&threadData.m_solverTempsStream,
				threadContext->m_world->isDeactivationEnabled() ? &threadData.m_activePairStream : HK_NULL );

			HK_MONITOR_ADD_VALUE( "NumConstraints",
				float( 1 + subtask.m_lastStateIndex - subtask.m_firstStateIndex), HK_MONITOR_TYPE_INT );

			HK_TIMER_END2( timerStream );
		}
		else
		{
			break;
		}
	}

	HK_TIMER_END2( timerStream );
}

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
