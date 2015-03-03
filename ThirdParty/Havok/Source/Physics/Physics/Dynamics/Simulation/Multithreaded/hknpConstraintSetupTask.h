/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CONSTRAINT_SETUP_TASK_H
#define HKNP_CONSTRAINT_SETUP_TASK_H

#include <Common/Base/Thread/Task/hkTask.h>

// Forward declarations
namespace hknpConstraintAtomSolverSetup
{
	struct ConstraintStates;
	struct SubTasks;
}


/// A task which gathers constraints into groups which can be processed independently.
class hknpGatherConstraintsTask : public hkTask
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		hknpGatherConstraintsTask( hknpSimulationContext& simulationContext );

		// hkTask implementation
		virtual void process() HK_OVERRIDE;

	public:

		// Input
		hknpSimulationContext& m_simulationContext;

		// Outputs
		hknpConstraintAtomSolverSetup::ConstraintStates m_constraintStates;	///< Sorted constraint states
		hknpConstraintAtomSolverSetup::SubTasks m_subTasks;					///< Subtasks sorted by estimated cost
};


/// A task which builds constraint Jacobians.
/// This task can be processed multiple times on different threads in order to process the subtasks in parallel.
class hknpConstraintSetupTask : public hkTask
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		hknpConstraintSetupTask( hknpSimulationContext& simulationContext, hknpSolverData& solverData );

		// hkTask implementation
		virtual void process() HK_OVERRIDE;

	public:

		// Inputs
		hknpSimulationContext& m_simulationContext;
		const hknpConstraintAtomSolverSetup::ConstraintStates* m_constraintStates;
		const hknpConstraintAtomSolverSetup::SubTasks* m_subTasks;

		// Current index into m_subTasks, atomically incremented during process()
		hkUint32 m_currentSubTaskIndex;

		// Output
		hknpSolverData& m_solverData;
};


#endif // HKNP_CONSTRAINT_SETUP_TASK_H

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
