/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SOLVER_UTIL_H
#define HKNP_SOLVER_UTIL_H

#include <Physics/Physics/Collide/NarrowPhase/hknpCollidePipeline.h>
#include <Physics/Physics/Dynamics/Solver/hknpConstraintSolver.h>
#include <Physics/Internal/Dynamics/Solver/Scheduler/hknpSolverScheduler.h>

class hknpSimulationThreadContext;
struct hknpLiveJacobianInfo;
class hknpLiveJacobianInfoStream;
class hknpSimulationContext;
class hknpConstraintAtomJacobianStream;
struct hknpConstraintSolverJacobianRange2;
struct hkcdManifold4;


/// Solver utility functions.
class hknpSolverUtil
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpSolverUtil );

		/// Solve the provided Jacobian grids single-threaded.
		/// Even though grids are intended for multi-threaded code, we use grids with a single cell in single-threaded mode to allow for unified code paths.
		static void HK_CALL solveSt(
			hknpSimulationContext& stepData, hknpSimulationThreadContext& tl,
			const hknpConstraintSolverSchedulerGridInfo* jacGridInfos,
			hknpConstraintSolverJacobianGrid** jacGrids,
			hknpLiveJacobianInfoStream* liveJacInfos,
			hknpSolverVelocity* HK_RESTRICT solverVelInOut, hknpSolverSumVelocity* HK_RESTRICT solverSumVelInOut, int numSolverVelicities,
			hknpConstraintSolverJacobianStream* HK_RESTRICT solverTempsStream, hknpConstraintSolverJacobianStream* HK_RESTRICT contactSolverTempsStream
			);

		/// Allocates solver temps for the provided Jacobian grids based on the
		/// hknpConstraintSolverJacobianRange2::SOLVER_TEMPS flags of each grid entry's range.
		static void HK_CALL allocateSolverTemps(
			const hknpSimulationThreadContext& tl,
			const hknpConstraintSolverSchedulerGridInfo* jacGridInfos, hknpConstraintSolverJacobianGrid** jacGrids, const int jacGridCount,
			hknpConstraintSolverJacobianStream* HK_RESTRICT solverTempsStream, hknpConstraintSolverJacobianStream* HK_RESTRICT contactSolverTempsStream
			);

		/// Calculate velocities at the contact point projected onto the contact normal using the velocities used to integrate the motions.
		static void HK_CALL calculateProjectedPointVelocitiesUsingIntegratedVelocities(
			const hkcdManifold4& manifold, const hknpMotion* motionA, const hknpMotion* motionB,
			hkVector4* HK_RESTRICT projectedPointVelocitiesOut
			);

		/// Calculate velocities at the contact point projected onto the contact normal using the current velocities
		static void HK_CALL calculateProjectedPointVelocities(
			const hkcdManifold4& manifold, const hknpMotion* motionA, const hknpMotion* motionB,
			hkVector4* HK_RESTRICT projectedPointVelocitiesOut
			);

		/// Sort the grids by ascending processing priority (see hknpDefaultConstraintSolverPriority) and return the results in sortedResultsOut.
		template< typename IndexArrayT>
		static HK_FORCE_INLINE void HK_CALL sortJacGrids(
			const hknpConstraintSolverSchedulerGridInfo* jacGridInfos, int jacGridCount,
			IndexArrayT& sortedResultsOut
			);
};

#include <Physics/Physics/Dynamics/Solver/hknpSolverUtil.inl>

#endif // HKNP_SOLVER_UTIL_H

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
