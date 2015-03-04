/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_CONTACT_SOLVER_H
#define HKNP_CONTACT_SOLVER_H

#include <Common/Base/Container/BlockStream/hkBlockStream.h>
#include <Physics/Physics/Dynamics/World/Grid/hknpGrid.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverInfo.h>
#include <Physics/Physics/Dynamics/Solver/hknpConstraintSolver.h>

// enable exporting of friction forces
#if !defined(HK_PLATFORM_HAS_SPU)
//#	define HKNP_SOLVER_EXPORT_FRICTION_IMPULSE
#endif
#define HK_ON_SOLVER_TIMER(X) //X


/// Structure holding force information for the hknpContactSolver
struct hknpContactSolverTemp
{
	enum { NUM_CONTACT_POINTS_PER_MANIFOLD = 4 };

	// This holds impulses for 4 contact points
	// when mxSize == 4 this is for 4 different manifolds, and the values are transposed
	//  (contactPoint0 for all 4 jacobians is in the first vector, contactPoint1 in the second, and so on)
	HK_ALIGN_REAL( hkReal m_impulseApplied[NUM_CONTACT_POINTS_PER_MANIFOLD][HKNP_NUM_MX_JACOBIANS] );

#if defined(HKNP_SOLVER_EXPORT_FRICTION_IMPULSE)
	// This holds impulses for linear0, linear1, and angular friction
	// when mxSize == 4 this is for 4 different manifolds, and the values are transposed
	//  (linear0 for all 4 jacobians is in the first vector, linear1 in the second, angular in the third)
#if defined(HKNP_MX_FRICTION)
	hkReal m_frictionImpulseApplied[3][HKNP_NUM_MX_JACOBIANS];
#else
	hkReal m_frictionImpulseApplied[HKNP_NUM_MX_JACOBIANS][3];
#endif
#endif
};

struct hknpManifoldSolverInfo;


/// Contact constraint implementation of hknpConstraintSolver.
class hknpContactSolver: public hknpConstraintSolver
{
	public:

		/// Allocate solver temps.
		static void allocateTempsImpl(const hknpConstraintSolverJacobianRange& jacobians, hkBlockStreamBase::Range& temps,
									  hknpConstraintSolverJacobianWriter* tempsWriter);

		/// Step the solver.
		/// Note that if a Jacobian has the linked flipped, it uses a flipped solverVelA and  solverVelB
		static void HK_CALL stepJacobianBatch(
			const hknpSimulationThreadContext& tl, const hknpSolverInfo* HK_RESTRICT solverInfo, const hknpSolverStep& solverStep,
			hknpSolverVelocity* HK_RESTRICT solverVelA, hknpSolverVelocity* HK_RESTRICT solverVelB,
			const hknpConstraintSolverJacobianRange* jacobians,
			const hkBlockStreamBase::Range* temps
			);

		/// Step the solver.
		template <int NUM_MANIFOLDS>
		HK_FORCE_INLINE static void HK_CALL _stepJacobianBatch(
			const hknpSimulationThreadContext& tl, const hknpSolverInfo* HK_RESTRICT solverInfo, const hknpSolverStep& solverStep,
			hknpSolverVelocity* HK_RESTRICT  solverVelAStream, hknpSolverVelocity* HK_RESTRICT  solverVelBStream,
			const hknpConstraintSolverJacobianRange* HK_RESTRICT jacobians,
			const hkBlockStreamBase::Range* HK_RESTRICT temps
			);


		// -------------------------------------------------------------------------------------------------------------
		// Base class methods.
		// -------------------------------------------------------------------------------------------------------------

		virtual void allocateTemps(const hknpConstraintSolverJacobianRange& jacobians, hkBlockStreamBase::Range& temps,
			hknpConstraintSolverJacobianWriter* tempsWriter);

		virtual void solveJacobians(const hknpSimulationThreadContext& tl, const hknpSolverStepInfo& stepInfo, const hknpSolverStep& solverStep,
			const hknpConstraintSolverJacobianRange2* jacobians,
			hknpSolverVelocity* HK_RESTRICT solverVelA, hknpSolverVelocity* HK_RESTRICT solverVelB,
			const hknpIdxRange& motionEntryA, const hknpIdxRange& motionEntryB);
};


#endif // HKNP_CONTACT_SOLVER_H

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
