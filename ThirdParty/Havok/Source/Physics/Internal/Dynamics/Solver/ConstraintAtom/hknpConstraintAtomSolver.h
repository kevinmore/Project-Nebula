/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CONSTRAINT_ATOM_SOLVER_H
#define HKNP_CONSTRAINT_ATOM_SOLVER_H

#include <Physics/Physics/hknpTypes.h>
#include <Common/Base/Container/BlockStream/hkBlockStream.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverInfo.h>
#include <Physics/Physics/Dynamics/Solver/hknpConstraintSolver.h>
#include <Physics/Physics/Dynamics/Constraint/hknpConstraint.h>

class hkpJacobianSchema;
class hkpConstraintData;
struct hkpConstraintAtom;
class hknpCdPairWriter;
class hknpSolverSumVelocity;


/// Atom constraint implementation of hknpConstraintSolver.
class hknpConstraintAtomSolver : public hknpConstraintSolver
{
	public:

		/// Constructor.
		hknpConstraintAtomSolver();

		/// Destructor.
		~hknpConstraintAtomSolver();

		/// Add a constraint.
		void addConstraint( hknpConstraint* constraint );

		/// Remove a constraint.
		void removeConstraint( hknpConstraint* constraint );

		/// Get the number of constraints.
		HK_FORCE_INLINE int getNumConstraints() const;

		/// Get all the constraints.
		HK_FORCE_INLINE hknpConstraint** getConstraints() const;

		/// Check if a constraint has been added.
		HK_FORCE_INLINE bool isConstraintAdded( const hknpConstraint* constraint ) const;

		/// Find all constraints using a given body.
		void findConstraintsUsingBody( hknpBodyId bodyId, hkArray<const hknpConstraint*>& instancesOut ) const;

		/// Set up the current constraints for solving.
		void setupConstraints(
			hknpSimulationThreadContext* tl, hknpConstraintSolverJacobianWriter& schemaWriter,
			hknpConstraintSolverJacobianWriter& solverTempsWriter, hknpCdPairWriter* activePairWriter );

		// hknpConstraintSolver implementation.
		virtual void solveJacobians(
			const hknpSimulationThreadContext& tl,
			const hknpSolverStepInfo& stepInfo, const hknpSolverStep& solverStep, const hknpConstraintSolverJacobianRange2* jacobians,
			hknpSolverVelocity* HK_RESTRICT solverVelA, hknpSolverVelocity* HK_RESTRICT solverVelB,
			const hknpIdxRange& motionEntryA, const hknpIdxRange& motionEntryB );

	protected:

		// Actual implementation of the solver function.
		static void solveJacobiansImpl(
			const hknpSimulationThreadContext* tl, const hknpSolverInfo& npInfo, const hknpSolverStep& solverStep,
			const hknpConstraintSolverJacobianRange* jacobians, const hkBlockStreamBase::Range* temps,
			const hknpSolverSumVelocity* HK_RESTRICT solverSumVelAStream, hknpSolverVelocity* HK_RESTRICT solverVelAStream,
			const hknpSolverSumVelocity* HK_RESTRICT solverSumVelBStream, hknpSolverVelocity* HK_RESTRICT solverVelBStream );

		// Implementation of the solver export function for exporting impulses and generating events.
		static void exportImpl(
			const hknpSimulationThreadContext* tl,
			const hknpSolverInfo& npInfo, const hknpConstraintSolverJacobianRange& jacobians, const hkBlockStreamBase::Range& tempsRange,
			const hknpSolverSumVelocity* HK_RESTRICT solverSumVelAStream, hknpSolverVelocity* HK_RESTRICT solverVelAStream,
			const hknpSolverSumVelocity* HK_RESTRICT solverSumVelBStream, hknpSolverVelocity* HK_RESTRICT solverVelBStream );

	protected:

#if !defined(HK_PLATFORM_SPU)
		hkArray< hkRefPtr<hknpConstraint> > m_constraints;	///< The constraints
#endif
};

#include <Physics/Internal/Dynamics/Solver/ConstraintAtom/hknpConstraintAtomSolver.inl>


#endif // HKNP_CONSTRAINT_ATOM_SOLVER_H

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
