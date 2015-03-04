/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CONSTRAINT_ATOM_SOLVER_SETUP_H
#define HKNP_CONSTRAINT_ATOM_SOLVER_SETUP_H

#include <Physics/Physics/hknpTypes.h>
#include <Physics/ConstraintSolver/Solve/hkpSolverResults.h>

struct hknpSolverInfo;
class hkpConstraintQueryStepInfo;
class hkpSolverResults;
class hknpConstraintSolverJacobianStream;
class hknpConstraintSolverJacobianWriter;
class hknpConstraintSolverJacobianGrid;
class hknpConstraintAtomJacobianStream;
class hknpConstraint;
class hknpConstraintAtomJacobianWriter;
class hknpConstraintAtomJacobianGrid;
class hknpCdPairWriter;
class hknpCdPairStream;


/// Atom constraint Jacobian header.
class hknpJacobianHeaderSchema
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpJacobianHeaderSchema );

		HK_FORCE_INLINE void initHeader( hkpSolverResults* sr, int solverResultStriding, hknpConstraint* constraint );

	private:

		HK_ALIGN_REAL( hkUint8 m_padding0 );	// type stored here .. or.

	public:

		hkUint8 m_solverResultStriding;		///< Striding of solver results in bytes.
		hknpConstraint::Flags m_flags;		///< Constraint flags.
		hknpImmediateConstraintId m_immediateConstraintId;	///< Immediate constraint ID.
		hknpSolverId m_bodyAIndex;			///< Index of the first body into solver velocities.
		hknpSolverId m_bodyBIndex;			///< Index of the second body into solver velocities.

		hknpCellIndex m_cellIndexA;			///< Space splitter cell of the first body.
		hknpCellIndex m_cellIndexB;			///< Space splitter cell of the second body.
		hkUchar m_isLinkFlipped;			///< Indicates which body is the first body in the constraint.
		hkUchar m_schemaSizeDiv16;			///< Size of the full schema divided by 16.

		hknpBodyFlags m_combinedBodyFlags;	///< The modifiers enabled for this constraint

		hkpSolverResults* m_solverResultInMainMemory;	///< Pointer to solver results in main memory.
		hknpConstraint* m_constraint;		///< Pointer to the instance of this schema (NULL for immediate constraints).
};


/// Setup of atom states.
namespace hknpConstraintAtomSolverSetup
{
	/// The state of a constraint, used during the solver setup.
	
	struct ConstraintState
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, ConstraintState );

		HK_FORCE_INLINE bool isActive() const												{ return m_islandIndex != hknpIslandId::InvalidValue; }
		HK_FORCE_INLINE void setActive(hkUint16 linkIndex)									{ m_linkIndex = linkIndex; }
		HK_FORCE_INLINE void setInactive(hkUint16 islandIndex, hkUint16 inactiveLinkIndex)	{ m_islandIndex = islandIndex; m_linkIndex = inactiveLinkIndex; }

		hkUint32 m_constraintIndex;	///< Index in the constraint solver's constraints array.
		hkUint16 m_islandIndex;		///< Index of deactivated island (if inactive). An invalid island id indicates an active constraint.
		hkUint16 m_linkIndex;		///< Index of grid link (if active).
	};

	/// An array of sorted ConstraintState structures.
	/// They are sorted first by activation state, then by index (either link or island).
	struct ConstraintStates : hkArray<ConstraintState>
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, ConstraintStates );

		ConstraintStates();
		void clear();

		/// Gather and sort all the enabled constraints from the given set.
		void gatherConstraints( hknpWorld* world, hknpConstraint** constraints, int numConstraints );

		/// Regroup the inactive constraints whose islands got activated.
		void regroupReactivatedConstraints( hknpWorld* world, hknpConstraint** constraints, const hkArray<hknpIslandId>& sortedActivatedIslands );

		HK_FORCE_INLINE int getFirstReactivatedIndex() const	{ return m_numActive; }
		HK_FORCE_INLINE int getLastReactivatedIndex() const		{ return m_numActive+m_numReactivated-1; }

		int m_numActive;		///< The number of active constraints
		int m_numReactivated;	///< The number of constraints that were reacativated during a call to regroupReactivated()
	};

	/// A group of ConstraintState's which have the same grid link index.
	/// This means it can be processed in parallel with other SubTasks.
	struct SubTask
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, SubTask );

		int m_firstStateIndex;			///< First index in ConstraintState array
		int m_lastStateIndex;			///< Last index in ConstraintState array
		unsigned int m_relativeCost;	///< Estimated setup cost, relative to other subtasks
	};

	/// An array of sorted SubTask structures.
	/// They are sorted by estimated setup cost, descending.
	struct SubTasks : hkArray<SubTask>
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, SubTasks );

		/// Group the given set of (sorted) states into tasks, and sort them by estimated setup cost.
		void create( hknpConstraint** constraints, const ConstraintStates& states, int firstIndex, int lastIndex );
	};


	/// Populate a hkpConstraintQueryStepInfo from a hknpSolverInfo.
	void HK_CALL setupConstraintStepInfo(
		const hknpSolverInfo& solverInfo, hkpConstraintQueryStepInfo* HK_RESTRICT inOut );

	/// Set up constraints for single-threaded solving.
	void HK_CALL setupConstraintsSt(
		hknpWorld* world, hknpSolverInfo& solverInfo, hknpConstraint** constraints, int numConstraints,
		hknpConstraintSolverJacobianWriter& schemaWriter, hknpConstraintSolverJacobianWriter& solverTempsWriter,
		hknpCdPairWriter* activePairWriter );

	/// Set up a single constraint for multi-threaded solving.
	void HK_CALL setupConstraintMt(
		const hknpSimulationThreadContext& tl, hknpWorld* world,
		hknpConstraint* constraint,
		hknpConstraintSolverJacobianGrid& grid, hknpConstraintSolverJacobianStream* stream,
		hknpConstraintSolverJacobianStream* solverTempsStream,
		hknpCdPairStream* activePairStream );

	/// Set up a group of constraint states for multi-threaded solving.
	void HK_CALL setupConstraintsMt(
		const hknpSimulationThreadContext& tl, hknpWorld* world, hknpConstraint** constraints,
		const hkArray<ConstraintState>& states, int firstIndex, int lastIndex,
		hknpConstraintSolverJacobianGrid& grid, hknpConstraintSolverJacobianStream* stream,
		hknpConstraintSolverJacobianStream* tempsStream,
		hknpCdPairStream* activeStream );
}


#endif // HKNP_CONSTRAINT_ATOM_SOLVER_SETUP_H

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
