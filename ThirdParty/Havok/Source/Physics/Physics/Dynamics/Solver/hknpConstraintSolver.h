/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CONSTRAINT_SOLVER_H
#define HKNP_CONSTRAINT_SOLVER_H

#include <Physics/Physics/hknpTypes.h>
#include <Common/Base/Container/BlockStream/hkBlockStream.h>
#include <Physics/Physics/Dynamics/World/Grid/hknpGrid.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverVelocity.h>


// -------------------------------------------------------------------------------------------------------------
// Basic solver types.
// -------------------------------------------------------------------------------------------------------------

/// Type definition for constraint solver identifiers (see hknpConstraintSolverType::Enum).
HK_DECLARE_HANDLE( hknpConstraintSolverId, hkUint8, 0xf ) ;


/// Type definition for constraint solver priority (see hknpDefaultConstraintSolverPriority).
typedef hkUint8 hknpConstraintSolverPriority;

/// Built-in constraint solver priorities.
/// These priorities determine the order in which constraints of different types are solved.
/// For custom solvers, one can choose custom constraint priorities and interleave them with these defaults.
/// Assigning equal priorities to different constraint types is supported.
struct hknpDefaultConstraintSolverPriority
{
	enum Enum
	{
		FIRST = 0,
		JOINTS = 10,			///< Default Priority assigned to joint constraints.
		MOVING_CONTACTS = 20,	///< Default Priority assigned to moving contact constraints.
		FIXED_CONTACTS = 30,	///< Default Priority assigned to fixed contact constraints.
		LAST = 40
	};
};



// -------------------------------------------------------------------------------------------------------------
// Solver Stream data structures.
// -------------------------------------------------------------------------------------------------------------

/// The base class for solver jacobian streams.

class hknpConstraintSolverJacobianStream : public hkBlockStream<hkUint8>
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_COLLIDE, hknpConstraintSolverJacobianStream );

		/// Constructor.
		hknpConstraintSolverJacobianStream( Allocator* tlAllocator, bool zeroNewBlocks = false );

		/// Constructor. You need to manually call initBlockStream() before you can use this class.
		hknpConstraintSolverJacobianStream();
};


/// Solver step information.
struct hknpSolverStep
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hknpSolverStep );

	enum Flags
	{
		FIRST_ITERATION = 1 << 0,	///< True if this is the first solver iteration.
		LAST_ITERATION = 1 << 1		///< True if this is the last solver iteration.
	};


	hkPadSpu<int>	m_currentStep;	///< The current integration step.
	hkSimdReal		m_intregratePositionFactor;
	hkPadSpu<int>	m_flags;

	/// Initialize using the given step index.
	void init( const hknpSolverInfo& solverInfo, int iStep, int iMicroStep );

	HK_FORCE_INLINE hkBool32 isFirstIteration() const	{ return m_flags & FIRST_ITERATION; }
	HK_FORCE_INLINE hkBool32 isLastIteration() const	{ return m_flags & LAST_ITERATION; }
};


/// Solver jacobian stream writer.
class hknpConstraintSolverJacobianWriter : public hknpConstraintSolverJacobianStream::Writer {};


/// Solver jacobian stream reader.
class hknpConstraintSolverJacobianReader : public hknpConstraintSolverJacobianStream::Reader {};


/// Base class for the Solver jacobian range.
/// A solver jacobian range can point to other ranges using LinkedRange.m_next forming a linked list.
/// In this case, all the ranges correspond to the same entry in the containing grid.
class hknpConstraintSolverJacobianRange : public hkBlockStreamBase::LinkedRange {};


/// Solver jacobian range used for grid based solver code.
struct hknpConstraintSolverJacobianRange2 : public hknpConstraintSolverJacobianRange
{
	public:

		/// Several flags pertaining to a jacobian range.
		enum Bits
		{
			LIVE_JACOBIANS = 1 << 0,	///< Use live jacobians when solving this range.
			
			SOLVER_TEMPS = 1 << 1		///< This range needs it's solver temps to be allocated after setup.
		};

		typedef hkFlags<Bits, hkUint8> Flags;

	public:

		/// Constructor.
		HK_FORCE_INLINE hknpConstraintSolverJacobianRange2();

		/// Initialize this range with a solver id and flags.
		HK_FORCE_INLINE void initRange( hknpConstraintSolverType::Enum solverId, Flags flags );

	public:

		hknpConstraintSolverId m_solverId;					///< The id of the solver to use with this range.
		Flags m_flags;										///< Flags for solver behavior
		mutable hkBlockStreamBase::Range m_solverTempRange;	///< The range of solver temps associated with the Jacobians of this range.
};



// -------------------------------------------------------------------------------------------------------------
// Solver Multi-threading data structures.
// -------------------------------------------------------------------------------------------------------------

/// Processing grid for jacobian constraint solvers.
class hknpConstraintSolverJacobianGrid : public hknpGrid<hknpConstraintSolverJacobianRange2> {};


/// Multi-threaded Scheduling info for a constraint jacobian grid.
class hknpConstraintSolverSchedulerGridInfo
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpConstraintSolverSchedulerGridInfo );

		/// Constructor.
		HK_FORCE_INLINE hknpConstraintSolverSchedulerGridInfo();

		/// Indicate that the grid is a grid of links (see hknpSpaceSplitter).
		HK_FORCE_INLINE void setIsLinkGrid();
		/// Returns true if the grid is a grid of links (see hknpSpaceSplitter).
		HK_FORCE_INLINE bool isLinkGrid() const;

		/// Indicate that the grid is an array of cells (see hknpSpaceSplitter).
		HK_FORCE_INLINE void setIsCellArray();
		/// Returns true if the grid is an array of cells (see hknpSpaceSplitter).
		HK_FORCE_INLINE bool isCellArray() const;

		/// Sets the priority of the grid (see hknpDefaultConstraintSolverPriority).
		HK_FORCE_INLINE void setPriority(hknpConstraintSolverPriority value);
		/// Gets the priority of the grid (see hknpDefaultConstraintSolverPriority).
		HK_FORCE_INLINE hknpConstraintSolverPriority getPriority() const;

	protected:

		/// Flags enum.
		enum FlagBits
		{
			CELL_ARRAY	= 1 << 0,	///< The grid is an array of processing cells.
			LINK_GRID	= 1 << 1,	///< The grid is a grid of links between cells.
		};

		typedef hkFlags<FlagBits, int> Flags;

		Flags m_flags;								///< The flags.
		hknpConstraintSolverPriority m_priority;	///< The priority.
};



// -------------------------------------------------------------------------------------------------------------
// Solver base class.
// -------------------------------------------------------------------------------------------------------------

class hknpSolverStepInfo;
class hknpSimulationThreadContext;
class hknpSolverVelocity;


/// Base class for constraint solvers.
#if !defined(HK_PLATFORM_SPU)
class hknpConstraintSolver : public hkReferencedObject
#else
class hknpConstraintSolver
#endif
{
	public:

#if !defined( HK_PLATFORM_SPU )
		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
#else
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpConstraintSolver );
#endif

		/// Destructor
#if !defined(HK_PLATFORM_SPU)
		virtual ~hknpConstraintSolver() {}
#endif

		/// This function should be implemented by solvers that use solver temps and want to allocate them after setup. This is indicated using hknpConstraintSolverJacobianRange2::SOLVER_TEMPS.
		/// Each solver implementation has the choice to either use no temps, allocate the temps during setup, or allocate them after setup
		virtual void allocateTemps(
			const hknpConstraintSolverJacobianRange& jacobians, hkBlockStreamBase::Range& temps,
			  hknpConstraintSolverJacobianWriter* tempsWriter );

		/// Solve the provided jacobians.
		virtual void solveJacobians(
			const hknpSimulationThreadContext& tl, const hknpSolverStepInfo& stepInfo, const hknpSolverStep& solverStep,
			const hknpConstraintSolverJacobianRange2* jacobians,
			hknpSolverVelocity* HK_RESTRICT solverVelA, hknpSolverVelocity* HK_RESTRICT solverVelB,
			const hknpIdxRange& motionEntryA, const hknpIdxRange& motionEntryB ) = 0;
};


#include <Physics/Physics/Dynamics/Solver/hknpConstraintSolver.inl>

#endif // HKNP_CONSTRAINT_SOLVER_H

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
