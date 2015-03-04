/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_INTERNAL_COMMANDS_H
#define HKNP_INTERNAL_COMMANDS_H

#include <Common/Base/Container/CommandStream/hkCommandStream.h>
#include <Physics/Physics/Dynamics/Solver/Event/hknpSolverEvents.h>
#include <Physics/Physics/Dynamics/Solver/hknpConstraintSolver.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverData.h>

class hknpConstraint;
class hknpConstraintSolverJacobianGrid;


/// Base class for all internal commands used during simulation.
class hknpInternalCommand : public hkCommand
{
	public:

		enum SecondaryType
		{
			CMD_CELL_INDEX_MODIFIED,
			CMD_ATOM_SOLVER_FORCE_CLIPPED,
			CMD_VALIDATE_TRIGGER_VOLUME_EVENT,
			CMD_MOTION_WELD_TOI,
			CMD_ADD_CONSTRAINT_RANGE,
			CMD_EMPTY
		};

	public:

		hknpInternalCommand( hknpBodyId bodyId, hkUint16 subType, int sizeInBytes )
		:	hkCommand( TYPE_PHYSICS_INTERNAL, subType, sizeInBytes )
		{
			m_bodyId = bodyId;
		}

		hknpBodyId m_bodyId;
};


struct hknpEmptyInternalCommand : public hknpInternalCommand
{
	hknpEmptyInternalCommand( hknpBodyId id ) : hknpInternalCommand( id, CMD_EMPTY, sizeof(*this) ) {}
	HK_FORCE_INLINE void printCommand( hknpWorld* world, hkOstream& stream ) const {}
	HK_FORCE_INLINE void checkIsEmptyCommand() const {}	/// This allows the compiler to check that all commands are dispatched
};


// Helper structures to allow for implementing command dispatching without vtables.
// See hknpApiCommandProcessor::exec() for how to do this.
#define HKNP_DECLARE_INTERNAL_COMMAND_DISCRIMINATOR( TYPE, ID )	\
	template <>		struct hknpInternalCommandTypeDiscriminator<hknpInternalCommand::ID> { typedef TYPE CommandType; }

template <int X>	struct hknpInternalCommandTypeDiscriminator { typedef hknpEmptyInternalCommand CommandType; };


struct hknpCellIndexModifiedCommand : public hknpInternalCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCellIndexModifiedCommand );

	hknpCellIndexModifiedCommand( hknpBodyId id, hknpCellIndex oldIndex, hknpCellIndex newIndex )
	:	hknpInternalCommand( id, CMD_CELL_INDEX_MODIFIED, sizeof(*this) )
	{
		m_oldCellIndex = oldIndex;
		m_newCellIndex = newIndex;
	}

	void printCommand( hknpWorld* world, hkOstream& stream ) const;

	hknpCellIndex m_oldCellIndex;
	hknpCellIndex m_newCellIndex;
};
HKNP_DECLARE_INTERNAL_COMMAND_DISCRIMINATOR( hknpCellIndexModifiedCommand, CMD_CELL_INDEX_MODIFIED );


struct hknpAtomSolverForceClippedCommand : public hknpInternalCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpAtomSolverForceClippedCommand );

	hknpAtomSolverForceClippedCommand( hknpConstraint* constraint, int atomIndex, hkSimdRealParameter impulseFactor )
	:	hknpInternalCommand( hknpBodyId(0), CMD_ATOM_SOLVER_FORCE_CLIPPED, sizeof(*this) )
	{
		m_constraint = constraint;
		m_atomIndex = atomIndex;
	}

	void printCommand( hknpWorld* world, hkOstream& stream ) const;

	hkReal m_impulseFactor;
	hknpConstraint* m_constraint;
	int m_atomIndex;
};
HKNP_DECLARE_INTERNAL_COMMAND_DISCRIMINATOR( hknpAtomSolverForceClippedCommand, CMD_ATOM_SOLVER_FORCE_CLIPPED );


struct hknpValidateTriggerVolumeEventCommand : public hknpContactSolverEvent
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpValidateTriggerVolumeEventCommand );

	hknpValidateTriggerVolumeEventCommand( hknpBodyId idA, hknpBodyId idB )
	:	hknpContactSolverEvent( hknpInternalCommand::CMD_VALIDATE_TRIGGER_VOLUME_EVENT, sizeof(*this), idA, idB )
	{
		m_primaryType = TYPE_PHYSICS_INTERNAL;
	}

	void printCommand( hknpWorld* world, hkOstream& stream ) const;
};
HKNP_DECLARE_INTERNAL_COMMAND_DISCRIMINATOR( hknpValidateTriggerVolumeEventCommand, CMD_VALIDATE_TRIGGER_VOLUME_EVENT );


struct hknpMotionWeldTOICommand : public hknpContactSolverEvent
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpMotionWeldTOICommand );

	hknpMotionWeldTOICommand( hknpBodyId idA, hknpBodyId idB)
	:	hknpContactSolverEvent( hknpInternalCommand::CMD_MOTION_WELD_TOI, sizeof(*this), idA, idB )
	{
		m_primaryType = TYPE_PHYSICS_INTERNAL;
	}

	void printCommand( hknpWorld* world, hkOstream& stream ) const;
};
HKNP_DECLARE_INTERNAL_COMMAND_DISCRIMINATOR( hknpMotionWeldTOICommand, CMD_MOTION_WELD_TOI );


struct hknpAddConstraintRangeCommand : public hknpInternalCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpAddConstraintRangeCommand );

	HK_FORCE_INLINE hknpAddConstraintRangeCommand( hknpBodyId idA );

	void printCommand( hknpWorld* world, hkOstream& stream ) const;

	hknpConstraintSolverJacobianRange2 m_range;
	hknpConstraintSolverJacobianGrid* m_grid;
	int m_linkIndex;
	hkUint32 m_sortKey;
};
HKNP_DECLARE_INTERNAL_COMMAND_DISCRIMINATOR( hknpAddConstraintRangeCommand, CMD_ADD_CONSTRAINT_RANGE );


///
class hknpInternalCommandProcessor : public hkSecondaryCommandDispatcher
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		hknpInternalCommandProcessor(hknpWorld* world) : m_world(world)
		{
			m_simulationThreadContext = HK_NULL;
			m_solverData = HK_NULL;
		}

		void beginDispatch( hknpSolverData* solverData, hknpSimulationThreadContext* tl )
		{
			m_solverData = solverData;
			m_simulationThreadContext = tl;
		}

		void endDispatch()
		{
			m_solverData = HK_NULL;
			m_simulationThreadContext = HK_NULL;
		}

		/// dispatch commands

		virtual void exec( const hkCommand& command );

		virtual void print( const hkCommand& command, hkOstream& stream ) const;

	public:

		hknpWorld* m_world;

		hknpSolverData* m_solverData;							// temporary variable
		hknpSimulationThreadContext* m_simulationThreadContext;	// temporary variable
};


#include <Physics/Physics/Dynamics/World/Commands/hknpInternalCommands.inl>


#endif	// HKNP_INTERNAL_COMMANDS_H

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
