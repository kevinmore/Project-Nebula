/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_WORLD_SIGNALS_H
#define HKNP_WORLD_SIGNALS_H

#include <Physics/Physics/hknpTypes.h>
#include <Common/Base/Types/hkSignalSlots.h>

class hknpSolverData;


/// A set of signals used by hknpWorld.
struct hknpWorldSignals
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpWorldSignals );

	//
	// World signals
	//

	/// Called before a world is destroyed.
	HK_DECLARE_SIGNAL( WorldDestroyedSignal, hkSignal1< hknpWorld* > );
	WorldDestroyedSignal m_worldDestroyed;

	/// Called after a world has been shifted.
	HK_DECLARE_SIGNAL( WorldShiftedSignal, hkSignal2< hknpWorld*, hkVector4Parameter > );
	WorldShiftedSignal m_worldShifted;

	//
	// Body/motion signals
	//

	/// Called before allocating a body if the buffer is full.
	HK_DECLARE_SIGNAL( BodyBufferFullSignal, hkSignal2< hknpWorld*, hknpBodyManager* > );
	BodyBufferFullSignal m_bodyBufferFull;

	/// Called after the body buffer has been reallocated / changed.
	HK_DECLARE_SIGNAL( BodyBufferChangedSignal, hkSignal2< hknpWorld*, hknpBodyManager* > );
	BodyBufferChangedSignal m_bodyBufferChanged;

	/// Called after a body has been allocated and initialized.
	HK_DECLARE_SIGNAL( BodyCreatedSignal, hkSignal3< hknpWorld*, const hknpBodyCinfo*, hknpBodyId > );
	BodyCreatedSignal m_bodyCreated;

	/// Called after a body has been added to the world.
	HK_DECLARE_SIGNAL( BodyAddedSignal, hkSignal2< hknpWorld*, hknpBodyId >	);
	BodyAddedSignal	m_bodyAdded;

	/// Called before a body is removed from the world.
	HK_DECLARE_SIGNAL( BodyRemovedSignal, hkSignal2< hknpWorld*, hknpBodyId > );
	BodyRemovedSignal m_bodyRemoved;

	/// Called before a body is destroyed.
	HK_DECLARE_SIGNAL( BodyDestroyedSignal, hkSignal2< hknpWorld*, hknpBodyId >	);
	BodyDestroyedSignal	m_bodyDestroyed;

	/// Called before allocating a motion if the buffer is full.
	HK_DECLARE_SIGNAL( MotionBufferFullSignal, hkSignal2< hknpWorld*, hknpMotionManager* > );
	MotionBufferFullSignal m_motionBufferFull;

	/// Called after the motion buffer has been reallocated / changed.
	HK_DECLARE_SIGNAL( MotionBufferChangedSignal, hkSignal2< hknpWorld*, hknpMotionManager* > );
	MotionBufferChangedSignal m_motionBufferChanged;

	/// Called after a motion has been allocated and initialized.
	HK_DECLARE_SIGNAL( MotionCreatedSignal, hkSignal3< hknpWorld*, const hknpMotionCinfo*, hknpMotionId > );
	MotionCreatedSignal m_motionCreated;

	/// Called before a motion is destroyed.
	HK_DECLARE_SIGNAL( MotionDestroyedSignal, hkSignal2< hknpWorld*, hknpMotionId > );
	MotionDestroyedSignal m_motionDestroyed;

	/// Called when a static body is moved.
	HK_DECLARE_SIGNAL( StaticBodyMovedSignal, hkSignal2< hknpWorld*, hknpBodyId > );
	StaticBodyMovedSignal m_staticBodyMoved;

	/// Called when a static body becomes dynamic or vice versa.
	HK_DECLARE_SIGNAL( BodySwitchStaticDynamicSignal, hkSignal3< hknpWorld*, hknpBodyId, bool /*isStatic*/> );
	BodySwitchStaticDynamicSignal m_bodySwitchStaticDynamic;

	/// Called when a body is about to get attached to another body
	HK_DECLARE_SIGNAL( BodyAttachToCompoundSignal, hkSignal3< hknpWorld*, hknpBodyId, hknpBodyId > );
	BodyAttachToCompoundSignal m_bodyAttached;

	/// Called after a body is detached from another body.
	HK_DECLARE_SIGNAL( BodyDetachToCompoundSignal, hkSignal3< hknpWorld*, hknpBodyId, hknpMotionId > );
	BodyDetachToCompoundSignal m_bodyDetached;

	/// Called when a body changes its shape.
	HK_DECLARE_SIGNAL( BodyShapeSetSignal, hkSignal2< hknpWorld*, hknpBodyId > );
	BodyShapeSetSignal m_bodyShapeChanged;

	/// Called when a body is changed in any way via the world API.
	HK_DECLARE_SIGNAL( BodyChangedSignal, hkSignal2< hknpWorld*, hknpBodyId > );
	BodyChangedSignal m_bodyChanged;

	//
	// Constraint signals
	//

	/// Called after a constraint has been added to the world.
	HK_DECLARE_SIGNAL( ConstraintAddedSignal, hkSignal2< hknpWorld*, hknpConstraint* > );
	ConstraintAddedSignal m_constraintAdded;

	/// Called after a constraint has been removed from the world.
	HK_DECLARE_SIGNAL( ConstraintRemovedSignal, hkSignal2< hknpWorld*, hknpConstraint* > );
	ConstraintRemovedSignal m_constraintRemoved;

	/// Called after an immediate constraint has been added to the hknpSolverData structure.
	HK_DECLARE_SIGNAL( ImmediateConstraintAddedSignal, hkSignal2< hknpWorld*, hknpConstraint* > );
	ImmediateConstraintAddedSignal m_immediateConstraintAdded;

	//
	// Simulation signals
	//

	/// Called at the start of the collision detection phase of a simulation step.
	HK_DECLARE_SIGNAL( PreCollideSignal, hkSignal1< hknpWorld* > );
	PreCollideSignal m_preCollide;

	/// Called at the end of the collision detection phase of a simulation step.
	HK_DECLARE_SIGNAL( PostCollideSignal, hkSignal2< hknpWorld*, hknpSolverData* > );
	PostCollideSignal m_postCollide;

	/// Called at the start of the solving phase of a simulation step.
	HK_DECLARE_SIGNAL( PreSolveSignal, hkSignal2< hknpWorld*, hknpSolverData* > );
	PreSolveSignal m_preSolve;

	/// Called at the end of the solving phase of a simulation step.
	HK_DECLARE_SIGNAL( PostSolveSignal, hkSignal1< hknpWorld* > );
	PostSolveSignal m_postSolve;
};


#endif // HKNP_WORLD_SIGNALS_H

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
