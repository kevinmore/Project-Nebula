/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_EVENTS_H
#define HKNP_EVENTS_H

#include <Common/Base/Container/CommandStream/hkCommandStream.h>
#include <Geometry/Internal/Types/hkcdManifold4.h>
#include <Physics/Physics/Dynamics/World/Events/hknpEventType.h>

struct hknpUnaryBodyEvent;
struct hknpBinaryBodyEvent;
struct hknpManifoldStatusEvent;
struct hknpManifoldProcessedEvent;
struct hknpContactImpulseEvent;
struct hknpContactImpulseClippedEvent;
struct hknpBodyActivationEvent;
struct hknpTriggerVolumeEvent;
struct hknpBodyExitedBroadPhaseEvent;
struct hknpConstraintForceEvent;
struct hknpConstraintForceExceededEvent;


/// A physics event.
struct hknpEvent : public hkCommand
{
	/// Constructor.
	hknpEvent( hkUint16 subType, int sizeInBytes ) : hkCommand( TYPE_PHYSICS_EVENTS, subType, sizeInBytes ) {}

	/// Is this event a hknpUnaryEvent?
	bool isUnaryEvent() const { return m_secondaryType > hknpEventType::END_OF_BINARY_BODY_EVENT && m_secondaryType < hknpEventType::END_OF_UNARY_BODY_EVENTS; }

	/// Is this event a hknpBinaryEvent?
	bool isBinaryEvent() const { return m_secondaryType < hknpEventType::END_OF_BINARY_BODY_EVENT; }

	//
	//	A set of convenience cast operators
	//

	/// Checked cast to hknpUnaryEvent
	const hknpUnaryBodyEvent& asUnaryEvent() const { HK_ASSERT( 0xf0345745, isUnaryEvent()); return (const hknpUnaryBodyEvent&)*this; }

	/// Checked cast to hknpBinaryEvent
	const hknpBinaryBodyEvent& asBinaryEvent() const { HK_ASSERT( 0xf0345745, isBinaryEvent()); return (const hknpBinaryBodyEvent&)*this; }

	/// Checked cast to hknpBodyExitedBroadPhaseEvent
	const hknpBodyExitedBroadPhaseEvent& asBodyExitedBroadPhaseEvent() const { HK_ASSERT( 0xf0345745, m_secondaryType == hknpEventType::BODY_EXITED_BROAD_PHASE); return (const hknpBodyExitedBroadPhaseEvent&)*this; }

	/// Checked cast to hknpManifoldStatusEvent
	const hknpManifoldStatusEvent& asManifoldStatusEvent() const { HK_ASSERT( 0xf0345745, m_secondaryType == hknpEventType::MANIFOLD_STATUS); return (const hknpManifoldStatusEvent&)*this; }

	/// Checked cast to hknpManifoldProcessedEvent
	const hknpManifoldProcessedEvent& asManifoldProcessedEvent() const { HK_ASSERT( 0xf0345745, m_secondaryType == hknpEventType::MANIFOLD_PROCESSED); return (const hknpManifoldProcessedEvent&)*this; }

	/// Checked cast to hknpContactImpulseEvent
	const hknpContactImpulseEvent& asContactImpulseEvent() const { HK_ASSERT( 0xf0345745, m_secondaryType == hknpEventType::CONTACT_IMPULSE); return (const hknpContactImpulseEvent&)*this; }

	/// Checked cast to hknpContactImpulseClippedEvent
	const hknpContactImpulseClippedEvent& asContactImpulseClippedEvent() const { HK_ASSERT( 0xf0345745, m_secondaryType == hknpEventType::CONTACT_IMPULSE_CLIPPED); return (const hknpContactImpulseClippedEvent&)*this; }

	/// Checked cast to hknpConstraintForceEvent
	const hknpConstraintForceEvent& asConstraintForceEvent() const { HK_ASSERT( 0xaf13e241, m_secondaryType == hknpEventType::CONSTRAINT_FORCE); return (const hknpConstraintForceEvent&)*this; }

	/// Checked cast to hknpConstraintForceExceededEvent
	const hknpConstraintForceExceededEvent& asConstraintForceExceededEvent() const { HK_ASSERT( 0xaf13e241, m_secondaryType == hknpEventType::CONSTRAINT_FORCE_EXCEEDED); return (const hknpConstraintForceExceededEvent&)*this; }

	/// Checked cast to hknpTriggerVolumeEvent
	const hknpTriggerVolumeEvent& asTriggerVolumeEvent() const { HK_ASSERT( 0xf0345745, m_secondaryType == hknpEventType::TRIGGER_VOLUME); return (const hknpTriggerVolumeEvent&)*this; }

	/// Checked cast to hknpBodyActivationEvent
	const hknpBodyActivationEvent& asBodyActivationEvent() const { HK_ASSERT( 0xf0345745, m_secondaryType == hknpEventType::BODY_ACTIVATION); return (const hknpBodyActivationEvent&)*this; }
};


/// An empty event, needed to test the event dispatcher.
struct hknpEmptyEvent : public hknpEvent
{
	hknpEmptyEvent( hknpBodyId id ) : hknpEvent( hknpEventType::END_OF_UNARY_BODY_EVENTS, sizeof(*this) ) {}

	HK_FORCE_INLINE void printCommand( hknpWorld* world, hkOstream& stream ) const {}
	HK_FORCE_INLINE void checkIsEmptyCommand() const {}	// This allows the compiler to check that all commands are dispatched
};


/// A base event, used if an event concerns only one body.
struct hknpUnaryBodyEvent : public hknpEvent
{
	hknpUnaryBodyEvent( hkUint16 subType, int sizeInBytes, hknpBodyId id) :
		hknpEvent( subType, sizeInBytes ), m_bodyId(id) {}

	hknpBodyId m_bodyId;		///< Body A
};


/// A base event, used if an event concerns two bodies.
struct hknpBinaryBodyEvent : public hknpEvent
{
	hknpBinaryBodyEvent( hkUint16 subType, int sizeInBytes, hknpBodyId idA, hknpBodyId idB):
		hknpEvent( subType, sizeInBytes )
	{
		m_bodyIds[0] = idA;
		m_bodyIds[1] = idB;
	}

	hknpBodyId m_bodyIds[2];	///< Body A and B
};


// Helper structures to do allow for implementing command dispatching without vtables.
// Check out hknpApiCommandProcessor::exec() for how to do this.
#define HKNP_DECLARE_EVENT_DISCRIMINATOR( TYPE, ID )	\
	template < >	struct hknpEventTypeDiscriminator<ID> { typedef TYPE CommandType; }

template <int X>	struct hknpEventTypeDiscriminator { typedef hknpEmptyEvent CommandType; };


/// Body activated/deactivated event.
struct hknpBodyActivationEvent : public hknpUnaryBodyEvent
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpBodyActivationEvent );

	hknpBodyActivationEvent( hknpBodyId id, bool isActive ) :
	hknpUnaryBodyEvent( hknpEventType::BODY_ACTIVATION, sizeof(*this), id ), m_activated(isActive) {}

	/// Print.
	void printCommand( hknpWorld* world, hkOstream& stream ) const;

	hkBool m_activated;		///< true = body activated, false = body deactivated
};
HKNP_DECLARE_EVENT_DISCRIMINATOR( hknpBodyActivationEvent, hknpEventType::BODY_ACTIVATION );


/// An event raised if a body moves beyond the broad phase extents.
struct hknpBodyExitedBroadPhaseEvent : public hknpUnaryBodyEvent
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpBodyExitedBroadPhaseEvent );

	hknpBodyExitedBroadPhaseEvent( hknpBodyId id ) :
		hknpUnaryBodyEvent( hknpEventType::BODY_EXITED_BROAD_PHASE, sizeof(*this), id ) {}

	/// Print.
	void printCommand( hknpWorld* world, hkOstream& stream ) const;
};
HKNP_DECLARE_EVENT_DISCRIMINATOR( hknpBodyExitedBroadPhaseEvent, hknpEventType::BODY_EXITED_BROAD_PHASE );


// Reserved event
struct hknpReserved0Event : public hknpEvent
{
	hknpReserved0Event() : hknpEvent( hknpEventType::RESERVED_0, sizeof(*this) ) {}

	void printCommand( hknpWorld* world, hkOstream& stream ) const;
};
HKNP_DECLARE_EVENT_DISCRIMINATOR( hknpReserved0Event, hknpEventType::RESERVED_0 );


// Reserved event
struct hknpReserved1Event : public hknpEvent
{
	hknpReserved1Event() : hknpEvent( hknpEventType::RESERVED_1, sizeof(*this) ) {}

	void printCommand( hknpWorld* world, hkOstream& stream ) const;
};
HKNP_DECLARE_EVENT_DISCRIMINATOR( hknpReserved1Event, hknpEventType::RESERVED_1 );


// Reserved event
struct hknpReserved2Event : public hknpEvent
{
	hknpReserved2Event() : hknpEvent( hknpEventType::RESERVED_2, sizeof(*this) ) {}

	void printCommand( hknpWorld* world, hkOstream& stream ) const;
};
HKNP_DECLARE_EVENT_DISCRIMINATOR( hknpReserved2Event, hknpEventType::RESERVED_2 );


// Reserved event
struct hknpReserved3Event : public hknpEvent
{
	hknpReserved3Event() : hknpEvent( hknpEventType::RESERVED_3, sizeof(*this) ) {}

	void printCommand( hknpWorld* world, hkOstream& stream ) const;
};
HKNP_DECLARE_EVENT_DISCRIMINATOR( hknpReserved3Event, hknpEventType::RESERVED_3 );


#endif	// HKNP_EVENTS_H

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
