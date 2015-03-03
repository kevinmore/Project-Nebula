/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_EVENT_DISPATCHER_H
#define HKNP_EVENT_DISPATCHER_H

#include <Physics/Physics/Dynamics/World/Events/hknpEvents.h>
#include <Physics/Physics/Collide/Event/hknpCollideEvents.h>
#include <Physics/Physics/Dynamics/Solver/Event/hknpSolverEvents.h>
#include <Common/Base/Types/hkSignalSlots.h>

class hknpSolverData;


/// Input structure for the event handler functions.
struct hknpEventHandlerInput
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpEventHandlerInput );

	/// Get the body which has the signal listener attached to.
	/// If you are using a global signal event handler, use getBodyA() or getBodyB() instead.
	HK_FORCE_INLINE hknpBodyId getThisBody( const hknpBinaryBodyEvent& event ) const
	{
		HK_ASSERT2( 0xf04346fe, m_bodiesAreReversed <= 1, "Cannot use getThisBody() with global events." );
		return event.m_bodyIds[m_bodiesAreReversed];
	}

	/// Get the other body than getThisBody().
	HK_FORCE_INLINE hknpBodyId getOtherBody( const hknpBinaryBodyEvent& event ) const
	{
		HK_ASSERT2( 0xf04346fe, m_bodiesAreReversed <= 1, "Cannot use getOtherBody() with global events." );
		return event.m_bodyIds[1-m_bodiesAreReversed];
	}

	/// Get the first body involved in an event.
	HK_FORCE_INLINE hknpBodyId getBodyA( const hknpBinaryBodyEvent& event ) const
	{
		return event.m_bodyIds[0];
	}

	/// Get the second body involved in an event.
	HK_FORCE_INLINE hknpBodyId getBodyB( const hknpBinaryBodyEvent& event ) const
	{
		return event.m_bodyIds[1];
	}


	/// The world.
	hknpWorld* m_world;

	/// The optional output of the collision detector, allows to add extra constraints.
	hknpSolverData* m_solverData;

	/// The optional command writer.
	hkBlockStream<hkCommand>::Writer* m_commandWriter;

	/// The optional simulation thread context.
	hknpSimulationThreadContext* m_simulationThreadContext;

	/// The body index this event to registered to:
	///		- if the event is registered to event.m_bodyIds[0] this is 0
	///		- if the event is registered to event.m_bodyIds[1] this is 1
	///		- else set to unsigned(-1) (also for unary events)
	hkUint32 m_bodiesAreReversed;
};


/// Declare signal function
HK_DECLARE_SIGNAL( hknpEventSignal, hkSignal2< const hknpEventHandlerInput&, const hknpEvent& > );


/// A simple dispatcher of hknpEvents to registered static functions.
class hknpEventDispatcher : public hkSecondaryCommandDispatcher
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Constructor.
		hknpEventDispatcher( hknpWorld* world );

		/// Get a signal for one event type for all bodies.
		hknpEventSignal& getSignal( hknpEventType::Enum eventType );

		/// Get a signal for one event type for just one body.
		hknpEventSignal& getSignal( hknpEventType::Enum eventType, hknpBodyId id );

		/// Unsubscribe all signals from a body.
		void unsubscribeAllSignals( hknpBodyId id );

		///
		HK_FORCE_INLINE void beginDispatch( hknpSolverData* solverData, hknpSimulationThreadContext* tl );

		///
		HK_FORCE_INLINE void endDispatch();

		/// Execute and remove all events remaining from previously applied merges.
		HK_FORCE_INLINE virtual void flushRemainingEvents();

		//
		// hkSecondaryCommandDispatcher implementation
		//

		/// Dispatch an event.
		HK_FORCE_INLINE virtual void exec( const hkCommand& command );

		/// Print an event.
		virtual void print( const hkCommand& command, hkOstream& stream ) const;

		//
		//	Internal section
		//

	public:

		typedef hkUint16 EntryIdx;
		enum { INVALID_ENTRY = 0xffff };

		// A linked list of event handlers for a given body
		struct Entry
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpEventDispatcher::Entry);

			HK_FORCE_INLINE void execEntry( const hknpEventHandlerInput& input, const hknpEvent& event )
			{
#if !defined ( HK_PLATFORM_SPU )
				if ( event.m_secondaryType == m_eventType )
				{
					m_signal.fire( input, event );
				}
#endif
			}

			EntryIdx m_nextEntry;
			hkEnum<hknpEventType::Enum,hkUint16> m_eventType;
			hknpEventSignal m_signal;
		};

	protected:

		Entry* allocateEntry( hknpBodyId id );

		void freeEntry( hknpBodyId id, Entry& entry );

	public:

		hknpWorld* m_world;
		hknpSolverData* m_solverData;			// temporary variable
		hkBlockStream<hkCommand>::Writer* m_commandWriter;	// temporary variable
		hknpSimulationThreadContext* m_simulationThreadContext;	// temporary variable

		EntryIdx m_firstFreeElement;		// points to the head of a linked list of free elements

		hkArray<Entry> m_entryPool;
		hkArray<EntryIdx> m_bodyToEntryMap;	// body0 means all bodies
};

#include <Physics/Physics/Dynamics/World/Events/hknpEventDispatcher.inl>


#endif	// HKNP_EVENT_DISPATCHER_H

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
