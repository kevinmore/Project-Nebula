/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Collide/TriggerVolume/hkpTriggerVolume.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Physics2012/Dynamics/Collide/ContactListener/Util/hkpCollisionCallbackUtil.h>
#include <Physics2012/Dynamics/Collide/hkpResponseModifier.h>

#include <Physics2012/Collide/Shape/hkpShapeContainer.h>
#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>

hkpTriggerVolume::hkpTriggerVolume( hkpRigidBody* triggerBody )
: m_triggerBody( triggerBody )
, m_sequenceNumber( 0 )
, m_isProcessingBodyOverlaps(false)
{
	HK_ASSERT2( 0x8a334fc1, !triggerBody->hasProperty( HK_PROPERTY_TRIGGER_VOLUME ), "This body is already the triggerBody of a triggerVolume." );
	triggerBody->addContactListener( this );
	triggerBody->addEntityListener( this );
	triggerBody->addProperty( HK_PROPERTY_TRIGGER_VOLUME, this );
	triggerBody->addProperty( HK_PROPERTY_DEBUG_DISPLAY_COLOR, HK_TRIGGER_VOLUME_DEBUG_COLOR );
	hkpWorld *const world = triggerBody->getWorld();
	if ( world )
	{
		triggerBodyEnteredWorld( world );
	}
	addReference();
}

hkpTriggerVolume::hkpTriggerVolume(hkFinishLoadedObjectFlag f) :	hkReferencedObject(f), m_overlappingBodies(f),
																	m_eventQueue(f), m_newOverlappingBodies(f)
{
}


void hkpTriggerVolume::entityAddedCallback( hkpEntity* entity )
{
	HK_ASSERT2( 0x8a334fc1, entity == m_triggerBody, "This object should only be an entity listener for the triggerBody." );
	hkpWorld *const world = entity->getWorld();
	triggerBodyEnteredWorld( world );
}

#ifdef HK_DEBUG
// Recursively search for the presence of boxes in the shape.
static hkBool findBoxesRecursively( const hkpShape* shape )
{
	if ( shape->getType() == hkcdShapeType::BOX )
	{
		return true;
	}

	const hkpShapeContainer *const container = shape->getContainer();
	if ( container )
	{
		hkpShapeBuffer buffer;
		for ( hkpShapeKey key = container->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = container->getNextKey( key ) )
		{
			const hkpShape *const childShape = container->getChildShape( key, buffer );
			if ( findBoxesRecursively( childShape ) )
			{
				return true;
			}
		}
	}
	return false;
}
#endif

void hkpTriggerVolume::triggerBodyEnteredWorld( hkpWorld* world )
{
	world->addWorldPostSimulationListener( this );
	hkpCollisionCallbackUtil::requireCollisionCallbackUtil( world );

#ifdef HK_DEBUG
	// Since the box-box agent doesn't handle deep-penetration well, we try to identify
	// those cases when it is registered and the triggerBody's shape includes a box.

	const hkpCollisionDispatcher *const collisionDispatcher = world->getCollisionDispatcher();
	// We determine whether box-box has a special agent by comparing it with box-convexVerticesShape.
	const hkBool boxBoxAgentRegistered = ( collisionDispatcher->getAgent3Type( hkcdShapeType::BOX, hkcdShapeType::BOX, false ) != collisionDispatcher->getAgent3Type( hkcdShapeType::BOX, hkcdShapeType::CONVEX_VERTICES, false ) );
	if ( boxBoxAgentRegistered )
	{
		// Notes:
		// * If you're running continuous physics, then the problem only applies to low quality collisions,
		//   since high quality box-box collisions will be handled by the predictive gsk agent.
		// * You can use hkpShapeGenerator::createConvexBox to generate a box-like convexVerticesShape.
		HK_WARN_ON_DEBUG_IF( findBoxesRecursively( m_triggerBody->getCollidable()->getShape() ), 0x331aad2e, \
			"The hkpBoxBoxAgent agent does not handle deep collisions well, so avoid using it with the hkpTriggerVolume. " \
			"We recommend either unregistering the box-box agent, or replacing any boxes in the hkpTriggerVolume by " \
			"equivalent hkpConvexVerticesShapes." );
	}
#endif
}

void hkpTriggerVolume::triggerBodyLeftWorld()
{
	hkpWorld *const world = m_triggerBody->getWorld();
	HK_ASSERT2( 0x8a3367e1, world, "Don't call this function unless the body is still in the world." );
	world->removeWorldPostSimulationListener( this );
	hkpCollisionCallbackUtil::releaseCollisionCallbackUtil( world );

	// Fire left events for any overlapping bodies.
	const int numOverlappingBodies = m_overlappingBodies.getSize();
	for ( int i = 0; i < numOverlappingBodies; ++i )
	{
		m_overlappingBodies[i]->removeEntityListener(this) ;
		triggerEventCallback( m_overlappingBodies[i], TRIGGER_BODY_LEFT_EVENT );
		m_overlappingBodies[i]->removeReference();
	}

	// Clear the queues.
	abandonEventQueue();
	m_overlappingBodies.clear();
}

void hkpTriggerVolume::entityShapeSetCallback( hkpEntity* entity)
{
	// Only call 'hkpTriggerVolume::entityRemovedCallback()' if 'hkpRigidBody::setShape()'
	// (and not 'hkpRigidBody::updateShape') was called. Note that this will effectively 
	// remove the entity from this TriggerVolume's overlapping bodies, reprocessing it in the next step
	// (In 'hkpTriggerVolume::postSimulationCallback(..);') with an ADDED_OP event which will be emitted
	// in the next physics world step (From 'hkpTriggerVolume::collisionAddedCallback()').
	int removedEventIndex = m_eventQueue.getSize() - 1;
	if ( !m_eventQueue.isEmpty() && m_eventQueue[removedEventIndex].m_operation == REMOVED_OP)
	{
		entityRemovedCallback(entity);
	}
}

void hkpTriggerVolume::entityRemovedCallback( hkpEntity* entity )
{
	// The TriggerVolume listens to its own removal or the removal of overlapping bodies
	if (entity == m_triggerBody)
	{
		triggerBodyLeftWorld();
	}
	else 
	{
		// An entity we were listening to (i.e. an overlapping rigid body) has been 
		// removed from the world, so we need to remove it now, because the user doesn't 
		// want to wait until the next simulation step. (HVK-6155)
		
		// Find the CollisionRemoved event (which should be the last event added to the queue) in the event queue.
		int eventIndex = m_eventQueue.getSize() - 1;		

		HK_ASSERT2(	0x492dfae0, eventIndex < m_eventQueue.getSize() && m_eventQueue[eventIndex].m_operation == REMOVED_OP,
					"Trying to execute instant removal for a body that was not scheduled for removal");

		// There are two cases to handle here:
		// 1) This callback is called while the 'hkpTriggerVolume::triggerEventCallback( body, ENTERED_EVENT );'
		//    gets triggered from 'postSimulationCallback()'/'updateOverlaps()' (ie.: Right after entity has
		//    been added to m_newOverlappingBodies).
		// 2) This callback is called after 'postSimulationCallback()'/'updateOverlaps()' has registered this
		//    TriggerVolume as an entity listener of 'entity' and m_overlappingBodies contains 'entity'.

		hkpRigidBody* body = static_cast<hkpRigidBody *> ( entity );

		// If this callback is called from 'hkpTriggerVolume::triggerEventCallback( body, ENTERED_EVENT );'.
		if ( m_isProcessingBodyOverlaps ) 
		{
			// Look for entity in the m_newOverlappingBodies array which stores all the overlapping bodies which
			// were selected in the 'postSimulationCallback()'/'updateOverlaps()' calling function.
			hkInt32 bodyIndex = hkAlgorithm::binarySearch(	body, m_newOverlappingBodies.begin(),
															m_newOverlappingBodies.getSize(), bodyOrderForArrays);

			// The user has requested to remove an entity that was already retrieved from
			// 'postSimulationCallback()'/'updateOverlaps()'.
			if (bodyIndex >= 0) 
			{
				// m_newOverlappingBodies, like m_overlappingBodies, needs to be kept sorted.
				m_newOverlappingBodies.removeAtAndCopy(bodyIndex);
			}
			// Else, the user has requested to remove an entity that is already in m_overlappingBodies
			// but wasn't yet selected to be put in m_newOverlappingBodies 
			// (see 'postSimulationCallback()'/'updateOverlaps()')
			else 
			{
				// Remove entity from m_overlappingBodies which is always assumed sorted.
				hkInt32 bodyIndexFound = hkAlgorithm::binarySearch(	body, m_overlappingBodies.begin(),
																m_overlappingBodies.getSize(), bodyOrderForArrays);
				HK_ASSERT2(0x492dfae1, bodyIndexFound != -1, "Trying to remove a body that was not overlapping");

				m_overlappingBodies.removeAtAndCopy(bodyIndexFound);

				// Compute the range [bodyEventStartIndex; bodyEventEndIndex] of events associated
				// to entity in m_eventQueue. 
				// Note that if (m_isProcessingBodyOverlaps) -> m_eventQueue is sorted; ( see postSimulationCallback() )
				hkInt32 bodyEventStartIndex;
				hkInt32 bodyEventEndIndex = eventIndex - 1;
				while ((bodyEventEndIndex  >= 0) && (m_eventQueue[bodyEventEndIndex].m_body != entity))
				{
					--bodyEventEndIndex;
				}
				if (bodyEventEndIndex >= 0) // If we at least found one event to remove
				{
					bodyEventStartIndex = bodyEventEndIndex;
					while ((bodyEventStartIndex  > 0) && (m_eventQueue[bodyEventStartIndex - 1].m_body == entity))
					{
						--bodyEventStartIndex;
					}
					int numBodyEvents = (bodyEventEndIndex - bodyEventStartIndex) + 1;

					// Remove references of entity attached in the event queue.
					for (int i = 0; i < numBodyEvents; i++) 
					{
						entity->removeReference();
					}
					// The event queue should stay sorted as assumed from 'postSimulationCallback()'/'updateOverlaps()'.
					m_eventQueue.removeAtAndCopy(bodyEventStartIndex, numBodyEvents);
				}
			}

			// Remove reference of the CollisionRemovedEvent from the event queue ( All other events attached to entity,
			// if any, should have been processed in 'postSimulationCallback()' ).
			m_eventQueue[eventIndex].m_body->removeReference();
			m_eventQueue.removeAtAndCopy(eventIndex);
		}
		else // The callback is called after 'postSimulationCallback()'/'updateOverlaps()' have registered entity in m_overlappingBodies
		{	 // and attached this TriggerVolume as one of its EntityListener.

			// Remove entity from m_overlappingBodies which needs to be kept sorted.
			hkInt32 bodyIndex = hkAlgorithm::binarySearch(	body, m_overlappingBodies.begin(),
															m_overlappingBodies.getSize(), bodyOrderForArrays);
			HK_ASSERT2(0x492dfae2, bodyIndex != -1, "Trying to remove a body that was not overlapping");
			
			m_overlappingBodies.removeAtAndCopy(bodyIndex);

			// Remove reference of all the events attached to entity from the event queue.
			for (; eventIndex >= 0; --eventIndex)
			{
				if (m_eventQueue[eventIndex].m_body == entity)
				{
					m_eventQueue[eventIndex].m_body->removeReference();
					m_eventQueue.removeAt(eventIndex); // The order will be reset in the next 'postSimulationCallback()' call.
				}
			}
		}

		// Now removing effectively the body from our list.
		triggerEventCallback(body, LEFT_EVENT);
		body->removeEntityListener(this);

		// Remove reference from the overlappingBodies array.
		body->removeReference();		
	}
}

void hkpTriggerVolume::entityDeletedCallback( hkpEntity* entity )
{
	HK_ASSERT2( 0x8a334fc1, entity == m_triggerBody, "This object should only be an entity listener for the triggerBody." );
	if ( m_triggerBody->getWorld() )
	{
		triggerBodyLeftWorld();
	}
	entity->removeEntityListener( this );
	entity->removeContactListener( this );
	// This object now deletes itself, unless a reference has been kept to it elsewhere.
	m_triggerBody = HK_NULL;
	removeReference();
}

hkpTriggerVolume::~hkpTriggerVolume()
{
	if ( m_triggerBody )
	{
		m_triggerBody->removeProperty( HK_PROPERTY_DEBUG_DISPLAY_COLOR );
		m_triggerBody->removeProperty( HK_PROPERTY_TRIGGER_VOLUME );
		m_triggerBody->removeEntityListener( this );
		m_triggerBody->removeContactListener( this );
	}
	
	// Removing the listening of overlapping entities
	for( hkpRigidBody** body = m_overlappingBodies.begin() ; body < m_overlappingBodies.end() ; ++body )
	{
		(*body)->removeEntityListener(this) ;
	}
	
	// Remove any additional references held to bodies.
	hkReferencedObject::removeReferences( m_overlappingBodies.begin(), m_overlappingBodies.getSize() );
	abandonEventQueue();
}

void hkpTriggerVolume::contactPointCallback( const hkpContactPointEvent& event )
{
	HK_ASSERT2( 0x8a334fc1, event.getBody( event.m_source ) == m_triggerBody, "This object should only be a contact listener for the triggerBody." );
	// In case a TOI occurs before a constraint has been created.
	event.m_contactPointProperties->m_flags |= hkContactPointMaterial::CONTACT_IS_DISABLED;

	hkpRigidBody* otherBody = event.getBody( 1 - event.m_source );

	if ( event.m_type == hkpContactPointEvent::TYPE_TOI )
	{
		// Because this is not a TYPE_MANIFOLD callback, we don't need to lock access to the array.
		addEvent( otherBody, TOI_OP );
	}
	else if ( event.m_type == hkpContactPointEvent::TYPE_EXPAND_MANIFOLD )
	{
		// Because this is not a TYPE_MANIFOLD callback, we don't need to lock access to the array.
		addEvent( otherBody, CONTACT_OP );
	}
}

void hkpTriggerVolume::collisionAddedCallback( const hkpCollisionEvent& event )
{
	HK_ASSERT2( 0x8a334fc1, event.getBody( event.m_source ) == m_triggerBody, "This object should only be a contact listener for the triggerBody." );
	
	// Disable the contact constraint. This can cause a "constraint is already disabled" warning
	// which is safe to ignore.
	{
		hkpConstraintInstance *const constraint = event.m_contactMgr->getConstraintInstance();
		hkpResponseModifier::disableConstraint( constraint, *constraint->getOwner() );
	}

	hkpRigidBody* otherBody = event.getBody( 1 - event.m_source );

	addEvent( otherBody, ADDED_OP );
}

void hkpTriggerVolume::collisionRemovedCallback( const hkpCollisionEvent& event )
{
	HK_ASSERT2( 0x8a334fc1, event.getBody( event.m_source ) == m_triggerBody, "This object should only be a contact listener for the triggerBody." );
	addEvent( event.getBody( 1 - event.m_source ), REMOVED_OP );
}

hkBool HK_CALL hkpTriggerVolume::bodyOrderForQueues( const hkpTriggerVolume::EventInfo& infoA, const hkpTriggerVolume::EventInfo& infoB )
{
	return infoA.m_sortValue < infoB.m_sortValue;
}

hkBool HK_CALL hkpTriggerVolume::bodyOrderForArrays( const hkpRigidBody* bodyA, const hkpRigidBody* bodyB )
{
	return bodyA->getUid() < bodyB->getUid();
}

void hkpTriggerVolume::postSimulationCallback( hkpWorld* world )
{
	m_isProcessingBodyOverlaps = true; // Flag that we are re-processing body overlaps.
	// Use m_newOverlappingBodies as a temp array for the m_overlappingBodies array.
	m_newOverlappingBodies.reserve( m_overlappingBodies.getSize() );

	// Sort the event queue in deterministic order.
	typedef hkBool (* BodyOrderType)( const EventInfo&, const EventInfo& );
	hkAlgorithm::quickSort<EventInfo, BodyOrderType>( m_eventQueue.begin(), m_eventQueue.getSize(), bodyOrderForQueues );

	// We use a state machine to process the events.
	enum States
	{
		ERROR_STATE,					// Processing error.
		START_IN_STATE,					// The body was inside the volume at the beginning of the frame.
		START_OUT_STATE,				// The body was outside the volume at the beginning of the frame.
		ADDED_STATE,					// The body and volume may have come into contact this frame (unless removed)
		ADDED_CONFIRMED_STATE,			// the body and the volume did come into contact this frame
		ADDED_CONFIRMED_REMOVED_STATE,	// ...but then separated.
		REMOVED_STATE,					// The body and the volume separated this frame.
		//
		NUM_STATES
	};

	// The state machine's transition function.
	static const States transitions[NUM_STATES][4] = 
	{
		// ADDED, REMOVED, CONTACT, TOI
		{ ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE },											// ERROR  
		{ ERROR_STATE, REMOVED_STATE, START_IN_STATE, START_IN_STATE },									// START_IN
		{ ADDED_STATE, ERROR_STATE, ERROR_STATE, ADDED_CONFIRMED_REMOVED_STATE },						// START_OUT
		{ ERROR_STATE, START_OUT_STATE, ADDED_CONFIRMED_STATE, ADDED_CONFIRMED_STATE },					// ADDED
		{ ERROR_STATE, ADDED_CONFIRMED_REMOVED_STATE, ADDED_CONFIRMED_STATE, ADDED_CONFIRMED_STATE },	// ADDED_CONFIRMED
		{ ADDED_CONFIRMED_STATE, ERROR_STATE, ERROR_STATE, ADDED_CONFIRMED_REMOVED_STATE },				// ADDED_CONFIRMED_REMOVED
		{ START_IN_STATE, ERROR_STATE, ERROR_STATE, REMOVED_STATE }										// REMOVED
	};

	// We traverse the overlappingBodies array in the same order as the event queue.
	hkpRigidBody** overlapCurrent = m_overlappingBodies.begin();
	hkpRigidBody** overlapsEnd = m_overlappingBodies.end(); 

	// Loop over all events.
	const EventInfo* event = m_eventQueue.begin(); 
	while( event < m_eventQueue.end() )
	{
		hkpRigidBody* body = event->m_body;

		// Use the m_overlappingBodies array to determine the starting state of the body.
		States state;
		{
			// Skip overlaps for which there is no event.
			while ( ( overlapCurrent < overlapsEnd ) && bodyOrderForArrays( *overlapCurrent, body ) )
			{
				m_newOverlappingBodies.pushBack( *overlapCurrent );
				++overlapCurrent;
			}
			if ( ( overlapCurrent == overlapsEnd ) || bodyOrderForArrays( body, *overlapCurrent ) )
			{
				state = START_OUT_STATE;
				// We need an additional reference here so the body lives at least until the callback is fired.
				body->addReference();
			}
			else
			{
				state = START_IN_STATE;
				++overlapCurrent;
			}
		}

		// Loop over the block containing all events concerning a single body
		do 
		{
			state = transitions[state][event->m_operation];
			// Bring the reference count due to the trigger volume down to exactly 1.
			body->removeReference();
			++event;
		} while ( ( event < m_eventQueue.end() ) && ( event->m_body == body ) );

		// Use the resulting state to issue events for the body.
		switch ( state )
		{
			case START_IN_STATE:
				m_newOverlappingBodies.pushBack( body );
				break;

			case START_OUT_STATE:
				body->removeReference();
				break;

			case ADDED_STATE:
			case ADDED_CONFIRMED_STATE:
				m_newOverlappingBodies.pushBack( body );
				body->addEntityListener(this);
				triggerEventCallback( body, ENTERED_EVENT );
				overlapsEnd = m_overlappingBodies.end(); 
				break;

			case ADDED_CONFIRMED_REMOVED_STATE:
				triggerEventCallback( body, ENTERED_AND_LEFT_EVENT );
				overlapsEnd = m_overlappingBodies.end(); 
				body->removeReference();
				break;

			case REMOVED_STATE:
				body->removeEntityListener(this);
				triggerEventCallback( body, LEFT_EVENT );
				overlapsEnd = m_overlappingBodies.end(); 
				body->removeReference();
				break;

			case ERROR_STATE:
			default:
				HK_ASSERT2( 0x341ef172, false, "State-machine error while processing trigger volume event queue." );

		}
	}
	// Copy any remaining overlaps in the m_overlappingBodies array.
	while ( overlapCurrent < overlapsEnd )
	{
		m_newOverlappingBodies.pushBack( *overlapCurrent );
		++overlapCurrent;
	}
	// Reset the event queue.
	m_eventQueue.clear();
	// Reset the sequence number.
	m_sequenceNumber = 0;

	// Use the newOverlapppingBodies array instead of m_overlappingBodies.
	m_overlappingBodies.swap( m_newOverlappingBodies );

	m_newOverlappingBodies.clear();
	m_isProcessingBodyOverlaps = false; // Flag that we are done re-processing body overlaps.
}

void hkpTriggerVolume::updateOverlaps()
{
	HK_ASSERT2( 0x8a334fd9, m_triggerBody->getWorld(), "Update overlaps should only be called when the trigger body is in the world." );
	m_isProcessingBodyOverlaps = true; // Flag that we are re-processing body overlaps.

	abandonEventQueue();

	// Obtain the current overlaps in the world.
	hkArray<hkpRigidBody*> overlapsInWorld;
	{
		// We sort for determinism below.
		hkArray<hkpLinkedCollidable::CollisionEntry>& collisionEntries = m_triggerBody->getLinkedCollidable()->getCollisionEntriesNonDeterministic();

		const int numCollisionEntries = collisionEntries.getSize();
		for ( int i = 0; i < numCollisionEntries; ++i )
		{
			const hkpLinkedCollidable::CollisionEntry& entry = collisionEntries[i];
			const hkpContactMgr *const mgr = entry.m_agentEntry->m_contactMgr;
			if ( mgr->m_type == hkpContactMgr::TYPE_SIMPLE_CONSTRAINT_CONTACT_MGR )
			{
				// Are their some contact points between the bodies.
				if ( ( static_cast<const hkpSimpleConstraintContactMgr*>( mgr ) )->m_contactConstraintData.getNumContactPoints() )
				{
					overlapsInWorld.pushBack( hkpGetRigidBody( entry.m_partner ) );
				}
			}
		}
		// Sort the overlaps for determinism.
		typedef hkBool (* BodyOrderType)( const hkpRigidBody*, const hkpRigidBody* );
		hkAlgorithm::quickSort<hkpRigidBody*, BodyOrderType>( overlapsInWorld.begin(), overlapsInWorld.getSize(), bodyOrderForArrays );
	}
	
	// Use m_newOverlappingBodies as a temp array for the overlapsInWorld array which will keep track of bodies which
	// were removed from the user callbacks and which need to be processed immediately.
	m_newOverlappingBodies.reserve( m_overlappingBodies.getSize() );

	// Traverse the arrays looking for differences.
	{
		hkpRigidBody** inWorld = overlapsInWorld.begin();
		hkpRigidBody** inVolume = m_overlappingBodies.begin();
		
		hkpRigidBody** endWorld = overlapsInWorld.end();
		hkpRigidBody** endVolume = m_overlappingBodies.end();


		while ( ( inWorld < endWorld ) && ( inVolume < endVolume ) )
		{
			if ( bodyOrderForArrays( *inWorld, *inVolume ) )
			{
				( *inWorld )->addReference();
				(*inWorld)->addEntityListener(this);
				m_newOverlappingBodies.pushBack( *inWorld );
				triggerEventCallback( *inWorld, ENTERED_EVENT );
				endVolume = m_overlappingBodies.end();
				++inWorld;
			}
			else if ( bodyOrderForArrays( *inVolume, *inWorld ) )
			{
				(*inVolume)->removeEntityListener(this);
				triggerEventCallback( *inVolume, LEFT_EVENT );
				endVolume = m_overlappingBodies.end();
				( *inVolume )->removeReference();
				++inVolume;
			}
			else
			{
				m_newOverlappingBodies.pushBack( *inWorld );
				++inWorld;
				++inVolume;
			}
		}
		while ( inWorld < endWorld )
		{
			( *inWorld )->addReference();
			(*inWorld)->addEntityListener(this);
			m_newOverlappingBodies.pushBack( *inWorld );
			triggerEventCallback( *inWorld, ENTERED_EVENT );
			endVolume = m_overlappingBodies.end();
			++inWorld;
		}
		while ( inVolume < endVolume )
		{
			(*inVolume)->removeEntityListener(this);
			triggerEventCallback( *inVolume, LEFT_EVENT );
			endVolume = m_overlappingBodies.end();
			( *inVolume )->removeReference();
			++inVolume;
		}
	}

	// Keep the overlapsInWorld array.
	m_overlappingBodies.swap( m_newOverlappingBodies );
	m_newOverlappingBodies.clear();

	m_isProcessingBodyOverlaps = false; // Flag that we are done re-processing body overlaps.
}

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
