/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Dynamics/Body/hknpBodyManager.h>
#include <Physics/Physics/Dynamics/Modifier/hknpModifier.h>
#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>

#include <Physics/Physics/Dynamics/World/Events/hknpEventDispatcher.h>

// Force explicit template instantiation
#include <Common/Base/Container/PointerMap/hkMap.cxx>
template class hkMapBase< hknpPropertyKey, hknpBodyManager::PropertyBuffer* >;

namespace hknpBodyManagerImpl
{
	struct Pointers
	{
		hknpBodyId m_nextFreeBodyId;
	};
}

HK_COMPILE_TIME_ASSERT( sizeof(hknpBody) >= sizeof(hknpBodyManagerImpl::Pointers) );

#define REINTERPRET_AS_POINTERS(body) (*(reinterpret_cast<hknpBodyManagerImpl::Pointers*>(&body)))
#define REINTERPRET_CONST_AS_POINTERS(body) (*(reinterpret_cast<hknpBodyManagerImpl::Pointers*>(const_cast<hknpBody*>(&body))))


hknpBodyManager::hknpBodyManager( hknpWorld* world, hknpBody* userBodyBuffer, hkUint32 capacity )
:	m_world(world),
	m_motionManager(HK_NULL),
	m_firstFreeBodyId(0),
	m_numAllocatedBodies(0),
	m_numMarkedBodies(0),
	m_peakBodyIndex(0)
{
#if !defined(HK_PLATFORM_SPU)

	// allocate the body buffer
	relocateBodyBuffer( userBodyBuffer, capacity );

	// create a dummy static body at index 0
	// note that this does not count as a usable static body
	{
		HK_COMPILE_TIME_ASSERT( hknpBodyId::WORLD == 0 );
		hknpBody& fixedBody = m_bodies[0];

		hkString::memClear16( &fixedBody, hkSizeOf(hknpBody)/16 );

		fixedBody.m_flags = hknpBody::IS_STATIC;
		fixedBody.m_motionId = hknpMotionId::STATIC;
		fixedBody.m_materialId = hknpMaterialId::DEFAULT;
		fixedBody.setTransform( hkTransform::getIdentity() );

		m_previousAabbs[0].setEmpty();
		m_bodyIdToCellIndexMap[0] = HKNP_INVALID_CELL_IDX;
		m_scheduledBodyChangeIndices[0] = INVALID_BODY_INDEX;

		m_firstFreeBodyId = hknpBodyId(1);
		m_peakBodyIndex = 0;
		m_numAllocatedBodies = 1;
	}

#endif	// !HK_PLATFORM_SPU
}

hknpBodyManager::~hknpBodyManager()
{
#if !defined(HK_PLATFORM_SPU)

	// deallocate the body buffer, if owned
	if( !m_bodyBufferIsUserOwned )
	{
		hkAlignedDeallocate<hknpBody>( m_bodies.begin() );
	}

	// deallocate any property buffers
	clearAllPropertiesFromAllBodies();

#endif	// !HK_PLATFORM_SPU
}

hkBool hknpBodyManager::relocateBodyBuffer( hknpBody* buffer, hkUint32 capacity )
{
	deleteMarkedBodies();

	const hkUint32 oldCapacity = m_bodies.getCapacity();	// if zero, the buffer is being allocated for the first time

	// Check if it is possible
	{
		if( capacity == 0 )
		{
			HK_WARN( 0x5c265501, "Body buffer capacity must be > 0. Relocation failed." );
			return false;
		}
		for( hkUint32 i=capacity; i<oldCapacity; ++i )
		{
			if( m_bodies[i].isValid() )
			{
				HK_WARN( 0x5c265502, "Requested body buffer capacity cannot fit existing bodies. Relocation failed." );
				return false;
			}
		}
	}

	//
	// Relocate the buffer
	//

	{
		const hknpBody* bufferIn = buffer;
		hknpBody* oldBuffer = m_bodies.begin();
		HK_ASSERT( 0x26a4e886, (oldCapacity == 0) ^ (oldBuffer != HK_NULL) );

		// Allocate if needed
		if( !buffer )
		{
			buffer = hkAlignedAllocate<hknpBody>( 128, capacity, HK_MEMORY_CLASS_PHYSICS );
		}

		HK_ASSERT2( 0x26a4e887,  ( (hkUlong)buffer & 0xf ) == 0, "Body buffer must be 16 byte aligned" );

		if( oldBuffer )
		{
			// Copy/move the body buffer to the new address
			const hkUint32 numToCopy = hkMath::min2( oldCapacity, capacity );
			if( !m_bodyBufferIsUserOwned )
			{
				// Was owned by manager.
				hkString::memCpy( buffer, oldBuffer, numToCopy * sizeof(hknpBody) );
				hkAlignedDeallocate<hknpBody>( oldBuffer );
				oldBuffer = HK_NULL;
			}
			else if( buffer != oldBuffer )
			{
				// Was owned by user. Possibly overlapping.
				hkString::memMove( buffer, oldBuffer, numToCopy * sizeof(hknpBody) );
			}
		}

		m_bodies.setDataUserFree( buffer, capacity, capacity );
		m_bodyBufferIsUserOwned = ( bufferIn != HK_NULL );
	}

	//
	// Resize other arrays
	
	//

	m_previousAabbs.setSize( capacity );
	m_bodyIdToCellIndexMap.setSize( capacity );
	m_scheduledBodyChangeIndices.setSize( capacity );

	// This needs 4 extra elements reserved. See getActiveBodies().
	m_activeBodyIds.reserve( capacity + 4 );

	//
	// Fix up everything
	//

	if( capacity > oldCapacity )
	{
		// Initialize the new bodies (as invalid)
		for( hkUint32 i=oldCapacity; i<capacity; ++i )
		{
			hknpBody& body = m_bodies[i];
			REINTERPRET_AS_POINTERS(body).m_nextFreeBodyId = hknpBodyId(i + 1);
			body.m_flags = 0;														// marked as invalid
			body.m_broadPhaseId = hknpBroadPhaseId(HKNP_INVALID_BROAD_PHASE_ID);	// marked as !addedToWorld()

			m_previousAabbs[i].setEmpty();
			m_bodyIdToCellIndexMap[i] = HKNP_INVALID_CELL_IDX;
			m_scheduledBodyChangeIndices[i] = INVALID_BODY_INDEX;
		}
		REINTERPRET_AS_POINTERS( m_bodies[capacity-1] ).m_nextFreeBodyId = hknpBodyId(0);

		// Link the existing free list (if any)
		if( oldCapacity )
		{
			if( m_firstFreeBodyId.value() == 0 )
			{
				// Existing buffer was full
				m_firstFreeBodyId = hknpBodyId(oldCapacity);
			}
			else
			{
				// Find the end of the free list then link it
				hknpBodyId current = m_firstFreeBodyId;
				while(1)
				{
					hknpBodyId& next = REINTERPRET_AS_POINTERS(m_bodies[current.value()]).m_nextFreeBodyId;
					if( next.value() == 0 )
					{
						next = hknpBodyId(oldCapacity);
						break;
					}
					current = next;
				}
			}
		}
	}
	else
	{
		// Need to make sure the free list is terminated and the peak index is within bounds.
		rebuildFreeList();
	}

	checkConsistency();

	// Success! Fire callbacks and return.
	if ( m_world )
	{
		m_world->m_signals.m_bodyBufferChanged.fire(m_world, this);
	}
	return true;
}

hknpBodyId hknpBodyManager::allocateBody()
{
	if( m_firstFreeBodyId.value() )
	{
		hknpBodyId id = m_firstFreeBodyId;

#ifdef HK_DEBUG
		// Sanity check to ensure the body is not already subscribed to any events.
		if( m_world && (hkUint32)m_world->getEventDispatcher()->m_bodyToEntryMap.getSize() > id.value() )
		{
			HK_ASSERT2( 0x34985ABF,
				m_world->getEventDispatcher()->m_bodyToEntryMap[id.value()] == hknpEventDispatcher::INVALID_ENTRY,
				"An invalid body was subscribed to events" );
		}
#endif

		m_firstFreeBodyId = REINTERPRET_AS_POINTERS(m_bodies[id.value()]).m_nextFreeBodyId;
		m_bodyIdToCellIndexMap[ id.value() ] = HKNP_INVALID_CELL_IDX;
		m_peakBodyIndex = hkMath::max2( m_peakBodyIndex, id.value() );
		m_numAllocatedBodies++;
		return id;
	}
	else
	{
		HK_ERROR( 0x506609a5, "Body buffer is full." );
		return hknpBodyId::invalid();
	}
}


hkResult hknpBodyManager::allocateBody( hknpBodyId bodyId )
{
	if ( !bodyId.isValid() || !(bodyId.value() < (hkUint32) m_bodies.getSize()) || m_bodies[bodyId.value()].isValid() )
	{
		// Caller requested an invalid ID, or the ID of an already allocated body.
		return HK_FAILURE;
	}

	hknpBodyId freeId = m_firstFreeBodyId;
	hknpBodyId* idToUpdate = &m_firstFreeBodyId;

	// Go through the linked list to find the insertion point.
	while (freeId.value() != 0 && freeId != bodyId )
	{
		idToUpdate = &(REINTERPRET_AS_POINTERS(m_bodies[freeId.value()]).m_nextFreeBodyId);
		freeId = REINTERPRET_AS_POINTERS(m_bodies[freeId.value()]).m_nextFreeBodyId;
	}
	if ( freeId.value() == 0 )
	{
		// If we have reached an invalid ID, that means we ran through the free list without finding the requested
		// bodyId, so it means it is not available yet.
		return HK_FAILURE;
	}

	// Update the linked list.
	*idToUpdate = REINTERPRET_AS_POINTERS(m_bodies[bodyId.value()]).m_nextFreeBodyId;

	m_peakBodyIndex = hkMath::max2( m_peakBodyIndex, bodyId.value() );
	m_numAllocatedBodies++;

	m_bodyIdToCellIndexMap[ bodyId.value() ] = HKNP_INVALID_CELL_IDX;
	return HK_SUCCESS;
}

void hknpBodyManager::initializeStaticBody( hknpBody* HK_RESTRICT body, hknpBodyId bodyId, const hknpBodyCinfo& cInfo )
{
	body->initialize( bodyId, cInfo );
	body->m_qualityId = cInfo.m_qualityId.isValid() ? cInfo.m_qualityId : hknpBodyQualityId::STATIC;
	body->m_motionId = hknpMotionId::STATIC;
	body->m_flags.orWith( hknpBody::IS_STATIC );

	setBodyName( bodyId, cInfo.m_name );
	clearAllPropertiesFromBody( bodyId );
}

void hknpBodyManager::initializeDynamicBody( hknpBody* HK_RESTRICT body, hknpBodyId bodyId, const hknpBodyCinfo& cInfo )
{
	body->initialize( bodyId, cInfo );
	body->m_qualityId = cInfo.m_qualityId.isValid() ? cInfo.m_qualityId : hknpBodyQualityId::DYNAMIC;
	body->m_flags.orWith( hknpBody::IS_DYNAMIC );

	setBodyName( bodyId, cInfo.m_name );
	clearAllPropertiesFromBody( bodyId );
}

void hknpBodyManager::getBodyCinfo( hknpBodyId bodyId, hknpBodyCinfo& cinfoOut ) const
{
	const hknpBody& body = m_bodies[ bodyId.value() ];
	HK_ASSERT( 0xf02d5ea9, body.isValid() );

	cinfoOut.m_shape						= body.m_shape;
	cinfoOut.m_qualityId					= body.m_qualityId;
	cinfoOut.m_materialId					= body.m_materialId;
	cinfoOut.m_motionId						= body.m_motionId;
	cinfoOut.m_collisionFilterInfo			= body.m_collisionFilterInfo;
	cinfoOut.m_flags						= body.m_flags.get( ~hkUint32(hknpBody::INTERNAL_FLAGS_MASK) );
	cinfoOut.m_spuFlags						= body.m_spuFlags.get();
	cinfoOut.m_collisionLookAheadDistance	= body.getCollisionLookAheadDistance();
	cinfoOut.m_position						= body.getTransform().getTranslation();
	cinfoOut.m_orientation					. set( body.getTransform().getRotation() );
	cinfoOut.m_name							= getBodyName( bodyId );
	cinfoOut.m_localFrame					= HK_NULL;
}

void hknpBodyManager::deleteMarkedBodies()
{
	for( int i=0, ei=m_scheduledBodyChanges.getSize(); i<ei; ++i )
	{
		if( m_scheduledBodyChanges[i].m_scheduledBodyFlags.get() & hknpBodyManager::TO_BE_DELETED )
		{
			const hknpBodyId id = m_scheduledBodyChanges[i].m_bodyId;
			m_bodies[id.value()].m_motionId = hknpMotionId::invalid();	

			REINTERPRET_AS_POINTERS(m_bodies[id.value()]).m_nextFreeBodyId = m_firstFreeBodyId;
			m_firstFreeBodyId = id;
			m_numAllocatedBodies--;

			m_scheduledBodyChanges[i].m_scheduledBodyFlags.clear( hknpBodyManager::TO_BE_DELETED );

			// We need to clear all event signals related to this body here.
			if (m_world)
			{
				m_world->getEventDispatcher()->unsubscribeAllSignals(id);
			}
		}
	}
	m_numMarkedBodies = 0;
}

void hknpBodyManager::prefetchActiveBodies()
{
	hknpBody* bodies = m_bodies.begin();
	for( int i = 0; i < m_activeBodyIds.getSize(); i++ )
	{
		hkMath::prefetch128( bodies + m_activeBodyIds[i].value() );
	}
}

const hkArray<hknpBodyId>& hknpBodyManager::getActiveBodies()
{
	hknpBodyId* padding = m_activeBodyIds.expandByUnchecked(4);
	padding[0] = hknpBodyId(0);
	padding[1] = hknpBodyId(0);
	padding[2] = hknpBodyId(0);
	padding[3] = hknpBodyId(0);
	m_activeBodyIds.popBack(4);
	return m_activeBodyIds;
}

void hknpBodyManager::rebuildFreeList()
{
	m_firstFreeBodyId = hknpBodyId(0);
	m_peakBodyIndex = 0;
	hkUint32 lastFreeBodyIndex = 0;
	for( hkUint32 i = 1, capacity = m_bodies.getCapacity(); i < capacity; i++ )
	{
		if( m_bodies[i].isValid() )
		{
			m_peakBodyIndex = i;
		}
		else
		{
			if( m_firstFreeBodyId.value() > 0 )
			{
				REINTERPRET_AS_POINTERS(m_bodies[lastFreeBodyIndex]).m_nextFreeBodyId = hknpBodyId(i);
			}
			else
			{
				m_firstFreeBodyId = hknpBodyId(i);
			}
			lastFreeBodyIndex = i;
		}
	}
	if( lastFreeBodyIndex > 0 )
	{
		REINTERPRET_AS_POINTERS(m_bodies[lastFreeBodyIndex]).m_nextFreeBodyId = hknpBodyId(0);
	}
}

void hknpBodyManager::rebuildActiveBodyArray()
{
	m_activeBodyIds.clear();
	for( int bodyId = 0; bodyId < m_bodies.getSize(); bodyId++ )
	{
		hknpBody& body = m_bodies[bodyId];
		if( body.isAddedToWorld() && body.isActive() )
		{
			body.m_indexIntoActiveListOrDeactivatedIslandId = hknpBodyId::Type( m_activeBodyIds.getSize() );
			m_activeBodyIds.pushBackUnchecked( hknpBodyId(bodyId) );
		}
	}
	checkConsistency();
}

void hknpBodyManager::rebuildBodyIdToCellIndexMap()
{
	for( int bodyId = 0; bodyId < m_bodies.getSize(); bodyId++ )
	{
		hknpBody& body = m_bodies[bodyId];
		if( !body.isAddedToWorld() )
		{
			continue;
		}

		hknpMotion& motion = m_motionManager->m_motions[ body.m_motionId.value() ];
		if( body.isActive() )
		{
			HK_ASSERT( 0xf0dfede4, body.isValid() );
			m_bodyIdToCellIndexMap[bodyId] = motion.m_cellIndex;
		}
	}
	checkConsistency();
}

void hknpBodyManager::updatePreviousAabbsOfActiveBodies()
{
	hknpBody* bodies = m_bodies.begin();
	int numActiveBodies = m_activeBodyIds.getSize();
	hkAabb16* HK_RESTRICT prevAabbs = m_previousAabbs.begin();
	HK_ASSERT( 0xf04fde45, m_previousAabbs.getSize() == m_bodies.getSize() );

	{
		int i=0;
		for ( ; i < numActiveBodies-8; i++ )
		{
			int activeBodyId = m_activeBodyIds[i].value();
			int prefetchActiveBodyId = m_activeBodyIds[i+8].value();
			prevAabbs[ activeBodyId ] = bodies[activeBodyId].m_aabb;
			hkMath::prefetch128( bodies + prefetchActiveBodyId );
			hkMath::prefetch128( prevAabbs + prefetchActiveBodyId );
		}
		for ( ; i < numActiveBodies; i++ )
		{
			int activeBodyId = m_activeBodyIds[i].value();
			prevAabbs[ activeBodyId ] = bodies[activeBodyId].m_aabb;
		}
	}
}

void hknpBodyManager::resetPreviousAabbsAndFlagsOfStaticBodies()
{
	hknpBody* bodies = m_bodies.begin();
	hkAabb16* HK_RESTRICT prevAabbs = m_previousAabbs.begin();
	HK_ASSERT( 0xf04fde45, m_previousAabbs.getSize() == m_bodies.getSize() );

	for( int i=0, ei=m_scheduledBodyChanges.getSize(); i<ei; ++i )
	{
		if( m_scheduledBodyChanges[i].m_scheduledBodyFlags.get() & hknpBodyManager::MOVED_STATIC )
		{
			int bodyId = m_scheduledBodyChanges[i].m_bodyId.value();
			hknpBody* body = &bodies[bodyId];
			if (body->isStatic())
			{
				prevAabbs[ bodyId ] = bodies[bodyId].m_aabb;

				body->m_flags.clear( hknpBody::TEMP_FLAGS_MASK );
				body->m_maxTimDistance = 0;
				body->m_timAngle = 0;
			}
		}
	}
}

void hknpBodyManager::addSingleBodyToActiveGroup( hknpBodyId bodyId )
{
	hknpBody& body = m_bodies[bodyId.value()];
	HK_ASSERT2( 0xf023defd, !body.isActive() && !body.isStatic(), "Body already active or static" );

	body.m_flags.orWith( hknpBody::IS_ACTIVE );
	body.m_indexIntoActiveListOrDeactivatedIslandId = hknpBodyId::Type( m_activeBodyIds.getSize() );
	m_activeBodyIds.pushBackUnchecked( bodyId );

	const hknpMotion& motion = m_motionManager->m_motions[ body.m_motionId.value()];

	m_bodyIdToCellIndexMap[ bodyId.value() ] = motion.m_cellIndex;

	// make sure we are really adding to an active body group
	HK_ASSERT( 0xf02fdefd, motion.isActive() );
}

void hknpBodyManager::addActiveBodyGroup( hknpBodyId firstId )
{
	hknpBody& firstBody = accessBody( firstId );
	hknpMotionId motionId = firstBody.m_motionId;
	hknpMotion& motion = m_motionManager->m_motions[ motionId.value() ];

	hknpBodyId attachedBodyId = firstId;
	do
	{
		hknpBody& attachedBody = accessBody( attachedBodyId );
		if (attachedBody.isAddedToWorld())
		{
			HK_ASSERT2( 0xf023defd, !attachedBody.isActive() && !attachedBody.isStatic(), "Body already active or static" );

			attachedBody.m_flags.orWith( hknpBody::IS_ACTIVE );
			attachedBody.m_indexIntoActiveListOrDeactivatedIslandId = hknpBodyId::Type(m_activeBodyIds.getSize());
			m_activeBodyIds.pushBackUnchecked( attachedBodyId );
			m_bodyIdToCellIndexMap[ attachedBodyId.value() ] = motion.m_cellIndex;
		}
		else
		{
			attachedBody.m_indexIntoActiveListOrDeactivatedIslandId = hknpBodyId::InvalidValue;
		}
		attachedBodyId = attachedBody.m_nextAttachedBodyId;
	}
	while( attachedBodyId != firstId );

	// activate motion
	m_motionManager->addActiveMotion( motion, motionId );
}

void hknpBodyManager::removeSingleBodyFromActiveList( hknpBodyId bodyId )
{
	hknpBody& body = m_bodies[bodyId.value()];
	HK_ASSERT( 0xf02d5ea1, body.isValid() && body.isActive() );

	int activeId = body.m_indexIntoActiveListOrDeactivatedIslandId;
	int lastActiveid = m_activeBodyIds.getSize()-1;
	// move last element in the list
	if ( activeId < lastActiveid )
	{
		hknpBodyId otherActiveBodyId = m_activeBodyIds[lastActiveid];
		hknpBody& otherActiveBody = m_bodies[otherActiveBodyId.value()];
		otherActiveBody.m_indexIntoActiveListOrDeactivatedIslandId = hknpBodyId::Type( activeId );
		m_activeBodyIds[ activeId ] = otherActiveBodyId;
	}
	m_activeBodyIds.popBack();

	body.m_flags.clear( hknpBody::IS_ACTIVE );
	body.m_indexIntoActiveListOrDeactivatedIslandId = hknpBodyId::InvalidValue;
	m_bodyIdToCellIndexMap[ bodyId.value() ] = HKNP_INVALID_CELL_IDX;
}



#if defined(HK_PLATFORM_PSVITA)
	#pragma control %push O
	#pragma control O=0
#endif

void hknpBodyManager::removeActiveBodyGroup( hknpBodyId firstId )
{
	hknpBodyId bodyId = firstId;
	do
	{
		const hknpBody& body = m_bodies[bodyId.value()];
		if (body.isAddedToWorld())
		{
			removeSingleBodyFromActiveList(bodyId);
		}
		bodyId = body.m_nextAttachedBodyId;
	} while (bodyId != firstId);

	// Deactivate motion
	const hknpMotionId motionId = m_bodies[firstId.value()].m_motionId;
	hknpMotion& motion = m_motionManager->m_motions[motionId.value()];
	m_motionManager->removeActiveMotion(motion, motionId);
}

#if defined(HK_PLATFORM_PSVITA)
	#pragma control %pop O
#endif

hknpBodyManager::PropertyBuffer* hknpBodyManager::PropertyBuffer::construct(
	int propertySize, int capacity, const PropertyBuffer* source )
{
	// Allocate
	const int headerSize = sizeof(PropertyBuffer);
	const int bitFieldSize = sizeof(hkUint32) * hkBitField::getNumWordsRequired( capacity );
	const int paddingSize = ( HK_REAL_ALIGNMENT - ( headerSize + bitFieldSize ) % HK_REAL_ALIGNMENT ) % HK_REAL_ALIGNMENT;
	const int propertiesSize = propertySize * capacity;
	const int bufferSize = headerSize + bitFieldSize + paddingSize + propertiesSize;
	PropertyBuffer* buffer = (PropertyBuffer*)hkMemHeapBlockAlloc<hkUchar>( bufferSize );

	// Initialize
	hkString::memSet( buffer, 0, bufferSize );
	buffer->m_bufferSize = bufferSize;
	buffer->m_propertySize = propertySize;
	new (&buffer->m_occupancy) hkBitField( (hkUint32*)((hkUlong)buffer + headerSize), 0, capacity );
	buffer->m_properties = (void*)( (hkUlong)buffer + headerSize + bitFieldSize + paddingSize );
	HK_ASSERT2( 0xf02d5ea4, hkUlong(buffer->m_properties) % HK_REAL_ALIGNMENT == 0, "Properties should be aligned" );

	// Copy
	if( source != HK_NULL )
	{
		HK_ASSERT( 0x7a5538e9, propertySize == source->m_propertySize );
		buffer->m_occupancy.copy(source->m_occupancy);

		const int numPropertiesToCopy = hkMath::min2( buffer->m_occupancy.getSize(), source->m_occupancy.getSize() );
		hkString::memCpy( buffer->m_properties, source->m_properties, numPropertiesToCopy * propertySize );
	}

	return buffer;
}

void hknpBodyManager::PropertyBuffer::destruct( PropertyBuffer* buffer )
{
	hkMemHeapBlockFree<hkUchar>( (hkUchar*)buffer, buffer->m_bufferSize );
}

void hknpBodyManager::clearProperty( hknpBodyId bodyId, hknpPropertyKey key )
{
	HK_ON_DEBUG( const hknpBody& body = m_bodies[bodyId.value()]; );
	HK_ASSERT( 0xf02d5ea3, body.isValid() );

	PropertyBuffer* buf;
	if( m_propertyMap.get( key, &buf ) == HK_SUCCESS )
	{
		if( bodyId.value() < (hkUint32)buf->m_occupancy.getSize() )
		{
			buf->m_occupancy.clear( bodyId.value() );
		}
	}
}

void hknpBodyManager::clearAllPropertiesFromBody( hknpBodyId bodyId )
{
	hkMap< hknpPropertyKey, PropertyBuffer* >::Iterator it;
	for( it = m_propertyMap.getIterator(); m_propertyMap.isValid(it); it = m_propertyMap.getNext(it) )
	{
		PropertyBuffer* buf = m_propertyMap.getValue(it);
		if( bodyId.value() < (hkUint32)buf->m_occupancy.getSize() )
		{
			buf->m_occupancy.clear( bodyId.value() );
		}
	}
}

void hknpBodyManager::clearPropertyFromAllBodies( hknpPropertyKey key )
{
	PropertyBuffer* buf;
	if( m_propertyMap.get( key, &buf ) == HK_SUCCESS )
	{
		m_propertyMap.remove( key );
		PropertyBuffer::destruct( buf );
	}
}

void hknpBodyManager::clearAllPropertiesFromAllBodies()
{
	hkMap< hknpPropertyKey, PropertyBuffer* >::Iterator it;
	for( it = m_propertyMap.getIterator(); m_propertyMap.isValid(it); it = m_propertyMap.getNext(it) )
	{
		PropertyBuffer::destruct( m_propertyMap.getValue(it) );
	}
	m_propertyMap.clear();
}

void hknpBodyManager::clearAllScheduledBodyChanges()
{
	for (int i=0, ei=m_scheduledBodyChanges.getSize(); i<ei; i++)
	{
		m_scheduledBodyChangeIndices[m_scheduledBodyChanges[i].m_bodyId.value()] = INVALID_BODY_INDEX;
	}
	m_scheduledBodyChanges.clear();
}

void hknpBodyManager::appendToPendingAddList( const hknpBodyId* ids, int numIds, ScheduledBodyFlags addFlag )
{
	hkArray<hknpBodyId>* bodyList = 0;
	if (addFlag.get()==hknpBodyManager::ADD_ACTIVE)
	{
		bodyList = &m_bodiesToAddAsActive;
	}
	else if (addFlag.get()==hknpBodyManager::ADD_INACTIVE)
	{
		bodyList = &m_bodiesToAddAsInactive;
	}

	for (int i=0; i<numIds; i++)
	{
		hknpBodyId bodyId = ids[i];
		hknpBodyManager::BodyIndexType bodyIndex = m_scheduledBodyChangeIndices[bodyId.value()];
		hknpBodyManager::ScheduledBodyChange* item;
		if (bodyIndex!=hknpBodyManager::INVALID_BODY_INDEX)
		{
			HK_ASSERT(0xcade11aa, m_scheduledBodyChanges[bodyIndex].m_bodyId==bodyId);
			item = &m_scheduledBodyChanges[bodyIndex];
		}
		else
		{
			m_scheduledBodyChangeIndices[bodyId.value()] = m_scheduledBodyChanges.getSize();
			item = &m_scheduledBodyChanges.expandOne();
			item->clear();
			item->m_bodyId = bodyId;
		}
		HK_ASSERT2(0xcade11ab, item->m_pendingAddIndex==hknpBodyManager::INVALID_BODY_INDEX, "body already in an add list");
		item->m_pendingAddIndex = (hknpBodyManager::BodyIndexType)(bodyList->getSize());
		item->m_scheduledBodyFlags.orWith(addFlag.get());
		bodyList->pushBack(bodyId);
	}
}

void hknpBodyManager::clearPendingAddLists()
{
	m_bodiesToAddAsActive.clear();
	m_bodiesToAddAsInactive.clear();

	for (int i=0, ei=m_scheduledBodyChanges.getSize(); i<ei; i++)
	{
		m_scheduledBodyChanges[i].m_scheduledBodyFlags.andWith(~ScheduledBodyFlags(ADD_ACTIVE|ADD_INACTIVE).get());
		m_scheduledBodyChanges[i].m_pendingAddIndex = INVALID_BODY_INDEX;
	}
}

void hknpBodyManager::checkConsistency() const
{
#if defined(HK_DEBUG)

	HK_TIME_CODE_BLOCK( "CheckBodyManagerConsistency", HK_NULL );

	// Check body buffer consistency
	if( m_numAllocatedBodies )	// skip during manager initialization
	{
		HK_ASSERT( 0xf0dfede5, m_bodies.getSize() == m_bodies.getCapacity() );
		HK_ASSERT( 0xf0dfede6, getNumAllocatedBodies() <= getCapacity() );
		HK_ASSERT( 0xf0dfede7, m_numMarkedBodies < m_numAllocatedBodies );

		// Gather "valid" bodies
		hkLocalBitField validBodies( m_bodies.getSize(), hkBitFieldValue::ZERO );
		{
			hkUint32 peakValidIndex = 0;
			for( hkUint32 bodyIndex = 0, capacity = (hkUint32)getCapacity(); bodyIndex < capacity; bodyIndex++ )
			{
				if( m_bodies[bodyIndex].isValid() )
				{
					validBodies.set( bodyIndex );
					peakValidIndex = hkMath::max2( peakValidIndex, bodyIndex );
				}
			}
			HK_ASSERT( 0x390e3103, m_peakBodyIndex >= peakValidIndex );
		}

		// Gather "free" bodies
		hkLocalBitField freeBodies( m_bodies.getSize(), hkBitFieldValue::ZERO );
		{
			hknpBodyId id = m_firstFreeBodyId;
			while( id.value() != 0 )
			{
				freeBodies.set( id.value() );
				hknpBodyId nextId = REINTERPRET_CONST_AS_POINTERS(m_bodies[id.value()]).m_nextFreeBodyId;
				id = nextId;
			}
		}

		hkBitField anded;
		anded = freeBodies;
		anded.andWith( validBodies );
		HK_ASSERT2( 0x3bc15dda, !anded.anyIsSet(), "Some valid bodies are in the free list" );

		// This check doesn't account for bodies marked for deletion
		//hkBitField ored;
		//ored = freeBodies;
		//ored.orWith( validBodies );
		//int numBits = ored.bitCount();
		//HK_ASSERT2( 0x1cff69a3, numBits == ored.getSize(), "Some bodies are neither free or valid!" );

		// All bodies should be exclusively either free or valid
		//freeBodies.xorWith( validBodies );
		//int numBits = freeBodies.bitCount();
		//HK_ASSERT( 0x39740cdd, numBits == freeBodies.getSize() );
	}

	// Check active body cache
	for (int i = 0; i < m_activeBodyIds.getSize(); ++i )
	{
		const hknpBody& body = m_bodies[m_activeBodyIds[i].value()];
		HK_ASSERT( 0xf0343455, body.isAddedToWorld() );
		HK_ASSERT( 0xf0343456, body.isDynamic() );
	}

	// Check motion consistency
	if( m_motionManager )
	{
		m_motionManager->checkConsistency();

		int numActiveBodies = 0;
		for( hkUint32 bodyIndex = 0; bodyIndex <= m_peakBodyIndex; bodyIndex++ )
		{
			const hknpBody& body = m_bodies[bodyIndex];
			if ( !body.isAddedToWorld() )
			{
				continue;
			}

			hknpMotion& motion = m_motionManager->m_motions[ body.m_motionId.value() ];

			if ( body.isStatic() )
			{
				HK_ASSERT( 0xf0dfedea, body.m_motionId == hknpMotionId::STATIC );
			}
			if ( body.isDynamic() )
			{
				HK_ASSERT( 0xf0dfede8, body.m_motionId != hknpMotionId::STATIC );
			}
			if ( body.isKeyframed() )
			{
				HK_ASSERT( 0xf0dfede8, motion.hasInfiniteMass() );
				//HK_ASSERT(0xf0dfede9, motion.m_motionPropertiesId.value() == hknpMotionPropertiesId::KEYFRAMED);
			}

			if ( body.isActive() )
			{
				HK_ASSERT( 0xf0dfede4, body.isValid() );
				int activeId = body.m_indexIntoActiveListOrDeactivatedIslandId;
				HK_ASSERT( 0xf0dfede5, m_activeBodyIds[activeId] == hknpBodyId(bodyIndex) );
				HK_ASSERT( 0xf0dfede6, m_bodyIdToCellIndexMap[bodyIndex] == motion.m_cellIndex );
				numActiveBodies++;
			}

			if (body.m_motionId != hknpMotionId::STATIC)
			{
				// Check that the bodyId is part of the linked list (ring) of its motion
				hknpBodyId firstAttachedId = m_motionManager->m_motions[ body.m_motionId.value() ].m_firstAttachedBodyId;
				bool foundFirstAttached = false;
				hknpBodyId bodyId = hknpBodyId(bodyIndex);
				while (1)
				{
					if( firstAttachedId == bodyId )
					{
						foundFirstAttached = true;
					}
					const hknpBody& pBody = getBody(bodyId);
					HK_ASSERT( 0xf0343454, pBody.m_nextAttachedBodyId.isValid() );
					if( pBody.m_nextAttachedBodyId.value() == bodyIndex )
					{
						break;
					}
					bodyId = pBody.m_nextAttachedBodyId;
				}
				HK_ASSERT( 0xf1343454, foundFirstAttached );
			}
		}
		HK_ASSERT( 0xf0dfede4, numActiveBodies == m_activeBodyIds.getSize() );
	}

#endif	// HK_DEBUG
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
