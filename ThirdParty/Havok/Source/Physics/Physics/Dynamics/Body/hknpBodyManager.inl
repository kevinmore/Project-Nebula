/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE hknpBodyManager::BodyIterator::BodyIterator( const hknpBodyManager& manager )
:	m_bodyManager(manager),
	m_bodyIndex(0)		// start on the special static body
{
	next();
}

HK_FORCE_INLINE void hknpBodyManager::BodyIterator::next()
{
	HK_ASSERT( 0x44f16e43, m_bodyIndex != -1 );

	const int peakIndex = m_bodyManager.getPeakBodyId().value();
	while( ++m_bodyIndex <= peakIndex )
	{
		if( m_bodyManager.m_bodies[m_bodyIndex].isValid() )
		{
			return;
		}
	}

#ifdef HK_DEBUG
	// Make sure there are no valid bodies in the rest of the buffer
	for( const int capacity = m_bodyManager.getCapacity(); m_bodyIndex < capacity; ++m_bodyIndex )
	{
		HK_ASSERT( 0x39a5ba5e, !m_bodyManager.m_bodies[m_bodyIndex].isValid() );
	}
#endif

	m_bodyIndex = -1;
}

HK_FORCE_INLINE bool hknpBodyManager::BodyIterator::isValid() const
{
	return m_bodyIndex != -1;
}

HK_FORCE_INLINE hknpBodyId hknpBodyManager::BodyIterator::getBodyId() const
{
	return hknpBodyId(m_bodyIndex);
}

HK_FORCE_INLINE const hknpBody& hknpBodyManager::BodyIterator::getBody() const
{
	return m_bodyManager.m_bodies[m_bodyIndex];
}


HK_FORCE_INLINE hkUint32 hknpBodyManager::getCapacity() const
{
	return (hkUint32)m_bodies.getCapacity();
}

HK_FORCE_INLINE const hknpBody* hknpBodyManager::getBodyBuffer() const
{
	return m_bodies.begin();
}

HK_FORCE_INLINE hknpBody* hknpBodyManager::accessBodyBuffer()
{
	return m_bodies.begin();
}

HK_FORCE_INLINE void hknpBodyManager::markBodyForDeletion( hknpBodyId bodyId )
{
	// Invalidate the body.
	HK_ASSERT2( 0xf02e5ea2, ( bodyId.value() >= hknpBodyId::NUM_PRESETS ) && m_bodies[bodyId.value()].isValid(),
		"Marking an invalid body for deletion" );
	m_bodies[bodyId.value()].m_flags.clear( hknpBody::INTERNAL_FLAGS_MASK );

	// Mark for delete
	setScheduledBodyFlags( bodyId, TO_BE_DELETED );
	m_numMarkedBodies++;
}

HK_FORCE_INLINE hkUint32 hknpBodyManager::getNumAllocatedBodies() const
{
	return m_numAllocatedBodies;
}

HK_FORCE_INLINE hkUint32 hknpBodyManager::getNumActiveBodies() const
{
	return m_activeBodyIds.getSize();
}

HK_FORCE_INLINE hknpBodyId hknpBodyManager::getPeakBodyId() const
{
	return hknpBodyId(m_peakBodyIndex);
}

HK_FORCE_INLINE hknpBodyIterator hknpBodyManager::getBodyIterator() const
{
	return BodyIterator(*this);
}

HK_FORCE_INLINE const hknpBody& hknpBodyManager::getBody( hknpBodyId id ) const
{
	return m_bodies[id.value()];
}

HK_FORCE_INLINE hknpBody& hknpBodyManager::accessBody( hknpBodyId id )
{
	return m_bodies[id.value()];
}

HK_FORCE_INLINE void hknpBodyManager::updateBodyToCellIndexTable( hknpBodyId firstId, hknpCellIndex cellIndex )
{
	hknpBodyId bodyId = firstId;
	do
	{
		hknpBody& body = m_bodies[bodyId.value()];
		m_bodyIdToCellIndexMap[ bodyId.value() ] = cellIndex;
		bodyId = body.m_nextAttachedBodyId;
	}
	while( bodyId != firstId );
}

HK_FORCE_INLINE void hknpBodyManager::setBodyName( hknpBodyId bodyId, const char* name )
{
	HK_ASSERT2( 0xf02d5ea2, ( bodyId.value() >= hknpBodyId::NUM_PRESETS ) && m_bodies[bodyId.value()].isValid(),
		"Tried to set the name of an invalid body" );

	if( name && (bodyId.value() >= (hkUint32)m_bodyNames.getSize()) )
	{
		// (re)allocate the array
		m_bodyNames.setSize( m_bodies.getCapacity(), HK_NULL );
	}

	if( bodyId.value() < (hkUint32)m_bodyNames.getSize() )
	{
		m_bodyNames[ bodyId.value() ] = name;
	}
}

HK_FORCE_INLINE const char* hknpBodyManager::getBodyName( hknpBodyId bodyId ) const
{
	HK_ASSERT2( 0x7171699f, ( bodyId.value() >= hknpBodyId::NUM_PRESETS ) && m_bodies[bodyId.value()].isValid(),
		"Tried to get the name of an invalid body" );

	if( bodyId.value() < (hkUint32)m_bodyNames.getSize() )
	{
		return m_bodyNames[ bodyId.value() ].cString();
	}
	else
	{
		return HK_NULL;
	}
}

HK_FORCE_INLINE hknpBodyId hknpBodyManager::findBodyByName( const char* name )
{
	const int max = hkMath::min2( m_bodies.getSize(), m_bodyNames.getSize() );
	for( int i = 0; i < max; i++ )
	{
		if( m_bodies[i].isValid() && (m_bodyNames[i].compareTo(name) == 0) )
		{
			return hknpBodyId(i);
		}
	}
	return hknpBodyId::invalid();
}

HK_FORCE_INLINE void hknpBodyManager::ScheduledBodyChange::clear()
{
	m_bodyId = hknpBodyId::invalid();
	m_pendingAddIndex = INVALID_BODY_INDEX;
	m_scheduledBodyFlags.clear();
}

HK_FORCE_INLINE hkBool32 hknpBodyManager::removeSingleBodyFromPendingAddLists( hknpBodyId bodyId )
{
	hknpBodyManager::BodyIndexType index = m_scheduledBodyChangeIndices[bodyId.value()];
	if( index == hknpBodyManager::INVALID_BODY_INDEX )
	{
		return false;
	}

	ScheduledBodyChange& item = m_scheduledBodyChanges[index];
	if( item.m_scheduledBodyFlags.noneIsSet(ADD_ACTIVE|ADD_INACTIVE) )
	{
		return false;
	}

	hkArray<hknpBodyId>* bodyList = HK_NULL;
	if( item.m_scheduledBodyFlags.anyIsSet(ADD_ACTIVE) )
	{
		bodyList = &m_bodiesToAddAsActive;
	}
	else
	{
		HK_ASSERT( 0x3bcd8355, item.m_scheduledBodyFlags.anyIsSet(ADD_INACTIVE) );
		bodyList = &m_bodiesToAddAsInactive;
	}

	hkUint32 lastElementIndex = bodyList->getSize() - 1;
	if( item.m_pendingAddIndex != lastElementIndex )
	{
		hknpBodyId lastBodyId = (*bodyList)[lastElementIndex];
		hknpBodyManager::ScheduledBodyChange& lastItem = m_scheduledBodyChanges[m_scheduledBodyChangeIndices[lastBodyId.value()]];
		lastItem.m_pendingAddIndex = item.m_pendingAddIndex;
		(*bodyList)[lastItem.m_pendingAddIndex] = lastBodyId;
	}
	bodyList->popBack();
	item.m_pendingAddIndex = hknpBodyManager::INVALID_BODY_INDEX;
	item.m_scheduledBodyFlags.clear(ADD_ACTIVE|ADD_INACTIVE);
	return true;
}

HK_FORCE_INLINE bool hknpBodyManager::isBodyWaitingToBeAdded( hknpBodyId bodyId ) const
{
	hknpBodyManager::BodyIndexType index = m_scheduledBodyChangeIndices[bodyId.value()];
	if( index == hknpBodyManager::INVALID_BODY_INDEX )
	{
		return false;
	}

	const ScheduledBodyChange& item = m_scheduledBodyChanges[index];
	return item.m_pendingAddIndex != INVALID_BODY_INDEX;
}

HK_FORCE_INLINE void hknpBodyManager::setScheduledBodyFlags( hknpBodyId bodyId, ScheduledBodyFlags flags )
{
	hknpBodyManager::BodyIndexType index = m_scheduledBodyChangeIndices[bodyId.value()];
	ScheduledBodyChange* item;
	if( index != hknpBodyManager::INVALID_BODY_INDEX )
	{
		HK_ASSERT( 0xcade11aa, m_scheduledBodyChanges[index].m_bodyId == bodyId );
		item = &m_scheduledBodyChanges[index];
	}
	else
	{
		m_scheduledBodyChangeIndices[bodyId.value()] = m_scheduledBodyChanges.getSize();
		item = &m_scheduledBodyChanges.expandOne();
		item->clear();
		item->m_bodyId = bodyId;
	}
	item->m_scheduledBodyFlags.orWith( flags.get() );
}

HK_FORCE_INLINE void hknpBodyManager::clearScheduledBodyFlags( hknpBodyId bodyId, ScheduledBodyFlags flags )
{
	hknpBodyManager::BodyIndexType index = m_scheduledBodyChangeIndices[bodyId.value()];
	if( index != hknpBodyManager::INVALID_BODY_INDEX )
	{
		HK_ASSERT( 0xcade11aa, m_scheduledBodyChanges[index].m_bodyId == bodyId );
		m_scheduledBodyChanges[index].m_scheduledBodyFlags.andWith( ~flags.get() );
	}
}

HK_FORCE_INLINE hknpBodyManager::ScheduledBodyFlags hknpBodyManager::getScheduledBodyFlags( hknpBodyId bodyId ) const
{
	hknpBodyManager::BodyIndexType index = m_scheduledBodyChangeIndices[bodyId.value()];
	if( index != hknpBodyManager::INVALID_BODY_INDEX )
	{
		HK_ASSERT( 0xcade11aa, m_scheduledBodyChanges[index].m_bodyId == bodyId );
		return m_scheduledBodyChanges[index].m_scheduledBodyFlags;
	}
	return ScheduledBodyFlags(0);
}

template< typename T >
HK_FORCE_INLINE void hknpBodyManager::setProperty( hknpBodyId bodyId, hknpPropertyKey key, const T& value )
{
	HK_ASSERT2( 0xf02d5ea3, ( bodyId.value() >= hknpBodyId::NUM_PRESETS ) && m_bodies[bodyId.value()].isValid(),
		"Tried to set a property of an invalid body" );

	PropertyBuffer* buf = m_propertyMap.getWithDefault( key, HK_NULL );
	if( !buf )
	{
		// allocate
		buf = PropertyBuffer::construct( sizeof(T), m_bodies.getCapacity() );
		m_propertyMap.insert( key, buf );
	}
	else if( bodyId.value() >= (hkUint32)buf->m_occupancy.getSize() )
	{
		// reallocate
		PropertyBuffer* buf2 = PropertyBuffer::construct( sizeof(T), m_bodies.getCapacity(), buf );
		PropertyBuffer::destruct( buf );
		buf = buf2;
		m_propertyMap.insert( key, buf );
	}

	HK_ASSERT2( 0xf02d5ea5, buf->m_propertySize == sizeof(T), "Property size mismatch" );
	HK_ASSERT2( 0xf02d5ea6,
		(hkUlong(buf) + buf->m_bufferSize) >= (hkUlong(buf->m_properties) + (bodyId.value()+1)*sizeof(T)),
		"Property buffer out of bounds" );

	buf->m_occupancy.set( bodyId.value() );
	T* property = (T*)( hkUlong(buf->m_properties) + (bodyId.value()*sizeof(T)) );
	*property = value;
}

template< typename T >
HK_FORCE_INLINE T* hknpBodyManager::getProperty( hknpBodyId bodyId, hknpPropertyKey key ) const
{
	HK_ASSERT2( 0xf02d5ea4, ( bodyId.value() >= hknpBodyId::NUM_PRESETS ) && m_bodies[bodyId.value()].isValid(),
		"Tried to get a property from an invalid body" );

	PropertyBuffer* buf = m_propertyMap.getWithDefault( key, HK_NULL );
	if( buf && ( (hkUint32)buf->m_occupancy.getSize() > bodyId.value() ) && buf->m_occupancy.get( bodyId.value() ) )
	{
		HK_ASSERT2( 0xf02d5ea5, sizeof(T) == buf->m_propertySize, "Property size mismatch" );
		HK_ASSERT2( 0xf02d5ea6,
			(hkUlong(buf) + buf->m_bufferSize) >= (hkUlong(buf->m_properties) + (bodyId.value()+1)*sizeof(T)),
			"Property buffer out of bounds" );

		return (T*)( hkUlong(buf->m_properties) + (bodyId.value()*sizeof(T)) );
	}

	return HK_NULL;
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
