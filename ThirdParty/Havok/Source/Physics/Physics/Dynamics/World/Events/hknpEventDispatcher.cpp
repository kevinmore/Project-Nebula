/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/World/Events/hknpEventDispatcher.h>
#include <Common/Base/Container/CommandStream/hkUnrollCaseMacro.h>


hknpEventDispatcher::hknpEventDispatcher( hknpWorld* world )
{
	m_world = world;
	m_solverData = HK_NULL;
	m_simulationThreadContext = HK_NULL;
	m_commandWriter = HK_NULL;
	m_firstFreeElement = INVALID_ENTRY;
	m_entryPool.reserve( 256 );
	m_bodyToEntryMap.setSize( m_world->getBodyCapacity(), INVALID_ENTRY );	// this can grow later
}

hknpEventDispatcher::Entry* hknpEventDispatcher::allocateEntry( hknpBodyId id )
{
	EntryIdx entryIdx;
	Entry* entry;
	if( m_firstFreeElement != INVALID_ENTRY )
	{
		entryIdx = m_firstFreeElement;
		entry = &m_entryPool[ entryIdx ];
		m_firstFreeElement = entry->m_nextEntry;
	}
	else
	{
		entryIdx = EntryIdx(m_entryPool.getSize());
		entry = m_entryPool.expandBy(1);
	}
	entry->m_nextEntry = m_bodyToEntryMap[id.value()];
	m_bodyToEntryMap[id.value()] = entryIdx;
	return entry;
}

void hknpEventDispatcher::freeEntry( hknpBodyId id, Entry& entry )
{
	int entryId = (int)((&entry) - m_entryPool.begin());

	// search previous entry
	EntryIdx previous = INVALID_ENTRY;
	for( EntryIdx eId = m_bodyToEntryMap[id.value()]; eId != INVALID_ENTRY && eId != entryId; eId = m_entryPool[eId].m_nextEntry )
	{
		previous = eId;
	}
	if( previous == INVALID_ENTRY )
	{
		m_bodyToEntryMap[id.value()] = entry.m_nextEntry;
	}
	else
	{
		m_entryPool[previous].m_nextEntry = entry.m_nextEntry;
	}

	entry.m_nextEntry = m_firstFreeElement;
	m_firstFreeElement = EntryIdx(entryId);
}

hknpEventSignal& hknpEventDispatcher::getSignal( hknpEventType::Enum eventType, hknpBodyId id )
{
	// May need to grow the map if the body buffer capacity increased since construction
	if( HK_VERY_UNLIKELY( (hkUint32)m_bodyToEntryMap.getSize() < m_world->getBodyCapacity() ) )
	{
		m_bodyToEntryMap.setSize( m_world->getBodyCapacity(), INVALID_ENTRY );
	}

	// Search for existing entry
	for( EntryIdx eId = m_bodyToEntryMap[id.value()]; eId != INVALID_ENTRY; eId = m_entryPool[eId].m_nextEntry )
	{
		if( m_entryPool[eId].m_eventType == eventType )
		{
			return m_entryPool[eId].m_signal;
		}
	}

	// Allocate a new entry
	Entry& entry =  *allocateEntry( id );
	entry.m_eventType = eventType;
	return entry.m_signal;
}

hknpEventSignal& hknpEventDispatcher::getSignal( hknpEventType::Enum eventType )
{
	// Search for existing entry
	for( EntryIdx eId = m_bodyToEntryMap[0]; eId != INVALID_ENTRY; eId = m_entryPool[eId].m_nextEntry )
	{
		if ( m_entryPool[eId].m_eventType == eventType )
		{
			return m_entryPool[eId].m_signal;
		}
	}

	// Allocate a new entry
	Entry& entry =  *allocateEntry( hknpBodyId(0) );
	entry.m_eventType = eventType;
	return entry.m_signal;
}

void hknpEventDispatcher::unsubscribeAllSignals( hknpBodyId id )
{
	if( id.value() < (hkUint32)m_bodyToEntryMap.getSize() )
	{
		hknpEventDispatcher::EntryIdx eId = m_bodyToEntryMap[id.value()];
		if (eId != hknpEventDispatcher::INVALID_ENTRY)
		{
			// Get the first entry
			hknpEventDispatcher::Entry* entry = &m_entryPool[eId];
			int entryId = (int)((entry) - m_entryPool.begin());

			// Disconnect the map from the linked list
			entry->m_signal.reset();
			m_bodyToEntryMap[id.value()] = hknpEventDispatcher::INVALID_ENTRY;

			// Walk down the link list to obtain the last element and clean the slots on the way
			do
			{
				entry = &m_entryPool[eId];
				entry->m_signal.reset();
				eId = entry->m_nextEntry;
			} while ( eId != hknpEventDispatcher::INVALID_ENTRY );
			entry->m_nextEntry = m_firstFreeElement;

			// Connect the linked list to the free list
			m_firstFreeElement = EntryIdx(entryId);
		}
	}
}

void hknpEventDispatcher::print( const hkCommand& command, hkOstream& stream ) const
{
#if !defined(HK_PLATFORM_CTR)
	switch( command.m_secondaryType )
	{
		HK_UNROLL_CASE_32(
		{
			typedef hknpEventTypeDiscriminator<UNROLL_I>::CommandType ct;
			const ct* c = reinterpret_cast<const ct*>( &command );
			c->printCommand( m_world, stream );
			break;
		}
		);
	}
	// check if our unroll macro is sufficient by checking if command 33 falls back to our empty command
	{
		typedef hknpEventTypeDiscriminator<33>::CommandType ct;
		const ct* c = reinterpret_cast<const ct*>( &command );
		c->checkIsEmptyCommand();
	}
#endif
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
