/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE hknpIdxRange::hknpIdxRange( int start, int numElements )
: m_start(start)
, m_numElements(numElements)
{
}

HK_FORCE_INLINE void hknpIdxRange::clearRange()
{
	m_start = -1;
	m_numElements = 0;
}

HK_FORCE_INLINE bool hknpIdxRange::isEmpty() const
{
	return m_numElements == 0;
}

template<typename Entry>
HK_FORCE_INLINE void hknpGrid<Entry>::setSize(int size)
{
	m_entries.setSize( size );
	clearGrid();
}

template<typename Entry>
HK_FORCE_INLINE int hknpGrid<Entry>::getSize()
{
	return m_entries.getSize();
}

template<typename Entry>
HK_FORCE_INLINE void hknpGrid<Entry>::clearGrid()
{
	for(int i=0; i<m_entries.getSize(); i++)
	{
		m_entries[i].clearRange();
	}
}

template<typename Entry>
HK_FORCE_INLINE void hknpGrid<Entry>::setInvalid()
{
	for(int i=0; i<m_entries.getSize(); i++)
	{
		m_entries[i].clearRange();
		m_entries[i].m_numElements = 0xf034fde3;
		m_entries[i].m_startBlock = HK_NULL;
	}
}

template<typename Entry>
HK_FORCE_INLINE bool hknpGrid<Entry>::isEmpty()const
{
	HK_ASSERT( 0xf0343454, m_entries.getSize() );
	for(int i=0; i<m_entries.getSize(); i++)
	{
		if ( !m_entries[i].isEmpty() )
		{
			return false;
		}
	}
	return true;
}


// special fast clear for hkUint32
template<>
HK_FORCE_INLINE void hknpGrid<hkUint32>::clearGrid()
{
	hkString::memSet4(m_entries.begin(), 0, m_entries.getSize());
}


template<typename Entry>
HK_FORCE_INLINE int hknpGrid<Entry>::getLinkedNumElements(int entryIndex)
{
	// Iterate over all linked ranges for this entry, summing their element counts.
	int numElements = 0;
	Entry* range = &(m_entries[entryIndex]);

	while (range != HK_NULL)
	{
		numElements += range->getNumElements();
		range = (Entry*) range->m_next;
	}

	return numElements;
}


template<typename Entry>
template<typename WRITER, typename RANGE>
HK_FORCE_INLINE void HK_CALL hknpGrid<Entry>::addRange(WRITER& writer, int entryIndex, const RANGE& range)
{
	if (!range.isEmpty())
	{
#if !defined(HK_PLATFORM_SPU)
		RANGE* HK_RESTRICT entryRange = &m_entries[entryIndex];
#else
		// Get grid entry to SPU.
		RANGE entryRangeSpu;
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( &entryRangeSpu, m_entries.begin() + entryIndex, sizeof(RANGE), hkSpuDmaManager::READ_WRITE );
		RANGE* HK_RESTRICT entryRange = &entryRangeSpu;
#endif

		if (entryRange->isEmpty())
		{
			HK_ASSERT2( 0xef5f13a6, entryRange->m_next == HK_NULL, "Empty entries should not be linked." );
			*entryRange = range;
		}
		else
		{
			// Create the new persistent range and set it.
			RANGE* HK_RESTRICT presistentRange = writer.template reserve<RANGE>();
			*presistentRange = range;

			// Append the new range.
			HK_ON_CPU( entryRange->appendPersistentRange( presistentRange ) );
			HK_ON_SPU( entryRange->appendPersistentRangeSpu( presistentRange, writer.spuToPpu(presistentRange) ) );
			writer.advance(sizeof(RANGE));
		}

#if defined(HK_PLATFORM_SPU)
		// Write the modified Spu grid entry back.
		hkSpuDmaManager::putToMainMemoryAndWaitForCompletion( m_entries.begin() + entryIndex, &entryRangeSpu, sizeof(RANGE), hkSpuDmaManager::WRITE_BACK );
		hkSpuDmaManager::performFinalChecks( m_entries.begin() + entryIndex, &entryRangeSpu, sizeof(RANGE) );
#endif
	}
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
