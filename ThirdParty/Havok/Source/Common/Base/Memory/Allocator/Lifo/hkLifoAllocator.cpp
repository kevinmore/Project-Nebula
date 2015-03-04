/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Allocator/Lifo/hkLifoAllocator.h>

class hkLifoAllocator::Implementation
{
	public:
		HK_DECLARE_PLACEMENT_ALLOCATOR();

		struct NonLifoFree
		{
			void* start;
			void* end;
			int slabIndex;
		};

		// The frees which are out of LIFO order.
		hkArrayBase<NonLifoFree> m_nonLifoFrees;
		// Slabs are the blocks we chop and return.
		hkArrayBase<void*> m_slabPtrs;

		// Number of large + slab allocations
		int m_numExternalAllocations;

		Implementation()
			: m_numExternalAllocations(0)
		{
		}
};

hkLifoAllocator::hkLifoAllocator(int slabSize)
	: m_impl(HK_NULL)
	, m_slabSize(slabSize)
{
}

void hkLifoAllocator::init(hkMemoryAllocator* slabAllocator, 
						   hkMemoryAllocator* largeAllocator,
						   hkMemoryAllocator* internalAllocator)
{
	HK_ASSERT(0x13ab261c, !m_impl);
	m_impl = new(internalAllocator->_blockAlloc<Implementation>(1)) Implementation;

	m_cur = HK_NULL;
	m_end = HK_NULL;
	m_firstNonLifoEnd = HK_NULL;
	m_cachedEmptySlab = HK_NULL;
	m_slabAllocator = slabAllocator;
	m_largeAllocator = largeAllocator;
	m_internalAllocator = internalAllocator;
}

void hkLifoAllocator::quit(hkMemoryAllocator* allocators[3])
{
	HK_ASSERT(0x76a17b03, m_impl->m_nonLifoFrees.isEmpty());
	if (m_cur)
	{
		HK_ASSERT(0x38873491, m_impl->m_slabPtrs.getSize() == 1);
		HK_ASSERT(0x12bb6e86, m_cur == m_impl->m_slabPtrs[0]);
		m_internalAllocator->blockFree(m_impl->m_slabPtrs[0], m_slabSize);
	}
	else
	{
		HK_ASSERT(0x38873491, m_impl->m_slabPtrs.isEmpty());
	}

	if( m_cachedEmptySlab )
	{
		m_internalAllocator->blockFree(m_cachedEmptySlab, m_slabSize);
	}

	m_impl->m_slabPtrs._clearAndDeallocate(*m_internalAllocator);
	m_impl->m_nonLifoFrees._clearAndDeallocate(*m_internalAllocator);
	m_impl->~Implementation();
	m_internalAllocator->_blockFree(m_impl, 1);
	m_impl = HK_NULL;

	// if requested return the allocators
	if(allocators)
	{
		allocators[0] = m_slabAllocator;
		allocators[1] = m_largeAllocator;
		allocators[2] = m_internalAllocator;
	}
}

void* hkLifoAllocator::blockAlloc(int numBytesIn)
{
	return fastBlockAlloc(numBytesIn);
}

void hkLifoAllocator::blockFree(void* p, int numBytesIn)
{
	fastBlockFree(p, numBytesIn);
}

void* hkLifoAllocator::bufAlloc( int& reqNumBytesInOut )
{
	reqNumBytesInOut = HK_NEXT_MULTIPLE_OF(16, reqNumBytesInOut);
	return fastBlockAlloc(reqNumBytesInOut);
}

void hkLifoAllocator::bufFree( void* p, int numBytes )
{
	fastBlockFree(p, numBytes);
}

void* hkLifoAllocator::bufRealloc( void* pold, int oldNumBytes, int& reqNumBytesInOut )
{
	int old16 = HK_NEXT_MULTIPLE_OF(16, oldNumBytes);
	reqNumBytesInOut = HK_NEXT_MULTIPLE_OF(16, reqNumBytesInOut);
	if( (hkAddByteOffset(pold, old16) == m_cur) ) // top of stack can just grow
	{
		if( (hkUlong)hkAddByteOffset(pold, reqNumBytesInOut) <= hkUlong(m_end) ) // but only if there's enough room
		{
			m_cur = hkAddByteOffset(pold, reqNumBytesInOut);
			return pold; // quick path return
		}
	}
	// slow path, alloc copy free
	{
		void* pnew = blockAlloc(reqNumBytesInOut); // nelem
		hkMemUtil::memCpy(pnew, pold, hkMath::min2(old16, reqNumBytesInOut));
		blockFree(pold, old16);
		return pnew;
	}
}

bool hkLifoAllocator::isEmpty() const
{
	return m_impl->m_nonLifoFrees.getSize() == 0 && m_impl->m_slabPtrs.getSize() <= 1;
}

int hkLifoAllocator::numExternalAllocations() const
{
	return m_impl->m_numExternalAllocations;
}

void* hkLifoAllocator::allocateFromNewSlab(int numBytes)
{
	if (numBytes > m_slabSize)
	{
		//HK_WARN_ON_DEBUG_IF( true, 0xf0345456, "You are allocating a very big block from the stack allocator. Falling back to system allocator." );
		m_impl->m_numExternalAllocations++;
		void* ret = m_largeAllocator->blockAlloc(numBytes);
		HK_WARN_ON_DEBUG_IF(ret == HK_NULL, 0x7d73ede7, "Couldn't allocate block. The code may crash soon after this.");
		return ret;
	}

	// get a new slab
	void* ret;
	if( m_cachedEmptySlab )
	{
		ret = m_cachedEmptySlab;
		m_cachedEmptySlab = HK_NULL;
	}
	else
	{
		if (m_impl->m_slabPtrs.getSize() < 2)
		{
			// use the internal allocator for the first two slabs
			ret = m_internalAllocator->blockAlloc(m_slabSize);
			HK_WARN_ON_DEBUG_IF(ret == HK_NULL, 0x1af0a345, "Couldn't allocate block. The code may crash soon after this.");
		}
		else
		{
			m_impl->m_numExternalAllocations++;
			ret = m_slabAllocator->blockAlloc(m_slabSize);
			HK_WARN_ON_DEBUG_IF(ret == HK_NULL, 0x1af0a345, "Couldn't allocate block. The code may crash soon after this.");
		}
	}

	if (!m_impl->m_slabPtrs.isEmpty())
	{
		// sneaky - reduce special casing by  pretending there's a non-lifo free at
		// the end of the current slab. When freeing, we don't have to check whether
		// we're at padding OR at a non-lifo free - we only need check the non-lifo case.
		Implementation::NonLifoFree& nl = m_impl->m_nonLifoFrees._expandOne(*m_internalAllocator);
		nl.end = ret;
		nl.start = m_cur;
		nl.slabIndex = m_impl->m_slabPtrs.getSize() - 1;
		m_firstNonLifoEnd = ret;
	}

	// Set up new members
	m_end = hkAddByteOffset(ret, m_slabSize);
	m_cur = hkAddByteOffset(ret, numBytes);
	m_impl->m_slabPtrs._pushBack(*m_internalAllocator, ret);


	return ret;
}

void hkLifoAllocator::popNonLifoFrees()
{
	void* cur = m_cur;

	// pop as many non lifo frees as we can
	for( int i = m_impl->m_nonLifoFrees.getSize()-1; i >= 0; --i )
	{
		const Implementation::NonLifoFree& nl = m_impl->m_nonLifoFrees[i];
		if( nl.end == cur )
		{
			cur = nl.start;
			m_impl->m_nonLifoFrees.popBack();
		}
		else
		{
			break;
		}
	}

	// free any now-empty slabs
	while(m_impl->m_slabPtrs.getSize() > 1 && 
		  (cur == m_impl->m_slabPtrs.back() || 
		   hkUlong(cur) - hkUlong(m_impl->m_slabPtrs.back()) > hkUlong(m_slabSize)))
	{
		if( m_cachedEmptySlab )
		{
			m_slabAllocator->blockFree(m_cachedEmptySlab, m_slabSize);
			m_impl->m_numExternalAllocations--;
		}
		m_cachedEmptySlab = m_impl->m_slabPtrs.back();
		m_impl->m_slabPtrs.popBack();
	}

	// and set up member vars
	m_cur = cur;
	m_end = m_impl->m_slabPtrs.getSize() ? hkAddByteOffset(m_impl->m_slabPtrs.back(), m_slabSize) : HK_NULL;
	m_firstNonLifoEnd = m_impl->m_nonLifoFrees.getSize() ? m_impl->m_nonLifoFrees.back().end : HK_NULL;
}

void hkLifoAllocator::insertNonLifoFree(void* pstart, int nbytes)
{
#ifdef HK_DEBUG
	if( m_impl->m_nonLifoFrees.getSize() == 100 )
	{
		HK_WARN(0x38d9223e, "Many non-lifo allocations at this point.\nThe temp allocator is probably being used incorrectly");
	}
#endif
	int slabIndex = -1;
	for( int i = m_impl->m_slabPtrs.getSize()-1; i >= 0; --i )
	{
		const void* slab = m_impl->m_slabPtrs[i];
		if( (hkUlong(pstart)-hkUlong(slab)) < hkUlong(m_slabSize) )
		{
			slabIndex = i;
			break;
		}
	}
	HK_ASSERT(0x552dcf6d, slabIndex >= 0);
	void* pend = hkAddByteOffset(pstart, nbytes);

	int insertPos = 0;
	for (int i = m_impl->m_nonLifoFrees.getSize() - 1; i >= 0; --i)
	{
		Implementation::NonLifoFree& nl = m_impl->m_nonLifoFrees[i];
		if( nl.slabIndex == slabIndex )
		{
			if( pstart == nl.end ) // merge with prev
			{
				nl.end = pend;
				goto end; // let's not cascade because that would need a memmove
			}
			else if( pend == nl.start ) // merge with next
			{
				nl.start = pstart;
				goto end; // let's not cascade because that would need a memmove
			}
			if( hkUlong(pstart) > hkUlong(nl.start) )
			{
				insertPos = i+1;
				break;
			}
		}
		else if( nl.slabIndex < slabIndex )
		{
			insertPos = i+1;
			break;
		}
	}
	HK_ASSERT(0x19655325, insertPos >= 0);
	{
		Implementation::NonLifoFree& n = *m_impl->m_nonLifoFrees._expandAt(*m_internalAllocator, insertPos, 1);
		n.start = pstart;
		n.end = pend;
		n.slabIndex = slabIndex;
	}
end:
	m_firstNonLifoEnd = m_impl->m_nonLifoFrees.getSize() ? m_impl->m_nonLifoFrees.back().end : HK_NULL;
}


void hkLifoAllocator::slowBlockFree(void* p, int numBytesIn)
{
	if (p)
	{
		if( numBytesIn <= m_slabSize )
		{
			int numBytes = HK_NEXT_MULTIPLE_OF(16, numBytesIn);
			if( hkAddByteOffset(p,numBytes) == m_cur ) // usual case, free top
			{
				m_cur = p;
				// the simple case has already been handled
				// so we can assume that there are nonlifo
				// frees to pop
				popNonLifoFrees();
			}
			else
			{
				insertNonLifoFree(p, numBytes);
			}
		}
		else
		{
			m_impl->m_numExternalAllocations--;
			HK_ASSERT(0x5403dabd, m_impl->m_numExternalAllocations >= 0);
			m_largeAllocator->blockFree(p, numBytesIn);
		}
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
