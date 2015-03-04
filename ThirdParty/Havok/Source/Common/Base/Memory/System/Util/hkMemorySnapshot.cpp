/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/System/Util/hkMemorySnapshot.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

HK_FORCE_INLINE hkBool hkMemorySnapshot::Allocation::operator==(const Allocation& rhs) const
{
	return (
		m_start == rhs.m_start &&
		m_size == rhs.m_size &&
		m_sourceId == rhs.m_sourceId &&
		m_status == rhs.m_status);
}

HK_FORCE_INLINE hkBool hkMemorySnapshot::Allocation::operator!=(const Allocation& rhs) const 
{ 
	return !(*this == rhs); 
}

hkMemorySnapshot::hkMemorySnapshot(hkMemoryAllocator* a)
	: m_mem(a),	m_callTree(a)
{
}

hkMemorySnapshot::hkMemorySnapshot(const hkMemorySnapshot& rhs)
	: m_mem(rhs.m_mem), m_callTree(rhs.m_callTree)
{
	m_allocations._append(*m_mem, rhs.m_allocations.begin(), rhs.m_allocations.getSize());
	for(int i = 0; i < rhs.m_providers.getSize(); ++i)
	{
		Provider& p = m_providers._expandOne(*m_mem);
		hkString::strNcpy(p.m_name, rhs.m_providers[i].m_name, sizeof(p.m_name));
		p.m_parentIndices._append(*m_mem, rhs.m_providers[i].m_parentIndices.begin(), rhs.m_providers[i].m_parentIndices.getSize());
	}
}

void hkMemorySnapshot::setAllocator(hkMemoryAllocator* a)
{
	m_mem = a;
	m_callTree.init(a);
}

void hkMemorySnapshot::swap(hkMemorySnapshot& m)
{
	m_mem = m.m_mem; m.m_mem = HK_NULL;
	m_allocations._setDataUnchecked( m.m_allocations.begin(), m.m_allocations.getSize(), m.m_allocations.getCapacityAndFlags() );
	m.m_allocations._setDataUnchecked(0,0,hkArray<char>::DONT_DEALLOCATE_FLAG);
	m_providers._setDataUnchecked( m.m_providers.begin(), m.m_providers.getSize(), m.m_providers.getCapacityAndFlags() );
	m.m_providers._setDataUnchecked(0,0,hkArray<char>::DONT_DEALLOCATE_FLAG);
	m_callTree.swap( m.m_callTree );
	hkString::memCpy(m_routerWiring, m.m_routerWiring, sizeof(m_routerWiring));
}

void hkMemorySnapshot::clear()
{
	m_allocations._clearAndDeallocate(*m_mem);
	for(int i = 0; i < m_providers.getSize(); ++i)
	{
		m_providers[i].m_parentIndices._clearAndDeallocate(*m_mem);
	}
	m_providers._clearAndDeallocate(*m_mem);
}

hkMemorySnapshot::~hkMemorySnapshot()
{
	clear();
}

hkMemorySnapshot::ProviderId hkMemorySnapshot::addProvider(const char* name, ProviderId parent)
{
	int id = m_providers.getSize();
	Provider& p = m_providers._expandOne(*m_mem);
	hkString::strNcpy(p.m_name, name, sizeof(p.m_name)-1); 
	p.m_name[sizeof(p.m_name)-1] = 0;
	if(parent != -1)
		p.m_parentIndices._pushBack(*m_mem, parent);
	return id;
}

void hkMemorySnapshot::addParentProvider(ProviderId provider, ProviderId parent)
{
	Provider& prov = m_providers[provider];
	prov.m_parentIndices._pushBack(*m_mem, parent);
}

hkMemorySnapshot::AllocationId hkMemorySnapshot::addItem(hkMemorySnapshot::ProviderId id, Status status, const void* address, int size)
{
	int aid = m_allocations.getSize();
	Allocation& a = m_allocations._expandOne(*m_mem);
	a.m_start = address;
	a.m_size = size;
	a.m_sourceId = id;
	a.m_traceId = -1;
	a.m_status = status;
	return aid;
}

void hkMemorySnapshot::setCallStack(hkMemorySnapshot::AllocationId id, const hkUlong* addresses, int numAddresses )
{
	m_allocations[id].m_traceId = m_callTree.insertCallStack(addresses, numAddresses);
}

void hkMemorySnapshot::setRouterWiring(ProviderId stack, ProviderId temp, ProviderId heap, ProviderId debug, ProviderId solver)
{
	m_routerWiring[0] = stack;
	m_routerWiring[1] = temp;
	m_routerWiring[2] = heap;
	m_routerWiring[3] = debug;
	m_routerWiring[4] = solver;
}

HK_FORCE_INLINE static hkBool32 _compareAllocations(const hkMemorySnapshot::Allocation& a, const hkMemorySnapshot::Allocation& b)
{
	// lowest address first, tie break by largest first.
	return (a.m_start < b.m_start) || ((a.m_start == b.m_start) && (a.m_size > b.m_size));
}

void hkMemorySnapshot::sort()
{
	hkSort( m_allocations.begin(), m_allocations.getSize(), _compareAllocations );
}

/*static*/ void HK_CALL hkMemorySnapshot::allocationDiff(const hkMemorySnapshot& snapA, const hkMemorySnapshot& snapB, hkArray<Allocation>& onlyInA, hkArray<Allocation>& onlyInB)
{
	onlyInA.clear();
	onlyInB.clear();

	const Allocation* curA = snapA.m_allocations.begin();
	const Allocation* endA = snapA.m_allocations.end();

	const Allocation* curB = snapB.m_allocations.begin();
	const Allocation* endB = snapB.m_allocations.end();

	while (curA < endA && curB < endB)
	{
		if (*curA == *curB)
		{
			curA++;
			curB++;
			continue;
		}

		if (curA->m_start == curB->m_start)
		{
			// Add them both.. as they are the same address
			onlyInA.pushBack(*curA);
			onlyInB.pushBack(*curB);

			curA++;
			curB++;
			continue;
		}

		HK_ASSERT(0x4322343, curA->m_start != curB->m_start);
		if (curA->m_start < curB->m_start)
		{
			onlyInA.pushBack(*curA);
			curA++;
		}
		else
		{
			onlyInB.pushBack(*curB);
			curB++;
		}
	}

	for (; curA < endA; curA++)
	{
		onlyInA.pushBack(*curA);
	}

	for (; curB < endB; curB++)
	{
		onlyInB.pushBack(*curB);
	}
}

void hkMemorySnapshot::dumpAllocation(const Allocation& alloc, hkOstream& stream) const
{
	stream << "Alloc: ";

	stream << (void*)alloc.m_start << 
		" Size: " << alloc.m_size << "\n";
	stream << (void*)alloc.m_start << 
		" Provider: " << alloc.m_sourceId << "\n";

	const char* status = HK_NULL;

	if(alloc.m_status == STATUS_OVERHEAD)
	{
		status = "overhead";
	}
	else if (alloc.m_status == STATUS_USED)
	{
		status = "used";
	}
	else // alloc.m_status == STATUS_UNUSED
	{
		status = "unused";
	}
	stream << (void*)alloc.m_start << 
		" Status: " << status << "\n";

	m_callTree.dumpTrace(alloc.m_traceId, stream);
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
