/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/System/StackTracer/hkStackTracer.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

hkStackTracer::CallTree::~CallTree() 
{
	quit();
}

void hkStackTracer::CallTree::quit() 
{ 
	if (m_allocator)
	{
		m_nodes._clearAndDeallocate(*m_allocator); 
		m_allocator = HK_NULL; 
		m_rootNode = -1;
		m_firstFreeNode = -1;
	}
}

hkStackTracer::CallTree::TraceId hkStackTracer::CallTree::insertCallStack( const hkUlong* addrs, int numAddrs )
{
	if( m_rootNode == -1 )
	{
		m_rootNode = getFreeNode();
	}
	int cur = m_rootNode;
	// stack traces are from the leaf backwards - we need the opposite way to get good compression
	for( int addrIndex = numAddrs-1; addrIndex >=0; --addrIndex )
	{
		hkUlong addr = addrs[addrIndex];
		int child = -1;
		for( int i = m_nodes[cur].m_firstChild; i > 0; i = m_nodes[i].m_next )
		{
			if( m_nodes[i].m_value == addr )
			{
				child = i;
				break;
			}
		}
		if( child == -1 )
		{
			child = getFreeNode();
			Node& n = m_nodes[child];
			n.m_value = addr;
			n.m_parent = cur;
			n.m_firstChild = -1;
			n.m_next = m_nodes[cur].m_firstChild;
			m_nodes[cur].m_firstChild = child;
		}
		cur = child;
	}
	// Only increment the usage count for the leaf address.
	++m_nodes[cur].m_usageCount;
	return cur;
}

void hkStackTracer::CallTree::getTraces(hkArray<hkUlong>& addrs, hkArray<int>& parents) const
{
	const int size = m_nodes.getSize();

	addrs.setSize(size);
	parents.setSize(size);

	if (size == 0)
	{
		return;
	}

	// Invalidate
	addrs[0] = 0;
	parents[0] = -1;

	for (int i = 1; i < size; i++)
	{
		const Node& n = m_nodes[i];

		addrs[i] = n.m_value;
		parents[i] = n.m_parent;
	}
}

static int capacityOrFlag(int cap)
{
	return cap ? cap : hkArrayBase<char>::DONT_DEALLOCATE_FLAG;
}

void hkStackTracer::CallTree::swap(hkStackTracer::CallTree& rhs)
{
	// Swap the nodes arrays... hamfisted because can't do a swap with hkArrayBase
	{
		Node* nodes = m_nodes.begin();
		int nodesSize = m_nodes.getSize();
		int nodesCapacity = m_nodes.getCapacity();

		m_nodes._setDataUnchecked(rhs.m_nodes.begin(), rhs.m_nodes.getSize(), capacityOrFlag(rhs.m_nodes.getCapacity()) );
		rhs.m_nodes._setDataUnchecked(nodes, nodesSize, capacityOrFlag(nodesCapacity) );
	}

	hkAlgorithm::swap(m_rootNode, rhs.m_rootNode);
	hkAlgorithm::swap(m_firstFreeNode, rhs.m_firstFreeNode);
	hkAlgorithm::swap(m_allocator, rhs.m_allocator);
}

hkStackTracer::CallTree::CallTree(const CallTree& rhs):
	m_allocator(rhs.m_allocator)
{
	m_nodes._insertAt(*m_allocator, 0, rhs.m_nodes.begin(), rhs.m_nodes.getSize());
	m_rootNode = rhs.m_rootNode;
	m_firstFreeNode = rhs.m_firstFreeNode;
}

void hkStackTracer::CallTree::operator=(const CallTree& rhs)
{
	m_nodes._clearAndDeallocate(*m_allocator);

	m_allocator = rhs.m_allocator;
	m_nodes._insertAt(*m_allocator, 0, rhs.m_nodes.begin(), rhs.m_nodes.getSize());
	m_rootNode = rhs.m_rootNode;
	m_firstFreeNode = rhs.m_firstFreeNode;
}


int hkStackTracer::CallTree::getCallStack( TraceId id, hkUlong* addrs, int maxAddrs ) const
{
	int i = 0;
	int cur = id;
	while( i < maxAddrs && cur > 0 )
	{
		addrs[i++] = m_nodes[cur].m_value;
		cur = m_nodes[cur].m_parent;
	}
	return i;
}

int hkStackTracer::CallTree::getCallStackSize(TraceId id) const
{
	int size = 0;

	int cur = id;
	while( cur > 0 )
	{
		size++;
		cur = m_nodes[cur].m_parent;
	}

	return size;
}

static void HK_CALL _dumpStackTrace(const char* text, void* context)
{
	hkOstream& stream = *(hkOstream*)context;
	stream << text;
}

void hkStackTracer::CallTree::dumpTrace(TraceId id, hkOstream& stream) const
{
	const int maxNumAddrs = 16;
	hkUlong addrs[maxNumAddrs];

	if (id >= 0)
	{
		const int numAddr = getCallStack( id, addrs, maxNumAddrs);

		hkStackTracer tracer;
		tracer.dumpStackTrace( addrs, numAddr, _dumpStackTrace, &stream);
	}
	else
	{
		stream << "No stack trace\n";
	}
}

hkStackTracer::CallTree::TraceId hkStackTracer::CallTree::insertCallStack( hkStackTracer& tracer )
{
	hkUlong trace[128];
	int numTrace = tracer.getStackTrace(trace, HK_COUNT_OF(trace));
	if( numTrace > 0 )
	{
		return insertCallStack(&trace[1], numTrace-1); // skip this
	}
	return -1;
}

void hkStackTracer::CallTree::releaseCallStack( TraceId id )
{
	if( id >= 0 )
	{
		--m_nodes[id].m_usageCount;
		removeNodeIfUnused( id );
	}
}

void hkStackTracer::CallTree::removeNodeIfUnused( int id )
{
	Node& n = m_nodes[id];
	if ( ( n.m_usageCount == 0 ) && ( n.m_firstChild == -1 ) )
	{
		if ( n.m_parent != -1 )
		{
			// Remove it from the children of its parent
			int* i = &m_nodes[n.m_parent].m_firstChild;
			while ( *i != id )
			{
				i = &m_nodes[*i].m_next;
			} 
			*i = n.m_next;

			// Recursively check the parent
			removeNodeIfUnused( n.m_parent );
		}
		else
		{
			// There is now no root node.
			m_rootNode = -1;
		}

		// Add the node to free list
		n.m_next = m_firstFreeNode;
		m_firstFreeNode = id;
	}
}

int hkStackTracer::CallTree::getFreeNode()
{
	int freeNode;
	if ( m_firstFreeNode != -1 )
	{
		freeNode = m_firstFreeNode;
		m_firstFreeNode = m_nodes[freeNode].m_next;
		
		m_nodes[freeNode].m_value = 0;
		m_nodes[freeNode].m_parent = -1;
		m_nodes[freeNode].m_firstChild = -1;
		m_nodes[freeNode].m_next = -1;
		m_nodes[freeNode].m_usageCount = 0;
	}
	else
	{
		freeNode = m_nodes.getSize();
		m_nodes._expandOne(*m_allocator);
		m_nodes[freeNode].m_value = 0;
		m_nodes[freeNode].m_parent = -1;
		m_nodes[freeNode].m_firstChild = -1;
		m_nodes[freeNode].m_next = -1;
		m_nodes[freeNode].m_usageCount = 0;
	}
	return freeNode;
}

hkBool hkStackTracer::CallTree::isEmpty() const
{
	return m_rootNode == -1;
}

#if 0
void print()
{
	std::vector<int> todoNow;
	std::vector<int> todoNext;
	todoNow.push_back(0);
	while( todoNow.size() )
	{
		for( int i = 0; i < (int)todoNow.size(); ++ i )
		{
			for( int j = todoNow[i]; j >= 0;  )
			{
				const Node& n = m_nodes[j];
				printf("%i ", n.m_value);
				j = n.m_next;
				if( n.m_firstChild > 0 )
				{
					todoNext.push_back( n.m_firstChild );
				}
			}

		}
		printf("\n");
		todoNow.swap( todoNext );
		todoNext.clear();
	}
}
#endif

#if defined(HK_PLATFORM_WIN32) && defined(HK_COMPILER_MSVC) && (HK_COMPILER_MSVC_VERSION >= 1300) && !defined(HK_PLATFORM_WINRT) && !defined(HK_PLATFORM_DURANGO)
#	include <Common/Base/System/StackTracer/Impl/hkStackTracerWin32.cxx> // uses Imagehlp, does not compile in VC6 at the moment
#elif (defined(HK_PLATFORM_LINUX) || defined(HK_PLATFORM_PSP)) && !defined(HK_PLATFORM_NACL)
	// backtrace does not work for mwerks HVK-1542
#	include <Common/Base/System/StackTracer/Impl/hkStackTracerGnu.cxx>
#elif defined(HK_ARCH_IA32) && !defined(HK_PLATFORM_NACL) // Xbox and VC6
#	include <Common/Base/System/StackTracer/Impl/hkStackTracerIa32.cxx>
#elif defined (HK_PLATFORM_XBOX360) && defined(HK_DEBUG)
#	include <Common/Base/System/StackTracer/Impl/hkStackTracerXBox360.cxx>
#elif defined (HK_PLATFORM_PS3_PPU)
#	include <Common/Base/System/StackTracer/Impl/hkStackTracerPs3.cxx>
#elif defined (HK_PLATFORM_PSVITA)
#	include <Common/Base/System/StackTracer/Impl/hkStackTracerVita.cxx>
#else
#	include <Common/Base/System/StackTracer/Impl/hkStackTracerNull.cxx>
#endif

#ifndef HK_STACKTRACER_EXTRA_DEFINED

void* hkStackTracer::getImplementation()
{
	return HK_NULL;
}

void hkStackTracer::replaceImplementation(void* newImpl)
{
}

void hkStackTracer::setNeedsSymInitialize(bool enabled)
{
}

#endif

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
