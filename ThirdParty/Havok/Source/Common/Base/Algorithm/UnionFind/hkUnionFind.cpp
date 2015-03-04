/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Algorithm/UnionFind/hkUnionFind.h>

/* Heres the deal, m_nodes contains all possible nodes in the graph.
 * we make sure all nodes are known before adding edges so that we
 * can do clever preallocation of m_parents.
 *
 * m_parents is a list of indices.
 * Let p = m_parents[i] for any i.
 * p>=0 means that the m_nodes[i] is connected to m_nodes[p]
 * p<0 means that m_nodes[i] is a root. Also m_nodes[i] has -p child nodes
 */


/*************************************************************** HELPER */


/*************************************************************** HELPER */

/*
#ifdef HK_DEBUG

hkOstream& operator << (hkOstream& os, const intArray& parents );

hkOstream& operator << (hkOstream& os, const intArray& parents )
{
	int* c = parents.begin();
	int* e = parents.end();
	os << "[ ";
	for( ; c!=e; ++c)
	{
		os << *c << " ";
	}
	os << "]\n\n";

	return os;
}

#endif //HK_DEBUG
*/

/*************************************************************** CONSTRUCT DESTRUCT */

hkUnionFind::hkUnionFind( IntArray& parents, int numnode )
:	m_parents(parents)
{
	m_numNodes = numnode;
	clear();
}

//
//	Clears the array of parents

void hkUnionFind::clear()
{
	for(int i = 0; i < m_numNodes; ++i)
	{
		m_parents[i] = -1;
	}

	m_isCollapsed = true;
	m_numRoots = -1;
	m_isAddingEdgesDebug = false;
}


/*************************************************************** PRIVATE FUNC */

// do path compression on the way up the tree
HK_FORCE_INLINE int hkUnionFind::_findRootOfNode(int i)
{
	// find root
	int root = i;
	while ( 1 )
	{
		if ( m_parents[root] < 0 )
		{
			break;
		}
		root = m_parents[root];
	}

	// set all
	while(m_parents[i]>=0)
	{
		int j = m_parents[i];
		m_parents[i] = root;
		i = j;
	}
	return i;
}

int hkUnionFind::findRootOfNode(int i)
{
	return _findRootOfNode(i);
}

/*************************************************************** PRIVATE FUNC */


// join two roots - make the one earlier in the list the new root
HK_FORCE_INLINE void hkUnionFind::_unionRoots(int r1, int r2)
{
	int n1 = m_parents[r1];
	int n2 = m_parents[r2];

	if(r1 < r2)
	{
		m_parents[r1] += n2;
		m_parents[r2] = r1;
	}
	else
	{
		m_parents[r2] += n1;
		m_parents[r1] = r2;
	}
}

void hkUnionFind::unionRoots(int r1, int r2)
{
	HK_ASSERT2(0xf0234345, r1 != r2 && m_parents[r1]<0 && m_parents[r2] < 0, "Your two input values are not valid" );
	_unionRoots(r1,r2);
}


void hkUnionFind::merge(const hkUnionFind& uf)
{
	beginAddEdges();

	HK_ASSERT2(0xad873711, m_numNodes == uf.m_numNodes, "Size of hkUnionFinds doesn't match.");
	for (int ni = 0; ni < m_numNodes; ni++)
	{
		if (uf.m_parents[ni] >= 0)
		{
			HK_ASSERT2(0xad374111, uf.m_parents[ni] != ni, "Internal check.");
			addEdge(ni, uf.m_parents[ni]);
		}
	}

	endAddEdges();
}





/*************************************************************** ADDEDGE */

void hkUnionFind::addEdge( int i1, int i2 )
{
	HK_ASSERT2( 0x2a09ea54, m_isAddingEdgesDebug != hkFalse32, "beginAddEdges must be called before addEdge."); 

	const int r1 = _findRootOfNode(i1);
	const int r2 = _findRootOfNode(i2);

	HK_ASSERT(0x2a09ea55, m_parents[r1]<0 && m_parents[r2]<0);

	if(r1!=r2)
	{
		_unionRoots(r1, r2);
	}
}

/*************************************************************** COLLAPSE */

int hkUnionFind::collapseTree()
{
	HK_ASSERT2( 0x2a09ea54, m_isAddingEdgesDebug == hkFalse32, "forgot to call endAddEdges?."); 

	if ( m_isCollapsed )
	{
		return m_numRoots;
	}
	
	// collapse nodes
	int rootCount = m_numNodes;
	int* c = m_parents.begin();
	int* e = c + m_numNodes;
	
	for(; c!=e; ++c)
	{
		if(*c>=0)
		{
			--rootCount;
			while( m_parents[*c]>=0 )
			{
				*c = m_parents[*c];
			}
		}
	}

	m_isCollapsed = 1;
	m_numRoots = rootCount;
	return m_numRoots;
}


hkResult hkUnionFind::assignGroups( hkArray<int>& elementsPerGroup )
{
	int numGroups = collapseTree();
	
	hkResult res = elementsPerGroup.reserve( numGroups );
	if (res != HK_SUCCESS)
	{
		return HK_FAILURE;
	}

	// populate the groups array
	int groupIndex = 0;
	for (int i = 0; i < m_numNodes; ++i)
	{
		int parent = m_parents[i];
		if ( parent < 0)
		{
			// a new root node - add a new group
			elementsPerGroup.pushBackUnchecked( -parent );
			m_parents[i] = groupIndex++;
		}
		else
		{
			int group = m_parents[parent];
			m_parents[i] = group;
			HK_ASSERT(0x27fc55e0, group >=0 && group < groupIndex);
		}
	}
	return HK_SUCCESS;
}




int hkUnionFind::moveBiggestGroupToIndexZero( hkArray<int>& elementsPerGroup )
{
	// resort the data so that the biggest group is zero
	int biggestSize = elementsPerGroup[0];
	int biggestIndex = 0;
	int ngroups = elementsPerGroup.getSize();
	{
		for (int i = 1; i < ngroups;i++)
		{
			if ( elementsPerGroup[i] > biggestSize)
			{
				biggestSize = elementsPerGroup[i];
				biggestIndex = i;
			}
		}
	}

	if ( biggestIndex == 0)
	{
		return 0;
	}

	// now swap 0 with biggest index
	hkArray<int>::Temp rindex(ngroups);
	{
		for (int i = 0; i < ngroups;i++)
		{
			rindex[i] = i;
		}
		rindex[0] = biggestIndex;
		rindex[biggestIndex] = 0;
		int h = elementsPerGroup[ biggestIndex ];
		elementsPerGroup[ biggestIndex ] = elementsPerGroup[0];
		elementsPerGroup[0] = h;
	}

	for (int i = 0; i < m_numNodes;i++)
	{
		int oldGroup = m_parents[i];
		int newGroup = rindex[oldGroup];
		m_parents[i] = newGroup;
	}
	return biggestIndex;
}

void hkUnionFind::reindex( const hkFixedArray<int>& rindex, int numNewGroups, hkArray<int>& elementsPerGroup )
{
	HK_ASSERT( 0xf0322345, rindex.getSizeDebug() == elementsPerGroup.getSize() );
	for (int i = 0; i < m_numNodes;i++)
	{
		int oldGroup = m_parents[i];
		int newGroup = rindex[oldGroup];
		m_parents[i] = newGroup;
	}

	hkArray<int>::Temp newSizes( numNewGroups );
	{ for (int i = 0; i < numNewGroups; i++){	 newSizes[i] = 0;	} }

	{
		for (int i = 0; i < elementsPerGroup.getSize(); i++)
		{
			int newGroup = rindex[i];
			newSizes[newGroup] += elementsPerGroup[i];
		}
	}

	elementsPerGroup.setSize( numNewGroups );
	{ for (int i = 0; i < numNewGroups; i++){	 elementsPerGroup[i] = newSizes[i];	} }

}

hkResult hkUnionFind::sortByGroupId(const hkArray<int>& elementsPerGroup,hkArray<int>& orderedGroups) const
{
    hkResult groupsRes = orderedGroups.trySetSize(m_numNodes);
	if (groupsRes != HK_SUCCESS)
	{
		return HK_FAILURE;
	}

    const int numGroups = elementsPerGroup.getSize();
    if (numGroups <= 0) return HK_SUCCESS;

    //
	hkArray<int>::Temp currentGroupIndexArray;
	hkResult arrayRes = currentGroupIndexArray.trySetSize(numGroups);
	if (arrayRes != HK_SUCCESS)
	{
		return HK_FAILURE;
	}

    // set up the start indices for each group
    int* currentGroupIndex = currentGroupIndexArray.begin();
    int index = 0;
    for (int i = 0; i < numGroups; i++)
    {
        currentGroupIndex[i] = index;
        index += elementsPerGroup[i];
    }

    // Where the indices are going to be written to
    int* orderedOut = orderedGroups.begin();

    // The ground index each element is in
    const int* elementGroupIndex = m_parents.begin();

    for (int i = 0; i < m_numNodes; i++)
    {
        // Get the group
        int group = elementGroupIndex[i];
        // Get the index to output to
        int outIndex = currentGroupIndex[group];
        // Update the current index
        currentGroupIndex[group] = outIndex + 1;

        //
        orderedOut[outIndex] = i;
    }

	return HK_SUCCESS;
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
