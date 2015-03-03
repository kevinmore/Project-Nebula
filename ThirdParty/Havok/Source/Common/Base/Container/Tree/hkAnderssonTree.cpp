/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Container/Tree/hkAnderssonTree.h>

hkAATree::hkAATree( compareFunc cmp, cloneFunc dup, destroyFunc rel )
	: m_cmp(cmp)
	, m_dup(dup)
	, m_rel(rel)
	, m_size(0)
{
	// Initialize the sentinel
	// every tree instance gets a different one
	m_nil = new Node;
	HK_ASSERT2(0x9476df4a,m_nil,"out of memory allocating sentinel\n");

	m_nil->m_data = HK_NULL; // Simplifies some ops
	m_nil->m_level = 0;
	m_nil->m_link[0] = m_nil->m_link[1] = m_nil;

	// Initialize the tree
	m_root = m_nil;
}

hkAATree::~hkAATree()
{
	clear();

	// Finalize destruction
	delete m_nil;
}

void hkAATree::clear()
{
	Node* it = m_root;
	Node* save;

	// Destruction by rotation
	while ( it != m_nil ) 
	{
		if ( it->m_link[0] == m_nil ) 
		{
			// Remove node
			save = it->m_link[1];
			m_rel ( it->m_data );
			delete it;
		}
		else 
		{
			// Rotate right
			save = it->m_link[0];
			it->m_link[0] = save->m_link[1];
			save->m_link[1] = it;
		}

		it = save;
	}

	m_root = m_nil;
}

void* hkAATree::find( void* data ) const
{
	Node* it = m_root;

	while ( it != m_nil ) 
	{
		int cmp = m_cmp ( it->m_data, data );

		if ( cmp == 0 )
			break;

		it = it->m_link[cmp < 0];
	}

	// nil->data == NULL
	return it->m_data;
}

// Insertion starts with a normal basic binary search tree insertion, after which 
// rebalancing is performed. If a right child increases in level (either because 
// the new node was inserted to its left, or because its left child's level increased, 
// or because it got a new left child with a higher level) then the possibility of 2 
// consecutive red nodes is handled with split (which may in turn cause the level of 
// the new parent to increase). If a node's level should increase, a skew is performed. 
// Then care must be taken because either the node's old parent (now child) or its 
// old sibling (new grandchild) could have been a red node. Then we are back to the 
// above situation where a split must be performed.
hkBool hkAATree::insert( void* data )
{
	if ( m_root == m_nil ) 
	{
		// Empty tree case
		m_root = newNode( data );
		if ( m_root == m_nil )
			return false;
	}
	else 
	{
		Node* it = m_root;
		Node* path[HK_AA_TREE_HEIGHT_LIMIT];
		int top = 0, dir;

		// Find a spot and save the path
		for ( ; ; ) 
		{
			path[top++] = it;
			dir = (m_cmp ( it->m_data, data ) < 0);

			if ( it->m_link[dir] == m_nil )
				break;

			it = it->m_link[dir];
		}

		// Create a new item
		it->m_link[dir] = newNode( data );
		if ( it->m_link[dir] == m_nil )
			return false;

		// Walk back and rebalance
		while ( --top >= 0 ) 
		{
			// Which child?
			if ( top != 0 )
				dir = (path[top - 1]->m_link[1] == path[top]);

			skew ( &(path[top]) );
			split( &(path[top]) );

			// Fix the parent
			if ( top != 0 )
				path[top - 1]->m_link[dir] = path[top];
			else
				m_root = path[top];
		}
	}

	++m_size;

	return true;
}

/*
// If the node is not a leaf node, we find one that appears just before or  
// just after it in the ordering. The properties of an AA tree guarantees that 
// it is possible. Then we swap the two and delete the leaf. For a left child, 
// its parent's level needs to decrease and for a right child, we must make sure
// there are no double red nodes. The difficult case occurs when seven elements 
// (a-b-c-d-e-f-g) are stored in the tree below and we want to delete 'a'.
// The numbers indicate the levels.
//     b,2
//    /   \ .
// a,1   e,2
//       /  \ . 
//     c,1  f,1
//       \    \ . 
//       d,1  g,1
*/
hkBool hkAATree::erase( void *data )
{
	if ( m_root == m_nil )
		return false;
	else 
	{
		Node* it = m_root;
		Node* path[HK_AA_TREE_HEIGHT_LIMIT];
		int top = 0, dir = 0, cmp;

		// Find node to remove and save path
		for ( ; ; ) 
		{
			path[top++] = it;

			if ( it == m_nil )
				return false;

			cmp = m_cmp( it->m_data, data );
			if ( cmp == 0 )
				break;

			dir = (cmp < 0);
			it = it->m_link[dir];
		}

		// Remove the found node
		if ( (it->m_link[0] == m_nil) || (it->m_link[1] == m_nil) )
		{
			// Single child case
			int dir2 = (it->m_link[0] == m_nil);

			// Unlink the item
			if ( --top != 0 )
				path[top - 1]->m_link[dir] = it->m_link[dir2];
			else
				m_root = it->m_link[1];

			m_rel( it->m_data );
			delete it;
		}
		else 
		{
			// Two child case
			Node* heir = it->m_link[1];
			Node* prev = it;

			while ( heir->m_link[0] != m_nil ) 
			{
				path[top++] = prev = heir;
				heir = heir->m_link[0];
			}

			// Order is important! (free item, replace item, free heir)
			m_rel( it->m_data );
			it->m_data = heir->m_data;
			prev->m_link[prev == it] = heir->m_link[1];
			delete heir;
		}

		// Walk back up and rebalance
		while ( --top >= 0 ) 
		{
			Node* up = path[top];

			if ( top != 0 )
				dir = (path[top - 1]->m_link[1] == up);

			// Rebalance (aka. black magic)
			if ( (up->m_link[0]->m_level < (up->m_level - 1)) || (up->m_link[1]->m_level < (up->m_level - 1)) )
			{
				if ( up->m_link[1]->m_level > --up->m_level )
					up->m_link[1]->m_level = up->m_level;

				// Order is important!
				skew ( &up );
				skew ( &(up->m_link[1]) );
				skew ( &(up->m_link[1]->m_link[1]) );
				split( &up );
				split( &(up->m_link[1]) );
			}

			// Fix the parent
			if ( top != 0 )
				path[top - 1]->m_link[dir] = up;
			else
				m_root = up;
		}
	}

	--m_size;

	return true;
}

// First step in traversal, handles min and max
void* hkAATree::Iterator::start( hkAATree* tree, int dir )
{
	m_tree = tree;
	m_it = tree->m_root;
	m_top = 0;

	// Build a path to work with
	if ( m_it != tree->m_nil ) 
	{
		while ( m_it->m_link[dir] != tree->m_nil ) 
		{
			m_path[m_top++] = m_it;
			m_it = m_it->m_link[dir];
		}
	}

	// Could be nil, but nil->data == NULL
	return m_it->m_data;
}

// Subsequent traversal steps, handles ascending and descending
void* hkAATree::Iterator::move( int dir )
{
	Node* nil = m_tree->m_nil;

	if ( m_it->m_link[dir] != nil ) 
	{
		// Continue down this branch
		m_path[m_top++] = m_it;
		m_it = m_it->m_link[dir];

		while ( m_it->m_link[!dir] != nil ) 
		{
			m_path[m_top++] = m_it;
			m_it = m_it->m_link[!dir];
		}
	}
	else 
	{
		// Move to the next branch
		Node* lastNode;

		do {
			if ( m_top == 0 ) 
			{
				m_it = nil;
				break;
			}

			lastNode = m_it;
			m_it = m_path[--m_top];
		} while ( lastNode == m_it->m_link[dir] );
	}

	// Could be nil, but nil->data == NULL
	return m_it->m_data;
}

int   HK_CALL hkAATree::defaultCompare(const void* p1, const void* p2)
{
	return int(int((hkUlong)p1) - int((hkUlong)p2));
}

void* HK_CALL hkAATree::defaultClone(void* p)
{
	return p;
}

void  HK_CALL hkAATree::defaultDestroy(void* p)
{
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
