/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Allocator/LargeBlock/hkLargeBlockAllocator.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/Memory/System/Util/hkMemorySnapshot.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

/*
Possible future improvements:

o Support for realloc style constructs
  + For example returning if a block can be extended, and its current actual size, extending into a freeblock
o Ideally there would be a realloc in there somewhere
o Debugging support
  + Having headers and footers
*/

hkLargeBlockAllocator::FixedMemoryBlockServer::FixedMemoryBlockServer(void* startIn, int size)
{
	const unsigned ALIGN = 16;
	char* start = ((char*)startIn);
	char* end = start + size;

	// Align them
	m_start = (char*)(hk_size_t(start + ALIGN-1) & ~hk_size_t(ALIGN - 1));
	m_end = (char*)(hk_size_t(end) & ~hk_size_t(ALIGN - 1));
	m_limit = m_end;
	m_break = m_start;
}

void* hkLargeBlockAllocator::FixedMemoryBlockServer::bufAlloc(int& size)
{
	/// If there is a break then we can allocate no more
	if (m_break != m_start)
	{
		return HK_NULL;
	}

	int totalSize = int(m_limit-m_start);
	if (totalSize < size)
	{
		return HK_NULL;
	}

	m_break = m_start + totalSize;

	size = totalSize;
	return (void*)m_start;
}

bool hkLargeBlockAllocator::FixedMemoryBlockServer::resize(void* data, int oldSize, hk_size_t newSize, hk_size_t& sizeOut)
{
	// Something must have been allocated, and it must start at the start of this block
	HK_ASSERT(0x32432432, data == (void*)m_start);
	HK_ASSERT(0x34234234, hk_size_t(m_break - m_start) == hk_size_t(oldSize));

	// Eh? We've given a block that doesn't belong to us
	if ((void*)m_start != data)
	{
		return false;
	}

	char* newBreak = m_start + newSize;
	// If its outside the range something has gone wrong
	if (newBreak < m_start || newBreak >m_limit)
	{
		return false;
	}
	/// Else thats out new break
	m_break = newBreak;
	sizeOut = newBreak - m_start;
	return true;
}

void hkLargeBlockAllocator::FixedMemoryBlockServer::bufFree(void* data, int size)
{
	if (m_break == m_start || int(m_break-m_start) != size)
	{
		// Eh? Nothings been allocated so how can I free it
		HK_BREAKPOINT(0);
		return;
	}
	HK_ASSERT(0x3424234, data == m_start);
	// Say its all deallocated
	m_break = HK_NULL;
}

void hkLargeBlockAllocator::FixedMemoryBlockServer::getMemoryStatistics(MemoryStatistics& stats) const
{
	stats.m_allocated = int(m_break - m_start);
	stats.m_inUse = stats.m_allocated;
	stats.m_peakInUse = stats.m_allocated;
	stats.m_available = int(m_end - m_break);
	stats.m_totalAvailable = stats.m_available;
	stats.m_largestBlock = int(m_end - m_start);
}

/*

Description of the underlying structure
----------------------------------------

Based on large block strategy of the Doug Lea allocator.

Larger chunks are kept in a form of bitwise digital trees (aka
tries) keyed on chunksizes.  Because 'tree chunks' are only for
free chunks greater than 256 bytes, their size doesn't impose any
constraints on user chunk sizes.  Each node looks like:

chunk->     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |             Size of previous chunk                            |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
`head:'     |             Size of chunk, in bytes                         |P|
mem->       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |             Forward pointer to next chunk of same size        |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |             Back pointer to previous chunk of same size       |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |             Pointer to left child (child[0])                  |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |             Pointer to right child (child[1])                 |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |             Pointer to parent                                 |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |             bin index of this chunk                           |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |             Unused space                                      .
            .                                                               |
nextchunk-> +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    `foot:' |             Size of chunk, in bytes                           |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Each tree holding treenodes is a tree of unique chunk sizes.  Chunks
of the same size are arranged in a circularly-linked list, with only
the oldest chunk (the next to be used, in our FIFO ordering)
actually in the tree.  (Tree members are distinguished by a non-null
parent pointer.)  If a chunk with the same size an an existing node
is inserted, it is linked off the existing node using pointers that
work in the same way as fd/bk pointers of small chunks.

Each tree contains a power of 2 sized range of chunk sizes (the
smallest is 0x100 <= x < 0x180), which is is divided in half at each
tree level, with the chunks in the smaller half of the range (0x100
<= x < 0x140 for the top nose) in the left subtree and the larger
half (0x140 <= x < 0x180) in the right subtree.  This is, of course,
done by inspecting individual bits.

Using these rules, each node's left subtree contains all smaller
sizes than its right subtree.  However, the node at the root of each
subtree has no particular ordering relationship to either.  (The
dividing line between the subtree sizes is based on trie relation.)
If we remove the last chunk of a given size from the interior of the
tree, we need to replace it with a leaf node.  The tree ordering
rules permit a node to be replaced by any leaf below it.

The smallest chunk in a tree (a common operation in a best-fit
allocator) can be found by walking a path to the leftmost leaf in
the tree.  Unlike a usual binary tree, where we follow left child
pointers until we reach a null, here we follow the right child
pointer any time the left one is null, until we reach a leaf with
both child pointers null. The smallest chunk in the tree will be
somewhere along that path.

The worst case number of steps to add, find, or remove a node is
bounded by the number of bits differentiating chunks within
bins. Under current bin calculations, this ranges from 6 up to 21
(for 32 bit sizes) or up to 53 (for 64 bit sizes). The typical case
is of course much better.
*/
hkLargeBlockAllocator::hkLargeBlockAllocator(hkMemoryAllocator* server):
	m_limitedListener(HK_NULL),
	m_fixedBlockServer(HK_NULL, 0),
	m_server(server)
{
    _init();
}

hkLargeBlockAllocator::hkLargeBlockAllocator(void* block, int size):
	m_limitedListener(HK_NULL),
	m_fixedBlockServer(block, size),
	m_server(&m_fixedBlockServer)
{
	_init();
}

hkLargeBlockAllocator::~hkLargeBlockAllocator()
{
    MemPage* cur = m_pages.m_next;
    MemPage* end = &m_pages;

    while (cur != end)
    {
        MemPage* next = cur->m_next;
        // Free the allocation
        m_server->bufFree(cur, cur->m_size);
        // Next page
        cur = next;
    }
}

void hkLargeBlockAllocator::_init()
{
    m_sumAllocatedSize = 0;
	m_sumAllocatedWithMgrOverhead = 0;

    m_treemap = 0;
    m_leastAddr = HK_NULL;

    for (int i = 0; i <int(sizeof(m_treebins) / sizeof(m_treebins[0])); i++)
    {
        m_treebins[i] = HK_NULL;
    }

    m_zeroChunk.head = hk_size_t(0 | MemChunk::PINUSE_BIT | MemChunk::CINUSE_BIT);
    m_zeroChunk.prevFoot = hk_size_t(0);

	// Its empty
    m_top = &m_zeroChunk;
    m_topsize = hk_size_t(0);

	m_pages.m_next = HK_NULL;
	m_pages.m_prev = HK_NULL;

	// Set up the pages list
    m_pages.m_next = &m_pages;
    m_pages.m_prev = &m_pages;
}


void hkLargeBlockAllocator::freeAll()
{
    MemPage* cur = m_pages.m_next;
    MemPage* end = &m_pages;

    while (cur != end)
    {
        MemPage* next = cur->m_next;
        // Free the allocation
        m_server->bufFree(cur, cur->m_size);
        // Next page
        cur = next;
    }

    //
    _init();
}

void hkLargeBlockAllocator::_insertLargeChunk( MemTreeChunk* x, hk_size_t s)
{
    BinIndex i = _computeTreeIndex(s);

	// Find the associated bin
    MemTreeChunk** h = &m_treebins[i];

	// Initialize X
    x->index = i;
    x->child[0] = x->child[1] = 0;

    // If the treemap isn't marked, then presumably that part of the tree is empty
    if (!_isTreeMapMarked(i))
    {
        // Mark up the tree
        _markTreeMap(i);
        // Point the treebin to this
        *h = x;
        // parent is not pointing to a parent block but to the tree bin that its currently in
        x->parent = (MemTreeChunk*)h;
        // Make the doubly linked list point to self
        x->next = x;
        x->prev = x;
    }
    else
    {
        MemTreeChunk* t = *h;
        hk_size_t k = s << _leftShiftForTreeIndex(i);
        for (;;)
        {
            if (t->getChunkSize() != s)
            {
                MemTreeChunk** c = &(t->child[(k >> (SIZE_T_BITSIZE- hk_size_t(1))) & 1]);
                k <<= 1;
                if (*c != HK_NULL)
                {
                    t = *c;
                }
                else
                {
                    HK_ASSERT(0x13423432,_isOkAddress(c));

                    *c = x;

                    x->parent = t;
                    x->next = x;
                    x->prev = x;
                    break;
                }
            }
            else
            {
                MemTreeChunk*  f = reinterpret_cast<MemTreeChunk*>(t->next);

                HK_ASSERT(0x32332432,_isOkAddress(t)&&_isOkAddress(f));

                t->next = f->prev = x;
                x->next = f;
                x->prev = t;
                x->parent = HK_NULL;
                break;
            }
        }
    }
}


//  Unlink steps:
//
//  1. If x is a chained node, unlink it from its same-sized fd/bk links
//     and choose its bk node as its replacement.
//  2. If x was the last node of its size, but not a leaf node, it must
//     be replaced with a leaf node (not merely one with an open left or
//     right), to make sure that lefts and rights of descendants
//     correspond properly to bit masks.  We use the rightmost descendant
//     of x.  We could use any other leaf, but this is easy to locate and
//     tends to counteract removal of leftmosts elsewhere, and so keeps
//     paths shorter than minimally guaranteed.  This doesn't loop much
//     because on average a node in a tree is near the bottom.
//  3. If x is the base of a chain (i.e., has parent links) relink
//     x's parent and children to x's replacement (or null if none).

void hkLargeBlockAllocator::_unlinkLargeChunk( MemTreeChunk* x)
{
    MemTreeChunk* xp = x->parent;
    MemTreeChunk* r;

    if (x->prev != x)
    {
        MemTreeChunk* f = reinterpret_cast<MemTreeChunk*>(x->next);
        r = reinterpret_cast<MemTreeChunk*>(x->prev);

        HK_ASSERT(0x34534534,_isOkAddress(f));

        // Detach
        f->prev = r;
        r->next = f;
    }
    else
    {
        MemTreeChunk** rp;
        if (((r = *(rp = &(x->child[1]))) != HK_NULL) ||
            ((r = *(rp = &(x->child[0]))) != HK_NULL))
        {
            MemTreeChunk** cp;
            while ((*(cp = &(r->child[1])) != 0) ||
                    (*(cp = &(r->child[0])) != 0))
            {
                r = *(rp = cp);
            }

            HK_ASSERT(0x34233434,_isOkAddress(rp));
            *rp = HK_NULL;
        }
    }


    if (xp != HK_NULL)
    {
        MemTreeChunk** h = &m_treebins[x->index];
        if (x == *h)
        {
            if ((*h = r) == HK_NULL)
			{
				_clearTreeMap( x->index);
			}
        }
        else
        {
            HK_ASSERT(0x543636,_isOkAddress(xp));
            if (xp->child[0] == x)
			{
                xp->child[0] = r;
			}
            else
			{
                xp->child[1] = r;
			}
        }

        if (r != HK_NULL)
        {
            HK_ASSERT(0x4534543,_isOkAddress(r));

            MemTreeChunk* c0, *c1;
            r->parent = xp;
            if ((c0 = x->child[0]) != HK_NULL)
            {
                HK_ASSERT(0x3234234,_isOkAddress(c0));
                r->child[0] = c0;
                c0->parent = r;
            }

            if ((c1 = x->child[1]) != HK_NULL )
            {
                HK_ASSERT(0x3435435,_isOkAddress(c1));
                r->child[1] = c1;
                c1->parent = r;
            }
        }
    }
}

// allocate a large request from the best fitting chunk in a treebin
void* hkLargeBlockAllocator::_allocLarge( hk_size_t nb)
{
    MemChunk* v = HK_NULL;

	/// This basically making the rsize, invalidly large, so the first valid block will be a fit
    hk_size_t rsize = hk_size_t(-((SignedSizeT)nb));


    int idx = _computeTreeIndex(nb);

    MemTreeChunk* t = m_treebins[idx];

    if (t)
    {
		// Traverse tree for this bin looking for node with size == nb
        hk_size_t sizebits = nb << _leftShiftForTreeIndex(idx);
		// The deepest untaken right subtree

        MemTreeChunk* rst = HK_NULL;
        for (;;)
        {

			// The amount of bytes at the end of the current chunk being examined
            hk_size_t trem = t->getChunkSize() - nb;
			// See if we have a better match
            if (trem < rsize)
            {
				// Save in v the best fit
                v = t;
				// If its an exact match we are done
                if ((rsize = trem) == 0)
				{
					break;
				}
            }

            MemTreeChunk* rt = t->child[1];

			// The shift here will return 0, unless the MSB is hit
            t = t->child[(sizebits >> (SIZE_T_BITSIZE- hk_size_t(1))) & 1];

			if (rt != HK_NULL && rt != t)
			{
				rst = rt;
			}

            if (t == HK_NULL)
            {
				// set t to least subtree holding sizes > nb
                t = rst;
                break;
            }
			// shift up
            sizebits += sizebits;
        }
    }

    if (t == HK_NULL && v == HK_NULL)
    {
        // set t to root of next non-empty treebin
        BinMap leftBits = _leftBits(_indexToBit(idx)) & m_treemap;

        if (leftBits != 0)
        {
            BinMap leastbit = _leastBit(leftBits);
            BinIndex i = _bitToIndex(leastbit);
            t = m_treebins[i];
        }
    }

    while (t != HK_NULL)
    {
        // find smallest of tree or subtree
        hk_size_t trem = t->getChunkSize() - nb;
        if (trem < rsize)
        {
            rsize = trem;
            v = t;
        }
        t = t->leftMostChild();
    }

	// If we couldn't find a block fail
	if (v== HK_NULL)
	{
		return HK_NULL;
	}

    //  If dv is a better fit, return 0 so malloc will use it
    //if (v != HK_NULL && rsize < (hk_size_t)(m_dvsize - nb))
    {
        HK_ASSERT(0x4243244,_isOkAddress(v));

        /* split */
        MemChunk*  r = v->chunkPlusOffset(nb);
        HK_ASSERT(0x3434344,v->getChunkSize() == rsize + nb);
        HK_ASSERT(0x3434342,_okNext(v,r));

        _unlinkLargeChunk(reinterpret_cast<MemTreeChunk*>(v));

		if (rsize < MIN_LARGE_SIZE)
		{
			v->setInuseAndPinuse(rsize + nb);

            // We add the remainder too
            m_sumAllocatedWithMgrOverhead += rsize + nb;
			m_sumAllocatedSize += rsize + nb - MemChunk::PAYLOAD_OFFSET;
		}
		else
		{
			v->setSizeAndPinuseOfInuseChunk(nb);

            // Alter m_used to keep up to date
            m_sumAllocatedWithMgrOverhead += nb;
			m_sumAllocatedSize += nb - MemChunk::PAYLOAD_OFFSET;

			r->setSizeAndPinuseOfFreeChunk(rsize);

			_insertLargeChunk( static_cast<MemTreeChunk*>(r),rsize);
		}

        return v->getPayload();
    }
}

hk_size_t hkLargeBlockAllocator::_findLargestTreeBlockSize(MemTreeChunk* t,hk_size_t largest) const
{
    while (t)
    {
        // The amount of bytes at the end of the current chunk being examined
        hk_size_t chunkSize = t->getChunkSize();

        if (chunkSize > largest) { largest = chunkSize; }

        // Go rightmost if we can
        if (t->child[1])
        {
            t = t->child[1];
            continue;
        }
        // Else try leftmost
        t = t->child[0];
    }
    return largest;
}

hk_size_t hkLargeBlockAllocator::findLargestBlockSize() const
{
    // Assume top is largest
    hk_size_t largest = m_topsize;

    // Theres no point searching buckets smaller than what we have
    int smallestBin = _computeTreeIndex(largest);

    // Lets search the tree
    for (int i = NTREEBINS - 1; i >= smallestBin; i--)
    {
        // Look for a bin with nodes in it
        MemTreeChunk* t = m_treebins[i];

        // Either the block we have or a block in this tree will be the largest
        if (t) 
		{
			return _findLargestTreeBlockSize(t,largest);
		}
    }
    // This must be the largest
    return largest;
}

void hkLargeBlockAllocator::_makeTopValid() const
{
    // Set is size + that its free
    m_top->head = m_topsize | MemChunk::PINUSE_BIT;
    MemChunk* footer = m_top->chunkPlusOffset(m_topsize);
    footer->prevFoot = m_topsize;
}

void* hkLargeBlockAllocator::_allocFromTop(hk_size_t nb)
{
    //HK_ASSERT(0x34243,m_used == _calculateUsed());

    // Allocated
    m_sumAllocatedWithMgrOverhead += nb;
	m_sumAllocatedSize += nb - MemChunk::PAYLOAD_OFFSET;

    // Split top
    hk_size_t rsize = m_topsize -= nb;

    MemChunk* p = m_top;

    MemChunk* r = m_top = p->chunkPlusOffset(nb);

    r->head = rsize | MemChunk::PINUSE_BIT;
    r->prevFoot = nb;

    p->setSizeAndPinuseOfInuseChunk(nb);

    void* mem = p->getPayload();
    //check_top_chunk(gm, gm->top);
    //check_malloced_chunk(gm, mem, nb);

    //HK_ASSERT(0x34243,m_used == _calculateUsed());

    return mem;
}


hkBool hkLargeBlockAllocator::_resizeSingleBlockServerPage(hk_size_t newSize)
{
	HK_ASSERT(0x34234324, _hasPages() && m_fixedBlockServer.inUse() && m_server == &m_fixedBlockServer );
	MemPage* page = m_pages.m_next;

	hk_size_t outSize = 0;
	if( m_fixedBlockServer.resize( page, page->m_size, newSize, outSize ) == false )
	{
		return false;
	}

    // It worked
    hk_size_t footsize = m_top->chunkPlusOffset(m_topsize)->getChunkSize();

    int change = int(outSize) - page->m_size;
    change &= ~MemChunk::ALIGN_MASK;

    m_topsize += change;

    page->m_size = int(outSize);
    page->m_end += change;

    // New footer?
    MemChunk* newFooter = m_top->chunkPlusOffset(m_topsize);
    newFooter->head = (footsize - change) | MemChunk::CINUSE_BIT;

    return true;
}

// nb is the padded amount of bytes

void* hkLargeBlockAllocator::_alloc(hk_size_t nb)
{
    if (m_treemap != 0)
    {
        //HK_ASSERT(0x34243,m_used == _calculateUsed());
        void* mem = _allocLarge(nb);
        if (mem)
        {
            //HK_ASSERT(0x34243,m_used == _calculateUsed());
            return mem;
        }
    }

    // See if there is enough memory in top
    if (nb + MIN_LARGE_SIZE <= m_topsize  )
    {
        //HK_ASSERT(0x34243,m_used == _calculateUsed());
        void* mem = _allocFromTop(nb);
        //HK_ASSERT(0x34243,m_used == _calculateUsed());
        return mem;
    }

    return HK_NULL;
}

void* hkLargeBlockAllocator::blockAlloc(int bytes)
{
    //   1. Find the smallest available binned chunk that fits, and use it
    //   2. If it is big enough, use the top chunk.
    //   3. Use the server to allocate more memory, and try again...

    hk_size_t nb = _padRequest(bytes);

    // This is the minimum allocation size
    if (nb < MIN_LARGE_SIZE) nb = MIN_LARGE_SIZE;

    if( void* p = _alloc(nb) )
	{
		return p;
	}

    // Okay this is where we are going to have to create a new block - if we have a single block server
    // things are substantially simpler
    if( m_fixedBlockServer.inUse() && _hasPages())
    {
        MemPage* page = m_pages.m_next;
        if (_resizeSingleBlockServerPage(page->m_size + nb))
        {
            // Allocate from top
            return _allocFromTop(nb);
        }

        if (m_limitedListener)
		{
			m_limitedListener->cannotAllocate(nb);
			if( void* p = _alloc(nb) )
			{
				return p;
			}

			// Try the resize again?
			if (_resizeSingleBlockServerPage(page->m_size + nb))
			{
				// Allocate from top
				return _allocFromTop(nb);
			}
			// Okay didn't work, time to cry
			m_limitedListener->allocationFailure(nb);
		}

        return HK_NULL;
    }

    // Okay we need a whole new block
    // We need space for both a header and a footer
    hk_size_t neededSize = nb + sizeof(MemPage) + MemChunk::PAYLOAD_OFFSET + MemChunk::ALIGN + MIN_LARGE_SIZE;
    int reqSize = int(neededSize);

	MemPage* newPage = (MemPage*)m_server->bufAlloc(reqSize);
    // If we couldn't create the page then we failed
    if (!newPage)
	{
		if (!m_limitedListener)
		{
			return HK_NULL;
		}

		// This could cause a garbage collect
		m_limitedListener->cannotAllocate(nb);
		if( void* p = _alloc(nb) )
		{
			return p;
		}

		// Work out the size again
		reqSize = int(neededSize);
		// Okay we can try and allocate a page again..
		newPage = (MemPage*)m_server->bufAlloc(reqSize);
		if (!newPage)
		{
			m_limitedListener->allocationFailure(nb);
			return HK_NULL;
		}
    }

    // Set up the page
    newPage->m_numAllocs = 0;
    newPage->m_size = reqSize;
    newPage->m_start = reinterpret_cast<char*>(newPage + 1);
    newPage->m_end = (reinterpret_cast<char*>(newPage)) + reqSize;
    // Align forward
    newPage->m_start = reinterpret_cast<char*>(((hk_size_t(newPage->m_start) + MemChunk::ALIGN_MASK) & ~MemChunk::ALIGN_MASK));
    // Align backward
    newPage->m_end = reinterpret_cast<char*>(((hk_size_t(newPage->m_end)) & ~MemChunk::ALIGN_MASK));

    // Find where we are going to insert

    MemPage* cur = m_pages.m_next;
    while (cur != &m_pages && cur < newPage)
	{
		cur = cur->m_next;
	}

    // we want to add the page before this page, as its either hit the end or the page its on is at a higher address

    newPage->m_next = cur;
    newPage->m_prev = cur->m_prev;
    newPage->m_prev->m_next = newPage;
    cur->m_prev = newPage;

    // If there is a top we need to add it too the tree
    if (m_top != &m_zeroChunk)
    {
        // Sets up stuff so its not all rubbish
        _makeTopValid();
        // Insert so its available
        _insertLargeChunk( static_cast<MemTreeChunk*>(m_top),m_top->getChunkSize());
    }

    // Work out the new top
    m_topsize = newPage->getMaxChunkSize();
    m_top = newPage->getFirstChunk();
    // make it valid
    _makeTopValid();

    // Mark the footer as in use
	{
		MemChunk* footerChunk = newPage->getFooter();
		HK_ASSERT(0x32423432,footerChunk == m_top->chunkPlusOffset(m_topsize));

		footerChunk->head = MemChunk::CINUSE_BIT;
		if (newPage->m_next != &m_pages)
		{
			// There is a page in front so attach to it
			MemChunk* nextChunk = newPage->m_next->getFirstChunk();
			HK_ASSERT(0x32423432,nextChunk->isPinuse());
			hk_size_t footerSize = (char*)nextChunk - (char*)footerChunk;
			HK_ASSERT(0x34234,(footerSize&MemChunk::INUSE_BITS)==0);
			footerChunk->head = MemChunk::CINUSE_BIT|footerSize;
		}
	}
	{
		// There may be a previous we need to sort
		if (newPage->m_prev != &m_pages)
		{
			MemChunk* prevFooter = newPage->m_prev->getFooter();
			hk_size_t chunkSize = (char*)m_top - (char*)prevFooter;
			HK_ASSERT(0x34234,(chunkSize&MemChunk::INUSE_BITS)==0);
			hk_size_t bits = (prevFooter->head&MemChunk::INUSE_BITS);
			prevFooter->head = bits|chunkSize;
			HK_ASSERT(0x34233423,prevFooter->isInuse() &&m_top->isInuse()==false&&m_top->isPinuse() );
		}
	}

        /// We can allocate from the top
    return _allocFromTop(nb);
}

void hkLargeBlockAllocator::incrementalGarbageCollect(int numBlocks)
{
	// Do nothing for now
}

hkResult hkLargeBlockAllocator::setMemorySoftLimit(hk_size_t maxMemory)
{
	return HK_FAILURE;
}

hk_size_t hkLargeBlockAllocator::getMemorySoftLimit() const
{
	return HK_FAILURE;
}

hk_size_t hkLargeBlockAllocator::getApproxTotalAllocated() const
{
	return m_sumAllocatedSize;
}

void hkLargeBlockAllocator::setScrubValues(hkUint32 allocValue, hkUint32 freeValue)
{
	// empty body, it is not passed through to the underlying allocator
}

bool hkLargeBlockAllocator::canAllocTotal( int numBytes )
{
	// Simple - just does the alloc and frees it. Could be done faster, but okay as used here.
	void* data = blockAlloc(numBytes);
	if (data)
	{
		blockFree(data, numBytes);
		return true;
	}

	return false;
}

hkResult hkLargeBlockAllocator::walkMemory(MemoryWalkCallback callback, void* param)
{
	Iterator iter = getIterator();
	for (; iter.isValid(); nextBlock(iter))
	{
		void* block = iter.getAddress();	
		// Do the callback
		callback(block, iter.getSize(), iter.isInuse(), 0, param);
	}

	return HK_SUCCESS;
}

int hkLargeBlockAllocator::addToSnapshot( hkMemorySnapshot& snap, hkMemorySnapshot::ProviderId parentId )
{
	hkMemorySnapshot::ProviderId lbaId = snap.addProvider("hkLargeBlockAllocator", parentId);
	
	// walk the list of blocks we've requested from the system
	for( MemPage* mp = m_pages.m_next; mp != &m_pages; mp = mp->m_next )
	{
		// add system allocation
		snap.addAllocation( parentId, mp, mp->m_size );

		// add memory page management overhead
		MemChunk* firstChunk = mp->getFirstChunk();
		MemChunk* footerChunk = mp->getFooter();
		const void* mpEnd = hkAddByteOffsetConst(mp, mp->m_size);
		snap.addOverhead(lbaId, mp, hkGetByteOffsetInt(mp, firstChunk));
		for(MemChunk* chunk = firstChunk; chunk != footerChunk; chunk = chunk->nextChunk())
		{
			// overhead for the chunk header containing two hk_size_t data members
			snap.addOverhead(lbaId, chunk, MemChunk::PAYLOAD_OFFSET);
			// actual chunk body (which might be free or used)
			hkMemorySnapshot::Status paylodStatus = chunk->isInuse()? hkMemorySnapshot::STATUS_USED : hkMemorySnapshot::STATUS_UNUSED;
			snap.addItem(lbaId, paylodStatus, chunk->getPayload(), static_cast<int>(chunk->getPayloadSize()));
		}
		snap.addOverhead(lbaId, footerChunk, hkGetByteOffsetInt(footerChunk, mpEnd));
	}
	return lbaId;
}

void hkLargeBlockAllocator::garbageCollect()
{
    // If we traverse, we want this set up correctly
    _makeTopValid();

    // Free any pages which are totally not in use
    {
        MemPage* page = m_pages.m_next;
        while (page != &m_pages)
        {
            // Lets see if all this memory is free
            MemChunk* chunk = page->getFirstChunk();
            if ((!chunk->isInuse())&&chunk->nextChunk() == page->getFooter())
            {
                MemPage* next = page->m_next;

                // This is a dead page.. we can free it
                // First delink
                page->m_prev->m_next = page->m_next;
                page->m_next->m_prev = page->m_prev;

				if (chunk == m_top)
				{
					// If we freed the block with the top block, make top block invalid.
					// No need to unlink the block as its never in the tree
					m_top = &m_zeroChunk; // HK_NULL;
					m_topsize = 0;
				}
				else
				{
					// We need to unlink this
					_unlinkLargeChunk((MemTreeChunk*)chunk);
				}

                // Free it
                m_server->bufFree( page, page->m_size );
                // Next
                page = next;
            }
            else
            {
                page = page->m_next;
            }
        }
    }

    if (!_hasPages())
    {
        m_top = &m_zeroChunk; // HK_NULL;
        m_topsize = 0;
        return;
    }

    if( m_fixedBlockServer.inUse() )
    {
        // We need to find the amount of space remaining in the top block
        MemPage* page = m_pages.m_next;

		// No point unless there is enough space
		if (m_topsize < 32*1024)
		{
			return;
		}

        hk_size_t usedSize = (((char*)m_top) + (MemChunk::PAYLOAD_OFFSET*2 + 256)) - page->getStart();
		//
        _resizeSingleBlockServerPage(usedSize);
        return;
    }

    // Lets go looking for a new top block - find the largest block in a page
    // NOTE! This means the top block is not the top of all allocations, merely the top of a page.
    {

		// If there is a top block, we insert into tree whilst we look for a new one
		if (m_top != &m_zeroChunk)
		{
			_insertLargeChunk((MemTreeChunk*)m_top, m_topsize);
			// Say there is no top block currently
			m_top = &m_zeroChunk; // HK_NULL;
			m_topsize = 0;
		}

		// Lets look for the best block

        MemChunk* bestChunk = HK_NULL;

        MemPage* page = m_pages.m_next;
        while (page != &m_pages)
        {
            MemChunk* chunk = page->getFooter();

            if (!chunk->isPinuse())
            {
                MemChunk* prev = chunk->previousChunk();
                if (bestChunk == HK_NULL || prev->getChunkSize() > bestChunk->getChunkSize())
                {
                    bestChunk = prev;
                }
            }
            page = page->m_next;
        }

        if (bestChunk)
        {
			// Need to unlink it - as it must be in the tree
			_unlinkLargeChunk((MemTreeChunk*)bestChunk);

			// Its the new top block
            m_top = bestChunk;
            m_topsize = bestChunk->getChunkSize();
        }
		else
		{
			m_top = &m_zeroChunk;
			m_topsize = 0;
		}
    }
}

hkBool hkLargeBlockAllocator::isValidAlloc(void* in)
{
    if (!MemChunk::isAligned(in))
	{
		return false;
	}
    MemChunk* p = MemChunk::toChunk(in);
    if (!p->isInuse())
	{
		return false;
	}

    // Okay lets see what page its in

    char* charAlloc = reinterpret_cast<char*>(in);

    MemPage* page = m_pages.m_next;
    while (page != &m_pages)
    {
        // Work out if its in the range
        char* start = page->getStart();
        char* end = page->getEnd();

        if (charAlloc >= start && charAlloc < end)
        {
            // Its in this block, but that doesn' make it valid

            MemChunk* cur = (MemChunk*)start;
            MemChunk* footer = (MemChunk*)(end - MemChunk::PAYLOAD_OFFSET);
            while (cur != footer)
            {
                hk_size_t chunkSize = cur->getChunkSize();
                if (cur == p)
				{
					return true;
				}

                // Next block
                cur = cur->chunkPlusOffset(chunkSize);
            }
            // We didn't find it, so its not valid
            return false;
        }
        page = page->m_next;
    }

    return false;
}


void hkLargeBlockAllocator::blockFree(void* mem, int)
{
    HK_ASSERT(0x3423432,MemChunk::isAligned(mem));

    //    Consolidate freed chunks with preceding or succeeding bordering
    //    free chunks, if they exist, and then place in a bin.  Intermixed
    //    with special cases for top.

    if (mem == HK_NULL)
	{
		return;
	}

    //HK_ASSERT(0x34243,m_used == _calculateUsed());

    MemChunk* p = MemChunk::toChunk(mem);

    HK_ASSERT2(0x3434343, _isOkAddress(p), "The pointer is not a valid allocation - the pointer is wrong, or the allocation has already been freed.");
    HK_ASSERT2(0x3443434, p->isInuse(), "The pointer is not a valid allocation - the pointer is wrong, or the allocation has already been freed.");

    hk_size_t psize = p->getChunkSize();

    // We've freed it
    m_sumAllocatedWithMgrOverhead -= psize;
	m_sumAllocatedSize -= p->getPayloadSize();

    //
    MemChunk* next = p->chunkPlusOffset(psize);

    if (!p->isPinuse())
    {
        hk_size_t prevsize = p->prevFoot;

        MemChunk* prev = p->chunkMinusOffset(prevsize);
        HK_ASSERT(0x54325443, _isOkAddress(prev));

        // Consolidate backward
        _unlinkLargeChunk(static_cast<MemTreeChunk*>(prev));

        // Work out the new size
        psize += prevsize;
        p = prev;
    }

    HK_ASSERT2(0x342432, _okNext(p, next) && next->isPinuse(), "The pointer is not a valid allocation - the pointer is wrong, or the allocation has already been freed.");
    if (!next->isInuse())
    {
        // consolidate forward
        if (next == m_top)
        {
            hk_size_t tsize = m_topsize += psize;
            m_top = p;

			p->head = tsize | MemChunk::PINUSE_BIT;

            // Check its not in use
            HK_ASSERT(0x3425425, !p->isInuse());
            //HK_ASSERT(0x34243,m_used == _calculateUsed());

			return;
        }
        else
        {
            hk_size_t nsize = next->getChunkSize();
            psize += nsize;

            // Unlink next
            _unlinkLargeChunk(static_cast<MemTreeChunk*>(next));

            p->setSizeAndPinuseOfFreeChunk(psize);
        }
    }
    else
    {
        p->setFreeWithPinuse(psize, next);
    }

    // Insert the chunk back
    _insertLargeChunk( static_cast<MemTreeChunk*>(p),psize);

    // Check the block is free
    //HK_ASSERT(0x2312321,_checkFreeChunk(p));
    //HK_ASSERT(0x34243,m_used == _calculateUsed());
}

int hkLargeBlockAllocator::getAllocatedSize( const void* mem, int size) const
{
    MemChunk* p = MemChunk::toChunk( const_cast<void*>(mem) );
    HK_ASSERT2(0x34243401, _checkUsedAlloc(mem), "Not a valid allocation");
    return (int)p->getPayloadSize();
}

void hkLargeBlockAllocator::forAllAllocs(MemBlockCallback callback, void* param)
{
	// We should write into top just so we don't have to special case it
	m_top->head = m_topsize | MemChunk::PINUSE_BIT;

	// Okay lets traverse the pages
	MemPage* page = m_pages.m_next;
	while (page != &m_pages)
	{
		MemChunk* cur = (MemChunk*)page->getStart();
		MemChunk* footer = (MemChunk*)(page->getEnd() - MemChunk::PAYLOAD_OFFSET);
		while (cur != footer)
		{
			hk_size_t chunkSize = cur->getChunkSize();
			if (cur->isInuse())
			{
				// Callback with the block
				callback(cur->getPayload(), cur->getPayloadSize(), param);
			}

			// Next block
			cur = cur->chunkPlusOffset(chunkSize);
		}

		// Onto the next page
		page = page->m_next;
	}
}

hkLargeBlockAllocator::Iterator hkLargeBlockAllocator::getIterator()
{
	m_top->head = m_topsize | MemChunk::PINUSE_BIT;

    // If there are no pages then there is no memory
    if (m_pages.m_next == &m_pages) return Iterator();

	MemPage* page = m_pages.m_next;
    MemChunk* start = (MemChunk*)page->getStart();
    MemChunk* end = (MemChunk*)(page->getEnd() - MemChunk::PAYLOAD_OFFSET);

    return Iterator(start,page,end);
}

hkBool hkLargeBlockAllocator::nextBlock(Iterator& iter)
{
    if (iter.m_chunk == HK_NULL) return false;
    // go to the next chunk
    iter.m_chunk = iter.m_chunk->nextChunk();
    // if we are at the end we need to go to the next page
    if (iter.m_chunk == iter.m_end)
    {
        MemPage* page = iter.m_page->m_next;
        if (page == &m_pages)
        {
            // Make invalid
            iter.m_chunk = HK_NULL;
            return false;
        }
        // Skip to the next page
        iter.m_page = page;
        iter.m_chunk = (MemChunk*)page->getStart();
        iter.m_end = (MemChunk*)(page->getEnd() - MemChunk::PAYLOAD_OFFSET);
    }
    // success
    return true;
}


hk_size_t hkLargeBlockAllocator::_calculateUsed() const
{
    hk_size_t used = 0;

    _makeTopValid();

	MemPage* page = m_pages.m_next;
	while (page != &m_pages)
	{
        MemChunk* cur = page->getFirstChunk();
        // The last chunk in this page
        MemChunk* footer = page->getFooter();
        while (cur != footer)
        {
            if (cur->isInuse())
            {
                used += cur->getChunkSize();
            }
            // Step to next block
            cur = cur->nextChunk();
        }
        // Onto the next page
		page = page->m_next;
	}
    return used;
}


void hkLargeBlockAllocator::getMemoryStatistics(hkMemoryAllocator::MemoryStatistics& stats) const
{
    stats.m_allocated = calculateMemoryUsedByThisAllocator();
    //HK_ASSERT(0x34234,m_used == _calculateUsed());
    stats.m_inUse			= m_sumAllocatedSize;
    stats.m_available		= stats.m_allocated - m_sumAllocatedSize;
    stats.m_largestBlock	= findLargestBlockSize();
    stats.m_totalAvailable = 0;
}

hk_size_t hkLargeBlockAllocator::calculateMemoryUsedByThisAllocator() const
{
    hk_size_t allocated = 0;
	MemPage* page = m_pages.m_next;
	while (page != &m_pages)
	{
        // Accumulate the size
        allocated += page->m_size;
        // Onto the next page
		page = page->m_next;
	}
    return allocated;
}

#if 0

/// This has been disabled, because theres a bug in there which causes errors. Will look into again when the structures
/// have been cleaned up abit

void
hkLargeBlockAllocator::resizeAlloc(void* in,hk_size_t newSize)
{
        /// Work out the size the new chunk would need to be
		/// We add the PAYLOAD_OFFSET as it could bigger than the sizeof(MemChunk), as it takes into account alignment
    hk_size_t newChunkSize = (newSize + MemChunk::PAYLOAD_OFFSET + MemChunk::ALIGN_MASK)&~MemChunk::ALIGN_MASK;
        /// Must at least be as big as the min block size
    if (newChunkSize < MIN_LARGE_SIZE) newChunkSize = MIN_LARGE_SIZE;

    MemChunk* p = MemChunk::toChunk(in);
    HK_ASSERT(0x34243403,_checkUsedAlloc(in));

        /// Get the chunk size
    hk_size_t chunkSize = p->getChunkSize();

        /// If the size is the same, we are done
    if (newChunkSize == chunkSize) return;

    HK_ASSERT(0x34242423,newChunkSize<chunkSize);
    if (newChunkSize>chunkSize) { return; }

        /// Work out the remainder size.
    hk_size_t remSize = chunkSize - newChunkSize;

        /// If the change is too small to make a new block, then we are done
    if (remSize <  MIN_LARGE_SIZE) { return; }

        /// This is where the new block will start
    MemChunk* remChunk = p->chunkPlusOffset(chunkSize - remSize);

        /// Get the next chunk
    MemChunk* next = p->chunkPlusOffset(chunkSize);

    HK_ASSERT(0x34243204,_okNext(p,next)&&next->isPinuse());
    if (!next->isInuse())
    {
        /* consolidate forward */
        if (next == m_top)
        {
                /// Make the top bigger
            m_topsize += remSize;
                /// Make this the new top
            m_top = remChunk;

                /// Set the new head size
            remChunk->head = m_topsize | MemChunk::PINUSE_BIT;

                /// Need to fix the old block
            p->head = (p->head&MemChunk::PINUSE_BIT)|MemChunk::CINUSE_BIT|(chunkSize - remSize);

                /// p remains in use, m_top is not in use
            HK_ASSERT(0x3425425,p->isInuse()&&!m_top->isInuse());
            return;
        }
        else
        {
                /// Add the next block
            remSize += next->getChunkSize();

                /// Unlink next
            _unlinkLargeChunk(static_cast<MemTreeChunk*>(next));

                /// Set up rem chunk
            remChunk->setSizeAndPinuseOfFreeChunk(remSize);
        }
    }
    else
    {
        remChunk->setFreeWithPinuse(remSize, next);
    }
    // Insert the chunk back
    _insertLargeChunk( static_cast<MemTreeChunk*>(remChunk),remSize);

    // Check the block is free
    HK_ASSERT(0x2312321,_checkFreeChunk(remChunk));
}
#endif


bool hkLargeBlockAllocator::_checkUsedAlloc( const void* mem) const
{
    MemChunk* p = MemChunk::toChunk(const_cast<void*>(mem));

    if (!MemChunk::isAligned(mem))
	{
		return false;
	}

    if (!_isOkAddress(p))
	{
		return false;
	}

    if (!p->isInuse())
	{
		return false;
	}
    // Its size must be at least as large as the smallest block
    hk_size_t minSize = _padRequest(0);

    hk_size_t size = p->getChunkSize();

    if (size < minSize)
	{
		return false;
	}

    /// Look at the next block - its pinuse should be true, and its prevFoot should be the size of this block
    MemChunk* next = p->chunkPlusOffset(size);

    if (!next->isPinuse()) 
	{
		return false;
	}

    // prevFoot is only set if the previous block is free
    if (!p->isPinuse())
    {
        MemChunk* prev = p->chunkMinusOffset(p->prevFoot);
        // Previous must be free (otherwise how could we be here
        if (prev->isInuse()!=false) 
		{
			return false;
		}
        // The previous block must be same size a prevFoot
        if (prev->getChunkSize() != p->prevFoot) 
		{
			return false;
		}
    }

    return true;
}

hkBool hkLargeBlockAllocator::_checkFreeChunk(MemChunk* p)
{
    if (!MemChunk::isAligned((void*)p)) 
	{
		return false;
	}

	// It can't be in use or how can it be free?
    if (p->isInuse()) 
	{
		return false;
	}

    if (!_isOkAddress(p)) 
	{
		return false;
	}

    // The block before and after must be used, because we always automatically combine

    hk_size_t size = p->getChunkSize();
    MemChunk* next = p->chunkPlusOffset(size);

    // The previous is not in use
    if (next->isPinuse()) 
	{
		return false;
	}
    // This must be in use
    if (!next->isInuse()) 
	{
		return false;
	}

    // The previous must be in use
    if (!p->isPinuse()) 
	{
		return false;
	}

    // prevFoot must be invalid - because prevFoot is only valid when Pinuse is false (ie when its free),
    // but it can't be free otherwise we would have consolidated it, as this block is supposedly free

    return true;
}

hkBool hkLargeBlockAllocator::checkAllocations(void** allocs,int size)
{
    for (int i=0;i<size;i++)
    {
        if (!_checkUsedAlloc(allocs[i])) 
		{
			return false;
		}
    }
    return true;
}

void hkLargeBlockAllocator::_addAlloc(void* data,hk_size_t size,void* param)
{
    hkArray<void*>& array = *(hkArray<void*>*)param;
    array.pushBack(data);
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// selfTest

#ifdef HK_DEBUG

void HK_CALL hkLargeBlockAllocator::allocatorTest(hkLargeBlockAllocator& allocator,int testSize)
{
	int maxPtrs = testSize / 1000;
	if (maxPtrs < 50 ) maxPtrs = 50;

    hkArray<void*> ptrs;
    hkPseudoRandomGenerator rand(12345);

	{
		for (int i = 0; i < testSize; i++)
		{
			int num = rand.getRand32();

			if ((num & 15) > 4 && ptrs.getSize() < maxPtrs)
			{
				// Add an allocation
				int size = rand.getRand32() % 50000;
				// do the allocation

				//if (size<100) size = 100;

				HK_ASSERT(0x4234324, allocator.checkAllocations(ptrs.begin(), ptrs.getSize()));

				void* alloc = allocator.blockAlloc(size);
				if (alloc)
				{
					if (!allocator._checkUsedAlloc(alloc))
					{
						HK_ASSERT(0x34234, allocator._checkUsedAlloc(alloc));
					}

					ptrs.pushBack(alloc);
					hkString::memSet(alloc, 0, (int)size);
				}
				else
				{
					// Failed to alloc

				}

				HK_ASSERT(0x4234324,allocator.checkAllocations(ptrs.begin(),ptrs.getSize()));
			}

	#if 0
			if ((num&0x1000)&&ptrs.getSize()>0)
			{
				// We need to remove an allocation
				int index = rand.getRand32()%ptrs.getSize();
				void* alloc = ptrs[index];

				hk_size_t blockSize = allocator.getAllocSize(alloc);

				for (hk_size_t j=blockSize;j>=0;j--)
				{
					//allocator.resizeAlloc(alloc,j);

					hkString::memSet(alloc,0,(int)(j));
					if (j==0) break;
				}
			}
	#endif

			if ((num & 0x100) && ptrs.getSize() > 0)
			{
				// We need to remove an allocation
				int index = rand.getRand32() % ptrs.getSize();

				void* alloc = ptrs[index];
				HK_ASSERT(0x4234324, allocator.checkAllocations(ptrs.begin(),ptrs.getSize()));

				allocator.blockFree(alloc, -1);
				ptrs.removeAt(index);

				HK_ASSERT(0x4234324, allocator.checkAllocations(ptrs.begin(),ptrs.getSize()));
			}

			if (((num & 0xff00) >> 8) > 253)
			{
				// Lets check the callback stuff
				hkArray<void*> foundPtrs;

				allocator.forAllAllocs(_addAlloc, &foundPtrs);

				HK_ASSERT(0x32432423,foundPtrs.getSize() == ptrs.getSize());

				// Sort both
				hkSort(ptrs.begin(), ptrs.getSize(), _comparePointers);
				hkSort(foundPtrs.begin(), foundPtrs.getSize(), _comparePointers);

				for (int j = 0; j < ptrs.getSize(); j++)
				{
					HK_ASSERT(0x34234234,ptrs[j] == foundPtrs[j]);
				}
			}


			num = (rand.getRand32())&0xffff;
			if (num > 4000 && num < 4002)
			{
				// Free them all
				HK_ASSERT(0x4234324,allocator.checkAllocations(ptrs.begin(),ptrs.getSize()));
				for (int k = 0; k < ptrs.getSize(); k++) 
				{
					allocator.blockFree(ptrs[k], -1);
				}
				ptrs.clear();
				HK_ASSERT(0x4234324,allocator.checkAllocations(ptrs.begin(),ptrs.getSize()));
			}
		}
	}

	// Free all at the end
	{
		for (int i = 0; i < ptrs.getSize(); i++) 
		{
			allocator.blockFree(ptrs[i], -1);
		}
	}
}

void HK_CALL hkLargeBlockAllocator::selfTest()
{
#if 0
    {
        hkSystemMemoryBlockServer server;
		hkLargeBlockAllocator allocator(&server);

        //allocatorTest(allocator,100000);
    }

	{
		const hk_size_t size = 1024*1024;
		void* mem = hkSystemMalloc(size,16);


		hkFixedMemoryBlockServer server(mem,size);
		hkLargeBlockAllocator allocator(&server);

		{
			// Right less put it to the test
			void* p0 = allocator.alloc(size-10*1024);
			HK_ASSERT(0x12313,p0);
			void* p1 = allocator.alloc(5*1024);
			HK_ASSERT(0x12313,p1);
			// Should push this on a bucket, because its not next to top
			allocator.free(p0);

			// Okay lets allocate something that must go into p0s old memory
			p0 = allocator.alloc(10*1024);
			HK_ASSERT(0x12313,p0);

			allocator.free(p1);
			allocator.free(p0);
		}

        allocatorTest(allocator,100000);

		// Shouldn't we just have the top block and thats it?

		allocator.freeAll();

        hkSystemFree(mem);
	}
	{
		const hk_size_t size = 4096;
		char mem[size];

		hkFixedMemoryBlockServer server(mem,size);
		hkLargeBlockAllocator allocator(&server);

		void* p1 = allocator.alloc(10);
		void* p2 = allocator.alloc(20);
		void* p3 = allocator.alloc(30);

		allocator.free(p2);

		p2 = allocator.alloc(20);

		allocator.free(p1);
		allocator.free(p2);
		allocator.free(p3);
	}
#endif
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
