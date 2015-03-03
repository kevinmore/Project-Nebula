/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_BLOCK_STREAM_ALLOCATOR_H
#define HK_BLOCK_STREAM_ALLOCATOR_H

#include <Common/Base/Container/BlockStream/hkBlockStream.h>

class hkBlockStreamAllocator: public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE );

		typedef hkBlockStreamBase::Block Block;

	#if !defined(HK_PLATFORM_SPU)

		/// Allocate blocks.
		/// Notes:
		///     - 'this' is a pointer to ppu.
		virtual void blockAllocBatch( Block** blocksOut, int nblocks ) = 0;

		/// Free blocks.
		/// Note: 'this' is a pointer to ppu
		virtual void blockFreeBatch( Block** blocks, int nblocks ) = 0;

		/// Get the current bytes used. Note that data in the thread local allocator will
		/// count as bytes used (as they are not available to other threads in a worst case
		/// scenario).
		virtual int getBytesUsed() const= 0;

		/// Get the peak usage.
		virtual int getMaxBytesUsed() const = 0;

		/// Get the maximum amount of memory that can be allocated through this allocator.
		virtual int getCapacity() const = 0;

		/// Sets all data to be free, even if there are outstanding allocations.
		virtual void freeAllRemainingAllocations() = 0;

		/// Returns an aggregate on memory statistics of this allocator.
		virtual void getMemoryStatistics( hkMemoryAllocator::MemoryStatistics& statsOut ) const = 0;

	#else

		// These methods just redirect to the hkFixedBlockStreamAllocator ones on SPU
		HK_FORCE_INLINE void blockAllocBatch(Block** blocksOut, int nblocks);
		HK_FORCE_INLINE void blockFreeBatch(Block** blocks, int nblocks);

	protected:

		// Protected constructor to prevent accidental construction on SPU
		hkBlockStreamAllocator() {}

	#endif
};

#if defined(HK_PLATFORM_SPU)
#	include <Common/Base/Container/BlockStream/Allocator/Fixed/hkFixedBlockStreamAllocator.h>
#endif
#include <Common/Base/Container/BlockStream/Allocator/hkBlockStreamAllocator.inl>

#endif // HK_BLOCK_STREAM_ALLOCATOR_H

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
