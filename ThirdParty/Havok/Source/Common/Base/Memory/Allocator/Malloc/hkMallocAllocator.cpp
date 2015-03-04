/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Allocator/Malloc/hkMallocAllocator.h>

static hkMallocAllocator s_defaultMallocAllocator;
hkMemoryAllocator* hkMallocAllocator::m_defaultMallocAllocator = &s_defaultMallocAllocator;

// If the platform has a builtin aligned allocation facility, we define
// aligned_malloc/aligned_free macros to be those functions. Otherwise, we define
// macros/functions for unaligned_malloc/unaligned free and manually align the
// allocations.
#if defined(HK_PLATFORM_WIN32) || defined(HK_PLATFORM_XBOX) || defined (HK_PLATFORM_XBOX360) || defined(HK_PLATFORM_LRBSIM)  

	#include <Common/Base/Fwd/hkcstdlib.h>
	#define aligned_malloc _aligned_malloc
	#define aligned_free _aligned_free

#elif defined(HK_PLATFORM_LINUX) || defined(HK_PLATFORM_PS3_PPU) || defined( HK_PLATFORM_PSVITA ) || defined(HK_PLATFORM_ANDROID) || defined(HK_PLATFORM_TIZEN) || defined(HK_PLATFORM_PS4) 

	#include <Common/Base/Fwd/hkcmalloc.h>
	#define aligned_malloc(size, align) memalign(align, size)
	#define aligned_free(p) free(p)

#elif defined(HK_PLATFORM_MAC386) || defined(HK_PLATFORM_MACPPC) || defined(HK_PLATFORM_LRB_XENSIM) || defined(HK_PLATFORM_LRB)
	#include <Common/Base/Fwd/hkcmalloc.h>
	#define unaligned_malloc(size) malloc(size) // native align only
	#define unaligned_free(p) free(p)

#elif defined(HK_PLATFORM_WIIU)
#	include <cafe/mem.h>
	#define aligned_malloc wiiuallocator_malloc
	#define aligned_free wiiudemoallocator_free

	static void* wiiuallocator_malloc(int size, int align)
	{
		return MEMAllocFromDefaultHeapEx(size, align);
	}

	static void wiiudemoallocator_free(void* p)
	{
		return MEMFreeToDefaultHeap(p);
	}
	
#elif defined(HK_PLATFORM_IOS) || defined(HK_PLATFORM_CTR)
	#include <Common/Base/Fwd/hkcmalloc.h>
	#define unaligned_malloc(size) malloc(size) // native align only, which on arm is 8 bytes not 16
	#define unaligned_free(p) free(p)


#elif defined(HK_PLATFORM_RVL)
#	include <revolution/mem.h>
 
	// Our demos use Mem2 for DemoAllocator2, and Mem1 for DemoAllocator1
	extern MEMAllocator	DemoAllocator2;
	extern MEMAllocator	DemoAllocator1;
	
	// Use Mem1 until exhausted, then use Mem2. 
	// Also, alloc in Mem1 only if number of bytes is less than some threshold, 16Mb here.
	const int MEM1_threshold = 16000000;
	
	#define aligned_malloc demoallocator_malloc
	#define aligned_free demoallocator_free

	static void* demoallocator_malloc(int size, int align)
	{
		HK_ASSERT(0x76a0c45d, sizeof(hk_size_t)==sizeof(char*));

		void* p;

		if ( size < MEM1_threshold ) 
		{
			p = MEMAllocFromAllocator( &DemoAllocator1, size );
			
			// If the allocation failed in Mem1, allocate the block from Mem2
			if( !p )
			{
				//HK_WARN(0x7da65775, "Tried and failed to allocate " << size << " bytes in Mem1, resorting to Mem2");
				p = MEMAllocFromAllocator( &DemoAllocator2, size );	
			}
		}
		else
		{
			HK_WARN(0x7da65775, "Large allocation of " << size << " bytes, resorting to Mem2");
			p = MEMAllocFromAllocator( &DemoAllocator2, size );
		}
		
		return p;
	}

	static void demoallocator_free(void* p)
	{
		if( p != HK_NULL )
		{
			hkUint32 arenaLoLocation = (u32)OSGetArenaLo();
			hkUint32 blockLocation = (u32)p;
			
			// Free to Mem2 if the block was not created in Mem1
			// Since the arenaLo is defined to be the top address of Mem1
			// anything with a larger address must have been allocated from Mem2
			if(blockLocation > arenaLoLocation)
			{
				MEMFreeToAllocator( &DemoAllocator2, p );
			}
			else
			{
				MEMFreeToAllocator( &DemoAllocator1, p );
			}
		}
	}

#elif defined (HK_PLATFORM_PS3_SPU)

	#define aligned_malloc no_malloc
	#define aligned_free HK_ASSERT(0x1a1da5f3,0)
	static HK_FORCE_INLINE void* no_malloc(int size, int align)
	{
		HK_ASSERT(0x64d61ce3,0);
		return HK_NULL;
	}

#else

#	error no Malloc & Free defined for this platform.

#endif


#if !defined(aligned_malloc)
	// we synthesize an aligned malloc from an unaligned one
	static inline void* aligned_malloc(int numBytes, int align)
	{
		// if numBytes == 0
		if (numBytes < 1)
		{
			return HK_NULL;
		}

		void* pvoid = unaligned_malloc(numBytes + sizeof(int) + (align>4? align : 4) ); // need extra int to store padding info, and pad for align
		if (pvoid == HK_NULL)
		{
			HK_WARN_ALWAYS(0x7da65776, "Failed to allocate " << numBytes << " bytes!");
			return HK_NULL;
		}
		
		HK_ASSERT2(0x38c3fe7c,  (reinterpret_cast<hkUlong>(pvoid) & 0x3) == 0, "Pointer was not 4 byte aligned");
		char* p = reinterpret_cast<char*>(pvoid);
		char* aligned = reinterpret_cast<char*>(
			HK_NEXT_MULTIPLE_OF( align,	reinterpret_cast<hkUlong>(p+1)) );
		reinterpret_cast<int*>(aligned)[-1] = (int)(aligned - p);
		return aligned;
	}
	static inline void aligned_free( void* p )
	{
		if (p)
		{
			int offset = reinterpret_cast<int*>(p)[-1];
			void* pvoid = static_cast<char*>(p) - offset;
			unaligned_free(pvoid);
		}
	}
#endif

void* hkMallocAllocator::blockAlloc( int numBytes )
{
	hkUint32 cur = hkCriticalSection::atomicExchangeAdd(&m_currentUsed, numBytes);
	if(cur + numBytes > m_peakUse) 
	{
		m_peakUse = m_currentUsed;
	}

	return aligned_malloc(numBytes, m_align);
}

void hkMallocAllocator::blockFree( void* p, int numBytes )
{
	hkCriticalSection::atomicExchangeAdd(&m_currentUsed,  -numBytes);

	aligned_free(p);
}

void hkMallocAllocator::getMemoryStatistics( MemoryStatistics& u ) const
{
	u.m_allocated = m_currentUsed;
	u.m_peakInUse = m_peakUse;
}

void hkMallocAllocator::resetPeakMemoryStatistics()
{
	m_peakUse = m_currentUsed;
}

int hkMallocAllocator::getAllocatedSize( const void* obj, int numBytes ) const
{
	return numBytes;
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
