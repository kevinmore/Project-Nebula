/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/System/Util/hkMemoryInitUtil.h>
#include <Common/Base/Memory/Allocator/FreeList/hkFreeListAllocator.h>
#include <Common/Base/Memory/Allocator/LargeBlock/hkLargeBlockAllocator.h>
#include <Common/Base/Memory/System/FreeList/hkFreeListMemorySystem.h>

#if defined HK_COMPILER_MSVC // Some of the deprecated interface passes through here
#	pragma warning(push)
#	pragma warning(disable: 4996)
#endif

namespace
{
	struct FreeList
	{
		HK_ALIGN16( hkUint8 m_largeBlock[sizeof(hkLargeBlockAllocator)] );
		HK_ALIGN16( hkUint8 m_allocator[sizeof(hkFreeListAllocator)] );
		HK_ALIGN16( hkUint8 m_system[sizeof(hkFreeListMemorySystem)] );
	} s_buffer;

	static hkMemoryAllocator*	s_largeBlockAllocator = HK_NULL;
	static hkMemoryAllocator*	s_freeListAllocator = HK_NULL;

	static void onQuit()
	{
		// System already destroyed
		HK_ASSERT( 0x3242a432, hkMemoryInitUtil::s_system == HK_NULL );

		// Destroy allocators
		if( s_freeListAllocator )
		{
			s_freeListAllocator->~hkMemoryAllocator();
			s_freeListAllocator = HK_NULL;
		}
		if( s_largeBlockAllocator )
		{
			s_largeBlockAllocator->~hkMemoryAllocator();
			s_largeBlockAllocator = HK_NULL;
		}
	}
}

hkMemoryRouter* HK_CALL hkMemoryInitUtil::initFreeListLargeBlock( hkMemoryAllocator *memoryAllocator, const hkMemorySystem::FrameInfo& info, const hkFreeListAllocator::Cinfo* cinfo, hkFreeListMemorySystem::SetupFlags flags )
{
	HK_ASSERT( 0x3242a432, s_system == HK_NULL );

	hkLargeBlockAllocator* largeBlockAllocator = new (s_buffer.m_largeBlock) hkLargeBlockAllocator( memoryAllocator );
	hkFreeListAllocator* allocator = new (s_buffer.m_allocator) hkFreeListAllocator( largeBlockAllocator, largeBlockAllocator, cinfo );
	hkFreeListMemorySystem* system = new (s_buffer.m_system) hkFreeListMemorySystem( memoryAllocator, allocator, allocator, flags );
	
	s_system = system;
	s_largeBlockAllocator = largeBlockAllocator;
	s_freeListAllocator = allocator;

	hkMemorySystem::replaceInstance( system );
	s_onQuitFunc = onQuit;

	return system->mainInit( info );
}

hkMemoryRouter* HK_CALL hkMemoryInitUtil::initFreeList( hkMemoryAllocator *memoryAllocator, hkMemoryAllocator::ExtendedInterface *memoryAllocatorExtended, const hkMemorySystem::FrameInfo& info, const hkFreeListAllocator::Cinfo* cinfo, hkFreeListMemorySystem::SetupFlags flags )
{
	HK_ASSERT( 0x3242a432, s_system == HK_NULL );

	hkFreeListAllocator* allocator = new (s_buffer.m_allocator) hkFreeListAllocator( memoryAllocator, memoryAllocatorExtended, cinfo );
	hkFreeListMemorySystem* system = new (s_buffer.m_system) hkFreeListMemorySystem( memoryAllocator, allocator, allocator, flags );

	s_system = system;
	s_freeListAllocator = allocator;

	hkMemorySystem::replaceInstance( system );
	s_onQuitFunc = onQuit;

	return system->mainInit( info );
}

hkMemoryRouter* HK_CALL hkMemoryInitUtil::initHeapAllocator( hkMemoryAllocator *memoryAllocator, hkMemoryAllocator *heapAllocator, hkMemoryAllocator::ExtendedInterface* heapInterface, const hkMemorySystem::FrameInfo& info )
{
	HK_ASSERT( 0x3242a432, s_system == HK_NULL );

	hkFreeListMemorySystem* system = new (s_buffer.m_system) hkFreeListMemorySystem( memoryAllocator, heapAllocator, heapInterface );
	s_system = system;

	hkMemorySystem::replaceInstance( system );

	return system->mainInit( info );
}

#if defined HK_COMPILER_MSVC
#	pragma warning(pop)
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
