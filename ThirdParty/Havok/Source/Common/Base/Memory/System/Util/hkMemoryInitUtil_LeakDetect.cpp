/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/System/Util/hkMemoryInitUtil.h>
#include <Common/Base/Memory/Allocator/Checking/hkLeakDetectAllocator.h>
#include <Common/Base/Memory/System/FreeList/hkFreeListMemorySystem.h>

namespace
{
	struct LeakDetect
	{
		HK_ALIGN16( hkUint8 m_allocator[sizeof(hkLeakDetectAllocator)] );
		HK_ALIGN16( hkUint8 m_system[sizeof(hkFreeListMemorySystem)] );
	} s_buffer;

	static void onQuit()
	{
		// System already destroyed
		HK_ASSERT( 0x3242a432, hkMemoryInitUtil::s_system == HK_NULL );

		// Destroy allocator
		{
			hkLeakDetectAllocator* allocator = reinterpret_cast<hkLeakDetectAllocator*>( &s_buffer.m_allocator );
			allocator->quit();
			allocator->~hkLeakDetectAllocator();
		}
	}
}

hkMemoryRouter* HK_CALL hkMemoryInitUtil::initLeakDetect( hkMemoryAllocator *memoryAllocator, const hkMemorySystem::FrameInfo& info )
{
	HK_ASSERT( 0x3242a432, s_system == HK_NULL );

	// Init the leak allocator
	hkLeakDetectAllocator* allocator = new (s_buffer.m_allocator) hkLeakDetectAllocator;
	allocator->init( memoryAllocator, memoryAllocator, outputDebugString, HK_NULL );

	// Create memory system, without thread memory
	hkFreeListMemorySystem::SetupFlags flags = hkFreeListMemorySystem::SetupFlags(hkFreeListMemorySystem::DEFAULT_SETUP_FLAGS & ~hkFreeListMemorySystem::USE_THREAD_MEMORY);
	hkFreeListMemorySystem* system = new (s_buffer.m_system) hkFreeListMemorySystem( memoryAllocator, allocator, HK_NULL, flags );
	s_system = system;

	hkMemorySystem::replaceInstance( system );
	s_onQuitFunc = onQuit;

	return system->mainInit( info );
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
