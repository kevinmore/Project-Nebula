/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/System/Util/hkMemoryInitUtil.h>
#include <Common/Base/Memory/System/Checking/hkCheckingMemorySystem.h>

namespace
{
	HK_ALIGN16( static hkUint8 s_buffer[sizeof(hkCheckingMemorySystem)] );
}

hkMemoryRouter* HK_CALL hkMemoryInitUtil::initChecking( hkMemoryAllocator *memoryAllocator, const hkMemorySystem::FrameInfo& info )
{
	HK_ASSERT( 0x3242a432, s_system == HK_NULL );
	hkCheckingMemorySystem* system = new (s_buffer) hkCheckingMemorySystem;
	s_system = system;

	system->init( memoryAllocator, outputDebugString, HK_NULL );
	hkMemorySystem::replaceInstance( system );

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
