/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/System/Util/hkMemoryInitUtil.h>
#include <Common/Base/Memory/System/Optimizer/hkOptimizerMemorySystem.h>

namespace
{
	HK_ALIGN16( static hkUint8 s_buffer[sizeof(hkOptimizerMemorySystem)] );
}

hkMemoryRouter* HK_CALL hkMemoryInitUtil::initOptimizer( hkMemoryAllocator *memoryAllocator, const hkMemorySystem::FrameInfo& info )
{
	HK_ASSERT( 0x3242a432, s_system == HK_NULL );
	hkOptimizerMemorySystem* system = new (s_buffer) hkOptimizerMemorySystem;
	s_system = system;

	system->init( memoryAllocator, outputDebugString, HK_NULL, hkOptimizerMemorySystem::DETECT_ALL );
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
