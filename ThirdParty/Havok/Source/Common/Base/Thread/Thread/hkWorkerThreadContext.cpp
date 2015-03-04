/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Thread/hkWorkerThreadContext.h>

#ifdef HK_PLATFORM_SPU
#	include <Common/Base/Thread/Thread/hkThreadLocalData.h>
#else
#	include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>
#	include <Common/Base/Memory/System/hkMemorySystem.h>
#	include <Common/Base/System/hkBaseSystem.h>
#	include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>
#endif

HK_THREAD_LOCAL( int ) hkThreadNumber;

#ifndef HK_PLATFORM_SPU
hkWorkerThreadContext::hkWorkerThreadContext(int threadId)
{
	// Flush all denormal/subnormal numbers (2^-1074 to 2^-1022) to zero.
	// Typically operations on denormals are very slow, up to 100 times slower than normal numbers.
#if defined(HK_COMPILER_HAS_INTRINSICS_IA32) && HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED	
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif

	// Initialize memory system	
	
	hkMemoryRouter& memoryRouter = m_memoryRouter;
	hkMemorySystem::getInstance().threadInit(memoryRouter, "hkWorkerThreadContext");
	hkBaseSystem::initThread(&m_memoryRouter);
	

	// Store thread id in thread local memory
	HK_THREAD_LOCAL_SET(hkThreadNumber, threadId);
}


hkWorkerThreadContext::~hkWorkerThreadContext()
{
	// Perform cleanup operations
	hkCheckDeterminismUtil::quitThread();

	
	hkBaseSystem::quitThread();
	hkMemorySystem::getInstance().threadQuit(m_memoryRouter);
	
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
