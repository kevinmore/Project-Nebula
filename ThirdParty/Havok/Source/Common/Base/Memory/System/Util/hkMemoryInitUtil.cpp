/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/System/Util/hkMemoryInitUtil.h>
#include <Common/Base/Memory/Allocator/Malloc/hkMallocAllocator.h>
#include <Common/Base/System/StackTracer/hkStackTracer.h>
#include <Common/Base/Fwd/hkcstdio.h>

using namespace std;

#if defined(HK_MEMORY_TRACKER_ENABLE)
#include <Common/Base/Memory/Tracker/Default/hkDefaultMemoryTracker.h>
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>
#endif

hkMemorySystem* hkMemoryInitUtil::s_system = HK_NULL;
hkMemoryInitUtil::onQuitFunc hkMemoryInitUtil::s_onQuitFunc = HK_NULL;

#ifdef HK_PLATFORM_CTR
	void HK_CALL hkMemoryInitUtil::outputDebugString(const char* s, void*)
	{
		nndbgDetailPrintf("%s",s); 
	}
#elif defined(HK_PLATFORM_WIN32)
	#include <Common/Base/Fwd/hkwindows.h>
	#include <stdlib.h>
	void HK_CALL hkMemoryInitUtil::outputDebugString(const char* s, void*)
	{
		#ifndef HK_PLATFORM_WINRT
			OutputDebugStringA(s);
			printf("%s",s); // Also output to console for automated testing
		#else
			// Unicode only 
			int sLen = hkString::strLen(s) + 1;
			wchar_t* wideStr = hkAllocateStack<wchar_t>( sLen );
			mbstowcs_s(HK_NULL, wideStr, sLen, s, sLen - 1); 
			OutputDebugString(wideStr);
			hkDeallocateStack<wchar_t>( wideStr, sLen);
		#endif
	}
#else
	#include <Common/Base/Fwd/hkcstdio.h>
	void HK_CALL hkMemoryInitUtil::outputDebugString(const char* s, void*)
	{
		using namespace std;
		printf("%s",s);
		fflush(stdout);
	}
#endif

hkResult HK_CALL hkMemoryInitUtil::quit()
{
	hkResult result = HK_SUCCESS;

	// Destroy the system
	HK_ASSERT( 0x23432aa3, s_system != HK_NULL );
	if( s_system )
	{
		result = s_system->mainQuit();
		s_system->~hkMemorySystem();
		s_system = HK_NULL;

		// Ensure it's not set
		hkMemorySystem::replaceInstance( HK_NULL );
	}

	// Do any remaining destruction
	if( s_onQuitFunc )
	{
		s_onQuitFunc();
		s_onQuitFunc = HK_NULL;
	}

	return result;
}

void HK_CALL hkMemoryInitUtil::refreshDebugSymbols()
{
	hkStackTracer tracer;
	tracer.refreshSymbols();
}

#if defined(HK_MEMORY_TRACKER_ENABLE)
void HK_CALL hkMemoryInitUtil::initMemoryTracker()
{
	if(hkMemorySystem::getInstancePtr())
	{
		// The memory tracker must be initialized before initializing the memory system
		HK_BREAKPOINT(0);
		return;
	}

	static hkMallocAllocator mallocAllocator;
	static hkUint8 s_tracker[sizeof(hkDefaultMemoryTracker)];

	hkDefaultMemoryTrackerAllocator::s_allocator = &mallocAllocator;
	hkDefaultMemoryTracker* tracker = new (s_tracker) hkDefaultMemoryTracker(&mallocAllocator);

	hkMemoryTracker::setInstance(tracker);
}

void HK_CALL hkMemoryInitUtil::quitMemoryTracker()
{
	if( hkMemoryTracker* tracker = hkMemoryTracker::getInstancePtr() )
	{
		tracker->~hkMemoryTracker();
		hkMemoryTracker::setInstance(HK_NULL);
	}
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
