/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Ext/hkBaseExt.h>
#include <Common/Base/Fwd/hkcstdio.h>
#include <Common/Base/Memory/Allocator/Checking/hkLeakDetectAllocator.h>
#include <Common/Base/Memory/Allocator/Malloc/hkMallocAllocator.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>

#define hkAllocate extAllocate
#define hkDeallocate extDeallocate
#define hkContainerHeapAllocator extContainerAllocator
#define hkInplaceArrayAligned16 extInplaceArrayAligned16
#define hkInplaceArray extInplaceArray
#define hkStringBuf extStringBuf
#define hkFixedSizeArray extFixedSizeArray
#define hkArrayBase extArrayBase
#define hkArray extArray
#define hkStringPtr extStringPtr
#include <Common/Base/Container/String/hkStringBuf.cpp>
#include <Common/Base/Container/String/hkStringPtr.cpp>

#include <Common/Base/Container/StringMap/hkStringMap.h>
#include <Common/Base/Container/StringMap/hkCachedHashMap.cxx>
template class hkCachedHashMap<hkStringMapOperations, extContainerAllocator>;

static hkMallocAllocator s_mallocAllocator(HK_REAL_ALIGNMENT);
static hkLeakDetectAllocator s_leakDetector;
static hkMemoryAllocator* s_activeAllocator;

#ifdef HK_PLATFORM_CTR
	static void HK_CALL outputDebugString(const char* s, void*)
	{
		nndbgDetailPrintf("%s",s); 
	}
#elif defined(HK_PLATFORM_WIN32)
	#include <Common/Base/Fwd/hkwindows.h>
	#include <stdlib.h>
	
	static void HK_CALL outputDebugString(const char* s, void*)
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
	static void HK_CALL outputDebugString(const char* s, void*)
	{
		using namespace std;
		printf("%s",s);
	}
#endif

hkMemoryAllocator& HK_CALL extAllocator::getInstance()
{
		// If user hasn't explicitly initialized, they get malloc
	if( HK_VERY_UNLIKELY( s_activeAllocator == HK_NULL) )
	{
		initDefault();
	}
	return *s_activeAllocator;
}

void HK_CALL extAllocator::initChecking()
{
	HK_ASSERT2( 0x62e69394, s_activeAllocator == HK_NULL, "Already initialized" );
	s_leakDetector.init(&s_mallocAllocator, &s_mallocAllocator, outputDebugString, HK_NULL);
	s_activeAllocator = &s_leakDetector;
}
void HK_CALL extAllocator::initDefault()
{
	HK_ASSERT2( 0x62e69395, s_activeAllocator == HK_NULL, "Already initialized" );
	s_activeAllocator = &s_mallocAllocator;
}
void HK_CALL extAllocator::quit()
{
	if( s_activeAllocator == &s_leakDetector )
	{
		s_leakDetector.quit();
	}
	s_activeAllocator = HK_NULL;
}

//
// extCriticalSection - a critical section without #including windows.h
//
extCriticalSection::extCriticalSection(int spinCount)
{
	HK_COMPILE_TIME_ASSERT( sizeof(m_section) >= sizeof(hkCriticalSection) );
	new (&m_section[0]) hkCriticalSection(spinCount);
}

extCriticalSection::~extCriticalSection()
{
	get()->~hkCriticalSection();
}

void extCriticalSection::enter()
{
	get()->enter();
}

void extCriticalSection::leave()
{
	get()->leave();
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
