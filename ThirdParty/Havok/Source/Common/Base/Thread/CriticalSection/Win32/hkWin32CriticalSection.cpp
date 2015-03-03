/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Config/hkConfigThread.h>
#include <Common/Base/System/hkBaseSystem.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>

#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED 

#pragma warning(push)
#pragma warning(disable:4355) // "this" in init list

// Win32 style only impl here for the moment.
hkCriticalSection::hkCriticalSection( int spinCount )
{
#if defined(HK_COMPILER_MSVC) && (HK_COMPILER_MSVC_VERSION < 1300)
	InitializeCriticalSection( &m_section );
#else // VC7 and higher
	if ( spinCount == 0 )
	{   
	    hkHardwareInfo	hwInfos; hkGetHardwareInfo(hwInfos);
		spinCount = hwInfos.m_numThreads*1000;
	}
	#ifndef HK_PLATFORM_WINRT
		InitializeCriticalSectionAndSpinCount( &m_section, spinCount );
	#else
		InitializeCriticalSectionEx( &m_section, spinCount, 
#ifndef HK_DEBUG
			CRITICAL_SECTION_NO_DEBUG_INFO
#else
			0
#endif
		);
	#endif

#endif
#	ifdef HK_PLATFORM_HAS_SPU
	m_this = this;
#	endif
}
#pragma warning(pop)

#ifdef HK_TIME_CRITICAL_SECTION_LOCKS

void hkCriticalSection::enter()
{
#if defined(HK_PLATFORM_SPU)
	// this is only allowed to be called by the simulated spu to access its own critical sections, eg. memory system
	HK_ASSERT2( 0xf0342323, this->m_this == this, "Call enter(*ppuAddress) instead" );
#endif

	if ( TryEnterCriticalSection(&m_section) )
	{
	}
	else
	{
		if ( HK_THREAD_LOCAL_GET(hkCriticalSection__m_timeLocks) )
		{
			HK_TIMER_BEGIN("CriticalLock", HK_NULL);
			EnterCriticalSection( &m_section );
			HK_TIMER_END();
		}
		else
		{
			EnterCriticalSection( &m_section );
		}
	}
}

#endif //! HK_TIME_CRITICAL_SECTION_LOCKS

#endif // HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED

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
