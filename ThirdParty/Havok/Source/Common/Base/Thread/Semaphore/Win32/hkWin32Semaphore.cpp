/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Semaphore/hkSemaphore.h>

#if defined(HK_PLATFORM_SIM)
#include <Common/Base/Spu/Simulator/hkSpuSimulator.h>
#endif

#ifdef HK_PLATFORM_WIN32
#	include <Common/Base/Fwd/hkwindows.h>
#elif defined(HK_PLATFORM_XBOX360)
# 	include <xtl.h>
#endif

#if defined(HK_PLATFORM_SIM)

	#if !defined(HK_PLATFORM_SPU)

		hkSemaphore::hkSemaphore( int initialCount, int maxCount, int numSpinIterations )
		{
			m_semaphore = CreateSemaphore( 
					NULL,			// default security attributes
					initialCount,	// initial count
					maxCount,		// maximum count
					NULL);			// unnamed semaphore
		}

		hkSemaphore::~hkSemaphore()
		{
			CloseHandle( m_semaphore );
		}

		void hkSemaphore::acquire()
		{
			HK_ON_DEBUG(DWORD dwWaitResult =) WaitForSingleObject( m_semaphore, INFINITE );          // zero-second time-out interval
			HK_ASSERT(0xf0324354, dwWaitResult == WAIT_OBJECT_0);
		}

		void hkSemaphore::release(int count)
		{
			HK_ON_DEBUG(BOOL success =) ReleaseSemaphore( 
				m_semaphore,	// handle to semaphore
				count,			// increase count by 'count'
				NULL);			// not interested in previous count

			HK_ASSERT2(0xad7633dd, success, "hkSemaphore::release() failed!!");
		}

	#endif

		// static function
	void hkSemaphore::acquire(hkSemaphore* semaphoreOnPpu)
	{
	#if !defined(HK_PLATFORM_SPU)
		semaphoreOnPpu->acquire();
	#else
		HK_ASSERT(0xaf436250, hkSpuSimulator::Client::getInstance());
		hkSpuSimulator::Client::getInstance()->acquireSemaphore( semaphoreOnPpu );
	#endif
	}

		// static function
	void hkSemaphore::release(hkSemaphore* semaphoreOnPpu, int count)
	{
	#if !defined(HK_PLATFORM_SPU)
		semaphoreOnPpu->release(count);
	#else
		HK_ASSERT(0xaf436251, hkSpuSimulator::Client::getInstance());
		hkSpuSimulator::Client::getInstance()->releaseSemaphore( semaphoreOnPpu, count );
	#endif
	}

#else // #elif !defined(HK_PLATFORM_SIM)

	hkSemaphore::hkSemaphore( int initialCount, int maxCount, int numSpinIterations )
	{
	#if !defined(HK_PLATFORM_WINRT) && !defined(HK_PLATFORM_DURANGO)
		m_semaphore = CreateSemaphore( 
			NULL,			// default security attributes
			initialCount,	// initial count
			maxCount,		// maximum count
			NULL);			// unnamed semaphore
	#else
		m_semaphore = CreateSemaphoreExW( 
			NULL,			// default security attributes
			initialCount,	// initial count
			maxCount,		// maximum count
			NULL, // unnamed (wchar_t here) semaphore
			0, //flags
			STANDARD_RIGHTS_ALL | SEMAPHORE_MODIFY_STATE  //desiredAccess
			);			
	#endif
	}

	hkSemaphore::~hkSemaphore()
	{
		CloseHandle( m_semaphore );
	}

	void hkSemaphore::acquire()
	{
		HK_ON_DEBUG( DWORD dwWaitResult = )
	#if !defined(HK_PLATFORM_WINRT) && !defined(HK_PLATFORM_DURANGO)
		WaitForSingleObject( m_semaphore, INFINITE );          // zero-second time-out interval
	#else
		WaitForSingleObjectEx( m_semaphore, INFINITE, FALSE ); 
	#endif
		HK_ASSERT(0xf0324354, dwWaitResult == WAIT_OBJECT_0);
	}

	void hkSemaphore::release(int count)
	{
		ReleaseSemaphore( 
				m_semaphore,	// handle to semaphore
				count,			// increase count by 'count'
				NULL);			// not interested in previous count
	}

		// static function
	void hkSemaphore::acquire(hkSemaphore* semaphore)
	{
		semaphore->acquire();
	}

		// static function
	void hkSemaphore::release(hkSemaphore* semaphore, int count)
	{
		semaphore->release(count);
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
