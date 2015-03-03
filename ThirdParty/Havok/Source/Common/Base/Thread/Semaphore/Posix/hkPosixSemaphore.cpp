/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Semaphore/hkSemaphore.h>

#ifndef HK_PLATFORM_LRB
#include <semaphore.h>
#endif

#include <Common/Base/Fwd/hkcstdio.h>
#include <Common/Base/Thread/Thread/Posix/hkPosixCheck.h>
#include <Common/Base/Thread/Thread/Posix/hkPthreadUtil.h>


hkSemaphore::hkSemaphore( int initialCount, int maxCount, int numSpinIterations )
{
	if (maxCount < 1 || initialCount > maxCount)
	{
		HK_BREAKPOINT(0);
		return;
	}

//	pthread_mutexattr_t mutex_attr;
//	HK_POSIX_CHECK(pthread_mutexattr_init(&mutex_attr) );
//	HK_POSIX_CHECK(pthread_mutexattr_setprotocol(&mutex_attr, PTHREAD_PRIO_INHERIT) );
//	HK_POSIX_CHECK(pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_RECURSIVE) ); 
//	HK_POSIX_CHECK(pthread_mutex_init(&m_semaphore.mutex, &mutex_attr) );

	HK_POSIX_CHECK(pthread_mutex_init(&m_semaphore.mutex, HK_NULL));

#if defined(HK_PLATFORM_LRB) || defined(HK_PLATFORM_NACL)
	// Attribute for LRB's pthread implementation currently (Aug 2008) must be NULL
  // NACL does not support cond attr either (Pepper16)
	HK_POSIX_CHECK( pthread_cond_init(&m_semaphore.cond, HK_NULL) );
#else
	pthread_condattr_t cond_attr;
	HK_POSIX_CHECK( pthread_condattr_init(&cond_attr) );
	HK_POSIX_CHECK( pthread_cond_init(&m_semaphore.cond, &cond_attr) );
#endif

	m_semaphore.curCount = initialCount;
	m_semaphore.maxCount = maxCount;
	m_semaphore.numSpinIterations = numSpinIterations;
}

hkSemaphore::~hkSemaphore()
{
	HK_POSIX_CHECK( pthread_cond_destroy(&m_semaphore.cond));
	HK_POSIX_CHECK( pthread_mutex_destroy(&m_semaphore.mutex));
}

void hkSemaphore::acquire()
{
	hkPthreadUtil::lockMutexWithSpinCount(m_semaphore.mutex, m_semaphore.numSpinIterations);

	while( m_semaphore.curCount <= 0 )
	{
		int error = pthread_cond_wait( &m_semaphore.cond, &m_semaphore.mutex);

		if (error)
		{
			perror("pthread_cond_wait failed" );
			HK_WARN(0x51640400, "pthread_cond_wait failed with " << error);
			HK_POSIX_CHECK( pthread_mutex_unlock(&m_semaphore.mutex) );
			return;
		}
	}

	m_semaphore.curCount--;

	HK_ASSERT2(0x49e37dc6, m_semaphore.curCount >= 0, "Illegal semaphore count value.");

	HK_POSIX_CHECK( pthread_mutex_unlock(&m_semaphore.mutex) );
}
	
void hkSemaphore::release(int count)
{
	hkPthreadUtil::lockMutexWithSpinCount(m_semaphore.mutex, m_semaphore.numSpinIterations);

	if (m_semaphore.curCount < m_semaphore.maxCount)
	{
		m_semaphore.curCount += count;
		if (m_semaphore.curCount > m_semaphore.maxCount)
		{
			m_semaphore.curCount = m_semaphore.maxCount;
		}	
	}
	else
	{
		HK_WARN(0x2baa4667, "Semaphore maxed out");
		HK_POSIX_CHECK( pthread_mutex_unlock(&m_semaphore.mutex) );
		return;
	}

	for( int i = 0; i < count; ++i )
	{
		int error = pthread_cond_signal(&m_semaphore.cond);

		if (error)
		{
			HK_WARN(0x1b4ded20, "Pthread cond signal failed" << error);
			HK_POSIX_CHECK( pthread_mutex_unlock(&m_semaphore.mutex) );
		}						
	}
	HK_POSIX_CHECK( pthread_mutex_unlock(&m_semaphore.mutex) ); 
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
