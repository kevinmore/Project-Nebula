/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Thread/hkThread.h>
#include <Common/Base/Thread/Thread/Posix/hkPosixCheck.h>
#include <Common/Base/Memory/System/hkMemorySystem.h>

#ifdef HK_PLATFORM_LRBSIM
#include <common/winpthread.h>
#else
#include <pthread.h>
#endif
#ifdef HK_PLATFORM_PS4
#include <pthread_np.h>
#endif
#ifdef HK_PLATFORM_LINUX
#include <sys/prctl.h>
#endif
#include <Common/Base/Fwd/hkcstdio.h>

hkThread::hkThread()
	: m_thread(HK_NULL), m_threadId(THREAD_NOT_STARTED)
{
}

hkThread::~hkThread()
{
	joinThread();
}

void hkThread::joinThread()
{
	if( m_thread )
	{
		HK_POSIX_CHECK( pthread_join((pthread_t)m_thread, HK_NULL) );
		m_thread = HK_NULL;
	}
}

#if defined(HK_PLATFORM_IOS) || defined(HK_PLATFORM_MAC386) || defined(HK_PLATFORM_LINUX)
// MacOS does not support naming another thread than the current one, so we need to wrap
// thread startup to define the thread name there. Linux with glibc < 2.12 does not provide
// pthread_setname_np so we need to use the prctl method which as well only works on the
// current thread.
namespace
{

struct hkThreadNamerParam
{
	// This class should not use instances of hkMemoryRouter, as they are not initialised yet
	// by the thread being named.

	void* operator new(size_t size)
	{
		return hkMemoryRouter::easyAlloc(*hkMemorySystem::getInstance().getUncachedLockedHeapAllocator(), size);
	}
	void operator delete(void* ptr)
	{
		hkMemoryRouter::easyFree(*hkMemorySystem::getInstance().getUncachedLockedHeapAllocator(), ptr);
	}

	hkThreadNamerParam(hkThread::StartFunction f, void* a, const char* n)
		: func(f), arg(a), name(hkString::strDup(n, *hkMemorySystem::getInstance().getUncachedLockedHeapAllocator()))
	{}
	~hkThreadNamerParam()
	{
		hkString::strFree(name, *hkMemorySystem::getInstance().getUncachedLockedHeapAllocator());
	}

	hkThread::StartFunction func;
	void* arg;
	char* name;
};

void* hkThreadNamerWrapper(void* param)
{
	hkThreadNamerParam* namerParam = reinterpret_cast<hkThreadNamerParam*>(param);
#if defined(HK_PLATFORM_LINUX)
	prctl(PR_SET_NAME, (unsigned long)namerParam->name, 0, 0, 0);
#else
	pthread_setname_np(namerParam->name);
#endif
	hkThread::StartFunction func = namerParam->func;
	void* arg = namerParam->arg;
	delete namerParam;
	return func(arg);
}

}
#endif

hkResult hkThread::startThread( hkThread::StartFunction func, void* arg, const char* name, /*UNUSED*/ int stackSize )
{
	pthread_t thread;
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

#if defined(HK_DEBUG) && defined(HK_PLATFORM_PS4)
	pthread_attr_setstacksize(&attr, 512*1024);
#endif

#ifdef HK_PLATFORM_PS4
	int error = pthread_create_name_np( &thread, &attr, func, arg, name );
#elif defined(HK_PLATFORM_IOS) || defined(HK_PLATFORM_MAC386) || defined(HK_PLATFORM_LINUX)
	int error = pthread_create(	&thread, &attr, hkThreadNamerWrapper,
								new hkThreadNamerParam(func, arg, name));
#else
	int error = pthread_create( &thread, &attr, func, arg );
#endif
	if (error)
	{
		perror("Thread Error\n" );
		return HK_FAILURE;
	}
#if defined(HK_PLATFORM_ANDROID) || defined(HK_PLATFORM_TIZEN)
	pthread_setname_np(thread, name);
#endif
	m_thread = (void*)thread;
	m_threadId = THREAD_RUNNING;

	return HK_SUCCESS;
}

hkThread::Status hkThread::getStatus()
{
	return static_cast<Status>(m_threadId);
}

hkUint64 hkThread::getChildThreadId()
{
	return hkUlong(m_thread);
}

void* hkThread::getHandle()
{
	return m_thread;
}

hkUint64 hkThread::getMyThreadId()
{
	pthread_t tid = pthread_self();
	return (hkUint64)tid;
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
