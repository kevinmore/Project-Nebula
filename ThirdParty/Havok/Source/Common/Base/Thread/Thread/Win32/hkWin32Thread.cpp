/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Thread/hkThread.h>

#ifdef HK_PLATFORM_WIN32
#	include <Common/Base/Fwd/hkwindows.h>
#elif defined(HK_PLATFORM_XBOX360)
# 	include <xtl.h>
#endif

hkThread::hkThread()
	: m_thread(HK_NULL)
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
		WaitForSingleObject( m_thread, INFINITE );
		CloseHandle( m_thread );
		m_thread = HK_NULL;
	}
}


// Taken from XeDK documentation, works on Win32 on Visual Studio 2008 too.
typedef struct tagTHREADNAME_INFO {
	DWORD dwType;     // Must be 0x1000
	LPCSTR szName;    // Pointer to name (in user address space)
	DWORD dwThreadID; // Thread ID (-1 for caller thread)
	DWORD dwFlags;    // Reserved for future use; must be zero
} THREADNAME_INFO;

hkResult hkThread::startThread( hkThread::StartFunction func, void* arg, const char* name, int stackSize)
{
	DWORD tempThreadId;
	m_thread = CreateThread(
		HK_NULL,						//LPSECURITY_ATTRIBUTES ThreadAttributes,
		stackSize ,						//DWORD StackSize,
		(LPTHREAD_START_ROUTINE)func,	//LPTHREAD_START_ROUTINE StartAddress,
		arg,							//LPVOID Parameter,
		0,								//DWORD CreationFlags,
		(LPDWORD)&tempThreadId			//LPDWORD ThreadId
		);

	if (m_thread == HK_NULL)
	{
		return HK_FAILURE;
	}
	else
	{
		m_threadId = (hkUint32) tempThreadId;

#if !defined(_MSC_VER) ||  (_MSC_VER < 1700) // Not understood by VS2012
		// Set thread name
		THREADNAME_INFO info;

		info.dwType = 0x1000;
		info.szName = name;
		info.dwThreadID = tempThreadId;
		info.dwFlags = 0;

		__try
		{
			RaiseException( 0x406D1388, 0, sizeof(info)/sizeof(DWORD), (CONST ULONG_PTR *)(&info) );
		}
		__except( GetExceptionCode()==0x406D1388 ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_EXECUTE_HANDLER )
		{
		}
#endif
		return HK_SUCCESS;
	}
}

hkThread::Status hkThread::getStatus()
{
	if (m_thread == HK_NULL)
	{
		return THREAD_NOT_STARTED;
	}
	DWORD exitCode;
	GetExitCodeThread( m_thread,&exitCode);
	if ( exitCode == STILL_ACTIVE )
	{
		return THREAD_RUNNING;
	}
	return THREAD_TERMINATED;
}


hkUint64 HK_CALL hkThread::getMyThreadId()
{
	return (hkUint64)GetCurrentThreadId();
}

hkUint64 hkThread::getChildThreadId()
{
	return m_threadId;
}

void* hkThread::getHandle()
{
	return m_thread;
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
