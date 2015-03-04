/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Thread/hkThread.h>
#include <Common/Base/Fwd/hkwindows.h>

#include <ppl.h>
#include <ppltasks.h>

using namespace std;
using namespace Platform;
using namespace Windows::Foundation;
using namespace Windows::System::Threading;


hkThread::hkThread()
	: m_thread(nullptr)
{
}

hkThread::~hkThread()
{
	
}

void hkThread::joinThread()
{
	if (m_thread)
	{
		if (m_thread->Status == AsyncStatus::Started)
		{
			// Could busy wait on Status here if u prefer

			try {
				concurrency::task<void> waitTask(m_thread);
				waitTask.wait();
			}
			catch (Platform::Exception^ e)
			{

			}
		}
		m_thread = nullptr;
	}
}


hkResult hkThread::startThread( hkThread::StartFunction func, void* arg, const char* name, int stackSize)
{
	auto workItemHandler = ref new WorkItemHandler([=](IAsyncAction^)
	{
		try
		{
			func(arg);
		}
		catch (...) { }

		// Clean up any TLS allocations made by this thread.
		//TlsShutdown();

	}, Platform::CallbackContext::Any);
	
	try
	{
		m_thread = ThreadPool::RunAsync(workItemHandler, WorkItemPriority::Normal, WorkItemOptions::TimeSliced );
	}
	catch (...)
	{
		HK_WARN_ALWAYS(0xabba5e2e, "Could not create worker thread!");
		m_thread = nullptr;
	}
	return HK_SUCCESS;
}

hkThread::Status hkThread::getStatus()
{
	if (m_thread == nullptr)
	{
		return THREAD_NOT_STARTED;
	}

	switch (m_thread->Status)
	{
		case AsyncStatus::Completed:
		case AsyncStatus::Canceled:
		case AsyncStatus::Error:
			return THREAD_TERMINATED;

		case AsyncStatus::Started:
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
	return reinterpret_cast<void*>( m_thread );
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
