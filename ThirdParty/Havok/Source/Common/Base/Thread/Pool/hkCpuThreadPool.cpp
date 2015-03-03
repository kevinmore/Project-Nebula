/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Pool/hkCpuThreadPool.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>
#include <Common/Base/Memory/System/hkMemorySystem.h>
#include <Common/Base/System/hkBaseSystem.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>
#include <Common/Base/Thread/Thread/hkWorkerThreadContext.h>

extern HK_THREAD_LOCAL( int ) hkThreadNumber;


hkCpuThreadPoolCinfo::hkCpuThreadPoolCinfo(hkCpuThreadPool::WorkerFunction workerFunction)
	: m_workerFunction(workerFunction), m_numThreads(1), m_timerBufferPerThreadAllocation(0)
{
	m_threadName = "HavokWorkerThread";
	m_stackSize = hkThread::HK_THREAD_DEFAULT_STACKSIZE;
}


hkCpuThreadPool::SharedThreadData::SharedThreadData()
:	m_workerThreadFinished( 0, HK_MAX_NUM_THREADS ), m_gcThreadMemoryOnCompletion(false)
{
}


hkCpuThreadPool::WorkerThreadData::WorkerThreadData() : m_semaphore(0,1)
{	
	m_killThread = false;
	m_threadId = -1;
	m_sharedThreadData = HK_NULL;
}


/// Initialize multi-threading sharedData and create threads.
hkCpuThreadPool::hkCpuThreadPool(const hkCpuThreadPoolCinfo& ci)
{
	m_isRunning = false;	
	m_threadName = ci.m_threadName;
	m_stackSize = ci.m_stackSize;

	m_sharedThreadData.m_workerFunction = ci.m_workerFunction;
	m_sharedThreadData.m_timerBufferAllocation = ci.m_timerBufferPerThreadAllocation;

	int numThreads = ci.m_numThreads;
	if (numThreads >= HK_MAX_NUM_THREADS)
	{
		HK_WARN( 0xf0defd23, "You requested more threads than the hkCpuThreadPool supports - see HK_MAX_NUM_THREADS" );
		numThreads = HK_MAX_NUM_THREADS - 1;
	}

	m_sharedThreadData.m_numThreads = numThreads;

#if defined(HK_PLATFORM_XBOX360) 
	int numCores = 3;
	int numThreadsPerCore = 2;
#elif defined(HK_PLATFORM_WIN32)
	hkHardwareInfo info;
	hkGetHardwareInfo( info );
	int numCores = info.m_numThreads;
	int numThreadsPerCore = 1; //XX Work out hyper threading (not just logical versus physical processors). Can work out physical procs using asm, but does not take into account proper Dual / Quad procs
#endif

	for (int i = 0; i < numThreads; i++ )
	{
		WorkerThreadData& threadData = m_workerThreads[i];
		threadData.m_sharedThreadData = &m_sharedThreadData;
		threadData.m_threadId = i + 1; // don't start with thread 0 (assume that is the calling thread)		
		threadData.m_killThread = false;
		threadData.m_clearTimers = false;
		threadData.m_context = HK_NULL;

#if defined(HK_PLATFORM_XBOX360) || defined(HK_PLATFORM_WIN32)
		if (ci.m_hardwareThreadIds.getSize() > 0)
		{
			HK_ASSERT2( 0x975fe134, ci.m_hardwareThreadIds.getSize() >= numThreads, "If you initialize hardware thread ids, you must give an ID to all threads");
			threadData.m_hardwareThreadId = ci.m_hardwareThreadIds[i];
		}
		else
		{
			//X360: { 2,4,1,3,5, 0, 2,4,.. }
			int procGroup = (threadData.m_threadId % numCores) * numThreadsPerCore;
			threadData.m_hardwareThreadId = procGroup + (numThreadsPerCore > 1? ((threadData.m_threadId / numCores) % numThreadsPerCore) : 0 );
		}
#endif

		threadData.m_thread.startThread( &threadMain, &threadData, m_threadName, m_stackSize );
	}
	hkReferencedObject::setLockMode( hkReferencedObject::LOCK_MODE_AUTO);
}

void* HK_CALL hkCpuThreadPool::threadMain(void* v)
{

#if defined HK_PLATFORM_PS3_PPU && defined HK_PS3_NO_TLS
	sys_ppu_thread_t id;
	sys_ppu_thread_get_id(&id);
	g_hkPs3PrxTls->registerThread((hkUlong) id);
#endif

	hkCpuThreadPool::WorkerThreadData& threadData = *static_cast<hkCpuThreadPool::WorkerThreadData*>(v);
	hkCpuThreadPool::SharedThreadData& sharedData = *threadData.m_sharedThreadData;
	{
		hkWorkerThreadContext threadContext(threadData.m_threadId);
		threadData.m_context = &threadContext;

		// Allocate a monitor stream for this thread (this  enables timers)
		if (sharedData.m_timerBufferAllocation > 0)
		{		
			hkMonitorStream::getInstance().resize(sharedData.m_timerBufferAllocation);
		}
		threadData.m_monitorStreamBegin = hkMonitorStream::getInstance().getStart();
		threadData.m_monitorStreamEnd = hkMonitorStream::getInstance().getEnd();

		hkCheckDeterminismUtil::initThread();

#ifdef HK_PLATFORM_XBOX360
		XSetThreadProcessor(GetCurrentThread(), threadData.m_hardwareThreadId );
#elif defined(HK_PLATFORM_DURANGO)
		PROCESSOR_NUMBER pp, prevPP;
		pp.Group = 0;
		pp.Number = (BYTE)threadData.m_hardwareThreadId;
		SetThreadIdealProcessorEx(GetCurrentThread(), &pp, &prevPP);
#elif defined(HK_PLATFORM_WIN32) && !defined(HK_PLATFORM_WINRT)

		SetThreadIdealProcessor(GetCurrentThread(), threadData.m_hardwareThreadId);
		// Can use SetThreadAffityMask to be more force-full.
#endif	

		// Wait for the main thread to release the worker thread
		threadData.m_semaphore.acquire();

		// The thread "main loop"
		while (threadData.m_killThread == false)
		{
			if (threadData.m_clearTimers)
			{
				hkMonitorStream::getInstance().reset();
				threadData.m_monitorStreamEnd = hkMonitorStream::getInstance().getEnd();
				threadData.m_clearTimers = false;
			}

			hkCheckDeterminismUtil::workerThreadStartFrame(false);

			// Enable timers for critical sections just during the step call
			hkCriticalSection::setTimersEnabled();

			// Work on the work load
			sharedData.m_workerFunction(sharedData.m_workLoad);

			// Disable timers for critical sections just during the step call
			hkCriticalSection::setTimersDisabled();

			// Note collected timer data	
			threadData.m_monitorStreamEnd = hkMonitorStream::getInstance().getEnd();

			hkCheckDeterminismUtil::workerThreadFinishFrame();

			// Perform garbage collection when requested
			if (sharedData.m_gcThreadMemoryOnCompletion)
			{
				hkMemorySystem::getInstance().garbageCollectThread(threadContext.m_memoryRouter);
			}

			// Release any thread (usually the main thread) which may be waiting for all worker threads to finish.
			sharedData.m_workerThreadFinished.release();

			// Immediately wait until the main thread releases the thread again
			threadData.m_semaphore.acquire();
		}

#if 0 //def HK_PLATFORM_PSVITA
		{
			hkUint32 id = (hkUint32)hkThread::getMyThreadId(); 
			hkStringBuf ef;
			ef.printf( "Havok thread shutting down (havok id:0x%x, system id:0x%x)", id, (hkUint32)sceKernelGetThreadId() );
			HK_REPORT( ef.cString() );
		}
#endif	
	}
#if defined HK_PLATFORM_PS3_PPU && defined HK_PS3_NO_TLS
	g_hkPs3PrxTls->unregisterThread((hkUlong) id);
#endif

	return 0;
}


// Destroy threads and delete sharedData.
hkCpuThreadPool::~hkCpuThreadPool()
{
	waitForCompletion();

	for (int i = 0; i < m_sharedThreadData.m_numThreads; i++)
	{
		WorkerThreadData& data = m_workerThreads[i];
		data.m_killThread = true;
		data.m_semaphore.release(); // sets the thread off to enable it to finish
	}

	for (int i = 0; i < m_sharedThreadData.m_numThreads; i++)
	{
		WorkerThreadData& data = m_workerThreads[i];
		data.m_thread.joinThread();
	}

	hkReferencedObject::setLockMode( hkReferencedObject::LOCK_MODE_NONE);
}

void hkCpuThreadPool::gcThreadMemoryOnNextCompletion()
{
	m_sharedThreadData.m_gcThreadMemoryOnCompletion = true;
}

void hkCpuThreadPool::addThread()
{
	HK_ASSERT2(0xad67bd88, ! m_isRunning, "You can only add or remove working threads via calls from the master thread and not between processWorkLoad() and waitForCompletion() calls. ");

#if defined(HK_PLATFORM_HAS_SPU)
	HK_ASSERT2(0xcede9735, m_sharedThreadData.m_numThreads < 2, "Only 2 PPU threads are supported on the PS3" );
#endif

	HK_ASSERT3(0xcede9734, m_sharedThreadData.m_numThreads < HK_MAX_NUM_THREADS, "A maximum of " << HK_MAX_NUM_THREADS << " threads are supported." );
	WorkerThreadData& data = m_workerThreads[m_sharedThreadData.m_numThreads];
	data.m_sharedThreadData = &m_sharedThreadData;
	data.m_threadId = m_sharedThreadData.m_numThreads+1;
	data.m_killThread = false;
	data.m_clearTimers = false;

#if (defined(HK_PLATFORM_WIN32) || defined(HK_PLATFORM_XBOX360)) 
#if defined HK_PLATFORM_XBOX360
	int numCores = 3;
	int numThreadsPerCore = 2; 
#elif defined(HK_PLATFORM_WIN32)
	SYSTEM_INFO sysInfo;
#	if !defined(HK_PLATFORM_WINRT)
	GetSystemInfo(&sysInfo);
#	else
	GetNativeSystemInfo(&sysInfo);
#	endif
	int numCores = sysInfo.dwNumberOfProcessors;
	int numThreadsPerCore = 1; //XX Work out hyper threading (not just logical versus physical processors)
	numCores /= numThreadsPerCore;
#endif

	//X360: { 2,4,1,3,5, 0, 2,4,.. }
	int procGroup = (data.m_threadId % numCores) * numThreadsPerCore;
	data.m_hardwareThreadId = procGroup + (numThreadsPerCore > 1? ((data.m_threadId / numCores) % numThreadsPerCore) : 0 );
#endif

	data.m_thread.startThread( &threadMain, &data, m_threadName, m_stackSize);
	m_sharedThreadData.m_numThreads++;
}


void hkCpuThreadPool::removeThread()
{
	HK_ASSERT2(0xad67bd89, !m_isRunning,
		"You can only add or remove working threads via calls from the master thread and not between processWorkLoad() and waitForCompletion() calls. ");

	HK_ASSERT2(0xcede9735, m_sharedThreadData.m_numThreads > 0, "You cannot set a negative number of threads" );

	--m_sharedThreadData.m_numThreads;

	WorkerThreadData& data = m_workerThreads[m_sharedThreadData.m_numThreads];

	// Signal the thread to be killed, and release the thread
	data.m_killThread = true;
	data.m_semaphore.release();

	
	// Close handle to thread to avoid resource leak
	data.m_thread.joinThread();
}


void hkCpuThreadPool::processWorkLoad(void * workLoad)
{
	HK_ASSERT2(0xad56dd77, m_isRunning == false,
		"Calling hkCpuThreadPool::processWorkLoad() for the second time, without having called hkCpuThreadPool::waitForCompletion().");
	m_isRunning = true;
	m_sharedThreadData.m_workLoad = workLoad;

	for (int i = m_sharedThreadData.m_numThreads - 1; i >=0; i--)
	{
		WorkerThreadData& data = m_workerThreads[i];
		data.m_semaphore.release();
	}
}


void hkCpuThreadPool::waitForCompletion()
{
	// This function does nothing if waitForStepWorldFinished() is called before startStepWorld()
	if ( m_isRunning )
	{
		for (int i = 0; i < m_sharedThreadData.m_numThreads; ++i)
		{
			m_sharedThreadData.m_workerThreadFinished.acquire();
		}
		m_isRunning = false;
		m_sharedThreadData.m_gcThreadMemoryOnCompletion = false;
	}
}


bool hkCpuThreadPool::isProcessing() const
{
	return m_isRunning;
}


void hkCpuThreadPool::appendTimerData(  hkArrayBase<hkTimerData>& timerDataOut, hkMemoryAllocator& alloc )
{
	for (int i = 0; i < m_sharedThreadData.m_numThreads; ++i)
	{
		hkTimerData& info = timerDataOut._expandOne(alloc);
		info.m_streamBegin = m_workerThreads[i].m_monitorStreamBegin;
		info.m_streamEnd = m_workerThreads[i].m_monitorStreamEnd;
	}
}


void hkCpuThreadPool::clearTimerData()
{
	for ( int i = 0; i < m_sharedThreadData.m_numThreads; ++i )
	{
		m_workerThreads[i].m_monitorStreamEnd = m_workerThreads[i].m_monitorStreamBegin;
		m_workerThreads[i].m_clearTimers = true;
	}
}


int hkCpuThreadPool::getNumThreads() const
{
	return m_sharedThreadData.m_numThreads;
}


void hkCpuThreadPool::setNumThreads(int numThreads)
{
	if ( numThreads >= HK_MAX_NUM_THREADS )
	{
		numThreads = HK_MAX_NUM_THREADS - 1;
	}
	while( m_sharedThreadData.m_numThreads < numThreads )
	{
		addThread();
	}

	while( m_sharedThreadData.m_numThreads > numThreads )
	{
		removeThread();
	}
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
