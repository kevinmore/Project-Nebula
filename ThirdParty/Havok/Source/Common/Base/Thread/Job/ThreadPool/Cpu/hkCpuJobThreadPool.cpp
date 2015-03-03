/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Job/ThreadPool/Cpu/hkCpuJobThreadPool.h>


hkCpuJobThreadPoolCinfo::hkCpuJobThreadPoolCinfo() : hkCpuThreadPoolCinfo(&hkCpuJobThreadPool::workerFunction) {}


	/// Initialize multi-threading sharedThreadData and create threads.
hkCpuJobThreadPool::hkCpuJobThreadPool(const hkCpuJobThreadPoolCinfo& ci) : m_threadPool(ci) {}


void hkCpuJobThreadPool::processAllJobs( hkJobQueue* queue, hkJobType firstJobType_unused )
{
#if defined(HK_PLATFORM_HAS_SPU)
	HK_ASSERT2(0x62f299e6, queue->m_hwSetup.m_cellRules == hkJobQueueHwSetup::PPU_CAN_TAKE_SPU_TASKS,
		"Must set hkJobQueue::m_hwSetup.m_cellRules to PPU_CAN_TAKE_SPU_TASKS if using hkCpuJobThreadPool instead of hkSpuJobThreadPool on PS3.");
#endif

	m_threadPool.processWorkLoad(queue);
}


bool hkCpuJobThreadPool::isProcessing() const
{
	return m_threadPool.isProcessing();
}


int hkCpuJobThreadPool::getNumThreads() const
{
	return m_threadPool.getNumThreads();
}


void hkCpuJobThreadPool::setNumThreads(int numThreads)
{
	m_threadPool.setNumThreads(numThreads);
}


void hkCpuJobThreadPool::waitForCompletion()
{
	m_threadPool.waitForCompletion();
}


void hkCpuJobThreadPool::appendTimerData(  hkArrayBase<hkTimerData>& timerDataOut, hkMemoryAllocator& alloc )
{
	m_threadPool.appendTimerData(timerDataOut, alloc);
}


void hkCpuJobThreadPool::clearTimerData()
{
	m_threadPool.clearTimerData();
}


void hkCpuJobThreadPool::gcThreadMemoryOnNextCompletion()
{
	m_threadPool.gcThreadMemoryOnNextCompletion();
}


void hkCpuJobThreadPool::workerFunction(void* workLoad)
{
	hkJobQueue* jobQueue = static_cast<hkJobQueue*>(workLoad);
	jobQueue->processAllJobs();
}


#if 0
static void* HK_CALL Function_For_Docs_Do_not_Delete(void *v)
{
	extern hkResult waitForStartSignal();
	extern void doWork();
	extern void sendWorkDoneSignal();
	
	hkMemorySystem& memSystem = hkMemorySystem::getInstance();
	hkMemoryRouter memRouter;
	memSystem.threadInit( memRouter, "worker", hkMemorySystem::FLAG_PERSISTENT );
	hkBaseSystem::initThread(&memRouter);

	while( waitForStartSignal() == HK_SUCCESS )
	{
		memSystem.threadInit( memRouter, "worker", hkMemorySystem::FLAG_TEMPORARY );
		doWork();
		memSystem.threadQuit( memRouter, hkMemorySystem::FLAG_TEMPORARY );
		sendWorkDoneSignal();
	}

	hkBaseSystem::quitThread();
	memSystem.threadQuit( memRouter, hkMemorySystem::FLAG_PERSISTENT );
	
	return 0;
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
