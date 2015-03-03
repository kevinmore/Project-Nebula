/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Internal/hkInternal.h>
#include <Common/Internal/GeometryProcessing/JobQueue/hkgpJobQueue.h>
#include <Common/Base/Thread/Thread/hkThread.h>
#include <Common/Base/Memory/System/hkMemorySystem.h>
#include <Common/Base/System/hkBaseSystem.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>
#include <Common/Base/Thread/Semaphore/hkSemaphore.h>

hkgpJobQueue::hkgpJobQueue(int numThreads)
{
#if defined(HK_PLATFORM_WIN32)
	if(numThreads<=0)
	{
		hkHardwareInfo	hwInfos;
		hkGetHardwareInfo(hwInfos);
		numThreads	=	hwInfos.m_numThreads;
	}
#else
	numThreads	=	1;
#endif	
	m_numPendings	=	0;
	m_jobsLock		=	new hkCriticalSection(0);
	m_newJobEvent	=	new hkSemaphore();
	m_endJobEvent	=	new hkSemaphore();
	m_endThreadEvent=	new hkSemaphore();
	m_threads.setSize(numThreads,0);
	if(m_threads.getSize()>1)
	{
		for(int i=0;i<numThreads;++i)
		{
			m_threads[i]=new hkThread();
			m_threads[i]->startThread(&threadStart,this,"");
		}
	}
}

//
hkgpJobQueue::~hkgpJobQueue()
{
	if(m_threads.getSize()>1)
	{
		m_jobsLock->enter();
		for(int i=0;i<m_threads.getSize();++i)
		{
			m_jobs.pushBack((IJob*)1);
		}
		m_jobsLock->leave();
		m_newJobEvent->release(m_threads.getSize());
		waitForCompletion();
		for(int i=0;i<m_threads.getSize();++i)
		{
			m_endThreadEvent->acquire();
		}
	}
	for(int i=0;i<m_threads.getSize();++i)
	{
		delete m_threads[i];
	}
	delete m_newJobEvent;
	delete m_endJobEvent;
	delete m_endThreadEvent;
	delete m_jobsLock;	
}

//
void		hkgpJobQueue::push(IJob* job)
{
	if(m_threads.getSize()>1)
	{
		m_jobsLock->enter();
		m_jobs.pushBack(job);		
		m_jobsLock->leave();
		m_newJobEvent->release();
	}
	else
	{
		job->run();
		delete job;
	}
}

//
void		hkgpJobQueue::waitForCompletion()
{
	if(m_threads.getSize()>1)
	{
		for(;;)
		{
			bool	wait=false;
			m_jobsLock->enter();
			wait	=	m_numPendings || m_jobs.getSize();
			m_jobsLock->leave();
			if(wait)
			{
				m_endJobEvent->acquire();
			}
			else
			{
				break;
			}
		}
	}		
}

//
void		hkgpJobQueue::threadMain()
{
	while(m_newJobEvent->acquire(),true)
	{
		IJob*	ijob=0;
		m_jobsLock->enter();
		if(m_jobs.getSize())
		{
			ijob=m_jobs.back();
			m_numPendings++;
			m_jobs.popBack();
			if(m_jobs.getSize()) m_newJobEvent->release();
		}
		m_jobsLock->leave();
		if(ijob)
		{				
			const bool finalJob=(ijob==(void*)1);
			if(!finalJob)
			{
				ijob->run();
				delete ijob;
			}
			m_jobsLock->enter();
			m_numPendings--;
			if(m_jobs.getSize()) m_newJobEvent->release();
			m_jobsLock->leave();
			m_endJobEvent->release();
			if(finalJob) break;
		}
	}
	m_endJobEvent->release();
	m_endThreadEvent->release();
}

//
void*		hkgpJobQueue::threadStart(void* arg)
{
	hkMemoryRouter memoryRouter;
	hkMemorySystem::getInstance().threadInit(memoryRouter, "hkgpJobsQueue");
	hkBaseSystem::initThread(&memoryRouter);
	((hkgpJobQueue*)arg)->threadMain();
	hkBaseSystem::quitThread();
	hkMemorySystem::getInstance().threadQuit(memoryRouter);
	return(0);
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
