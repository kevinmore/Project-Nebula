/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Common/Base/hkBase.h>
#include <Common/Base/Types/hkBaseTypes.h>
#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>

hkMemoryExceptionTestingUtil* hkMemoryExceptionTestingUtil::s_instance = HK_NULL;

hkMemoryAllocator::MemoryState hkOutOfMemoryState = hkMemoryAllocator::MEMORY_STATE_OK;
#if defined(HK_PLATFORM_HAS_SPU)
#	include <Common/Base/Thread/JobQueue/hkJobQueue.h>
#endif

void hkSetOutOfMemoryState( hkMemoryAllocator::MemoryState state )
{
#if defined(HK_PLATFORM_HAS_SPU)
	// signal the spus
	if ( hkJobQueue::s_instance )
	{
		hkJobQueue* queue = hkJobQueue::s_instance;

		HK_ALIGN(char dynamicDataStorage[sizeof(hkJobQueue::DynamicData)], 16);
		hkJobQueue::DynamicData* data = queue->lockQueue( dynamicDataStorage );
		data->m_outOfMemory = (state != hkMemoryAllocator::MEMORY_STATE_OK);
		queue->unlockQueue( data );
	}

#endif
	hkOutOfMemoryState = state;
}


hkMemoryExceptionTestingUtil::hkMemoryExceptionTestingUtil()
{
	m_frameCounter = 0;
	m_frameCounter = 0;
	m_outOfMemory = false;
	m_allowMemoryExceptions = false;
}

hkMemoryExceptionTestingUtil::~hkMemoryExceptionTestingUtil()
{

}

void hkMemoryExceptionTestingUtil::create()
{
	delete s_instance;
	s_instance = new hkMemoryExceptionTestingUtil();
}

void hkMemoryExceptionTestingUtil::destroy()
{
	delete s_instance;
	s_instance = HK_NULL;
}

hkMemoryExceptionTestingUtil& hkMemoryExceptionTestingUtil::getInstance()
{
	return *s_instance;
}

void hkMemoryExceptionTestingUtil::startNewDemoImpl()
{
	m_frameCounter = 0;

	for (int i = 0; i < MAX_CHECK_ID; i++)
	{
		m_wasCheckIdThrown[i] = false;
	}

}


void hkMemoryExceptionTestingUtil::startFrameImpl()
{
	//getInstance().m_idToCauseOutOfMemoryException++;

	// reset out of memory conditions
	m_outOfMemory = false;
	m_frameCounter++;
}

void hkMemoryExceptionTestingUtil::endFrameImpl()
{
	// do nothing
}

void hkMemoryExceptionTestingUtil::allowMemoryExceptionsImpl(bool allowMemExceptions)
{
	m_allowMemoryExceptions = allowMemExceptions;
}

bool hkMemoryExceptionTestingUtil::isMemoryAvailableImpl(int id)
{
	if (m_outOfMemory && m_allowMemoryExceptions)
	{
		return false;
	}

	if (m_frameCounter > 50 && m_frameCounter % 3 == 0 && m_allowMemoryExceptions)
	{
		if (!m_wasCheckIdThrown[id])
		{
			m_wasCheckIdThrown[id] = true;
			m_outOfMemory = true;
			return false;
		}
	}
	
	return true;
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
