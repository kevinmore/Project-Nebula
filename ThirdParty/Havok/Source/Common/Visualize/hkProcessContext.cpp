/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Visualize/hkVisualize.h>
#include <Common/Visualize/hkProcessContext.h>
#include <Common/Visualize/hkVisualDebugger.h>
#include <Common/Base/Thread/Job/ThreadPool/hkJobThreadPool.h>


void hkProcessContext::syncTimers( hkThreadPool* threadPool )
{
	m_monitorStreamBegins.clear();
	m_monitorStreamEnds.clear();
	m_monitorStreamBegins.pushBack( hkMonitorStream::getInstance().getStart() );
	m_monitorStreamEnds.pushBack( hkMonitorStream::getInstance().getEnd() );
	if ( threadPool != HK_NULL )
	{
		addThreadPoolTimers( threadPool );
	}
}

void hkProcessContext::addThreadPoolTimers( hkThreadPool* threadPool )
{
	hkArray<hkTimerData>::Temp data;
	threadPool->appendTimerData( data, hkMemoryRouter::getInstance().temp() );
	for ( int i = 0; i < data.getSize(); ++i )
	{
		m_monitorStreamBegins.pushBack( data[i].m_streamBegin );
		m_monitorStreamEnds.pushBack( data[i].m_streamEnd );
	}
}

hkProcessContext::~hkProcessContext()
{
	if (m_owner)
	{
		m_owner->removeContext( this );
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
