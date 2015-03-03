/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Visualize/hkVisualize.h>
#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Base/Memory/Tracker/Report/hkVdbStreamReportUtil.h>
#include <Common/Base/Memory/Tracker/ScanCalculator/hkTrackerSnapshotUtil.h>
#include <Common/Base/System/Io/Writer/VdbCommand/hkVdbCommandWriter.h>
#include <Common/Visualize/hkProcessHandler.h>
#include <Common/Visualize/hkVisualDebuggerProtocol.h>
#include <Common/Visualize/Process/hkMemorySnapshotProcess.h>

int hkMemorySnapshotProcess::m_tag = 0;

void HK_CALL hkMemorySnapshotProcess::registerProcess()
{
	m_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
}

hkProcess* HK_CALL hkMemorySnapshotProcess::create( const hkArray<hkProcessContext*>& contexts )
{
	return new hkMemorySnapshotProcess( contexts );
}

hkMemorySnapshotProcess::hkMemorySnapshotProcess( const hkArray<hkProcessContext*>& contexts )
: hkProcess( true )
{

}

void hkMemorySnapshotProcess::step( hkReal frameTimeInMs )
{
	// step once, send the HKX file in full (in native console format, so less likely to hit any serialization issues)
	if (m_outStream)
	{
		hkVdbCommandWriter writer( m_outStream->getStreamWriter(), hkVisualDebuggerProtocol::MEMORY_SNAPSHOT );
		
		hkOstream vdbOut( &writer );

		hkTrackerScanSnapshot* memorySnapshot = hkTrackerSnapshotUtil::createSnapshot();
		
		hkVdbStreamReportUtil::generateReport( memorySnapshot, vdbOut );

		memorySnapshot->removeReference();
	}

	// Now we have done the work, we can turn ourselves off.
	if (m_processHandler)
	{
		m_processHandler->deleteProcess( m_tag );

		//as our name has a '*' in it, the VDB clients will expect us to have deleted ourselves, so should be fine.
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
