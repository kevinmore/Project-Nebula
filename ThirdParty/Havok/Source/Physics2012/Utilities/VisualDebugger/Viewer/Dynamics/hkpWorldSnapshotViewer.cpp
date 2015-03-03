/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/hkBase.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>

//#define HK_DISABLE_DEBUG_DISPLAY
#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>
#include <Common/Visualize/hkProcessHandler.h>

#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpWorldSnapshotViewer.h>
#include <Physics2012/Utilities/Serialize/hkpHavokSnapshot.h>

#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>

#include <Common/Visualize/hkVisualDebuggerProtocol.h>

/* static */int hkpWorldSnapshotViewer::s_tags[TYPE_COUNT_OF];

void HK_CALL hkpWorldSnapshotViewer::registerViewer()
{
	s_tags[TYPE_BINARY_TAGFILE] = hkProcessFactory::getInstance().registerProcess( getNameBinaryTagfile(), createBinaryTagfile );
	s_tags[TYPE_XML_TAGFILE] = hkProcessFactory::getInstance().registerProcess( getNameXmlTagfile(), createXmlTagfile  );
}

hkProcess* HK_CALL hkpWorldSnapshotViewer::createBinaryTagfile(const hkArray<hkProcessContext*>& contexts)
{
	return new hkpWorldSnapshotViewer(contexts, TYPE_BINARY_TAGFILE);
}

hkProcess* HK_CALL hkpWorldSnapshotViewer::createXmlTagfile(const hkArray<hkProcessContext*>& contexts)
{
	return new hkpWorldSnapshotViewer(contexts, TYPE_XML_TAGFILE);
}

hkpWorldSnapshotViewer::hkpWorldSnapshotViewer(const hkArray<hkProcessContext*>& contexts, Type type)
: hkpWorldViewerBase( contexts), 
m_type(type)
{
	
}

void hkpWorldSnapshotViewer::init()
{
	if (m_context)
	{
		for (int i=0; i < m_context->getNumWorlds(); ++i)
			worldAddedCallback( m_context->getWorld(i));
	}
}

hkpWorldSnapshotViewer::~hkpWorldSnapshotViewer()
{
	if (m_context)
	{
		for (int i=0; i < m_context->getNumWorlds(); ++i)
		{
			worldRemovedCallback( m_context->getWorld(i));
		}
	}
}

void hkpWorldSnapshotViewer::step( hkReal frameTimeInMs )
{
	// step once, send the HKX file in full (in native console format, so less likely to hit any serialization issues)
	if (m_outStream)
	{
		hkArray<char> storage;
		storage.reserve(100*1024); // 100KB
		hkArrayStreamWriter sw(&storage, hkArrayStreamWriter::ARRAY_BORROW);

		//m_convert? hkStructureLayout::MsvcWin32LayoutRules :

		const hkStructureLayout::LayoutRules& layout =  hkStructureLayout::HostLayoutRules;
		
		for (int w=0; w < m_context->getNumWorlds(); ++w)
		{
			storage.setSize(0);

			hkpHavokSnapshot::Options options(m_type == TYPE_BINARY_TAGFILE ? hkpHavokSnapshot::SNAPSHOT_BINARY_TAGFILE : hkpHavokSnapshot::SNAPSHOT_TEXT);

			hkpHavokSnapshot::save(m_context->getWorld(w), &sw, options);
			//hkpHavokSnapshot::save(m_context->getWorld(w), &sw, true, &layout);

			// if (m_convert)
			int snapShotLength = storage.getSize();
			if (snapShotLength < 1)
				continue;
			
			const int packetSize = (1*5) + 4 + snapShotLength;

			// the packet header
			m_outStream->write32u(packetSize);
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_SNAPSHOT);
			
			// As tagfile I don't need this information - but I'll keep for now. 
			// As could be used to send native etc to if necessary. If the m_bytesInPointer == 0 means its a tagfile
			// 
			
			if (true)
			{
				m_outStream->write8u(0);							// Bytes in pointer. 0 means its a tagfile
				// Binary is 0, Xml 1
				m_outStream->write8u((m_type == TYPE_BINARY_TAGFILE) ? 0 : 1);
				m_outStream->write8u(0);
				m_outStream->write8u(0);
			}
			else
			{
				// For native snapshots...
				// the layout the snapshot is in
				m_outStream->write8u(layout.m_bytesInPointer);
				m_outStream->write8u(layout.m_littleEndian);
				m_outStream->write8u(layout.m_reusePaddingOptimization);
				m_outStream->write8u(layout.m_emptyBaseClassOptimization);
			}
			
			// Snapshot itself
			m_outStream->write32(snapShotLength);
			m_outStream->writeRaw(storage.begin(), snapShotLength);
			
		//	bool streamOK = (m_outStream && m_outStream->isOk());
		}
	}

	// Now we have done the work, we can turn ourselves off.
	if (m_processHandler)
	{
		m_processHandler->deleteProcess( s_tags[m_type]);

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
