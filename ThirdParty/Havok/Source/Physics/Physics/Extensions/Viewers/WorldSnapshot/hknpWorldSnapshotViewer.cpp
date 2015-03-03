/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Extensions/Viewers/WorldSnapshot/hknpWorldSnapshotViewer.h>
#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>
#include <Physics/Physics/Extensions/WorldSnapshot/hknpWorldSnapshot.h>
#include <Common/Serialize/Util/hkSerializeUtil.h>
#include <Common/Visualize/hkVisualDebuggerProtocol.h>
#include <Common/Visualize/hkProcessHandler.h>

extern const hkClass hknpWorldSnapshotClass;

/* static */int hknpWorldSnapshotViewer::s_tags[TYPE_COUNT_OF];

void HK_CALL hknpWorldSnapshotViewer::registerViewer( hkProcessFactory& factory )
{
	s_tags[TYPE_BINARY_TAGFILE] = hkProcessFactory::getInstance().registerProcess( getNameBinaryTagfile(), createBinaryTagfile );
	s_tags[TYPE_XML_TAGFILE] = hkProcessFactory::getInstance().registerProcess( getNameXmlTagfile(), createXmlTagfile  );
}

hkProcess* HK_CALL hknpWorldSnapshotViewer::createBinaryTagfile(const hkArray<hkProcessContext*>& contexts)
{
	return new hknpWorldSnapshotViewer(contexts, TYPE_BINARY_TAGFILE);
}

hkProcess* HK_CALL hknpWorldSnapshotViewer::createXmlTagfile(const hkArray<hkProcessContext*>& contexts)
{
	return new hknpWorldSnapshotViewer(contexts, TYPE_XML_TAGFILE);
}

hknpWorldSnapshotViewer::hknpWorldSnapshotViewer(const hkArray<hkProcessContext*>& contexts, Type type)
:	hknpViewer( contexts )
,	m_type(type)
{
}

void hknpWorldSnapshotViewer::step(hkReal deltaTime)
{
	// step once
	if (m_outStream)
	{
		hkArray<char> storage;
		storage.reserve(100*1024); // 100KB
		hkArrayStreamWriter sw(&storage, hkArrayStreamWriter::ARRAY_BORROW);

		for (int w=0; w < m_context->getNumWorlds(); ++w)
		{
			storage.setSize(0);

			hknpWorldSnapshot snapshot (*m_context->getWorld(w));

			hkSerializeUtil::SaveOptions saveOptions;
			if (m_type == TYPE_BINARY_TAGFILE)
			{
				saveOptions.useBinary(true);
			}
			else
			{
				saveOptions.useBinary(false);
			}

			hkSerializeUtil::saveTagfile(&snapshot, hknpWorldSnapshotClass, &sw, HK_NULL, saveOptions);

			int snapShotLength = storage.getSize();
			if (snapShotLength < 1)
				continue;

			const int packetSize = (1*5) + 4 + snapShotLength;

			// the packet header
			m_outStream->write32u(packetSize);
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_SNAPSHOT);
			m_outStream->write8u(0);							// Bytes in pointer. 0 means its a tagfile
			// Binary is 0, Xml 1
			m_outStream->write8u((m_type == TYPE_BINARY_TAGFILE) ? 0 : 1);
			m_outStream->write8u(0);
			m_outStream->write8u(0);

			// Snapshot itself
			m_outStream->write32(snapShotLength);
			m_outStream->writeRaw(storage.begin(), snapShotLength);
		}
	}

	// Now we have done the work, we can turn ourselves off.
	if (m_processHandler)
	{
		m_processHandler->deleteProcess(s_tags[m_type]);

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
