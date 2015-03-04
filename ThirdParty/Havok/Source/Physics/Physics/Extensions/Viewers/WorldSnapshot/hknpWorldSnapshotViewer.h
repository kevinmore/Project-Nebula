/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_WORLD_SNAPSHOT_VIEWER_H
#define HKNP_WORLD_SNAPSHOT_VIEWER_H

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Extensions/Viewers/hknpViewer.h>

class hkProcessFactory;


/// Creates and streams a world snapshot
class hknpWorldSnapshotViewer : public hknpViewer
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		enum Type
		{
			TYPE_XML_TAGFILE = 0,
			TYPE_BINARY_TAGFILE,
			TYPE_COUNT_OF,
		};

		/// Creates a hknpWorldSnapshotViewer.
		static hkProcess* HK_CALL createXmlTagfile(const hkArray<hkProcessContext*>& contexts);
		static hkProcess* HK_CALL createBinaryTagfile(const hkArray<hkProcessContext*>& contexts);

		/// Register this viewer with a factory.
		static void HK_CALL registerViewer(hkProcessFactory& factory);

		static inline const char* HK_CALL getNameXmlTagfile() { return "* Grab NP World Snapshot (Xml Tagfile)"; }
		static inline const char* HK_CALL getNameBinaryTagfile() { return "* Grab NP World Snapshot (Binary Tagfile)"; }

		/// Create a hknpWorldSnapshotViewer.
		static hkProcess* HK_CALL create(const hkArray<hkProcessContext*>& contexts);

		virtual ~hknpWorldSnapshotViewer() {}

		//
		// hknpViewer implementation
		//

		/// Gets the tag associated with this viewer type
		virtual int getProcessTag() { return s_tags[m_type]; }

		virtual void step(hkReal deltaTime);

	protected:

		hknpWorldSnapshotViewer(const hkArray<hkProcessContext*>& contexts, Type type);

		static int s_tags[TYPE_COUNT_OF];

		Type m_type;
};

#endif // HKNP_WORLD_SNAPSHOT_VIEWER_H

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
