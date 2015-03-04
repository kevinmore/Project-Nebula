/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CELL_VIEWER_H
#define HKNP_CELL_VIEWER_H

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Extensions/Viewers/hknpViewer.h>

class hkProcessFactory;


/// Displays AABB's of all active bodies with a color derived from the cell index.
/// It uses the physics post simulation signal to do its job.
class hknpCellViewer : public hknpViewer
{
	public:

		static inline const char* HK_CALL getName() { return HKNP_CELL_VIEWER_NAME; }

		/// Register this viewer with a factory.
		static void HK_CALL registerViewer( hkProcessFactory& factory );

		/// Create a hknpCellViewer.
		static hkProcess* HK_CALL create( const hkArray<hkProcessContext*>& contexts );

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		virtual ~hknpCellViewer() {}

		//
		// hknpViewer implementation
		//

		virtual int getProcessTag() { return s_tag; }

		virtual void step( hkReal deltaTime );

	protected:

		hknpCellViewer( const hkArray<hkProcessContext*>& contexts );

	protected:

		hkArray< hknpWorld* > m_worlds;

		static int s_tag;
};

#endif // HKNP_CELL_VIEWER_H

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
