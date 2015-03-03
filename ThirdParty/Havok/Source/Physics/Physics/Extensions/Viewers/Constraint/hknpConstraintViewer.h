/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_ATOM_CONSTRAINT_VIEWER_H
#define HKNP_ATOM_CONSTRAINT_VIEWER_H

#include <Physics/Physics/Extensions/Viewers/hknpViewer.h>

class hkProcessFactory;


/// Displays atom constraints.
class hknpConstraintViewer : public hknpViewer
{
	public:

		static inline const char* HK_CALL getName() { return HKNP_CONSTRAINT_VIEWER_NAME; }

		/// Register this viewer with a factory.
		static void HK_CALL registerViewer( hkProcessFactory& factory );

		/// Create a hknpConstraintViewer
		static hkProcess* HK_CALL create( const hkArray<hkProcessContext*>& contexts );

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		virtual ~hknpConstraintViewer();

		// Signal handler
		void onImmediateConstraintAddedSignal( hknpWorld* world, const hknpConstraint* constraint );

		//
		// hknpViewer implementation
		//

		virtual int getProcessTag() { return s_tag; }

		virtual void step( hkReal deltaTime );

		virtual void worldAddedCallback( hknpWorld* world );

		virtual void worldRemovedCallback( hknpWorld* world );

	protected:

		hknpConstraintViewer( const hkArray<hkProcessContext*>& contexts );

	public:

		static hkReal m_scale;

	protected:

		static int s_tag;
};


#endif	// HKNP_CONSTRAINT_VIEWER_H

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
