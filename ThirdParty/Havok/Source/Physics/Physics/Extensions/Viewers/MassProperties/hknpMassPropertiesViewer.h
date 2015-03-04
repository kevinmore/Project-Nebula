/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MASS_PROPERTIES_VIEWER_H
#define HKNP_MASS_PROPERTIES_VIEWER_H

#include <Physics/Physics/Extensions/Viewers/hknpViewer.h>


/// Shows the mass and the inertia tensor of all dynamic rigid bodies added to the world.
/// The inertia tensor is displayed as a box whose dimensions are such that it would have the same inertia
/// tensor as the body in question, if it was also of equal mass. In other words, it shows an object which
/// would behave identically from a dynamic point of view (though obviously, not from a collision detection
/// point of view).
/// In the case of a rigid body being simulated as an oriented particle this box will always be a cube aligned in local space.
/// In the case of a rigid body being simulated with a DIAGONAL inertia tensor this box may have different edge
/// lengths, but will still be aligned in local space.
/// Note: We actually display a box which is *slightly bigger* than the one defined above by a factor of 1.01, just for
/// ease of display.
class hknpMassPropertiesViewer : public hknpViewer
{
	public:

		static inline const char* HK_CALL getName() { return HKNP_MASS_PROPERTIES_VIEWER_NAME; }

		/// Create a hknpMassPropertiesViewer.
		static hkProcess* HK_CALL create( const hkArray<hkProcessContext*>& contexts );

		/// Register this viewer with a factory.
		static void HK_CALL registerViewer( hkProcessFactory& factory );

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		virtual ~hknpMassPropertiesViewer() {}

		/// Set the size of the buffer used to build display objects before sending them to the display handler.
		/// Defaults to 32KB.
		void setBufferSize( int size );

		//
		// hknpViewer implementation
		//

		virtual int getProcessTag() { return s_tag; }

		virtual void step( hkReal deltaTime );

	protected:

		hknpMassPropertiesViewer( const hkArray<hkProcessContext*>& contexts );

	protected:

		static int s_tag;

		int m_bufferSize;
};

#endif // HKNP_MASS_PROPERTIES_VIEWER_H

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
