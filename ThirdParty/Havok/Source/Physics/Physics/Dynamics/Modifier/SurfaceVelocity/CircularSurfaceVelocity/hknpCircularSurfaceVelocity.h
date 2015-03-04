/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CIRCULAR_SURFACE_VECLOCITY_H
#define HKNP_CIRCULAR_SURFACE_VECLOCITY_H

#include <Physics/Physics/Dynamics/Modifier/SurfaceVelocity/hknpSurfaceVelocity.h>


/// An implementation of a hknpSurfaceVelocity which only has angular movement.
class hknpCircularSurfaceVelocity : public hknpSurfaceVelocity
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Constructor
		hknpCircularSurfaceVelocity(
			hkVector4Parameter pivot = hkVector4::getZero(),
			hkVector4Parameter angVel = hkVector4::getZero() );

		/// Serialization constructor
		hknpCircularSurfaceVelocity( hkFinishLoadedObjectFlag f ) : hknpSurfaceVelocity(f) {}

		/// hknpSurfaceVelocity implementation
		void calcSurfaceVelocity(
			hkVector4Parameter positionWs, hkVector4Parameter normalWs, const hkTransform& shapeTransform,
			hkVector4* linearSurfaceVelocityWsOut,	hkVector4* angularSurfaceVelocityWsOut	) const;

	public:

		hkBool m_velocityIsLocalSpace;

		hkVector4 m_pivot;
		hkVector4 m_angularVelocity;
};

#endif	// HKNP_CIRCULAR_SURFACE_VECLOCITY_H

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
