/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SURFACE_VECLOCITY_H
#define HKNP_SURFACE_VECLOCITY_H


/// An interface to calculate a velocity at a given position for an object.
class hknpSurfaceVelocity : public hkReferencedObject
{
	public:

		/// Helper define for derived classes.
		enum Space
		{
			USE_LOCAL_SPACE,	///< Data is defined in local space.
			USE_WORLD_SPACE,	///< Data is defined in world space.
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Destructor
		virtual ~hknpSurfaceVelocity() {}

		/// Calculate the surface velocity
		virtual void calcSurfaceVelocity(
			hkVector4Parameter positionWs, hkVector4Parameter normalWs, const hkTransform& shapeTransform,
			hkVector4* linearSurfaceVelocityWsOut,	hkVector4* angularSurfaceVelocityWsOut ) const = 0;

	protected:

		/// Empty constructor
		hknpSurfaceVelocity() {}

		/// Serialization constructor
		hknpSurfaceVelocity( hkFinishLoadedObjectFlag f ) : hkReferencedObject(f) {}
};

#endif	// HKNP_SURFACE_VECLOCITY_H

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
