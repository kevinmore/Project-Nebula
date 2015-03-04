/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_LINEAR_SURFACE_VECLOCITY_H
#define HKNP_LINEAR_SURFACE_VECLOCITY_H

#include <Physics/Physics/Dynamics/Modifier/SurfaceVelocity/hknpSurfaceVelocity.h>


/// An implementation of a hknpSurfaceVelocity which only has linear movement.
class hknpLinearSurfaceVelocity: public hknpSurfaceVelocity
{
	public:

		/// How to rescale the velocity vector if the velocity is not parallel to the surface.
		enum ProjectMethod
		{
			VELOCITY_PROJECT,		///< just project the velocity onto the surface, this is the fastest method and should be used if the velocity
			VELOCITY_RESCALE,		///< keep velocity constant: adjust the velocity so that the projected velocity vector length matches the m_velocity length.
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Constructor
		hknpLinearSurfaceVelocity(
			hkVector4Parameter velocity = hkVector4::getZero(),
			Space space = USE_WORLD_SPACE,
			ProjectMethod projectMethod = VELOCITY_PROJECT );

		/// Serialization constructor
		hknpLinearSurfaceVelocity( hkFinishLoadedObjectFlag f ) : hknpSurfaceVelocity(f) {}

		/// hknpSurfaceVelocity implementation
		virtual void calcSurfaceVelocity(
			hkVector4Parameter positionWs, hkVector4Parameter normalWs, const hkTransform& shapeTransform,
			hkVector4* linearSurfaceVelocityWsOut,	hkVector4* angularSurfaceVelocityWsOut	) const;

	public:

		hkEnum<Space,hkUint8> m_space;					///< local or world space.

		// projection

		hkEnum<ProjectMethod,hkUint8> m_projectMethod;	///< how to rescale velocity if the m_velocity does not l
		hkReal m_maxVelocityScale;						///< when projecting velocity, never scale the velocity up more than this factor.

		/// If set (!= 0,0,0) and m_projectMethod == VELOCITY_RESCALE,
		/// the final velocity will be rescaled so that its length projected onto this plane will match the input m_velocity.
		hkVector4 m_velocityMeasurePlane;

		/// desired velocity.
		hkVector4 m_velocity;
};

#endif	// HKNP_LINEAR_SURFACE_VECLOCITY_H

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
