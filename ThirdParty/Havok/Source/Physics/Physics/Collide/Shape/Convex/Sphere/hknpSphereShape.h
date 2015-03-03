/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SPHERE_SHAPE_H
#define HKNP_SPHERE_SHAPE_H

#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>

extern const hkClass hknpSphereShapeClass;


/// Sphere shape.
/// Basically a convex vertices shape with 4 points in the center.
class hknpSphereShape : public hknpConvexShape
{
	public:

		/// Create a sphere shape.
		static hknpSphereShape* HK_CALL createSphereShape( hkVector4Parameter center, hkReal radius );

		/// Create a sphere shape in the given buffer. \a buffer must be 16 byte aligned.
		/// Advanced use only.
		static HK_FORCE_INLINE hknpSphereShape* createInPlace(
			hkVector4Parameter center, hkReal radius, hkUint8* buffer, int bufferSize );

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Serialization constructor.
		hknpSphereShape( class hkFinishLoadedObjectFlag flag );

		//
		// hknpShape implementation
		//

		virtual hknpShapeType::Enum getType() const HK_OVERRIDE { return hknpShapeType::SPHERE; }

		virtual hkReal calcMinAngleBetweenFaces() const HK_OVERRIDE;

#if !defined(HK_PLATFORM_SPU)

		virtual void buildMassProperties( const MassConfig& massConfig, hkDiagonalizedMassProperties& massPropertiesOut ) const HK_OVERRIDE;

		virtual hkResult buildSurfaceGeometry( const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut ) const HK_OVERRIDE;

#endif

	protected:

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpSphereShape, hknpConvexShape );

		/// Protected constructor. Use createSphereShape() instead.
		HK_FORCE_INLINE hknpSphereShape( hkVector4Parameter center, hkReal radius );

		/// Used to initialize a sphere shape without constructing it. Required for SPU
		HK_FORCE_INLINE void init( hkVector4Parameter center );
};

#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.inl>


#endif // HKNP_SPHERE_SHAPE_H

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
