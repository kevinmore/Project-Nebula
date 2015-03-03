/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CAPSULE_SHAPE_H
#define HKNP_CAPSULE_SHAPE_H

#include <Physics/Physics/Collide/Shape/Convex/Polytope/hknpConvexPolytopeShape.h>

extern const hkClass hknpCapsuleShapeClass;

/// Size of a capsule shape not including vertices and face information
#define HKNP_CAPSULE_BASE_SIZE		HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeof(hknpCapsuleShape))


/// Capsule shape.
/// Basically a convex shape with 8 vertices along the capsule axis.
class hknpCapsuleShape : public hknpConvexPolytopeShape
{
	public:

		/// Create a capsule shape.
		/// Note: If \a posA == \a posB, this function returns an hknpSphereShape.
		static hknpCapsuleShape* HK_CALL createCapsuleShape(
			hkVector4Parameter posA, hkVector4Parameter posB, hkReal radius );

		/// Create a capsule shape in the given buffer. \a buffer must be 16 byte aligned.
		/// Advanced use only.
		static HK_FORCE_INLINE hknpCapsuleShape* createInPlace(
			hkVector4Parameter posA, hkVector4Parameter posB, hkReal radius, hkUint8* buffer, int bufferSize );

	public:

		HK_DECLARE_REFLECTION();
		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Serialization constructor.
		hknpCapsuleShape( class hkFinishLoadedObjectFlag flag );

		//
		// hknpShape implementation
		//

		virtual hknpShapeType::Enum getType() const HK_OVERRIDE { return hknpShapeType::CAPSULE; }

		virtual int calcSize() const HK_OVERRIDE;

		virtual int getNumberOfSupportVertices() const HK_OVERRIDE;

		virtual const hkcdVertex* getSupportVertices( hkcdVertex* vertexBuffer, int bufferSize ) const HK_OVERRIDE;

		virtual void getSignedDistances( const hknpShape::SdfQuery& query, SdfContactPoint* contactsOut ) const HK_OVERRIDE;

		virtual int getSignedDistanceContacts(
			const hknpSimulationThreadContext& tl, const hknpShape* queryShape, const hkTransform& sdfFromQueryTransform,
			hkReal maxDistance, int vertexIdOffset, SdfContactPointWriter& contactPointsOut ) const HK_OVERRIDE;

#if !defined(HK_PLATFORM_SPU)

		virtual void buildMassProperties( const MassConfig& massConfig, hkDiagonalizedMassProperties& massPropertiesOut ) const HK_OVERRIDE;

		virtual hkResult buildSurfaceGeometry( const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut ) const HK_OVERRIDE;

#endif

	protected:

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpCapsuleShape, hknpConvexPolytopeShape );

		/// For internal use, use createCapsuleShape instead
		HK_FORCE_INLINE hknpCapsuleShape( hkVector4Parameter posA, hkVector4Parameter posB, hkReal radius );

		/// Used to initialize a capsule shape without constructing it. Required for SPU
		HK_FORCE_INLINE void init( hkVector4Parameter a, hkVector4Parameter b );

	public:

		hkVector4 m_a;
		hkVector4 m_b;
};

#include <Physics/Physics/Collide/Shape/Convex/Capsule/hknpCapsuleShape.inl>


#endif // HKNP_CAPSULE_SHAPE_H

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
