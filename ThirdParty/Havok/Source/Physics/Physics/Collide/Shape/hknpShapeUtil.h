/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SHAPE_UTIL_H
#define HKNP_SHAPE_UTIL_H

#include <Physics/Physics/Collide/Shape/hknpShape.h>

class hkgpConvexHull;
class hkDisplayGeometry;


/// Helper class that holds utility methods specific to hknpShape.
class hknpShapeUtil
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpShapeUtil );

		/// Calculate the mass properties for an AABB.
		static hkResult buildAabbMassProperties( const hknpShape::MassConfig& massConfig,
			const hkAabb& aabb, hkDiagonalizedMassProperties& massPropertiesOut );

		/// Build a set of display geometries representing the surface of the given shape.
		static void HK_CALL buildShapeDisplayGeometries(
			const hknpShape* shape, const hkTransform& transform, hkVector4Parameter scale,
			hknpShape::ConvexRadiusDisplayMode radiusMode, hkArray<hkDisplayGeometry*>& displayGeometriesOut );

		/// Create the geometry of the convex hull of the shape if possible, and appends it to the input geometry.
		static hkResult HK_CALL createConvexHullGeometry(
			const hknpShape& shape, hknpShape::ConvexRadiusDisplayMode radiusMode,
			hkGeometry* geometryInOut, int material = HKNP_INVALID_SHAPE_TAG );

		/// Create the geometry of the convex hull of the vertices if possible, and appends it to the input geometry.
		static hkResult HK_CALL createConvexHullGeometry(
			const hkVector4* vertices, int numVertices, hkReal convexRadius, hknpShape::ConvexRadiusDisplayMode radiusMode,
			hkGeometry* geometryInOut, int material = HKNP_INVALID_SHAPE_TAG );

		/// Compute scaling parameters for a convex shape given the scale mode, desired scale and convex radius.
		/// Advanced use only.
		/// \note \p numVertices must be a multiple of 4.
		static void HK_CALL calcScalingParameters(
			const hknpConvexShape& shape, hknpShape::ScaleMode mode,
			hkVector4* HK_RESTRICT scaleInOut, hkReal* HK_RESTRICT radiusInOut, hkVector4* HK_RESTRICT offsetOut );

		/// Find all of the convex shapes in a shape hierarchy.
		/// The transforms are the local to reference transforms for each of the shapes.
		/// NOTE! This method will not work properly with shape hierarchies that have shape collections which do not
		/// return actual shape pointers (as opposed to storage being in the hkpShapeBuffer) - as the buffers will go
		/// out of scope before the method returns.
		static void HK_CALL flattenIntoConvexShapes(
			const hknpShape* shape, const hkTransform& worldFromParent,
			hkArray<const hknpConvexShape*>& shapesOut, hkArray<hkTransform>& worldFromShapesOut );
};


#endif // HKNP_SHAPE__UTIL_H

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
