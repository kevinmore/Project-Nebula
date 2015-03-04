/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CONVEX_SHAPE_UTIL_H
#define HKNP_CONVEX_SHAPE_UTIL_H

#include <Common/Base/Math/Vector/Mx/hkMxVector.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

#include <Physics/Physics/Collide/Shape/hknpShape.h>
#include <Physics/Physics/Collide/Shape/hknpShapeUtil.h>
#include <Physics/Physics/Collide/Shape/Convex/Polytope/hknpConvexPolytopeShape.h>


/// Helper class that holds utility methods specific to hknpConvexShape.
class hknpConvexShapeUtil
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpConvexShapeUtil );

		/// Calculate the number of unique vertex IDs found in the shape's vertices.
		static int HK_CALL getNumberOfUniqueVertices( const hknpConvexShape* shape );

		/// Calculate the closest points between two convex shapes.
		/// Output parameters \a normal and \a pointOnTarget are both in target space.
		static bool HK_CALL getClosestPoints(
			const hknpConvexShape* queryCvx, const hknpConvexShape* targetCvx,
			const hkTransform& queryToTarget, hkSimdReal* distance,
			hkVector4* HK_RESTRICT normal, hkVector4* HK_RESTRICT pointOnTarget );

		/// Calculate the closest points between two convex shapes each one passed as a set of vertices and a convex
		/// radius. Useful if you don't already have a hknpConvexShape and want to avoid the overhead of constructing it.
		/// Output parameters \a normal and \a pointOnTarget are both in target space.
		HK_FORCE_INLINE static bool getClosestPoints(
			const hkcdVertex* queryVertices, int numQueryVertices, hkReal queryRadius,
			const hkcdVertex* targetVertices, int numTargetVertices, hkReal targetRadius,
			const hkTransform& queryToTarget, hkSimdReal* distance,
			hkVector4* HK_RESTRICT normal, hkVector4* HK_RESTRICT pointOnTarget );

		/// Calculate the closest points between two hknpConvexShape shapes.
		static bool HK_CALL getClosestPoints(
			const hknpConvexShape* queryCvx, const hknpQueryFilterData& queryShapeFilterData, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpConvexShape* targetCvx, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget,
			hknpCollisionQueryCollector* collector );

		/// Calculate the closest points between two sets of vertices, taking scaling of the vertex sets into account.
		HK_FORCE_INLINE static bool HK_CALL getClosestPointsWithScale(
			const hkcdVertex* queryVertices, int numQueryVertices, const hknpQueryFilterData& queryShapeFilterData, const hknpShapeQueryInfo& queryShapeInfo,
			const hkcdVertex* targetVertices, int numTargetVertices, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget,
			hknpCollisionQueryCollector* HK_RESTRICT collector );

		/// Calculate the closest point between a point and a triangle
		HK_FORCE_INLINE static bool HK_CALL getClosestPointsToTriangleWithScale(
			const hkcdVertex& queryShapeVertex, const hknpQueryFilterData& queryShapeFilterData, const hknpShapeQueryInfo& queryShapeInfo,
			const hkcdVertex* targetShapeVertices, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget,
			hknpCollisionQueryCollector* HK_RESTRICT collector );

		/// Calculate the closest points between two hknpShape shapes.
		static bool HK_CALL getClosestPointsUsingConvexHull(
			const hknpShape* queryShape, const hknpQueryFilterData& queryShapeFilterData, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget,
			hknpCollisionQueryCollector* collector );

		///
		HK_FORCE_INLINE static void HK_CALL getSupportingFace(
			const hkVector4* HK_RESTRICT planeA, int facesA,
			hkVector4Parameter surfacePointA, int& faceAOut );

		///
		HK_FORCE_INLINE static void HK_CALL getSupportingFaces(
			const hkVector4* HK_RESTRICT planeA, int facesA,
			const hkVector4* HK_RESTRICT planeB, int facesB,
			hkVector4Parameter surfacePointA,
			hkVector4Parameter surfacePointB,
			int& faceAOut, int& faceBOut  );

		/// Very fast implementation of getSupportingVertex for two convex shapes.
		HK_FORCE_INLINE static void getSupportingVertices(
			const hknpConvexShape& shapeA, hkVector4Parameter directionA,
			const hknpConvexShape& shapeB, const hkTransform& aTb,
			hkcdVertex* HK_RESTRICT vertexAinAOut, hkcdVertex* HK_RESTRICT vertexBinBOut,
			hkVector4* HK_RESTRICT vertexBinAOut );

		/// Very fast implementation of getSupportingVertex for two convex shapes.
		// directionA is in A-space, directionB in B-space
		HK_FORCE_INLINE static void getSupportingVertices2(
			const hkVector4* fvA, int numA,
			const hkVector4* fvB, int numB,
			hkVector4Parameter directionA, hkVector4Parameter directionB,
			hkVector4* vertexAinAOut, hkVector4* vertexBinBOut );

		///
		HK_FORCE_INLINE static void getFaceVertices(
			const hknpConvexPolytopeShape& shapeA, const int faceIndexA,
			hkVector4& planeOutA, int& verticesOutA, hkVector4* const HK_RESTRICT vertexBufferA,
			const hknpConvexPolytopeShape& shapeB, const int faceIndexB,
			hkVector4& planeOutB, int& verticesOutB, hkVector4* const HK_RESTRICT vertexBufferB );

		///
		HK_FORCE_INLINE static void getFaceVertices(
			const hknpConvexPolytopeShape& shapeA, const int faceIndexA,
			int& verticesOutA, hkVector4* const HK_RESTRICT vertexBufferA);

		//
		// Mass properties
		//

		/// Calculate the mass properties for a sphere.
		HK_FORCE_INLINE static hkResult buildSphereMassProperties( const hknpShape::MassConfig& massConfig,
			hkVector4Parameter center, hkReal radius, hkDiagonalizedMassProperties& massPropertiesOut );

		/// Calculate the mass properties for a capsule.
		HK_FORCE_INLINE static hkResult buildCapsuleMassProperties( const hknpShape::MassConfig& massConfig,
			hkVector4Parameter axisStart, hkVector4Parameter axisEnd, hkReal radius, hkDiagonalizedMassProperties& massPropertiesOut );

		/// Calculate the mass properties for a triangle.
		HK_FORCE_INLINE static hkResult buildTriangleMassProperties( const hknpShape::MassConfig& massConfig,
			const hkVector4* vertices, hkReal radius, hkDiagonalizedMassProperties& massPropertiesOut );

		/// Calculate the mass properties for a planar quad.
		HK_FORCE_INLINE static hkResult buildQuadMassProperties( const hknpShape::MassConfig& massConfig,
			const hkVector4* vertices, hkReal radius, hkDiagonalizedMassProperties& massPropertiesOut );

		/// Calculate the mass properties for a hull.
		HK_FORCE_INLINE static hkResult buildHullMassProperties( const hknpShape::MassConfig& massConfig,
			const hkVector4* vertices, const int numVertices, hkReal radius, hkDiagonalizedMassProperties& massPropertiesOut );

	protected:

		static HK_ALIGN16( const hkUint32 s_curIndices[4] );	
};

#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShapeUtil.inl>


#endif // HKNP_CONVEX_SHAPE_UTIL_H

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
