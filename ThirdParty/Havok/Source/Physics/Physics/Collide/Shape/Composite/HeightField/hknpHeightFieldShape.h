/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_HEIGHT_FIELD_SHAPE_H
#define HKNP_HEIGHT_FIELD_SHAPE_H

#include <Physics/Physics/Collide/Shape/Composite/hknpCompositeShape.h>
#include <Physics/Physics/Collide/Shape/Composite/HeightField/hknpMinMaxQuadTree.h>

struct hknpClosestPointsQuery;


/// Construction info for a hknpHeightFieldShape.
class hknpHeightFieldShapeCinfo
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpHeightFieldShapeCinfo );

		/// Constructor. Sets default values.
		hknpHeightFieldShapeCinfo();

	public:

		/// Determines how to collide with other shapes. Must be either:
		/// - DISTANCE_FIELD : This will collide the other shape's vertices against the height field.
		/// This is the fastest method, and no welding will be needed.
		/// - COMPOSITE : This will collide the other shape against the height field triangles (two per quad).
		/// This can be more accurate when the other shape is large, but welding may be required if objects slide across the height field.
		hknpCollisionDispatchType::Enum m_dispatchType;

		/// How the height field scales from integer space (where x,z as integers correspond to vertex coordinates)
		/// to real space. The y-component can be used to scale the heights.
		hkVector4 m_scale;

		/// The resolution (number of vertices) along x. The number of edges is m_xRes-1.
		hkInt32 m_xRes;

		/// The resolution (number of vertices) along z (y up).
		hkInt32 m_zRes;

		/// Determines the coarseness of the min-max quadtree (the bounding volume hierarchy).
		/// The edge size at at the finest level of the quadtree will be 1<<m_minMaxTreeCoarseness.
		/// So if m_minMaxTreeCoarseness is set to 3 (the default) it will store min and max height values for
		/// every 8 X 8 tile.
		hkInt32 m_minMaxTreeCoarseness;

		/// The minimum height returned by the height field. buildMinMaxTree() will calculate this automatically.
		hkReal m_minHeight;

		/// The maximum height returned by the height field. buildMinMaxTree() will calculate this automatically.
		hkReal m_maxHeight;
};


/// Base class for 2D sampled height field shapes.
/// It uses the y coordinate for the up axis (height). The x and z coordinates are used to lookup a height information.
/// If you want to use this class, you need to subclass it and implement the getQuadInfoAt method:
/// \code void getQuadInfoAt(int x, int z, hkVector4* heightsOut, hknpShapeTag* shapeTagOut, hkBool32 *triangleFlipOut) const \endcode
/// It returns the 4 corner heights around the quad [x,x+1] X [z,z+1], shapeTag (for filtering and materials) and
/// how to split the quad into triangles..
/// See hknpCompressedHeightFieldShape for a simple concrete implementation.
class hknpHeightFieldShape : public hknpCompositeShape
{
	//+version(1)

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Constructor.
		hknpHeightFieldShape( const hknpHeightFieldShapeCinfo& cinfo );

#if !defined(HK_PLATFORM_SPU)

		/// Serialization constructor.
		hknpHeightFieldShape( hkFinishLoadedObjectFlag f ) : hknpCompositeShape(f), m_minMaxTree(f) {}

#endif

		/// This method will traverse the height field to calculate min and max height and build the coarse min-max
		/// quadtree.
		/// Unless a custom getCoarseMinMax method is used you are required to call this to build the min-max quadtree.
		void buildMinMaxTree();

		/// Returns information about the quad.
		///	heightOut should contain the heights for (x,z), (x+1,z), (x,z+1), (x+1,z+1) packed into the 4 components
		/// of the vector.
		/// shapeTagOut is the shapeTag for the quad (can be set to HKNP_INVALID_SHAPE_TAG).
		/// Set triangleFlipOut to true if the two triangles of the quad share the edge share the
		/// edge p(x,z) - p(x+1,z+1).
#if !defined ( HK_PLATFORM_SPU )
		virtual void getQuadInfoAt(
			int x, int z,
			hkVector4* heightsOut, hknpShapeTag* shapeTagOut, hkBool32 *triangleFlipOut ) const = 0;
#else
		void getQuadInfoAt(
			int x, int z,
			hkVector4* heightsOut, hknpShapeTag* shapeTagOut, hkBool32 *triangleFlipOut ) const;
#endif

		/// Returns 4 minimum and 4 maximum height values for 4 cells packed into the xyzw components of the vectors.
		/// The four cells have coordinates (2*x,2*z), (2*x+1,2*z), (2*x+1,2*z+1) and (2*x,2*z+1) in the coarse grid
		/// with length 1<<level (two to the level'th power). It might help to think of level as the mipmap level of a
		/// texture.
		/// Has a default implementation that uses a min-max quadtree (see buildMinMaxTree()).
#if !defined ( HK_PLATFORM_SPU )
		virtual void getCoarseMinMax( int level, int x, int z, hkVector4* HK_RESTRICT minOut, hkVector4* HK_RESTRICT maxOut ) const;
#else
		void getCoarseMinMax( int level, int x, int z, hkVector4* HK_RESTRICT minOut, hkVector4* HK_RESTRICT maxOut ) const;
#endif

		//
		// hknpShape implementation
		//

		virtual hknpShapeType::Enum getType() const HK_OVERRIDE { return hknpShapeType::HEIGHT_FIELD; }

		virtual void calcAabb( const hkTransform& transform, hkAabb& aabbOut ) const HK_OVERRIDE;

#if !defined(HK_PLATFORM_SPU)
		virtual hkRefNew<hknpShapeKeyIterator> createShapeKeyIterator( const hknpShapeKeyMask* mask = HK_NULL ) const HK_OVERRIDE;
#else
		virtual hknpShapeKeyIterator* createShapeKeyIterator( hkUint8* buffer, int bufferSize, const hknpShapeKeyMask* mask = HK_NULL ) const HK_OVERRIDE;
#endif

		virtual void getLeafShape( hknpShapeKey key, hknpShapeCollector* collector ) const HK_OVERRIDE;

		virtual void getSignedDistances(
			const hknpShape::SdfQuery& query, hknpShape::SdfContactPoint* contactsOut ) const HK_OVERRIDE;

		virtual void castRayImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpRayCastQuery& query,
			const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hknpCollisionQueryCollector* collector ) const HK_OVERRIDE;

		virtual void queryAabbImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hkArray<hknpShapeKey>* hits, hknpQueryAabbNmp* nmpInOut ) const HK_OVERRIDE;

		virtual void queryAabbImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hknpCollisionQueryCollector* collector, hknpQueryAabbNmp* nmpInOut ) const HK_OVERRIDE;

#if !defined(HK_PLATFORM_SPU)

		virtual hkResult buildSurfaceGeometry(
			const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut ) const HK_OVERRIDE;

#endif

		//
		// Internals
		//

		/// Internal. Cast a shape against a hknpHeightFieldShape.
		static void castShapeImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpShapeCastQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector );

		/// Internal. Get the closest points between a shape and a hknpHeightFieldShape.
		static void getClosestPointsImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpHeightFieldShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector );

		/// Internal. Get the closest points between two hknpHeightFieldShapes.
		static void getClosestPointsToHeightfieldImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpHeightFieldShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector );

		/// This method is used by mesh vs height field getClosestPointsImpl().
		/// It returns <true> if the lower estimate (returned in \a distSquaredOut) is less than \a maxSquaredDist.
		hkBool32 calcClosestDistanceSquaredLowerEstimate(
			const hkAabb& queryAabb, hkSimdRealParameter maxSquaredDist, hkSimdReal* HK_RESTRICT distSquaredOut ) const;

		/// This method is used by mesh vs height field getClosestPointsImpl() when the heightfield is 'externally'
		/// scaled (e.g. when embedded in a hknpCompoundShape.)
		/// It returns <true> if the lower estimate (returned in \a distSquaredOut) is less than \a maxSquaredDist.
		hkBool32 calcClosestDistanceSquaredLowerEstimateWithScale(
			const hkTransform& heightfieldToWorld, const hkTransform& worldToHeightfield, const hkAabb& queryAabb,
			hkSimdRealParameter maxSquaredDist, hkSimdReal* HK_RESTRICT distSquaredOut ) const;

	protected:

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpHeightFieldShape, hknpCompositeShape );

	public:

		/// The local AABB for the height field. The y-components can be used to get minimum and maximum height.
		hkAabb m_aabb;

		/// Used when scaling to integer space (coordinates).
		hkVector4 m_floatToIntScale;

		/// Used when scaling from integer space (coordinates).
		hkVector4 m_intToFloatScale;

		/// Number of edges in the x direction.
		int m_intSizeX;

		/// Number of edges in the z direction.
		int m_intSizeZ;

		/// The number of bits needed to represent the X resolution (at high resolution).
		int m_numBitsX;

		/// The number of bits needed to represent the Z resolution (at high resolution).
		int m_numBitsZ;

		/// Quadtree for storing min-max bounding volumes.
		hknpMinMaxQuadTree m_minMaxTree;

		/// Determines the coarseness of the min-max quadtree.
		hkInt32 m_minMaxTreeCoarseness;
};


#endif // HKNP_HEIGHT_FIELD_SHAPE_H

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
