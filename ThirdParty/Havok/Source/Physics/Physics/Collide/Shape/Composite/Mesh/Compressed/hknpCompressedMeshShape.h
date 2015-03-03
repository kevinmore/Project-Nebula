/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COMPRESSED_MESH_SHAPE_H
#define HKNP_COMPRESSED_MESH_SHAPE_H

#include <Physics/Physics/Collide/Shape/Composite/hknpCompositeShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Compressed/hknpCompressedMeshShapeCinfo.h>
#include <Common/Base/Container/BitField/hkBitField.h>

extern const class hkClass hknpCompressedMeshShapeClass;
extern const class hkClass hknpCompressedMeshShapeTreeClass;
extern const class hkClass hknpCompressedMeshShapeTreeExtClass;


/// A mesh shape storing its triangles in a very efficient way.
/// Stores geometry and collision information's in a format optimized for memory and query speed.
/// Compression rate between 10 and 12 bytes per triangle.
///
/// Notes:
///	- Supports efficient sparse (i.e. number of unique data less than the number of triangles) data storage for
///   each triangle.
///	- Maximum triangle capacity: 2^23.
class hknpCompressedMeshShape : public hknpCompositeShape
{
	//+version(3)

	public:

		// Constants
		enum
		{
			// Maximum number of vertices that can be returned for a child convex shape
		#ifndef HK_PLATFORM_SPU
			MAX_VERTICES_PER_CONVEX_SHAPE = hknpConvexShape::MAX_NUM_VERTICES,
		#else
			MAX_VERTICES_PER_CONVEX_SHAPE = 128,
		#endif

			// Size of the static mesh tree (in bytes)
		#if (HK_POINTER_SIZE == 8)
		#	ifdef HK_REAL_IS_DOUBLE
				INTERNALS_SIZE = 224,
		#	else
				INTERNALS_SIZE = 160,
		#	endif
		#else
		#	ifdef HK_REAL_IS_DOUBLE
				INTERNALS_SIZE = 224,
		#	else
				INTERNALS_SIZE = 144,
		#	endif
		#endif
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Constructor.
		hknpCompressedMeshShape( const hknpCompressedMeshShapeCinfo& cInfo );

		/// Serialization constructor.
		hknpCompressedMeshShape( hkFinishLoadedObjectFlag flag );

		/// Destructor.
		~hknpCompressedMeshShape();

		/// Optimize for speed.
		/// This allocates extra memory (~4x larger) in order optimize mesh queries (~3x faster).
		/// Note that setting this to true can trigger a long operation - it is recommended to pre-compute it at construction time.
		void optimizeForSpeed( bool enable );

		/// Return true if the mesh as been optimized for speed.
		bool isOptimizedForSpeed() const;

		/// This method flags a primitive (triangle/quad) if:
		/// - none of its neighboring primitives has been flagged yet
		/// - the angle between the two adjacent primitives is less than \a angleThreshold [radians]
		/// This method is called from the constructor with a default angle threshold of 45deg.
		void flagInternalEdges( hkReal angleThreshold );

		/// Return true if the given triangle has been flagged as internal.
		HK_FORCE_INLINE bool isTriangleInternal( int triangleIndex ) const;

		//
		// hknpShape implementation
		//

		virtual hknpShapeType::Enum getType() const HK_OVERRIDE { return hknpShapeType::COMPRESSED_MESH; }

		virtual void calcAabb( const hkTransform& transform, hkAabb& aabbOut ) const HK_OVERRIDE;

		virtual int calcSize() const HK_OVERRIDE;

#if !defined( HK_PLATFORM_SPU )

		virtual hknpShapeKeyMask* createShapeKeyMask() const HK_OVERRIDE;

		virtual hkRefNew<hknpShapeKeyIterator> createShapeKeyIterator( const hknpShapeKeyMask* mask = HK_NULL ) const HK_OVERRIDE;

#endif

		virtual void getLeafShape( hknpShapeKey key, hknpShapeCollector* collector ) const HK_OVERRIDE;

		virtual void queryAabbImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hkArray<hknpShapeKey>* hits, hknpQueryAabbNmp* nmpInOut) const HK_OVERRIDE;

		virtual void queryAabbImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hknpCollisionQueryCollector* collector, hknpQueryAabbNmp* nmpInOut) const HK_OVERRIDE;

		virtual void castRayImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpRayCastQuery& query,
			const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hknpCollisionQueryCollector* collector ) const HK_OVERRIDE;

#if !defined(HK_PLATFORM_SPU)

		virtual hkResult buildSurfaceGeometry(
			const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut ) const HK_OVERRIDE;

#endif

		/// Internal. Cast a shape against a hknpCompressedMeshShape.
		static void castShapeImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpShapeCastQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector );

	protected:

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpCompressedMeshShape, hknpCompositeShape );

	protected:

		/// One bit per quad, defining whether it is flat.
		/// Note that to access this array we remove the lowest bit.
		hkBitField m_quadIsFlat;

		/// One bit per triangle, defining whether all its edges are internal/concave mesh edges.
		hkBitField m_triangleIsInternal;

		/// Internal data.
		HK_ALIGN16( hkUint8 m_data[INTERNALS_SIZE] );	//+overridetype(class hknpCompressedMeshShapeTree)

		/// Optional extended internal data.
		struct hknpCompressedMeshShapeTreeExt* m_extendedData;

		friend struct hknpCompressedMeshShapeInternals;
		friend class hknpCompressedMeshShapeUtil;
};

#include <Physics/Physics/Collide/Shape/Composite/Mesh/Compressed/hknpCompressedMeshShape.inl>


#endif // HKNP_COMPRESSED_MESH_SHAPE_H

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
