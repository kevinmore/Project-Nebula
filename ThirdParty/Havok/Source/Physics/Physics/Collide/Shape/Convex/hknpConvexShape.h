/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CONVEX_SHAPE_H
#define HKNP_CONVEX_SHAPE_H

#include <Physics/Physics/Collide/Shape/hknpShape.h>

#include <Common/Base/Types/Geometry/hkStridedVertices.h>
#include <Common/Base/Container/RelArray/hkRelArray.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

#include <Geometry/Internal/Types/hkcdVertex.h>
#include <Geometry/Internal/Algorithms/Gsk/hkcdGsk.h>

extern const hkClass hknpConvexShapeClass;

/// The base size of a convex shape, NOT including its vertices.
#define HKNP_CONVEX_SHAPE_BASE_SIZE		HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeof(hknpConvexShape))


/// A convex shape represented only as vertices, without face information.
class hknpConvexShape : public hknpShape
{
	public:

		/// Constants.
		enum
		{
			MAX_NUM_VERTICES = 252,	///< The maximum allowed number of vertices in a convex shape.
		};

		/// A vertex index. Maximum of MAX_NUM_VERTICES. 0xFF means invalid.
		typedef hkUint8 VertexIndex;

		/// Configuration structure for the createFromXxx() functions.
		struct BuildConfig
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpConvexShape::BuildConfig );

			/// Constructor. Sets default values.
			BuildConfig();

			/// If true, the shape will be a hknpConvexPolytopeShape, which includes face information used to
			/// optimize collision detection. If false, the shape will be a hknpConvexShape. (default = true).
			hkBool m_buildFaces;

			/// If true, mass properties will be built and attached to the shape (default = true).
			hkBool m_buildMassProperties;

			/// If m_buildMassProperties is set to true, this configures how the mass properties are built
			/// (default = high quality, unit density).
			MassConfig m_massConfig;

			/// If true, the shape will be shrunk by the convex radius during construction (default = true).
			hkBool m_shrinkByRadius;

			/// If m_shrinkByRadius is set to true, this value between zero and one controls how much to
			/// respect any sharp corners and edges in the shape when shrinking it.
			/// If set to zero, vertex displacement is unrestricted. Sharp features may shift significantly.
			/// If set to one, vertex displacement is restricted by the convex radius value.
			/// (default = 0.0)
			hkReal m_featurePreservationFactor;

			/// The maximum number of vertices in the resulting shape. If there are more vertices than this,
			/// the least significant ones will be dropped. (default = MAX_NUM_VERTICES).
			hkUint8 m_maxNumVertices;

			/// An optional transform to apply to the vertices during construction (default = NULL).
			hkTransform* m_extraTransform;

			/// Advanced. If set > 0, the shape will be simplified by expanding it by this value (default = 0.0).
			hkReal m_simplifyExpansionDistance;

			/// Internal. Must be set to the base size of the convex shape implementation.
			hkUint32 m_sizeOfBaseClass;
		};

	public:

#if !defined(HK_PLATFORM_SPU)

		//
		// Construction
		//

		/// Construct a convex shape from a point cloud.
		static hknpConvexShape* HK_CALL	createFromVertices(
			const struct hkStridedVertices& vertices, hkReal radius = HKNP_SHAPE_DEFAULT_CONVEX_RADIUS,
			const BuildConfig& config = BuildConfig() );

		/// Construct a convex shape from a point cloud. Vertices are indexed by the given index buffer.
		static hknpConvexShape* HK_CALL	createFromIndexedVertices(
			const hkVector4* HK_RESTRICT vertexBuffer, const hkUint16* HK_RESTRICT indexBuffer, int numVertices,
			hkReal radius = HKNP_SHAPE_DEFAULT_CONVEX_RADIUS, const BuildConfig& config = BuildConfig() );

		/// Construct a convex shape from half extents.
		static hknpConvexShape* HK_CALL	createFromHalfExtents(
			hkVector4Parameter halfExtent, hkReal radius = HKNP_SHAPE_DEFAULT_CONVEX_RADIUS,
			const BuildConfig& config = BuildConfig() );

		/// Construct a convex shape from an AABB.
		static hknpConvexShape* HK_CALL	createFromAabb(
			const hkAabb& aabb, hkReal radius = HKNP_SHAPE_DEFAULT_CONVEX_RADIUS,
			const BuildConfig& config = BuildConfig() );

		/// Construct a convex shape from a cylinder.
		/// \a halfExtent defines the cylinders half height along y and its radii along x and z.
		/// \a numVertices needs to be 8 or more and has to be an even number.
		static hknpConvexShape* HK_CALL	createFromCylinder(
			hkVector4Parameter halfExtent, int numVertices, hkReal radius = HKNP_SHAPE_DEFAULT_CONVEX_RADIUS,
			const BuildConfig& config = BuildConfig() );

#endif	// !HK_PLATFORM_SPU

		/// Construct a convex shape in the given buffer configured to have the given number of vertices.
		/// The vertices of the returned shape must be properly initialized before use. \a buffer must be
		/// 16 byte aligned. Advanced use only.
		static HK_FORCE_INLINE hknpConvexShape* createInPlace(
			int numVertices, hkReal radius, hkUint8* buffer, int bufferSize,
			int sizeOfBaseClass = HKNP_CONVEX_SHAPE_BASE_SIZE );

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Serialization constructor.
		hknpConvexShape( hkFinishLoadedObjectFlag flag );

		/// Calculate the AABB of the vertices in \a transform space, excluding the convex radius.
		HK_FORCE_INLINE void calcAabbNoRadius( const hkTransform& transform, hkAabb& aabbOut ) const;

		//
		// hknpShape implementation
		//

		virtual hknpShapeType::Enum getType() const HK_OVERRIDE { return hknpShapeType::CONVEX; }
		virtual void calcAabb( const hkTransform& transform, hkAabb& aabbOut ) const HK_OVERRIDE;
		virtual int calcSize() const HK_OVERRIDE;

		virtual int					getNumberOfSupportVertices() const HK_OVERRIDE;
		virtual const hkcdVertex*	getSupportVertices( hkcdVertex* vertexBuffer, int bufferSize ) const HK_OVERRIDE;
		HK_FORCE_INLINE void	getSupportingVertex( hkVector4Parameter direction, hkcdVertex* vertexBufferOut ) const;
		HK_FORCE_INLINE void	convertVertexIdsToVertices( const hkUint8* ids, int numIds, hkcdVertex* verticesOut ) const;

		virtual int				getNumberOfFaces() const HK_OVERRIDE { return 0; }
		virtual int				getFaceVertices( const int faceId, hkVector4& planeOut, hkcdVertex* vertexBufferOut ) const HK_OVERRIDE;
		HK_FORCE_INLINE void	getFaceInfo( const int faceId, hkVector4& planeOut, int& minAngleOut ) const;
		HK_FORCE_INLINE int		getSupportingFace(
			hkVector4Parameter direction, const hkcdGsk::Cache* gskCache, bool useB,
			hkVector4& planeOut, int& minAngleOut, hkUint32 &prevFaceId ) const;

#if !defined(HK_PLATFORM_SPU)
		virtual void getAllShapeKeys(
			const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask,
			hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const HK_OVERRIDE;
#else
		virtual void getAllShapeKeys(
			const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask,
			hkUint8* shapeBuffer, int shapeBufferSize,
			hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const HK_OVERRIDE;
#endif

#if !defined(HK_PLATFORM_SPU)
		virtual hkRefNew<hknpShapeKeyIterator> createShapeKeyIterator(
			const hknpShapeKeyMask* mask ) const HK_OVERRIDE;
#else
		virtual hknpShapeKeyIterator* createShapeKeyIterator(
			hkUint8* buffer, int bufferSize,
			const hknpShapeKeyMask* mask ) const HK_OVERRIDE;
#endif

		virtual void getLeafShape( hknpShapeKey key, hknpShapeCollector* collector ) const HK_OVERRIDE;

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

		virtual void buildMassProperties(
			const MassConfig& massConfig, hkDiagonalizedMassProperties& massPropertiesOut ) const HK_OVERRIDE;

		virtual hkResult buildSurfaceGeometry(
			const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut ) const HK_OVERRIDE;

#endif

	public:

		// Vertex accessors
		HK_FORCE_INLINE int					getNumberOfVertices() const { return m_vertices.getSize(); }
		HK_FORCE_INLINE hkcdVertex*			getVertices()				{ return m_vertices.begin(); }
		HK_FORCE_INLINE const hkcdVertex*	getVertices() const			{ return m_vertices.begin(); }
		HK_FORCE_INLINE const hkcdVertex&	getVertex(int index) const	{ return getVertices()[index]; }

		/// Slow reference implementation of getSupportingVertex() for debugging purposes.
		void getSupportingVertexRef( hkVector4Parameter direction, hkcdVertex& supportingVertexOut ) const;

	protected:

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpConvexShape, hknpShape );

		/// Protected constructor. Use the createFromXxx() methods instead.
		HK_FORCE_INLINE	hknpConvexShape( int numVertices, hkReal radius, int sizeOfBaseClass = HKNP_CONVEX_SHAPE_BASE_SIZE );

		/// Calculate the AABB of the vertices in local space, excluding the convex radius.
		hkAabb calcAabbNoRadius();

		/// Initializes the convex shape. Required to construct shapes in place in SPU.
		HK_FORCE_INLINE void init( int numVertices, hkReal radius, int sizeOfBaseClass = HKNP_CONVEX_SHAPE_BASE_SIZE );

		/// Returns the size of a convex shape with the given characteristics.
		static HK_FORCE_INLINE int calcConvexShapeSize( int numVertices, int sizeofBaseClass = HKNP_CONVEX_SHAPE_BASE_SIZE );

		/// Allocates a buffer for a convex shape with the given characteristics. Placement new with hknpConvexShape
		/// constructor must be performed in the buffer before any access to it as a convex shape.
		static HK_FORCE_INLINE void* allocateConvexShape( int numVertices, int sizeOfBaseClass, int& shapeSizeOut );

	private:

		// Copying is not allowed
		HK_FORCE_INLINE hknpConvexShape( const hknpConvexShape& other );
		HK_FORCE_INLINE void operator=( const hknpConvexShape& other );

	protected:

		hkRelArray<hkcdVertex> m_vertices;	//+overridetype(hkRelArray<hkVector4>) 	///< Offset to vertices stream.

		static HK_ALIGN16( const hkUint32 s_curIndices[4] );	
};

#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.inl>


#endif // HKNP_CONVEX_SHAPE

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
