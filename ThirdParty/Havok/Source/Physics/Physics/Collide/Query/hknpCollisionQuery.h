/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COLLISION_QUERY_H
#define HKNP_COLLISION_QUERY_H

#include <Common/Base/Container/Array/hkFixedCapacityArray.h>
#include <Common/Base/Container/Array/hkFixedInplaceArray.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Geometry/Internal/Types/hkcdRay.h>
#include <Physics/Physics/Collide/Query/Collector/hknpCollisionQueryCollector.h>
#include <Physics/Physics/Collide/Shape/Convex/Triangle/hknpTriangleShape.h>
#include <Physics/Physics/Dynamics/Body/hknpBody.h>

class hknpShapeTagCodec;
struct hknpShapeKeyMask;
class hknpCollisionFilter;
struct hknpAabbQueryResult;
struct hknpClosestPointsQueryResult;
struct hknpRayCastQueryResult;
struct hknpShapeCastQueryResult;
class hknpCollisionQueryDispatcherBase;

#define HKNP_DEFAULT_SHAPE_CAST_ACCURACY 0.001f


struct hknpCollisionQueryType
{
	enum Enum
	{
		UNDEFINED,
		RAY_CAST,
		SHAPE_CAST,
		GET_CLOSEST_POINTS,
		QUERY_AABB,
	};
};


/// This struct holds a shape tag and its immediate context as returned by hknpShape::getLeafShape().
struct hknpShapeTagPathEntry
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpShapeTagPathEntry );

	HK_PAD_ON_SPU( hknpShapeTag )		m_shapeTag;		///< A shape tag.
	HK_PAD_ON_SPU( const hknpShape* )	m_parentShape;	///< Parent shape to the 'tagged' shape.
	HK_PAD_ON_SPU( hknpShapeKey )		m_shapeKey;		///< Shape key of the 'tagged' shape.
	HK_PAD_ON_SPU( const hknpShape* )	m_shape;		///< The tagged shape itself.
};


/// For performance reasons a Shape Tag Path is passed using a fixed inplace array. Its maximum depth is hardcoded
/// to 8 levels. If you have shape hierarchies deeper than 8 levels you can adjust this value here accordingly.
#define HKNP_MAXIMUM_SHAPE_TAG_PATH_DEPTH 8
typedef hkFixedInplaceArray<hknpShapeTagPathEntry, HKNP_MAXIMUM_SHAPE_TAG_PATH_DEPTH> hknpShapeTagPath;


/// A storage structure that will be passed down the collision query pipeline and that holds pre-allocated or
/// persistent data.
/// If you intend to multithread your collision queries, make sure to create dedicated query contexts for each
/// query (as the triangle shapes cannot be shared between concurrent queries.)
struct hknpCollisionQueryContext
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCollisionQueryContext );

		/// Creates two empty triangles and sets the shape tag codec to HK_NULL.
		/// The \a shapeTagCodec is only required if filtering is enabled for the query.
		hknpCollisionQueryContext();

		/// This variant allows to pass in external triangle shapes instead of creating them.
		/// The \a shapeTagCodec is only required if filtering is enabled for the query.
		HK_FORCE_INLINE hknpCollisionQueryContext( hknpTriangleShape* queryTriangle, hknpTriangleShape* targetTriangle );

		/// Deletes the two allocated triangles if they weren't passed in from the outside.
		HK_FORCE_INLINE ~hknpCollisionQueryContext();

	public:

		/// The collision query dispatcher to use for pairwise shape queries.
		HK_PAD_ON_SPU( const hknpCollisionQueryDispatcherBase* ) m_dispatcher;

		/// The codec to use for decoding shape tags during a collision query.
		/// Some values in hknpCollisionResult will only be valid if a shape tag codec has been provided.
		/// This value *has* to be set if filtering is enabled for a spatial query.
		HK_PAD_ON_SPU( const hknpShapeTagCodec* ) m_shapeTagCodec;

		/// A pre-allocated but uninitialized triangle shape.
		/// This can be used temporarily by the collision (query) pipeline.
		HK_PAD_ON_SPU( hknpTriangleShape* ) m_queryTriangle;

		/// A pre-allocated but uninitialized triangle shape.
		/// This can be used temporarily by the collision (query) pipeline.
		HK_PAD_ON_SPU( hknpTriangleShape* ) m_targetTriangle;

	protected:

		/// Set to true if the two triangle shapes were passed in from the outside.
		HK_PAD_ON_SPU( hkBool32 ) m_externallyAllocatedTriangles;
};


/// A collection of data on a shape that is partaking in a collision query and that could be relevant to a filter.
struct hknpQueryFilterData
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpQueryFilterData );

		/// Initializes this struct to default values.
		HK_FORCE_INLINE hknpQueryFilterData();

		/// Initializes this struct from data taken from \a body.
		HK_FORCE_INLINE hknpQueryFilterData( const hknpBody& body );

		/// Sets all members to data taken from \a body.
		HK_FORCE_INLINE void setFromBody( const hknpBody& body );

	public:

		hknpMaterialId			m_materialId;			///< The material id associated with the shape. Allowed to be 'invalid'.
		hkPadSpu< hkUint32 >	m_collisionFilterInfo;	///< The collision filter info associated with the shape.
		hkPadSpu< hkUint64 >	m_userData;				///< The user data associated with the shape.
};


/// Additional information on a shape that is partaking in a collision query.
struct hknpShapeQueryInfo
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpShapeQueryInfo );

		/// Initializes this struct to default values.
		HK_FORCE_INLINE hknpShapeQueryInfo();

		/// Constructor. Copies values from another hknpShapeQueryInfo. Resets m_shapeKeyMask to HK_NULL!
		HK_FORCE_INLINE hknpShapeQueryInfo( const hknpShapeQueryInfo* other );

		/// Initialize this struct from an existing hknpShapeQueryInfo. Resets m_shapeKeyMask to HK_NULL!
		HK_FORCE_INLINE	void setFromInfo( const hknpShapeQueryInfo& other );

		/// Initialize this struct from an existing \a body.
		HK_FORCE_INLINE void setFromBody( const hknpBody& body );

	private:

		// Don't allow this class to be copied. Use the explicit constructor to pass in a source hknpShapeQueryInfo.
		hknpShapeQueryInfo( const hknpShapeQueryInfo& other );
		hknpShapeQueryInfo& operator=( const hknpShapeQueryInfo& other );

	public:

		hkPadSpu< const hknpBody* >			m_body;				///< The shape's governing body. Can be set to HK_NULL.
		hkPadSpu< const hknpShape* >		m_rootShape;		///< The shape's topmost ancestor. Identical to the shape in m_body (if available).
		hkPadSpu< const hknpShape* >		m_parentShape;		///< The shape's parent shape. HK_NULL for top-level shapes.
		hknpShapeKeyPath					m_shapeKeyPath;		///< The shape's key path.
		hkPadSpu< const hkTransform* >		m_shapeToWorld;		///< The shape's world transform. This value is MANDATORY!
		hkPadSpu< const hknpShapeKeyMask* >	m_shapeKeyMask;		///< The shape's (optional) shape key mask.

		hkPadSpu< hkReal >					m_shapeConvexRadius;///< This value will be automatically set by the engine. The convex radius for the shape. This value will be ignored if all shapes involved in a spatial query are unscaled.
		hkPadSpu< hkBool32 >				m_shapeIsScaled;	///< This value will be automatically set by the engine. Non-zero if shape is scaled.
		hkVector4							m_shapeScale;		///< This value will be automatically set by the engine. The scaling vector for the shape (if scaling is enabled).
		hkVector4							m_shapeScaleOffset;	///< This value will be automatically set by the engine. The offset for the shape (if scaling is enabled).
};


// =====================================================================================================================
//
// QUERY INPUT STRUCTURES
//
// =====================================================================================================================

/// Input structure for RAY CAST queries.
struct hknpRayCastQuery : public hkcdRay
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpRayCastQuery );

		/// Initializes the query to default values.
		HK_FORCE_INLINE	hknpRayCastQuery();

		/// Initializes the ray to be cast from \a start position (world space) to \a end position (world space)
		/// and sets the remaining members to their default values.
		HK_FORCE_INLINE	hknpRayCastQuery( hkVector4Parameter start, hkVector4Parameter end );

		/// Initializes the ray to be cast from \a start position (world space) along the \a direction (normalized)
		/// up to a maximum \a length and sets the remaining members to their default values.
		HK_FORCE_INLINE	hknpRayCastQuery(
			hkVector4Parameter start, hkVector4Parameter direction, hkSimdRealParameter length );

		/// (Re)Set the ray to be cast from \a start position (world space) to \a end position (world space).
		HK_FORCE_INLINE	void setStartEnd( hkVector4Parameter start, hkVector4Parameter end );

		/// (Re)Set the ray to be cast from \a start position (world space) along the \a direction (normalized)
		/// up to a maximum \a length.
		HK_FORCE_INLINE	void setStartDirectionLength(
			hkVector4Parameter start, hkVector4Parameter direction, hkSimdRealParameter length );

	protected:

		/// Sets all members to their default values.
		HK_FORCE_INLINE void init();

	public:

		/// Flags for customizing the ray behavior.
		hkcdRayQueryFlags::Enum m_flags;

		/// Additional filter data associated with the ray.
		hknpQueryFilterData m_filterData;

		/// An optional filter associated with this query.
		hkPadSpu< hknpCollisionFilter* > m_filter;
};


/// Input structure for SHAPE CAST queries.
struct hknpShapeCastQuery : public hknpRayCastQuery
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpShapeCastQuery );

		/// Initializes this query with default values.
		HK_FORCE_INLINE	hknpShapeCastQuery();

		/// Initializes this query to cast a \a shape (positioned at \a castStartPosition) along the \a castDirection,
		/// up to a (maximum) \a castDistance and sets the remaining members to their default values.
		/// See hknpWorld::castShape() and hknpShapeQueryInterface::castShape() for details on the coordinate space
		/// of start position, cast direction and maximum distance.
		HK_FORCE_INLINE	hknpShapeCastQuery(
			const hknpShape& shape, hkVector4Parameter castStartPosition,
			hkVector4Parameter castDirection, hkSimdRealParameter maximumDistance );

		/// Initializes this query to cast a \a body from its current (world space) position along the \a castDirection,
		/// up to a (maximum) \a castDistance and sets the remaining members to their default values.
		/// See hknpWorld::castShape() and hknpShapeQueryInterface::castShape() for details on the coordinate space
		/// of cast direction and maximum distance.
		/// Note that using this constructor before calling hknpShapeQueryInterface::castShape() will give incorrect
		/// results unless you manually transform the underlying hkcdRay's m_origin to target shape space as well.
		HK_FORCE_INLINE	hknpShapeCastQuery(
			const hknpBody& body, hkVector4Parameter castDirection, hkSimdRealParameter maximumDistance );

	public:

		/// The shape to cast.
		hkPadSpu< const hknpShape* > m_shape;

		/// An optional body to which \a m_shape belongs.
		/// If set, this pointer will be used to avoid 'self collisions' during world queries (i.e. to avoid getting
		/// \a m_shape itself reported as a hit) as well as a return parameter in the collision result.
		hkPadSpu< const hknpBody* > m_body;

		/// Shape casting uses an iterative approach in order to find the exact collision point. This parameter
		/// determines how accurate that point should be. Lower values result in more accurate results, but with
		/// potentially more iterations and higher CPU cost.
		/// Defaults to 0.001.
		hkPadSpu< hkReal > m_accuracy;
};


/// Input structure for CLOSEST POINTS queries.
struct hknpClosestPointsQuery
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpClosestPointsQuery );

		/// Initializes the query to default values.
		HK_FORCE_INLINE	hknpClosestPointsQuery();

		/// Initializes the query's shape and the maximum distance from the input and sets the remaining members to
		/// their default values.
		HK_FORCE_INLINE	hknpClosestPointsQuery( const hknpShape& shape, hkReal maxDistance );

		/// Initializes the query's shape, associated body and filter data from the input body, initializes the
		/// maximum distance from the input parameter and sets the remaining members to their default values.
		HK_FORCE_INLINE	hknpClosestPointsQuery( const hknpBody& body, hkReal maxDistance );

	protected:

		/// Sets all members to their default values.
		HK_FORCE_INLINE	void init();

	public:

		/// Shape to compute closest points against.
		hkPadSpu< const hknpShape* > m_shape;

		/// An optional body to which \a m_shape belongs.
		/// If set, this pointer will be used to avoid 'self collisions' during world queries (i.e. to avoid getting
		/// \a m_shape reported as the closest point to \a m_shape).
		hkPadSpu< const hknpBody* > m_body;

		/// Only those points that are closer or equal to this distance will be reported.
		hkPadSpu< hkReal > m_maxDistance;

		/// An optional filter associated with this query.
		hkPadSpu< hknpCollisionFilter* > m_filter;

		/// Additional filter data associated with \a m_shape.
		hknpQueryFilterData m_filterData;
};


/// Input structure for AABB queries.
struct hknpAabbQuery
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS, hknpAabbQuery);

		/// Initializes the query to default values.
		HK_FORCE_INLINE	hknpAabbQuery();

		/// Initializes the embedded AABB from the input and sets the remaining members to their default values.
		HK_FORCE_INLINE	hknpAabbQuery( const hkAabb& aabb );

	protected:

		/// Sets all members to their default values.
		HK_FORCE_INLINE void init();

	public:

		/// The AABB to query. This AABB should be in:
		///  - shape space when using hknpShapeQueryInterface::queryAabb().
		///  - world space when using hknpWorld::queryAabb().
		hkAabb m_aabb;

		/// An optional filter associated with this query.
		hkPadSpu< hknpCollisionFilter* > m_filter;

		/// Additional filter data associated with the AABB.
		hknpQueryFilterData m_filterData;
};


/// The output structure for collision queries (Ray Cast, Shape Cast, Closest Points, AABB).
struct hknpCollisionResult
{
	public:

		/// Combined information on a body and its shape (see m_queryBodyInfo and m_hitBodyInfo for details).
		struct BodyInfo
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, BodyInfo );

			hknpBodyId					m_bodyId;
			hknpMaterialId				m_shapeMaterialId;
			hkPadSpu< hknpShapeKey >	m_shapeKey;
			hkPadSpu< hkUint32 >		m_shapeCollisionFilterInfo;
			hkPadSpu< hkUint64 >		m_shapeUserData;
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCollisionResult );

		/// Initializes the structure to default values.
		HK_FORCE_INLINE hknpCollisionResult();

		/// Comparison operator required for sorting, checks the fractions.
		HK_FORCE_INLINE hkBool operator<( const hknpCollisionResult& other ) const;

		/// Sets all members to their default values.
		/// You should call this before re-using the structure for another collision query call.
		HK_FORCE_INLINE void clear();

		/// Interpret these results as AABB query results.
		/// You can use this specific interface for easier access to the results.
		HK_FORCE_INLINE const hknpAabbQueryResult& asAabb() const;

		/// Interpret these results as CLOSEST POINTS query results.
		/// You can use this specific interface for easier access to the results.
		HK_FORCE_INLINE const hknpClosestPointsQueryResult& asClosestPoints() const;

		/// Interpret these results as RAY CAST query results.
		/// You can use this specific interface for easier access to the results.
		HK_FORCE_INLINE const hknpRayCastQueryResult& asRayCast() const;

		/// Interpret these results as SHAPE CAST query results.
		/// You can use this specific interface for easier access to the results.
		HK_FORCE_INLINE const hknpShapeCastQueryResult& asShapeCast() const;

	public:

		/// <ul>
		/// <li><b>Ray Cast Query :</b>       <ul><li>Ray hit position (world space).</ul>
		/// <li><b>Shape Cast Query :</b>     <ul><li>Contact position (world space) between query shape and hit shape.</ul>
		/// <li><b>Closest Points Query :</b> <ul><li>The closest point (world space) on the hit shape.</ul>
		/// <li><b>AABB Query :</b>           <ul><li>Undefined.</ul>
		/// </ul>
		hkVector4 m_position;

		/// <ul>
		/// <li><b>Ray Cast Query :</b>       <ul><li>Surface normal (world space) at ray hit position.</ul>
		/// <li><b>Shape Cast Query :</b>     <ul><li>Contact normal (world space) at contact position (pointing from hit shape to query shape).</ul>
		/// <li><b>Closest Points Query :</b> <ul><li>Normalized separating vector (world space) between both shapes (pointing from hit shape to query shape).</ul>
		/// <li><b>AABB Query :</b>           <ul><li>Undefined.</ul>
		/// </ul>
		hkVector4 m_normal;

		/// <ul>
		/// <li><b>Ray Cast Query :</b>       <ul><li>Relative distance to the hit position along the cast direction.\n
		///                                           Absolute distance = length of (\a m_fraction * cast direction)</ul>
		/// <li><b>Shape Cast Query :</b>     <ul><li>Relative distance the shape has traveled along the cast direction until it hit.\n
		///                                           Absolute distance = length of (\a m_fraction * cast direction)</ul>
		/// <li><b>Closest Points Query :</b> <ul><li>Absolute shortest distance between the query shape and the found neighbor shape.</ul>
		/// <li><b>AABB Query :</b>           <ul><li>Undefined.</ul>
		/// </ul>
		hkPadSpu< hkReal > m_fraction;

		/// <ul>
		/// <li><b>Ray Cast Query :</b>       <ul><li>Body is undefined.
		///                                       <li>Shape key is undefined.
		///                                       <li>User Data, Collision Filter Info and Material Id associated with the query.</ul>
		/// <li><b>Shape Cast Query :</b>     <ul><li>Body associated with the query shape (if available).
		///                                       <li>Shape key at the collision point.
		///                                       <li>User Data, Collision Filter Info and Material Id at the collision point. This data is only available if shape tag decoding has been enabled by passing a valid hknpShapeTagCodec to the query via the hknpCollisionQueryContext.</ul>
		/// <li><b>Closest Points Query :</b> <ul><li>Body associated with the query shape (if available).
		///                                       <li>Shape key at the position closest to the hit shape.
		///                                       <li>User Data, Collision Filter Info and Material Id at the position closest to the hit shape. This data is only available if shape tag decoding has been enabled by passing a valid hknpShapeTagCodec to the query via the hknpCollisionQueryContext.</ul>
		/// <li><b>AABB Query :</b>           <ul><li>Body is undefined.
		///                                       <li>Shape key is undefined.
		///                                       <li>User Data, Collision Filter Info and Material Id associated with the query.</ul>
		/// </ul>
		BodyInfo m_queryBodyInfo;

		/// <ul>
		/// <li><b>Ray Cast Query :</b>       <ul><li>Body of the hit shape (if available).
		///                                       <li>Shape key at the ray's hit position.
		///                                       <li>User Data, Collision Filter Info and Material Id at the ray's hit position. This data is only available if shape tag decoding has been enabled by passing a valid hknpShapeTagCodec to the query via the hknpCollisionQueryContext.</ul>
		/// <li><b>Shape Cast Query :</b>     <ul><li>Body of the hit shape (if available).
		///                                       <li>Shape key at the collision point.
		///                                       <li>User Data, Collision Filter Info and Material Id at the collision point. This data is only available if shape tag decoding has been enabled by passing a valid hknpShapeTagCodec to the query via the hknpCollisionQueryContext.</ul>
		/// <li><b>Closest Points Query :</b> <ul><li>Body of the shape closest to the query shape (if available).
		///                                       <li>Shape key at the position closest to the query shape.
		///                                       <li>User Data, Collision Filter Info and Material Id at the position closest to the query shape. This data is only available if shape tag decoding has been enabled by passing a valid hknpShapeTagCodec to the query via the hknpCollisionQueryContext.</ul>
		/// <li><b>AABB Query :</b>           <ul><li>Body of the shape overlapping with the AABB (if available).
		///                                       <li>Shape key of the shape overlapping with the AABB.
		///                                       <li>User Data, Collision Filter Info and Material Id of the shape overlapping with the AABB. This data is only available if shape tag decoding has been enabled by passing a valid hknpShapeTagCodec to the query via the hknpCollisionQueryContext.</ul>
		/// </ul>
		BodyInfo m_hitBodyInfo;

		/// The type of the source query.
		hknpCollisionQueryType::Enum m_queryType;

		/// The ray cast return value, See \ref RayCastDocumentation.
		hkPadSpu< hkInt32 > m_hitResult;
};


// This include needs to be here as its classes derive from hknpCollisionResult.
#include <Physics/Physics/Collide/Query/hknpCollisionResultAccessors.h>


// =====================================================================================================================
//
// PIPELINE-INTERNAL OR SHAPE-LEVEL STRUCTURES.
//
// =====================================================================================================================

/// This (internally used) utility structure provides an interface to some templated queryAabb() methods.
struct hknpAabbQueryUtil
{
	HK_FORCE_INLINE static void addHit(
		const hknpBody* HK_RESTRICT queryBody, const hknpQueryFilterData& queryShapeFilterData,
		const hknpBody* HK_RESTRICT targetBody, hknpShapeKey targetShapeKey, const hknpQueryFilterData& targetShapeFilterData,
		hkArray<hknpShapeKey>* collector );

	HK_FORCE_INLINE static void addHit(
		const hknpBody* HK_RESTRICT queryBody,
		const hknpBody* HK_RESTRICT targetBody, hknpShapeKey targetShapeKey,
		hkArray<hknpShapeKey>* collector );

	HK_FORCE_INLINE static void addHit(
		const hknpBody* HK_RESTRICT queryBody, const hknpQueryFilterData& queryShapeFilterData,
		const hknpBody* HK_RESTRICT targetBody, hknpShapeKey targetShapeKey, const hknpQueryFilterData& targetShapeFilterData,
		hknpCollisionQueryCollector* collector );

	HK_FORCE_INLINE static void addHit(
		const hknpBody* HK_RESTRICT queryBody,
		const hknpBody* HK_RESTRICT targetBody, hknpShapeKey targetShapeKey,
		hknpCollisionQueryCollector* collector );
};


#include <Physics/Physics/Collide/Query/hknpCollisionQuery.inl>


#endif // HKNP_COLLISION_QUERY_H

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
