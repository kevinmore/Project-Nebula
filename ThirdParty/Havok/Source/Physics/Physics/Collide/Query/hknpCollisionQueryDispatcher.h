/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COLLISION_QUERY_DISPATCHER_H
#define HKNP_COLLISION_QUERY_DISPATCHER_H

#include <Physics/Physics/Collide/Query/hknpCollisionQuery.h>

class hknpCollisionQueryCollector;
class hknpConvexShape;


/// Collision query dispatcher.
/// Performs collision queries between pairs of shapes, based on a set of registered dispatch functions.
class hknpCollisionQueryDispatcherBase
{
	public:

		/// Base shape types, used in the dispatch tables.
		enum BaseType
		{
			// Convex types
			CONVEX,

			// Composite types
			COMPRESSED_MESH,
			EXTERN_MESH,
			STATIC_COMPOUND,
			DYNAMIC_COMPOUND,

			// Height field types
			HEIGHT_FIELD,

			// Wrapper types
			MASKED_COMPOSITE,
			SCALED_CONVEX,

			// User defined types
			USER,

			NUM_BASE_TYPES
		};

		/// A bitfield where each bit represent one of BaseType. Used to configure the query dispatcher initialization.
		typedef hkUint32 ShapeMask;

#if defined(HK_PLATFORM_SPU)
		static const ShapeMask defaultShapeMask = 0 \
			| 1 << hknpCollisionQueryDispatcherBase::CONVEX \
			| 1 << hknpCollisionQueryDispatcherBase::COMPRESSED_MESH \
			| 1 << hknpCollisionQueryDispatcherBase::STATIC_COMPOUND \
			| 1 << hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND \
			/* 1 << hknpCollisionQueryDispatcherBase::HEIGHT_FIELD*/ \
			| 1 << hknpCollisionQueryDispatcherBase::MASKED_COMPOSITE \
			| 1 << hknpCollisionQueryDispatcherBase::SCALED_CONVEX \
			/*| 1 << hknpCollisionQueryDispatcherBase::USER*/;
#else
		static const ShapeMask defaultShapeMask = 0xFFFFFFFF;
#endif

		/// A dispatch table for a given function type.
		template<typename FUNCTION_TYPE>
		struct DispatchTable
		{
			/// Set the function for querying one shape type against another.
			HK_FORCE_INLINE void setFunction(
				const BaseType query, const BaseType target, FUNCTION_TYPE func, ShapeMask shapeMask = defaultShapeMask );

			/// Get the function for querying one shape type against another.
			HK_FORCE_INLINE FUNCTION_TYPE getFunction( const BaseType query, const BaseType target ) const;

			FUNCTION_TYPE m_dispatchTable[NUM_BASE_TYPES][NUM_BASE_TYPES];
		};

		/// A function to cast one shape against another.
		typedef void (HK_CALL *ShapeCastFunc)(
			hknpCollisionQueryContext* queryContext,
			const hknpShapeCastQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector );

		/// A function to get the closest points between a pair of shapes.
		typedef void (HK_CALL *ClosestPointsFunc)(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo &queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo &targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector );

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCollisionQueryDispatcherBase );

		/// Constructor.
		hknpCollisionQueryDispatcherBase();

		/// Destructor.
		~hknpCollisionQueryDispatcherBase();

		/// Set the base type for a shape type.
		/// NOTE: shapes with the given shape type must derive from shapes with the given base type.
		HK_FORCE_INLINE void setBaseShapeType( hknpShapeType::Enum shapeType, BaseType baseType );

		/// Get the base type for a shape type.
		HK_FORCE_INLINE BaseType getBaseShapeType( hknpShapeType::Enum shapeType ) const;

		/// Cast a shape against another shape.
		void castShape(
			hknpCollisionQueryContext* queryContext,
			const hknpShapeCastQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector ) const;

		/// Find the set of closest points between two shapes.
		void getClosestPoints(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector ) const;

	protected:

		/// A map of concrete shape types to base types used for dispatching.
		BaseType m_baseTypeMap[hknpShapeType::NUM_SHAPE_TYPES];

	public:

		/// A dispatch table for closest point queries.
		DispatchTable<ClosestPointsFunc> m_closestPointsDispatchTable;

		/// A dispatch table for shape cast queries.
		DispatchTable<ShapeCastFunc> m_shapeCastDispatchTable;
};


/// The default collision query dispatcher implementation.
/// Handles both shape cast and get closest point queries.
class hknpCollisionQueryDispatcher : public hknpCollisionQueryDispatcherBase
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCollisionQueryDispatcher );

		/// Constructor.
		hknpCollisionQueryDispatcher( ShapeMask shapeMask = defaultShapeMask );
};


/// A collision query dispatcher implementation which handles only shape cast queries.
class hknpShapeCastQueryDispatcher : public hknpCollisionQueryDispatcherBase
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpShapeCastQueryDispatcher );

		/// Constructor.
		hknpShapeCastQueryDispatcher( ShapeMask shapeMask = defaultShapeMask );
};


/// A collision query dispatcher implementation which handles only get closest points queries.
class hknpGetClosestPointsQueryDispatcher : public hknpCollisionQueryDispatcherBase
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpGetClosestPointsQueryDispatcher );

		/// Constructor.
		hknpGetClosestPointsQueryDispatcher( ShapeMask shapeMask = defaultShapeMask );
};


#include <Physics/Physics/Collide/Query/hknpCollisionQueryDispatcher.inl>

#endif // HKNP_COLLISION_QUERY_DISPATCHER_H

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
