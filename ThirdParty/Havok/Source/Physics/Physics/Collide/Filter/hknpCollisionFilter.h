/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COLLISION_FILTER_H
#define HKNP_COLLISION_FILTER_H

#include <Physics/Physics/Collide/Query/hknpCollisionQuery.h>

class hknpCompoundShape;


/// Base class for all collision filters.
class hknpCollisionFilter : public hkReferencedObject
{
	public:

		/// Type, mostly used for debugging and SPU.
		enum FilterType
		{
			ALWAYS_HIT_FILTER,
			CONSTRAINT_FILTER,
			GROUP_FILTER,
			PAIR_FILTER,
			USER_FILTER
		};

		/// This input structure is passed to the shape level isCollisionEnabled() callback and holds contextual
		/// information on a colliding shape.
		struct FilterInput
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, FilterInput );

			hknpQueryFilterData						m_filterData;	///< Basic filter data associated with the colliding shape (like collision filter info, material id, user data).
			HK_PAD_ON_SPU( const hknpBody* )		m_body;			///< The colliding shape's governing body. Can be HK_NULL.
			HK_PAD_ON_SPU( const hknpShape* )		m_rootShape;	///< The colliding shape's topmost ancestor. Identical to the shape in m_body (if available).
			HK_PAD_ON_SPU( const hknpShape* )		m_parentShape;	///< The 'partner' shape's parent shape. HK_NULL for top-level shapes.
			HK_PAD_ON_SPU( hknpShapeKey )			m_shapeKey;		///< The full shape key for the colliding shape (or HKNP_INVALID_SHAPE_KEY if it is a convex shape or if no shape is available).
			HK_PAD_ON_SPU( const hknpShape* )		m_shape;		///< The colliding shape (if available).
		};

	public:

		/// Utility method to get the input data on the original query shape (where available).
		/// You only need to call this if your code depends on knowing which shape is which.
		static HK_FORCE_INLINE const FilterInput& getQueryShapeInput(
			bool targetShapeIsB, const FilterInput& shapeInputA, const FilterInput& shapeInputB );

		/// Utility method to get the input data on the original target shape.
		/// You only need to call this if your code depends on knowing which shape is which.
		static HK_FORCE_INLINE const FilterInput& getTargetShapeInput(
			bool targetShapeIsB, const FilterInput& shapeInputA, const FilterInput& shapeInputB );

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Constructor.
		hknpCollisionFilter( FilterType type ) : m_type( type ) {}

		/// Serialization constructor.
		hknpCollisionFilter( hkFinishLoadedObjectFlag flag ) : hkReferencedObject( flag ) {}

#if !defined( HK_PLATFORM_SPU )

		/// Destructor.
		virtual ~hknpCollisionFilter() {}

		/// Reduce the list of body pairs passed in through the \a pairs buffer; return the number of pairs left in buffer.
		/// Called after new overlaps have been found by the broad phase.
		virtual int filterBodyPairs(
			const hknpSimulationThreadContext& context, hknpBodyIdPair* pairs, int numPairs ) const = 0;

#endif

		/// Filter collisions at BODY TYPE level.
		/// Called from: hknpWorld collision queries.
		virtual bool isCollisionEnabled(
			hknpCollisionQueryType::Enum queryType,
			hknpBroadPhaseLayerIndex layerIndex ) const = 0;

		/// Filter collisions at BODY VS BODY level.
		/// Called from: hknpWorld::getClosestPoints() and hknpWorld::castShape() queries.
		virtual bool isCollisionEnabled(
			hknpCollisionQueryType::Enum queryType,
			hknpBodyId bodyIdA, hknpBodyId bodyIdB ) const = 0;

		/// Filter collisions at BODY level.
		/// Called from: hknpWorld collision queries.
		virtual bool isCollisionEnabled(
			hknpCollisionQueryType::Enum queryType,
			const hknpQueryFilterData& queryFilterData, const hknpBody& body ) const = 0;

		/// Filter collisions at SHAPE level.
		/// Called from: collision pipeline and collision queries.
		/// If /a targetShapeIsB is TRUE then /a shapeInputB holds data on the target shape and /a shapeInputA holds
		/// data on the query shape (if available).
		virtual bool isCollisionEnabled(
			hknpCollisionQueryType::Enum queryType, bool targetShapeIsB,
			const FilterInput& shapeInputA, const FilterInput& shapeInputB ) const = 0;

	public:

		/// The filter type.
		hkEnum<FilterType,hkUint8> m_type;
};


#include <Physics/Physics/Collide/Filter/hknpCollisionFilter.inl>


#endif // HKNP_COLLISION_FILTER_H

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
