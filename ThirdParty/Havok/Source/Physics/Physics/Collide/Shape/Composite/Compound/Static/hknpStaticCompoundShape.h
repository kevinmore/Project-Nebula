/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_STATIC_COMPOUND_SHAPE_H
#define HKNP_STATIC_COMPOUND_SHAPE_H

#include <Physics/Physics/Collide/Shape/Composite/Compound/hknpCompoundShape.h>

struct hknpClosestPointsQuery;

extern const hkClass hknpStaticCompoundShapeClass;
extern const hkClass hknpStaticCompoundShapeTreeClass;


/// A shape that stores a static set of child shape instances.
/// The instance shapes and transforms can only be set at build time, however the instances can be enabled or disabled
/// at any time.
class hknpStaticCompoundShape : public hknpCompoundShape
{
	//+hk.PostFinish("hknpStaticCompoundShape::postFinish")

	public:

		// Constants
		enum
		{
			// This is the value of sizeof(hknpStaticCompoundShapeTree).
			// We can't use that directly because hknpStaticCompoundShapeTree isn't accessible from public headers.
			#if defined(HK_REAL_IS_DOUBLE)
				TREE_SIZE = 128
			#else
				TREE_SIZE = 48
			#endif
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

#if !defined(HK_PLATFORM_SPU)

		/// Construct a shape from a given set of child shape instances.
		/// If massConfig is set, mass properties will be built and attached to the shape.
		/// If isMutable is set to true, instances can be safely disabled/enabled after construction.
		/// If instanceIdsOut is set, it will be populated with a list of instance IDs which map to the input instance indices.
		/// Notes:
		///  - the input instances are copied into internal storage.
		///  - if any child shapes already have mass properties, those will be used instead of building with massConfig.
		hknpStaticCompoundShape( const hknpShapeInstance* instances, int numInstances,
			const MassConfig* massConfig = HK_NULL, bool isMutable = false,
			hknpShapeInstanceId* instanceIdsOut = HK_NULL );

		/// Serialization constructor.
		hknpStaticCompoundShape( hkFinishLoadedObjectFlag flag );

		/// Post finish serialization constructor.
		static void postFinish( void* data );

		/// Destructor.
		virtual ~hknpStaticCompoundShape();

		/// hkReferencedObject implementation.
		virtual const hkClass* getClassType() const HK_OVERRIDE;

#endif // !HK_PLATFORM_SPU

		//
		// hknpCompoundShape implementation
		//

		virtual bool updateAabb() HK_OVERRIDE;

		virtual hknpShapeKey getChildShape( hknpShapeKey key, hknpShapeCollector* collector ) const HK_OVERRIDE;

		//
		// hknpShape implementation
		//

		virtual hknpShapeType::Enum getType() const HK_OVERRIDE { return hknpShapeType::STATIC_COMPOUND; }

		virtual void calcAabb( const hkTransform& transform, hkAabb& aabbOut ) const HK_OVERRIDE;

		virtual int calcSize() const HK_OVERRIDE;

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
			hknpCollisionQueryCollector* collector, hknpQueryAabbNmp* nmpInOut ) const HK_OVERRIDE;

		virtual void queryAabbImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hkArray<hknpShapeKey>* hits, hknpQueryAabbNmp* nmpInOut ) const HK_OVERRIDE;

#if !defined(HK_PLATFORM_SPU)

		virtual hknpShapeKeyMask* createShapeKeyMask() const HK_OVERRIDE;

#else

		static void createShapeKeyMaskVTable();

		void patchShapeKeyMaskVTable( hknpShapeKeyMask* mask ) const HK_OVERRIDE;

#endif // !HK_PLATFORM_SPU

		/// Internal. Cast a shape against a hknpStaticCompoundShape.
		static void castShapeImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpShapeCastQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector );

		/// Internal. Calculate the set of closest points between a hknpStaticCompoundShape and a hknpConvexShape.
		static void HK_CALL getClosestPointsImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpStaticCompoundShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector );

	protected:

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpStaticCompoundShape, hknpCompoundShape );

	protected:

		/// Internal bounding volume tree representation.
		HK_ALIGN16( hkUint8 m_tree[TREE_SIZE] ); //+overridetype(class hknpStaticCompoundShapeTree)

		friend class hknpStaticCompoundShapeInternals;
};


#endif // HKNP_STATIC_COMPOUND_SHAPE_H

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
