/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_DYNAMIC_COMPOUND_SHAPE_H
#define HKNP_DYNAMIC_COMPOUND_SHAPE_H

#include <Physics/Physics/Collide/Shape/Composite/Compound/hknpCompoundShape.h>

struct hknpClosestPointsQuery;

extern const hkClass hknpDynamicCompoundShapeClass;
extern const hkClass hknpDynamicCompoundShapeTreeClass;


/// A shape that stores a dynamic set of child shape instances.
/// Each instance can be enabled or disabled individually and their transforms can be changed.
class hknpDynamicCompoundShape : public hknpCompoundShape
{
	//+hk.PostFinish("hknpDynamicCompoundShape::postFinish")

	public:

		// Constants
		enum
		{
			// This is the value of sizeof(hknpDynamicCompoundShapeTree).
			// We can't use that directly because hknpDynamicCompoundShapeTree isn't accessible from public headers.
			#if defined(HK_REAL_IS_DOUBLE)
				TREE_SIZE = 128
			#else
			#	if (HK_POINTER_SIZE == 4)
					TREE_SIZE = 28
			#	elif defined(HK_COMPILER_CLANG) || defined(HK_COMPILER_GCC)
					TREE_SIZE = 32
			#	else
					TREE_SIZE = 40
			#	endif
			#endif
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

#if !defined( HK_PLATFORM_SPU )

		/// Construct an empty shape.
		hknpDynamicCompoundShape( int capacity );

		/// Construct a shape from a given set of child shape instances.
		/// If massConfig is set, mass properties will be built and attached to the shape.
		/// If instanceIdsOut is set, it will be populated with a list of instance IDs which map to the input instance indices.
		/// Notes:
		///  - the input instances are copied into internal storage.
		///  - the shape is optimized during construction. There is no need to call rebuild() afterward.
		///  - if any child shapes already have mass properties, those will be used instead of building with massConfig.
		///  - the mass properties may need to be rebuilt manually if the shape is significantly changed after construction.
		hknpDynamicCompoundShape( const hknpShapeInstance* instances, int numInstances, int capacity,
			const MassConfig* massConfig = HK_NULL, hknpShapeInstanceId* instanceIdsOut = HK_NULL );

		/// Serialization constructor.
		hknpDynamicCompoundShape( hkFinishLoadedObjectFlag flag );

		/// Post finish serialization constructor.
		static void postFinish( void* data );

		/// Destructor.
		virtual ~hknpDynamicCompoundShape();

		/// hkReferencedObject implementation.
		virtual const hkClass* getClassType() const HK_OVERRIDE;

#endif

		/// Add a set of instances to the shape.
		/// If the instanceIdsOut buffer is provided, it must be sized for numInstances. If any instance could not
		/// be added (because the shape is full), its returned instance ID will be hknpShapeInstanceId::invalid().
		void addInstances( const hknpShapeInstance* instances, int numInstances,
			hknpShapeInstanceId* instanceIdsOut = HK_NULL );

		/// Remove a set of instances from the shape.
		void removeInstances( const hknpShapeInstanceId* instanceIds, int numInstances );

		/// Replace a set of existing instances.
		void updateInstances( const hknpShapeInstanceId* instanceIds, int numInstances,
			const hknpShapeInstance* instances );

		/// Update the transforms and/or scales of a set of existing instances.
		/// Either the transforms or the scales can be set to NULL.
		/// This does a quick update of the bounding volume tree (leaf AABBs only). If you need to keep the tree
		/// optimal you should call optimize() regularly.
		void setInstanceTransforms( const hknpShapeInstanceId* instanceIds, int numInstances,
			const hkTransform* transforms, const hkVector4* scales = HK_NULL );

		/// Optimize the bounding volume tree.
		/// This method can be called if a lot of instances have been updated.
		void optimize();

		/// Rebuild and optimize the bounding volume tree.
		/// This method should be called if a lot of instances have been added.
		void rebuild();

		//
		// hknpCompoundShape implementation
		//

		virtual bool updateAabb() HK_OVERRIDE;

		virtual hknpShapeKey getChildShape( hknpShapeKey key, hknpShapeCollector* collector ) const HK_OVERRIDE;

		//
		// hknpShape implementation
		//

		virtual hknpShapeType::Enum getType() const HK_OVERRIDE { return hknpShapeType::DYNAMIC_COMPOUND; }

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
			hkArray<hknpShapeKey>* hits, hknpQueryAabbNmp* nmpInOut ) const HK_OVERRIDE;

		virtual void queryAabbImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hknpCollisionQueryCollector* collector, hknpQueryAabbNmp* nmpInOut ) const HK_OVERRIDE;

		/// Internal. Cast a shape against a hknpDynamicCompoundShape.
		static void castShapeImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpShapeCastQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector );

		/// Internal. Calculate the set of closest points between a hknpDynamicCompoundShape and a hknpConvexShape.
		static void HK_CALL getClosestPointsImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpDynamicCompoundShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector );

	protected:

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpDynamicCompoundShape, hknpCompoundShape );

	protected:

		/// Internal bounding volume tree representation.
		HK_ALIGN16( hkUint8 m_tree[TREE_SIZE] );	//+overridetype(class hknpDynamicCompoundShapeTree)

		friend struct hknpDynamicCompoundShapeInternals;
};


#endif // HKNP_DYNAMIC_COMPOUND_SHAPE_H

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
