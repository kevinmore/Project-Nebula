/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_EXTERN_MESH_SHAPE_UTIL_H
#define HKNP_EXTERN_MESH_SHAPE_UTIL_H

class hknpExternMeshShape;

/// Helper class that holds utility methods specific to hknpExternMeshShape.
class hknpExternMeshShapeUtil
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpExternMeshShapeUtil );

		/// Calculate the set of closest points between two hknpExternMeshShapes.
		/*static void HK_CALL getClosestPointsToCompressedMesh(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpExternMeshShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector );*/

		/// Calculate the set of closest points between an hknpExternMeshShape and an hknpConvexShape.
		static void HK_CALL getClosestPointsToConvex(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpExternMeshShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector );

		/// Calculate the set of closest points between an hknpExternMeshShape and an hknpHeightFieldShape.
		/*static void HK_CALL getClosestPointsToHeightfield(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpExternMeshShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector );*/
};


#endif // HKNP_EXTERN_MESH_SHAPE_UTIL_H

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
