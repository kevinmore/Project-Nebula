/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_GSK_MANIFOLD_UTIL_H
#define HK_COLLIDE2_GSK_MANIFOLD_UTIL_H

#include <Physics2012/Internal/Collide/Gjk/GskManifold/hkpGskManifold.h>

struct hkpGskManifold;
class hkpCdBody;
struct hkpProcessCollisionInput;
struct hkpExtendedGskOut;
class hkpGskCache;
class hkpContactMgr;
#include <Geometry/Internal/Types/hkcdVertex.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Physics2012/Collide/Shape/Convex/hkpConvexShape.h>
#include <Physics2012/Internal/Collide/Gjk/hkpGsk.h>

struct hkpGskManifoldWork
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MANIFOLD, hkpGskManifoldWork );

	hkcdVertex m_vertices[16];
	hkVector4 m_masterNormal;
	hkPadSpu<hkReal>    m_radiusA;
	hkPadSpu<hkReal>    m_radiusB;
	hkPadSpu<hkReal>    m_keepContact;
	hkPadSpu<hkReal>    m_radiusSumSqrd;

	inline const hkVector4& getVertexA( hkUint32 vertexIdMult16 ) const
	{
		return *hkAddByteOffsetConst( &m_vertices[0], vertexIdMult16 * (sizeof(hkVector4)>>4) );
	}

	inline const hkVector4& getVertexB( hkUint32 vertexIdMult16 ) const
	{
		return *hkAddByteOffsetConst( &m_vertices[0], vertexIdMult16 * (sizeof(hkVector4)>>4));
	}
};

enum hkpGskManifoldAddStatus
{
	HK_GSK_MANIFOLD_POINT_REPLACED0,	// means it replace point 0 in the result array
	HK_GSK_MANIFOLD_POINT_REPLACED1,
	HK_GSK_MANIFOLD_POINT_REPLACED2,
	HK_GSK_MANIFOLD_POINT_REPLACED3,
	HK_GSK_MANIFOLD_POINT_ADDED,
	HK_GSK_MANIFOLD_POINT_REJECTED,
	HK_GSK_MANIFOLD_TWO_POINT2_REJECTED,
};

enum hkpGskManifoldUtilMgrHandling
{
	HK_GSK_MANIFOLD_CREATE_ID_ALWAYS,
	HK_GSK_MANIFOLD_NO_ID_FOR_POTENTIALS
};

enum hkpGskManifoldPointExistsFlags
{
	HK_GSK_MANIFOLD_POINT_NOT_IN_MANIFOLD = 0,
	HK_GSK_MANIFOLD_POINT_IN_MANIFOLD = 1, // must be 1
	HK_GSK_MANIFOLD_FEATURE_WITHIN_KEEP_DISTANCE_REMOVED = 2,
};

extern "C"
{
		/// Searches the manifold for a given point. Returns false (=0) if the point can't be found;
		/// returns true otherwise and moves the found point to the first element in the manifold.
		/// (This is useful, as it allows for skipping the first point in hkGskManifold_verifyAndGetPoints,
		/// because you have the worldspace information already available.)
	hkpGskManifoldPointExistsFlags HK_CALL hkGskManifold_doesPointExistAndResort(  hkpGskManifold& manifold, const hkpGskCache& newPoint );

		/// Initializes the work variables
	HK_FORCE_INLINE void HK_CALL hkGskManifold_init( const hkpGskManifold& manifold, const hkVector4& masterNormal, const hkpCdBody& cA, const hkpCdBody& cB, hkReal keepContactMaxDist, hkpGskManifoldWork& work );

		/// Verify the points from firstPointIndex to manifol.m_numContactPoints.
		/// If (valid) put world space point into the output array.
		/// Else free contact point id and remove point.
		/// The return value is true when a feature was removed, that was within the contact point keep distance.
		/// This is used by GSK agents to explicitly allow new points into the manifold in the current frame.
	int HK_CALL hkGskManifold_verifyAndGetPoints( hkpGskManifold& manifold, const hkpGskManifoldWork& work, int firstPointIndex, hkpProcessCollisionOutput& out, hkpContactMgr* contactMgr );


		/// Tries to add a point to the manifold.
		/// Allocates and frees contact point ids as necessary.
		/// The new point will always by point 0 in the contactPoint Array in the gskManifold
	hkpGskManifoldAddStatus HK_CALL hkGskManifold_addPoint(  hkpGskManifold& manifold, const hkpCdBody& bodyA,		const hkpCdBody& bodyB, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& output, 
		                                                    const hkpGskCache& newPoint, hkpProcessCdPoint* newCdPointInResultArray, hkpProcessCdPoint* resultPointArray,
															hkpContactMgr* mgr, hkpGskManifoldUtilMgrHandling mgrHandling );
		
		/// Removes a point from the manifold
	void HK_CALL hkGskManifold_removePoint( hkpGskManifold& manifold, int index );
	
		/// Remove all contact points and reset the manifold
	void HK_CALL hkGskManifold_cleanup( hkpGskManifold& manifold, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner );

}


HK_FORCE_INLINE void HK_CALL hkGskManifold_init( const hkpGskManifold& manifold, const hkVector4& masterNormal, const hkpCdBody& cA, const hkpCdBody& cB, hkReal keepContactMaxDist, hkpGskManifoldWork& work )
{
	work.m_keepContact = keepContactMaxDist;
	const hkpConvexShape* shapeA = static_cast<const hkpConvexShape*>(cA.getShape());
	const hkpConvexShape* shapeB = static_cast<const hkpConvexShape*>(cB.getShape());
	work.m_radiusA = shapeA->getRadius();
	work.m_radiusB = shapeB->getRadius();
	const hkReal maxDist = work.m_keepContact + work.m_radiusA + work.m_radiusB;
	work.m_radiusSumSqrd = maxDist * maxDist;
	work.m_masterNormal = masterNormal;

	if ( ! manifold.m_numContactPoints )
	{
		return;
	}

	const hkpGskManifold::VertexId* vertexIds = manifold.getVertexIds();

	shapeA->convertVertexIdsToVertices( vertexIds, manifold.m_numVertsA, &work.m_vertices[0] );
	hkVector4Util::transformPoints( cA.getTransform(), &work.m_vertices[0], manifold.m_numVertsA, &work.m_vertices[0] );

	int b = manifold.m_numVertsA;
	hkcdVertex* verts = &work.m_vertices[b];
	shapeB->convertVertexIdsToVertices( vertexIds+ b, manifold.m_numVertsB, verts );
	hkVector4Util::transformPoints( cB.getTransform(), verts, manifold.m_numVertsB, verts );
}


#endif //HK_COLLIDE2_GSK_MANIFOLD_UTIL_H

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
