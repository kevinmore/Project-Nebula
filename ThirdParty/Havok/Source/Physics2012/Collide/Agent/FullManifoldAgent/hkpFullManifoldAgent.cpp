/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Physics2012/Collide/Shape/Query/hkpRayShapeCollectionFilter.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Query/hkpRayHitCollector.h>
#include <Physics2012/Collide/Shape/HeightField/hkpSphereRepShape.h>
#include <Physics2012/Collide/Shape/HeightField/hkpHeightFieldShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesConnectivity.h>
#include <Physics2012/Collide/Agent/FullManifoldAgent/hkpFullManifoldAgent.h>



// --------------------------------------------------------------------------------------
// 
//  Polytope collision using SAT. The basic algorithm goes like this:
// 
//  - Find face normal of max separation on A - return if separating axis is found
//  - Find face normal of max separation on B - return if separation axis is found
//  - Choose reference face as min(minA, minB)
//  - Find incident face
//  - Clip incident face polygon against side planes of reference face
// 
//  For the general algorithm see the ODE box collision code and Erin Catto's
//  GDC presentations. Christer Ericson has an nice presentation about the 
//  mathematical foundation of SAT.
// 
// 
//  So far we ignore edge contacts since they have a hight performance hit. In 
//  a final version the algorithm should go like this if the narrow phase becomes
//  a bottleneck (using maybe the DCEL or QuadEdge data structures):
// 
//  - Build difference vector between shape positions dp = p2 - p1
//  - Find supporting face on A in direction of dp
//  - Search neighbors for a (maybe local) minimum face separation
//  - Test face normal and all edges attached to vertices defining the face
//  - Find support face on B in direction of -dp
//  - Search neighbors as above
//  - Test possible separating axes above
//  - Create either edge contact or face contact through clipping
// 
//  A note on a support face. As a support vertex a support face is the face 
//  whose distance of the centroid in a specific direction is extremal. The
//  same concept would work for edges as well.
// 
//  With this approach it is theoretically possible to get false positives. I 
//  suggest to assert that the polytopes are intersecting using a boolean query 
//  as post-condition. If you run into this assert you move it to the beginning
//  of the function and exit early. If this problem doesn't occur in praxis
//  you can save the additional intersection text.
// 
//  The above algorithm is a so called local search starting from an initial guess
//  (here the support points in direction of the difference vector dp = p2 - p1)
//  Alternatively we can try to cull away possible separating axis. One idea is
//  to clip the polytopes and only perform the tests on the partial sets. E.g. 
//  we could clip the first polytope against P1: dp/|dp| * x - dp/|dp| * S2( dp )
//  and the second polytope against P2 : -dp/|dp| * x + dp/|dp| * S1( -dp )
// 
//  For an example see P. Tierdeman's blog at www.codercorner.com
//
// --------------------------------------------------------------------------------------



//
// SAT utility functions
//
union hkpFeature
{
	hkUint32 m_id;
	
	struct Bits
	{
		hkUint8 m_refFace;
		hkUint8 m_incFace;
		hkUint8 m_incVertex;
		hkUint8 m_flags;
	} m_bits;	
};


struct hkpMaxFaceSeparationQueryResult
{
	hkReal m_separation;
	int m_planeIndex;
};


struct hkpClipPoint
{
	HK_DECLARE_POD_TYPE();
	hkVector4 m_position;
	int m_feature;
};


static inline
void hkpQueryMaxFaceSeparation( hkpMaxFaceSeparationQueryResult& out, const hkTransform& transformA, const hkpConvexVerticesShape* convexA, const hkTransform& transformB, const hkpConvexVerticesShape* convexB )
{
	hkTransform a2b, b2a;
	a2b.setMulInverseMul( transformB, transformA );
	b2a.setMulInverseMul( transformA, transformB );

	hkSimdReal d_max = hkSimdReal_MinusMax;
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkIntVector i_max; i_max.setZero();
	hkIntVector i_counter; i_counter.setZero();
#else
	int i_max = 0;
#endif

	const hkArray< hkVector4 >& planes = convexA->getPlaneEquations();
	for ( int i = 0; i < planes.getSize(); ++i )
	{	
		// Get plane in local space of A
		hkVector4 P = planes[ i ];

		// Transform plane normal into local space of B
		hkVector4 v;
		v._setRotatedDir( a2b.getRotation(), P );
		v.setNeg<3>( v );
		
		// Find local supporting vertex on surface of B
		hkcdVertex S;
		convexB->getSupportingVertex( v, S );

		// Transform supporting vertex into local space of A
		hkVector4 p;
		p._setTransformedPos( b2a, S );

		// Compute distance of support point in local space of A
		const hkSimdReal d = P.dot4xyz1( p );
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
		i_max.setSelect(d.greater(d_max), i_counter, i_max);
		i_counter.setAddS32(i_counter, hkIntVector::getConstant<HK_QUADINT_1>());
#else
		if ( d.isGreater(d_max) )
		{
			// DIRK_TODO: We could actually break here if we find a separating normal
			i_max = i;
		}
#endif
		d_max.setMax(d_max, d);
	}

	d_max.store<1>(&out.m_separation);
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	i_max.store<1, HK_IO_NATIVE_ALIGNED>((hkUint32*)out.m_planeIndex);
#else
	out.m_planeIndex = i_max;
#endif
}


static inline 
void hkpBuildClipPlanes( hkArray< hkVector4 >& out, const hkTransform& refT, const hkpConvexVerticesShape* refConvex, int refFace )
{
	const hkArray< hkVector4 >& refPlanes = refConvex->getPlaneEquations();
	hkVector4 refPlane = refPlanes[ refFace ];

	hkVector4 n;
	n._setRotatedDir( refT.getRotation(), refPlane );

	const hkpConvexVerticesConnectivity* refConnectivity = refConvex->getConnectivity();
	HK_ASSERT(0x5f83254f, refConnectivity->isClosed() );

	int faceStart = 0;
	for ( int i = 0; i < refFace; ++i )
	{
		int numFaceIndices = refConnectivity->m_numVerticesPerFace[ i ];
		faceStart += numFaceIndices;
	}
	

	hkArray< hkVector4 > refVertices;
	refConvex->getOriginalVertices( refVertices );

	int numFaceIndices = refConnectivity->m_numVerticesPerFace[ refFace ];
	out.reserve( numFaceIndices );

	for ( int i = 0; i < numFaceIndices; ++i )
	{
		int idx0 = refConnectivity->m_vertexIndices[ faceStart + i ];
		int idx1 = i == numFaceIndices - 1 ? refConnectivity->m_vertexIndices[ faceStart ] : refConnectivity->m_vertexIndices[ faceStart + i + 1 ];

		hkVector4 v0, v1;
		v0._setTransformedPos( refT, refVertices[ idx0 ] );
		v1._setTransformedPos( refT, refVertices[ idx1 ] );

		hkVector4 e;
		e.setSub( v1, v0 );

		hkVector4 P;
		P.setCross( e, n );
		P.normalize<3>();

		P.setW( -P.dot<3>( v0 ) );
		
		out.pushBackUnchecked( P );


		// *** DEBUG ***
		hkVector4 center;
		center.setAdd( v0, v1 );
		center.mul( hkSimdReal_Half );

		HK_DISPLAY_LINE( v0, v1, hkColor::RED );
		HK_DISPLAY_ARROW( center, P, hkColor::YELLOW );
	}
}


static inline
void hkpBuildClipFace( hkArray< hkpClipPoint >& out, const hkTransform& refT, const hkpConvexVerticesShape* refConvex, int refFace, const hkTransform& incT, const hkpConvexVerticesShape* incConvex )
{
	// Find the incident face
	const hkArray< hkVector4 >& refPlanes = refConvex->getPlaneEquations();
	const hkArray< hkVector4 >& incPlanes = incConvex->getPlaneEquations();

	hkTransform ref2inc;
	ref2inc.setMulInverseMul( incT, refT );

	hkVector4 refNormal;
	refNormal._setRotatedDir( ref2inc.getRotation(), refPlanes[ refFace ] );

	
	hkSimdReal d_min = hkSimdReal_Max;
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkIntVector i_min; i_min.setZero();
	hkIntVector i_counter; i_counter.setZero();
#else
	int i_min = 0;
#endif

	for ( int i = 0; i < incPlanes.getSize(); ++i )
	{
		hkVector4 n = incPlanes[ i ];

		const hkSimdReal d = refNormal.dot<3>( n );
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
		i_min.setSelect(d.less(d_min), i_counter, i_min);
		i_counter.setAddS32(i_counter, hkIntVector::getConstant<HK_QUADINT_1>());
#else
		if ( d.isLess(d_min) )
		{
			i_min = i;
		}
#endif
		d_min.setMin(d_min, d);
	}

#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	int minFace;
	i_min.store<1, HK_IO_NATIVE_ALIGNED>((hkUint32*)&minFace);
#else
	int minFace = i_min;
#endif

	// Build clip polygon
	const hkpConvexVerticesConnectivity* incConnectivity = incConvex->getConnectivity();
	HK_ASSERT(0x253af8cf, incConnectivity->isClosed() );

	int faceStart = 0;
	for ( int i = 0; i < minFace; ++i )
	{
		int numFaceIndices = incConnectivity->m_numVerticesPerFace[ i ];
		faceStart += numFaceIndices;
	}

	hkArray< hkVector4 > incVertices;
	incConvex->getOriginalVertices( incVertices );

	int numFaceIndices = incConnectivity->m_numVerticesPerFace[ minFace ];
	out.reserve( numFaceIndices );

	for ( int i = 0; i < numFaceIndices; ++i )
	{
		int idx = incConnectivity->m_vertexIndices[ faceStart + i ];

		hkpFeature feature;
		
		feature.m_bits.m_refFace   = (hkUint8)refFace;
		feature.m_bits.m_incFace   = (hkUint8)minFace;
		feature.m_bits.m_incVertex = (hkUint8)idx;
		feature.m_bits.m_flags     = 0;

		hkpClipPoint clipPoint;
		clipPoint.m_position._setTransformedPos( incT, incVertices[ idx ] );
		clipPoint.m_feature = feature.m_id;

		out.pushBackUnchecked( clipPoint );

		// *** DEBUG ***
		HK_DISPLAY_POINT( clipPoint.m_position, hkColor::RED );
	}
}


static inline
void hkdClipPolygon( hkArray< hkpClipPoint >& out, const hkArray< hkpClipPoint >& v, hkVector4Parameter P )
{
	HK_ASSERT(0x1def0b77, v.getSize() > 2 );

	out.clear();
	
	hkpClipPoint v1 = v.back(); 
	hkSimdReal d1 = P.dot4xyz1( v1.m_position );

	for ( int i = 0; i < v.getSize(); ++i )
	{
		hkpClipPoint v2 = v[ i ];
		hkSimdReal d2 = P.dot4xyz1( v2.m_position ); 

		hkBool32 d1LE0 = d1.isLessEqualZero();
		hkBool32 d2LE0 = d2.isLessEqualZero();

		if ( d1LE0 && d2LE0 )
		{
			// Both points are behind plane - keep point B
			out.pushBack( v2 );
		}
		else if ( d1LE0 && !d2LE0 )
		{
			// First point behind and second in front of plane -> intersect
			const hkSimdReal t = d1 / ( d1 - d2 );

			hkVector4 dv;
			dv.setSub( v2.m_position, v1.m_position );

			hkVector4 x;
			x.setAddMul( v1.m_position, dv, t );

			// Keep intersection point
			hkpClipPoint newV;
			newV.m_position = x;
			newV.m_feature = v2.m_feature;

			out.pushBack( newV );
		}
		else if ( d2LE0 && !d1LE0 )
		{
			// Second point behind and first in front of plane -> intersect
			const hkSimdReal t = d1 / ( d1 - d2 );

			hkVector4 dv;
			dv.setSub( v2.m_position, v1.m_position );

			hkVector4 x;
			x.setAddMul( v1.m_position, dv, t );

			// Keep intersection point...
			hkpClipPoint newV;
			newV.m_position = x;
			newV.m_feature = v1.m_feature;

			out.pushBack( newV );

			// ...and also keep point B
			out.pushBack( v2 );
		}


		// Keep v2 as starting point for next edge
		v1 = v2;
		d1 = d2;
	}
}


static inline
void hkdCullPoints( hkArray< hkpClipPoint >& out, const hkArray< hkpClipPoint >& contactPoints, hkVector4Parameter normal )
{
	if ( contactPoints.getSize() < 4 )
	{
		out = contactPoints;
		return;
	}

	// Find the manifold center
	hkVector4 center;
	center.setZero();

	for ( int i = 0; i < contactPoints.getSize(); ++i )
	{
		const hkpClipPoint& cp = contactPoints[ i ];
		center.add( cp.m_position );
	}

	center.mul( hkSimdReal::fromFloat(1.0f / contactPoints.getSize()) );

	
	// Find minimum and maximum projection onto arbitrary tangent vector
	hkVector4 tangent;
	hkVector4Util::calculatePerpendicularVector( normal, tangent );

	hkSimdReal d1_min = hkSimdReal::getConstant<HK_QUADREAL_MAX>();
	int i1_min = 0;
	hkSimdReal d1_max = -d1_min;
	int i1_max = 0;

	for ( int i = 0; i < contactPoints.getSize(); ++i )
	{
		const hkpClipPoint& cp = contactPoints[ i ];

		hkVector4 v;
		v.setSub( cp.m_position, center );

		const hkSimdReal d = v.dot<3>( tangent );

		if ( d < d1_min )
		{
			d1_min = d;
			i1_min = i;

			continue;
		}

		if ( d > d1_max )
		{
			d1_max = d;
			i1_max = i;

			continue;
		}
	}

	HK_ASSERT(0x3fa7bae3, i1_min != i1_max );

	hkVector4 vMin = contactPoints[ i1_min ].m_position;
	hkVector4 vMax = contactPoints[ i1_max ].m_position;


	// Find minimum and maximum projection onto perpendicular difference vector
	hkVector4 dv;
	dv.setSub( vMax, vMin );
	dv.normalize<3>();

	hkVector4 binormal;
	binormal.setCross( normal, dv );

	hkSimdReal d2_min = hkSimdReal::getConstant<HK_QUADREAL_MAX>();
	int i2_min = 0;
	hkSimdReal d2_max = -d2_min;
	int i2_max = 0;

	for ( int i = 0; i < contactPoints.getSize(); ++i )
	{
		if ( i == i1_min || i == i1_max )
		{
			continue;
		}

		const hkpClipPoint& cp = contactPoints[ i ];

		hkVector4 v;
		v.setSub( cp.m_position, center );

		const hkSimdReal d = v.dot<3>( binormal );

		if ( d < d2_min )
		{
			d2_min = d;
			i2_min = i;
		}

		if ( d > d2_max )
		{
			d2_max = d;
			i2_max = i;
		}
	}

	HK_ASSERT(0x368264f9, i2_min != i2_max );


	// Save result
	out.setSize( 4 );
	out[ 0 ] = contactPoints[ i1_max ];
	out[ 1 ] = contactPoints[ i2_max ];
	out[ 2 ] = contactPoints[ i1_min ];
	out[ 3 ] = contactPoints[ i2_min ];

}


hkpFullManifoldAgent::hkpFullManifoldAgent(const hkpCdBody& A, const hkpCdBody& B, const hkpCollisionInput& input, hkpContactMgr* mgr)
: hkpCollisionAgent( mgr )
{
}


void HK_CALL hkpFullManifoldAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createFullManifoldAgent;
		af.m_getPenetrationsFunc = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = false;
	    dispatcher->registerCollisionAgent(af, hkcdShapeType::CONVEX_VERTICES, hkcdShapeType::CONVEX_VERTICES);	
	}
}


hkpCollisionAgent* hkpFullManifoldAgent::createFullManifoldAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, 
															  const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpFullManifoldAgent* agent = new hkpFullManifoldAgent(bodyA, bodyB, input, mgr);
	return agent;
}



void hkpFullManifoldAgent::destroyContactPoints( hkCollisionConstraintOwner& constraintOwner )
{
	if ( m_contactMgr )
	{
		// Remove any unneeded contact points
		for (int i=0; i< m_contactPointIds.getSize(); i++)
		{
			// Remove from contact manager
			m_contactMgr->removeContactPoint(m_contactPointIds[i].m_contactPointId, constraintOwner );
		}
	}
}

void hkpFullManifoldAgent::cleanup( hkCollisionConstraintOwner& constraintOwner )
{
	destroyContactPoints( constraintOwner );
	delete this;
}


void hkpFullManifoldAgent::processCollision( const hkpCdBody& csBodyA, const hkpCdBody& csBodyB, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& processOutput )
{
	HK_ASSERT2(0x7371164d,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );
	HK_TIME_CODE_BLOCK( "Convex SAT", HK_NULL );

	const hkTransform& transformA = csBodyA.getTransform();
	const hkpConvexVerticesShape* convexA = static_cast< const hkpConvexVerticesShape* >( csBodyA.getShape() );
	HK_ASSERT(0x714a1d3b, convexA->getConnectivity() != HK_NULL );

	const hkTransform& transformB = csBodyB.getTransform();
	const hkpConvexVerticesShape* convexB = static_cast< const hkpConvexVerticesShape* >( csBodyB.getShape() );
	HK_ASSERT(0x69daa8b0, convexB->getConnectivity() != HK_NULL );

 
	//
	// Test face normals
	//
	hkpMaxFaceSeparationQueryResult resultA;
	hkpQueryMaxFaceSeparation( resultA, transformA, convexA, transformB, convexB );
	if ( resultA.m_separation > 0.0f )
	{
		// We found a separating axis
		destroyContactPoints( *processOutput.m_constraintOwner.val() );
		return;
	}

	hkpMaxFaceSeparationQueryResult resultB;
	hkpQueryMaxFaceSeparation( resultB, transformB, convexB, transformA, convexA );

	if ( resultB.m_separation > 0.0f )
	{
		// We found a separating axis
		destroyContactPoints( *processOutput.m_constraintOwner.val() );
		return;
	}


	//
	// Identify the reference and incident shape
	//
	hkpMaxFaceSeparationQueryResult referenceQueryResult;
	
	hkTransform referenceTransform;
	const hkpConvexVerticesShape* referenceConvex;
	
	hkTransform incidentTransform;
	const hkpConvexVerticesShape* incidentConvex;

	// Use weighting to avoid face jitter
	const hkReal kRelativeTol = 0.98f;
	const hkReal kAbsoluteTol = 0.001f;

	if ( resultB.m_separation > kRelativeTol * resultA.m_separation + kAbsoluteTol )
	{
		referenceQueryResult = resultB;

		referenceTransform = transformB;
		referenceConvex = convexB;

		incidentTransform = transformA;
		incidentConvex = convexA;
	}
	else
	{
		referenceQueryResult = resultA; 

		referenceTransform = transformA;
		referenceConvex = convexA;

		incidentTransform = transformB;
		incidentConvex = convexB;
	}


	//
	// Build clip planes on reference convex
	//
	hkArray< hkVector4 > clipPlanes;
	hkpBuildClipPlanes( clipPlanes, referenceTransform, referenceConvex, referenceQueryResult.m_planeIndex );

	//
	// Build clip polygon on incident convex
	//
	hkArray< hkpClipPoint > clipFace1, clipFace2;
	hkpBuildClipFace( clipFace1, referenceTransform, referenceConvex, referenceQueryResult.m_planeIndex, incidentTransform, incidentConvex );

	// 
	// Clip incident face against side planes of reference face
	//
	hkArray< hkpClipPoint >* pIn  = &clipFace1;
	hkArray< hkpClipPoint >* pOut = &clipFace2;

	for ( int i = 0; i < clipPlanes.getSize(); ++i )
	{
		hkdClipPolygon( *pOut, *pIn, clipPlanes[ i ] );

		if ( pOut->getSize() < 3 )
		{
			// We clipped away all contact points and can exit
			destroyContactPoints( *processOutput.m_constraintOwner.val() );
			return;
		}

		// Swap arrays
		hkArray< hkpClipPoint >* pTemp = pIn;
		pIn = pOut;
		pOut = pTemp;
	}


	//
	// Finally keep points below reference face and do point management
	//
	const hkArray< hkVector4 >& refPlanes = referenceConvex->getPlaneEquations();

	hkVector4 refPlane;
	refPlane = refPlanes[ referenceQueryResult.m_planeIndex ];
	hkVector4Util::transformPlaneEquation( referenceTransform, refPlane, refPlane );
	
	const hkArray< hkpClipPoint >& clipFace = *pIn;
	
	hkArray<hkUchar> usedPoints( m_contactPointIds.getSize() );
	hkString::memSet( usedPoints.begin(), 0, usedPoints.getSize() );
	
	for ( int i = 0; i < clipFace.getSize(); ++i )
	{
		const hkSimdReal separation = refPlane.dot4xyz1( clipFace[ i ].m_position );

		if ( separation.isLessEqualZero() )
		{
			int feature = clipFace[ i ].m_feature;

			hkVector4 cp;
			cp.setAddMul( clipFace[ i ].m_position, refPlane, -separation );

			hkContactPointId contactPointId = HK_INVALID_CONTACT_POINT;
			for ( int k = 0; k < usedPoints.getSize(); k++ )	// do not test the brand new points
			{
				 if ( m_contactPointIds[k].m_feature == feature )
				 {
					 contactPointId = m_contactPointIds[k].m_contactPointId;
					 usedPoints[ k ] = 1;
				 }
			}


			// Add point to manifold
			hkpProcessCdPoint& point = *processOutput.reserveContactPoints( 1 );
			point.m_contact.setPositionNormalAndDistance( cp, refPlane, separation );

			// If the normal comes from body A we need to flip the contact
			if ( referenceConvex == convexA )
			{
				point.m_contact.flip();
			}

			// If this point does not already exist
			if( contactPointId != HK_INVALID_CONTACT_POINT )
			{
				processOutput.commitContactPoints( 1 );
			}
			else
			{
				// Add it to the contact manager
				contactPointId = m_contactMgr->addContactPoint( csBodyA, csBodyB, input, processOutput, HK_NULL, point.m_contact );
			}
			
			if( contactPointId == HK_INVALID_CONTACT_POINT )
			{
				 processOutput.abortContactPoints( 1 );
			}
			else
			{
				 ContactPoint& new_cp = m_contactPointIds.expandOne();
				 new_cp.m_contactPointId = contactPointId;
				 new_cp.m_feature = feature;

				 processOutput.commitContactPoints( 1 );
			}

		 // Update ID
		 point.m_contactPointId = contactPointId;
		 }
	}


	// Remove old contacts
	for ( int l = usedPoints.getSize() - 1 ; l >= 0; l-- )
	{
		if ( usedPoints[l] == 0 )
		{
			m_contactMgr->removeContactPoint( m_contactPointIds[l].m_contactPointId, *processOutput.m_constraintOwner.val() );
			m_contactPointIds.removeAt( l );
		}
	}
}


			// hkpCollisionAgent interface implementation.
void hkpFullManifoldAgent::getPenetrations(const hkpCdBody& csBody, const hkpCdBody& hfBody, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	staticGetPenetrations( csBody, hfBody, input, collector );
}

			// hkpCollisionAgent interface implementation.
void HK_CALL hkpFullManifoldAgent::staticGetPenetrations(const hkpCdBody& csBody, const hkpCdBody& hfBody, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
}

			// hkpCollisionAgent interface implementation.
void hkpFullManifoldAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	staticGetClosestPoints( bodyA, bodyB, input, collector );
}
			
			// hkpCollisionAgent interface implementation.
void HK_CALL hkpFullManifoldAgent::staticGetClosestPoints( const hkpCdBody& csBody, const hkpCdBody& hfBody, const hkpCollisionInput& input, class hkpCdPointCollector& collector )
{
	HK_TIMER_BEGIN_LIST( "Convex SAT", "closest points" );


	HK_TIMER_END_LIST();

}

class hkHeightFieldRayForwardingCollector : public hkpRayHitCollector
{
	public:
		virtual void addRayHit( const hkpCdBody& cdBody, const hkpShapeRayCastCollectorOutput& hitInfo )
		{
			hkContactPoint contact;
			hkVector4 cpPos; cpPos.setAddMul( m_currentFrom, m_path, hkSimdReal::fromFloat(hitInfo.m_hitFraction) );
			cpPos.addMul( hkSimdReal::fromFloat(-m_currentRadius), contact.getNormal());
			contact.setPosition(cpPos);
			hkVector4 cpN; cpN._setRotatedDir( cdBody.getTransform().getRotation(), hitInfo.m_normal );
			contact.setSeparatingNormal(cpN);
			contact.setDistance( hitInfo.m_hitFraction );

			hkpCdPoint point( m_csBody, cdBody, contact );
			m_collector.addCdPoint( point );
			m_earlyOutHitFraction = hkMath::min2( m_collector.getEarlyOutDistance(), m_earlyOutHitFraction);
		}

		hkHeightFieldRayForwardingCollector( const hkpCdBody& csBody, const hkVector4& path, hkpCdPointCollector& collector )
			: m_path(path), m_csBody( csBody), m_collector( collector )
		{

		}

		hkVector4 m_currentFrom;
		hkReal    m_currentRadius;
		hkVector4 m_path;

		const hkpCdBody&  m_csBody;
		hkpCdPointCollector& m_collector;

};


void hkpFullManifoldAgent::staticLinearCast( const hkpCdBody& csBody, const hkpCdBody& hfBody, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	HK_TIMER_BEGIN_LIST( "hkpFullManifoldAgent", "ClosestPoints" );

	HK_TIMER_END_LIST();
}

void hkpFullManifoldAgent::linearCast( const hkpCdBody& csBody, const hkpCdBody& hfBody, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	staticLinearCast( csBody, hfBody, input, collector, startCollector );
}

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
