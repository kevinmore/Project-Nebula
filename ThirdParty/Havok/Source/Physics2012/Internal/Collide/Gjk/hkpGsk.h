/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_GSK
#define HK_COLLIDE2_GSK

#include <Physics2012/Collide/Util/hkpCollideTriangleUtil.h>
#include <Common/Base/Algorithm/Collide/LineSegment/hkLineSegmentUtil.h>

#include <Physics2012/Internal/Collide/Gjk/hkpGskCache.h>
#include <Physics2012/Collide/Shape/Convex/hkpConvexShape.h>
#include <Geometry/Internal/Types/hkcdVertex.h>
class hkpConvexShape;
class hkpTriangleShape;
class hkpGskCache;

// #define HK_DEBUG_GSK

enum hkpGskStatus
{
	// General success indication
	HK_GSK_OK = 0,

	// Penetration depth algorithm errors
	HK_GSK_PD_HEURISTIC_SAMPLING, // The output is an approximation at best
	HK_GSK_PD_OUT_OF_MEMORY,	  // The output is an approximation at best
	HK_GSK_PD_UNSOLVABLE,		  // The output is not valid, due to numerical limitations

	HK_GSK_PENETRATING
};


const int HK_GSK_MAX_VERTICES = 4;


/// Low level GJK and GSK output data
struct hkpGskOut
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CDINFO, hkpGskOut );

	hkVector4 m_normalInA;
	hkVector4 m_pointAinA;
	hkPadSpu<hkReal> m_distance;
};



	/// add secondary values and adjust points by radii
struct hkpExtendedGskOut: public hkpGskOut
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CDINFO, hkpExtendedGskOut );

	hkVector4 m_pointBinB;
	hkVector4 m_normalInWorld;
	hkVector4 m_unweldedNormalInWorld;
};



class hkpGsk
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CDINFO, hkpGsk );

		HK_FORCE_INLINE hkpGsk(): m_doNotHandlePenetration(false){}

			/// Get the closest feature (Note: \p in can be set to \p out).
			/// Additional results you can get from this class:
			///     - \a m_support: The unscaled normal.
			///     - \a m_featureChanged: Set to 1 if the feature actually has changed.
		HK_FORCE_INLINE void init ( const hkpConvexShape* shapeA, const hkpConvexShape* shapeB, const hkpGskCache& cache );

		hkpGskStatus getClosestFeature( const hkpConvexShape* shapeA, const hkpConvexShape* shapeB, const hkTransform& btoa, hkVector4& separatingNormalOut );

		void convertFeatureToClosestDistance( const hkVector4& separatingNormal, struct hkpGskOut& gjkOut );

		struct GetClosesetPointInput
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CDINFO, hkpGsk::GetClosesetPointInput );

			hkPadSpu<const hkTransform*>	m_aTb;
			hkPadSpu<const hkTransform*>	m_transformA;
			hkPadSpu<const hkpConvexShape*>	m_shapeA;
			hkPadSpu<const hkpConvexShape*>	m_shapeB;
			hkPadSpu<hkReal>		 		m_collisionTolerance;

		};
			/// Get the closest point between two objects. Always returns the separatingNormalOut.
			/// Only return pointOnBOut if distance < m_collisionTolerance.
		static hkResult HK_CALL getClosestPoint( const GetClosesetPointInput& input, hkpGskCache& cache, hkVector4& separatingNormalOut, hkVector4& pointOnBOut );

			/// Get a point anywhere on the closest dist in A space
		HK_FORCE_INLINE void getClosestPointAinA( const hkVector4& separatingNormalA, hkVector4& pointInAOut );

		HK_FORCE_INLINE void checkForChangesAndUpdateCache( hkpGskCache& cache ) const;
	
		
		//
		//	Internal section
		//
	public:

		enum
		{
			GSK_NONE_NONE      = (0<<3) + 0,
			GSK_POINT_POINT    = (1<<3) + 1,
			GSK_POINT_EDGE     = (1<<3) + 2,
			GSK_EDGE_POINT     = (2<<3) + 1,
			GSK_POINT_TRIANGLE = (1<<3) + 3,
			GSK_TRIANGLE_POINT = (3<<3) + 1,
			GSK_EDGE_TRIANGLE  = (2<<3) + 3,
			GSK_TRIANGLE_EDGE  = (3<<3) + 2,
			GSK_POINT_TETRA    = (1<<3) + 4,
			GSK_TETRA_POINT    = (4<<3) + 1,
			GSK_EDGE_EDGE      = (2<<3) + 2
		};

		enum NextCase
		{
			NEXT_IS_BREAK,
			NEXT_IS_PENETRATION,
			NEXT_IS_EDGE_EDGE
		};

		enum SupportTypes
		{
			BUILD_NEG_SUPPORT = -hkVector4ComparisonMask::MASK_X,
			BUILD_NO_SUPPORT =  0,
			BUILD_POS_SUPPORT = hkVector4ComparisonMask::MASK_X
		};

		enum SupportState
		{
			SUPPORT_CHANGED     = 0,
			DONT_CALL_GET_SUPPORTING_VERTEX_A = BUILD_POS_SUPPORT,
			DONT_CALL_GET_SUPPORTING_VERTEX_B = BUILD_NEG_SUPPORT
		};

	public:
		hkPadSpu<int> m_dimA;
		hkPadSpu<int> m_dimB;
		hkPadSpu<int> m_maxDimA;
		hkPadSpu<int> m_maxDimB;

		hkPadSpu<hkUint32> m_doNotHandlePenetration;	// actually a bool

		hkPadSpu<int> m_featureChange;

			// warning: the layout of the next variables is fixed, do not change
		hkcdVertex m_verticesA[HK_GSK_MAX_VERTICES];
		hkVector4 m_dummy[HK_GSK_MAX_VERTICES];
		hkVector4 m_verticesBinA[HK_GSK_MAX_VERTICES];
		hkcdVertex m_verticesBinB[HK_GSK_MAX_VERTICES];

			// the next two variables are output variables of getClosestFeature
			// and can be used to quickly calculate the contact point
		hkVector4 m_support;
		hkLineSegmentUtil::ClosestLineSegLineSegResult m_closestLineLineResult;

#if !defined(HK_PLATFORM_SPU)
		static void HK_CALL codeBegin();
		static void HK_CALL codeEnd();
#endif

			///	Print current two simplexes.
		void print();

	protected:
		// temporary data, this holds the dot products of the edge plane equations and
		// the point (if all values are negative, point is inside triangle)
		// m_checkTriangleDots to the unscaled distance of the point projected onto the triangle and the edges
		hkVector4 m_checkTriangleDots;
		hkPadSpu<int> m_lastDimB; // m_dimA + m_dimB == m_lastDimA + m_lastDimB + reduced_dimension ? 1 : 0
		SupportState m_supportState;

		enum ReduceDimensionResult
		{
			REDUCE_DIMENSION_OK,
			REDUCE_DIMENSION_PENETRATION
		};

		/// Checks m_dimA + m_dimB <= 4 for correctness
		/// and allows for repeated calls for m_dimA + m_dimB == 5
		HK_FORCE_INLINE ReduceDimensionResult reduceDimension();

		/// Reduce dimension for all cases not handled by reduceDimimension().
		/// The idea is to try to select a subset of current features, which would give the
		/// same collision normal as the penetration depth call.
		/// This is identical to finding a valid point-face or a valid edge-edge case
		/// that reduces (m_dimA + m_dimB >= 4) to (m_dimA + m_dimB < 4).
		void reduceDimensionExtended();
#ifdef HK_DEBUG_GSK
		void validateGsk(hkReal& currentValidateDistance);
#endif

		void handlePenetration(const hkpConvexShape* shapeA, const hkpConvexShape* shapeB, const hkTransform& aTb);

		inline void setFeatureChange(int f)
		{
			m_featureChange = f;
		}

	public:
		void exitAndExportCacheImpl( hkpGskCache& cache ) const ;

	protected:
		/*
		** functions used by reduceDimension
		*/

		/// Checks if a point is within the boundaries of a triangle.
		///
		/// Input:
		///		a point and a triangle
		///
		/// Output:
		///		a mask, of which each bit is telling whether the point is inside an edge or not
		/// 
		///	Sideeffects:
		///		It sets \a m_checkTriangleDots to the unscaled distance of the point projected onto the triangle and the edges
		///
		///		If checkSupport is set:
		///			- \a m_support is set to the current normal of the triangle.
		///			- If point is on the wrong side of the support, it flips m_support and also
		///			  flips point 0 and 1.
		///		
		int checkTriangleBoundaries( const hkVector4& a, hkVector4* threeVertices, SupportTypes support );

		NextCase processEdgeTriangle( hkVector4* vertexA, hkVector4* vertexB, hkPadSpu<int>& dimA, hkPadSpu<int>& dimB, hkBool comingFromEdgeEdge, SupportTypes supportType );
};


void hkpGsk::init( const hkpConvexShape* shapeA, const hkpConvexShape* shapeB, const hkpGskCache& cache )
{
	HK_ASSERT(0xf049ad2e, unsigned(cache.m_dimA + cache.m_dimB) <= 4 );
	m_dimA = cache.m_dimA;
	m_dimB = cache.m_dimB;
	m_maxDimA = cache.m_maxDimA;
	m_maxDimB = cache.m_maxDimB;

	setFeatureChange( 0 );
	{
		shapeA->convertVertexIdsToVertices( &cache.m_vertices[0],		m_dimA, &m_verticesA[0] );
		shapeB->convertVertexIdsToVertices( &cache.m_vertices[m_dimA],	m_dimB, &m_verticesBinB[0] );
	}
}

void hkpGsk::checkForChangesAndUpdateCache( hkpGskCache& cache ) const
{
	if (!m_featureChange)
	{
		return;
	}
	exitAndExportCacheImpl( cache );
}

void hkpGsk::getClosestPointAinA( const hkVector4& separatingNormal, hkVector4& pointInA )
{
	if ( m_dimA == 1 )
	{
		pointInA = m_verticesA[0];
	}
	else if ( m_dimB == 1 )
	{
		pointInA = this->m_verticesBinA[0];
		pointInA.addMul( separatingNormal.getComponent<3>(), separatingNormal  );
	}
	else
	{
		pointInA = m_closestLineLineResult.m_closestPointA;
	}
}

extern "C"
{
	void HK_CALL hkGskRecalcContactInternal( hkVector4* verticesA, hkVector4* verticesB, int dimA, int dimB, hkVector4Parameter masterNormal, hkVector4& pointA, hkVector4& support );

	HK_FORCE_INLINE void HK_CALL hkGskRecalcContact(  hkpGsk& gsk, hkVector4Parameter masterNormalA, hkVector4& pointBinAOut, hkVector4& supportOut )
	{
		hkGskRecalcContactInternal( gsk.m_verticesA, gsk.m_verticesBinA, gsk.m_dimA, gsk.m_dimB, masterNormalA, pointBinAOut, supportOut );
	}
}

	/// Returns either HK_GSK_OK, HK_GSK_PD_HEURISTIC_SAMPLING or HK_GSK_PD_OUT_OF_MEMORY, never HK_GSK_PD_UNSOLVABLE
hkpGskStatus HK_CALL hkCalculatePenetrationDepth( const hkpConvexShape* shapeA, const hkpConvexShape* shapeB,	const hkTransform& btoa, hkReal epsTol,
												 hkcdVertex* pointsAinA, hkcdVertex* pointsBinB, hkVector4* simplex, int simplexSize, hkpGskOut& output );

hkResult HK_CALL hkCalcMultiPenetrationDepth( const hkTransform& transformA, const hkpConvexShape* shapeA, const hkpConvexShape** shapeBs, int numShapeBs, const hkTransform& aTb, hkContactPoint** pointsOut );


#endif // HK_COLLIDE2_GSK

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
