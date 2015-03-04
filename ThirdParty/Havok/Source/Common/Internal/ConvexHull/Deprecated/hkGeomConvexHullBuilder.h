/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_GENERATE_HULL_H
#define HK_GENERATE_HULL_H

#include <Common/Internal/ConvexHull/Deprecated/hkGeomHull.h>
#include <Common/Internal/ConvexHull/Deprecated/hkGeomConvexHullTolerances.h>
#include <Common/Internal/ConvexHull/hkGeometryUtility.h>
#include <Common/Base/Types/Color/hkColor.h>

class hkTextDisplay;
struct hkGeometry;
class hkAabb;

struct hkpGeomConvexHullConfig
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_GEOMETRY, hkpGeomConvexHullConfig );

	hkReal m_maxAngle;
	hkReal m_lineLengthImportance;
	int m_maxIterations;
};

/// Builds the convex hull of a set of points in 3-space.
///
/// The functionality of this class should be accessed through the hkGeometryUtility interface.
///
/// This algorithm for constructing convex hulls works as follows:
///
/// Sort the set of points by their x-coordinate, divide the two sets,
/// recursively construct the hull of each half, and merge.
///
/// The merge consists of two steps:
///     - wrapping a plane around the two hulls so that the combined shape is a convex hull
///     - removing faces from both hulls that are no longer visible.
class hkGeomConvexHullBuilder
{
	public:
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,hkGeomConvexHullBuilder);
		/// Generates a valid convex hull. This function keeps calling the "buildConvexHull" function with
		/// different tolerances checking the validity of the generated hull by calling the isValidHull test
		/// until a valid hull is generated.
		static void HK_CALL generateConvexHull( const hkVector4* verts, int numVertices,
											  hkGeomHull& hullOut, hkArray<hkVector4>& usedVerticesOut, hkGeomConvexHullMode mode );

		// Creates the convex hull of the input vertices verts.  The hull is stored
		// as a hkGeomHull.  The vertices used are stored in usedVerticesOut.
		static void HK_CALL buildConvexHull( const hkGeomConvexHullTolerances&  tolerances, const hkVector4* verts, int numVertices,
											  hkGeomHull& hullOut, hkArray<hkVector4>& usedVerticesOut );


		// Generate plane equations and geometries from a hkGeomHull.
		static hkBool HK_CALL buildPlaneEquations( const hkGeomConvexHullTolerances&  tolerances, hkGeomHull& hull, 
				const hkArray<hkVector4>& usedVertices, hkArray<hkVector4>& planeEquationsOut );

		// Return the maximum distance from a points set and a convex hull
		static hkReal HK_CALL getMaximumDistance( hkGeomHull& hull, const hkArray<hkVector4>& usedVertices, const hkArray<hkVector4>& set);
		
		static void HK_CALL buildGeometry( hkGeomHull& hull, hkGeometry& geometryOut );


		static void HK_CALL draw( hkGeomHull& hull, hkColor::Argb color, hkTextDisplay* textDisplay );

		static hkReal HK_CALL getAngleBetweenVertexAndPlane( const hkGeomConvexHullTolerances&  tolerances, const hkVector4& vertex, const hkVector4& planeEquation, const hkVector4& tangentStart, const hkVector4& tangentEnd );

		static void HK_CALL weldXsortedVertices( hkReal weldTolerance, hkArray<hkVector4>& verts, int& numVertices );

		static void HK_CALL removeFlaggedVertices( hkArray<hkVector4>& vertices );

		static void HK_CALL getAabb( const hkArray<hkVector4>& verts, hkAabb& aabb );

		static void HK_CALL convertToUnitCube( hkArray<hkVector4>& usedVerticesOut, hkVector4& extents, hkVector4& aabbCenter );
		static void HK_CALL convertFromUnitCube( hkArray<hkVector4>& usedVerticesOut, hkVector4& extents, hkVector4& aabbCenter );

		static hkBool HK_CALL vectorLessAndMergeCoordinates( hkVector4& v1, hkVector4& v2 );
	public:

		// 
		struct WeightedLine
		{
			public:
				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_GEOMETRY, hkGeomConvexHullBuilder::WeightedLine );
			hkBool edgeVerticesEqual( const WeightedLine& other )
			{
//				return ( (m_leftEdge->m_vertex == other.m_leftEdge->m_vertex) &&
//						 (m_rightEdge->m_vertex == other.m_rightEdge->m_vertex) );
				return ( (m_leftEdge == other.m_leftEdge) &&
						 (m_rightEdge == other.m_rightEdge) );
			}
			hkBool edgesAndVertexEqual( const WeightedLine& other )
			{
				return ( (m_leftEdge == other.m_leftEdge) &&
						 (m_rightEdge == other.m_rightEdge) &&
						 m_lastVertex == other.m_lastVertex);
			}
			hkBool vertsAndVertexEqual( const WeightedLine& other )
			{
				return ( (m_leftEdge->m_vertex == other.m_leftEdge->m_vertex) &&
						 (m_rightEdge->m_vertex == other.m_rightEdge->m_vertex) &&
						 m_lastVertex == other.m_lastVertex);
			}

			hkGeomEdge*	m_leftEdge;
			hkGeomEdge*	m_rightEdge;

			WeightedLine* m_source;
			hkUint32      m_lastVertex;

			hkReal m_weight;
		};

		struct WrappingLine
		{
			public:
				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_GEOMETRY, hkGeomConvexHullBuilder::WrappingLine );
			hkGeomEdge*	m_leftEdge;
			hkGeomEdge*	m_rightEdge;
			
			hkUint16 m_leftVertex;
			hkUint16 m_rightVertex;

			hkUint16 m_leftNextEdgeVertex;
			hkUint16 m_rightNextEdgeVertex;
		};


		struct WeightedNeighbour
		{
			public:
				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_GEOMETRY, hkGeomConvexHullBuilder::WeightedNeighbour );
			hkGeomEdge*	m_edge;
			hkReal m_weight;
		};
	
	//protected:
	
		//friend class hkGeomConvexHullTester;

		enum VisitedEdgeInfo
		{
			NOT_VISITED,
			VISITED_VISIBLE,	// edge of a visible triangle
			VISITED_BOUNDARY,	// visible edge of invisible triangles
			VISITED_HIDDEN,		// hidden edge
		};

		

		//  Used during the generation of the plane equations from the convex hull.

		class PlaneAndPoints
		{
			public:
				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_GEOMETRY, hkGeomConvexHullBuilder::PlaneAndPoints );
		public: 
			void sort();

			// returns true iff
			// - the vertices are equal
			// - the edges are different
			// - the mirrors are different

			#define EDGE_OK( E1, E2 )  ( ((E1) != (E2)) && ((E1)->m_vertex == (E2)->m_vertex) ) ? ( (E1)->getMirror( edgeBase )->m_vertex != (E2)->getMirror( edgeBase )->m_vertex ) : true
			#define MIRROR_OK( E1, E2 )  ( ((E1)->getMirror( edgeBase )->m_vertex == (E2)->m_vertex) && ((E2)->getMirror( edgeBase )->m_vertex == (E1)->m_vertex) ) ? ( (E1)->getMirror( edgeBase ) == (E2) ) : true
			
			static hkBool findPair( hkGeomEdge* edgeBase, hkGeomEdge* p11, hkGeomEdge* p21, hkGeomEdge* p12, hkGeomEdge* p22, PlaneAndPoints& p1, PlaneAndPoints& p2 );

			hkVector4 m_planeEquation;
			hkGeomEdge* m_v0;
			hkGeomEdge* m_v1;
			hkGeomEdge* m_v2;
			
			hkUint16    m_info:16;
		};

		static hkBool HK_CALL buildPlaneEquations( const hkGeomConvexHullTolerances&  tolerances, 
			hkGeomHull& hull, const hkArray<hkVector4>& usedVertices, 
			hkVector4& planarPlaneEquationOut, hkBool& isPlanarOut, hkArray<hkVector4>& planeEquationsOut, 
			hkArray<PlaneAndPoints>& tangentPlanesOut );

		static void HK_CALL buildConvexSubHull(const hkGeomConvexHullTolerances&  tolerances, hkArray<hkVector4>& xSortedVerts, int startVertex, int endVertex, hkGeomHull& hullOut);
			static void HK_CALL removeUnusedVertices( hkGeomHull& hull, hkArray<hkVector4>& vertices );

		static hkResult HK_CALL mergeHulls( const hkGeomConvexHullTolerances&  tolerances, hkGeomHull& lhull, hkGeomHull& rhull, hkGeomHull& hullOut);
			static hkBool HK_CALL isSingleLine( hkReal degenerateTolerance, hkGeomHull& lhull, hkGeomHull& rhull, hkGeomHull& hullOut );


		// helper functions
		static void HK_CALL getCommonTangent( hkGeomHull& lhull, hkGeomHull& rhull, WeightedLine& weightedLineOut, hkVector4& tangentPlaneEquationOut);
			static void HK_CALL getPlaneEquationZaxis( const hkVector4& a, const hkVector4& b, hkVector4& planeEquationOut);

		static void HK_CALL findWeightedNeighbours( const hkGeomConvexHullTolerances&  tolerances, hkGeomHull& hull, const hkVector4& tangentPlaneEquation, const hkUint16 lastVertex, const hkGeomEdge* startPoint, const hkVector4* tangentStart, const hkVector4* tangentEnd, hkArray<WeightedNeighbour>& neighboursOut );

			static void HK_CALL removeCoPlanarNeighbours( const hkGeomConvexHullTolerances&  tolerances, hkGeomHull& hull, const hkVector4& tangentPlaneEquation, const hkGeomEdge* startPoint, const hkVector4* tangentStart, const hkVector4* tangentEnd, hkArray<WeightedNeighbour>& neighboursOut );

			static void HK_CALL calculateNewNeighbours( const hkVector4* vertexBase, const hkGeomConvexHullTolerances& tolerances, hkReal lowestWeight,
				const hkVector4& tangentPlaneEquation, const hkGeomEdge* startPoint, const hkVector4* tangentStart, const hkVector4* tangentEnd, hkArray<WeightedNeighbour>& neighboursOut );

			static void HK_CALL validateNeighbours( const hkGeomConvexHullTolerances& tolerances, const hkVector4* vertexBase, const hkVector4& tangentPlaneEquation, const hkUint16 lastVertex, WeightedLine* sourceLine, const hkVector4* tangentStart, const hkVector4* tangentEnd, hkArray<WeightedNeighbour>& leftNeighboursOut, hkArray<WeightedNeighbour>& rightNeighboursOut );
			
		static void HK_CALL createBevelPlane( const hkVector4& planeNormal, const hkVector4& vertex0, const hkVector4& vertex1, const hkVector4& vertex2, hkArray<hkVector4>& planeEquationsOut );

		static void HK_CALL addWrappingLines( const hkpGeomConvexHullConfig& config, WeightedLine* sourceLine, hkArray<WeightedNeighbour>& leftN, hkArray<WeightedNeighbour>& rightN, hkArray<WeightedLine>& wrappingLinesOut );

		static void HK_CALL findWrapping(     hkGeomHull& lhull, hkGeomHull& rhull, const hkArray<WeightedLine>& lastTangents, hkArray<WrappingLine>& wrappingOut );
		static hkBool HK_CALL isValidEdgeCheck( hkGeomHull& hull, hkGeomEdge* edge, int edgeInfo );
		static hkBool HK_CALL isValidWrapping(  hkGeomHull& hull, hkGeomEdge* otherEdgeBase, hkBool usingLeftWrapping, hkArray<WrappingLine>& wrapping );

		static void HK_CALL printEdges( hkGeomHull& hull );

		static void HK_CALL appendAndReindexEdges( hkGeomHull& hull, hkBool usingLeftWrapping, hkGeomHull& hullOut, hkArray<hkUint16>& hullReindexMap );
		static void HK_CALL stitchHulls( hkGeomHull& lhull, hkGeomHull& rhull, hkArray<WrappingLine>& wrapping, hkGeomHull& hullOut );


			/// A pre-filter function that ensures there are no three collinear vertices in the input set.
		static void HK_CALL removeCollinearVertices( hkArray< hkVector4>& vertices, hkReal degenerateTolerance );

			/// 
		static void HK_CALL postFilterVertices( hkGeomHull& hull, int startVertex, int endVertex, 
												 const hkGeomConvexHullTolerances&  tolerances, 
												 hkBool& vertsHaveChanged );

		static void HK_CALL drawPlane( const hkVector4& planeEqn, const hkVector4& centrePoint, hkColor::Argb color );

		static hkSimdReal HK_CALL getPseudoAngle( hkSimdRealParameter cosTheta, hkSimdRealParameter sinTheta );

		static void HK_CALL generateHullFromPlanarPoints(const hkVector4& planeEqn, const hkVector4 *verticesin, int numverts, hkArray<hkVector4>& usedVertices, hkArray<hkVector4>& planeEquationsOut);

// 		static hkBool HK_CALL approximatePlanesFromVerticesWithKDop(const float *verticesin, int numverts, int vertexStriding, hkArray<hkVector4>& usedVertices);
};

#endif //HK_GENERATE_HULL_H

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
