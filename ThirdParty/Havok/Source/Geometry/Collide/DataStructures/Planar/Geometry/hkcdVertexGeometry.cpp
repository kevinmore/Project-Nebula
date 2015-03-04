/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/DataStructures/IntAabb/hkcdIntAabb.h>
#include <Geometry/Collide/DataStructures/Planar/Geometry/hkcdVertexGeometry.h>

#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

//
//	Constructor

hkcdVertexGeometry::hkcdVertexGeometry(PlanesCollection* planesCollection, hkcdPlanarEntityDebugger* debugger)
:	hkcdPlanarEntity(debugger)
,	m_planes(planesCollection)
{}

//
//	Destructor

hkcdVertexGeometry::~hkcdVertexGeometry()
{}

//
//	Constructor

hkcdVertexGeometry::VPolygonCollectionBase::VPolygonCollectionBase()
{
	clear();
}

//
//	Copy constructor

hkcdVertexGeometry::VPolygonCollectionBase::VPolygonCollectionBase(const VPolygonCollectionBase& other)
:	hkcdPlanarGeometryPrimitives::Collection<hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_BIT>()
{
	copy(other);
}

//
//	Utility functions

namespace hkcdVertexGeometryImpl
{
	//
	//	Types

	typedef hkcdPlanarEntity::Plane						Plane;
	typedef hkcdPlanarEntity::PlaneId					PlaneId;
	typedef hkcdPlanarGeometry::Polygon					Polygon;
	typedef hkcdPlanarGeometry::PolygonId				PolygonId;
	typedef hkcdPlanarGeometryPredicates::Orientation	Orientation;
	typedef hkcdPlanarGeometryPredicates::Winding		Winding;
	typedef hkcdPlanarGeometryPlanesCollection			PlanesCollection;
	typedef hkcdVertexGeometry::Vertex					Vertex;
	typedef hkcdVertexGeometry::Line					Line;
	typedef hkcdVertexGeometry::LineId					LineId;
	typedef hkcdVertexGeometry::Edge					Edge;
	typedef hkcdVertexGeometry::EdgeId					EdgeId;
	typedef hkcdVertexGeometry::VPolygon				VPolygon;
	typedef hkcdVertexGeometry::VPolygonId				VPolygonId;
	typedef hkcdVertexGeometry::VPolygonCollection		VPolygonCollection;

	//
	//	Compares two int vectors by XYZ

	static HK_FORCE_INLINE hkBool32 HK_CALL vecLess_XYZ(hkIntVectorParameter vA, hkIntVectorParameter vB)
	{
		const hkVector4Comparison cmpL = vA.compareLessThanS32(vB);	// [* az < bz, ay < by, ax < bx]
		const hkVector4Comparison cmpE = vA.compareEqualS32(vB);	// [*, az == bz, ay == by, ax == bx]

		const int code = ((cmpL.getMask() << 2) & 0x1C) | (cmpE.getMask() & 3);
		return (0xFAF8FAF0 >> code) & 1;
	}

	//
	//	Compares two Vertices

	static HK_FORCE_INLINE hkBool32 HK_CALL vertexLess(const Vertex& vA, const Vertex& vB)
	{
		return vecLess_XYZ(vA.getPlanes(), vB.getPlanes());
	}

	//
	//	Compares two Lines

	static HK_FORCE_INLINE hkBool32 HK_CALL lineLess(const Line& lA, const Line& lB)
	{
		return vecLess_XYZ(lA.getPlanes(), lB.getPlanes());
	}

	//
	//	Compares two edges

	static HK_FORCE_INLINE hkBool32 HK_CALL edgeLess(const Edge& eA, const Edge& eB)
	{
		return vecLess_XYZ(eA.getVertices(), eB.getVertices());
	}

	//
	//	Returns true if two vertices coincide

	static HK_FORCE_INLINE hkBool32 HK_CALL verticesCoincide(const PlanesCollection* geomPlanes, hkIntVectorParameter planeIdsA, hkIntVectorParameter planeIdsB)
	{
		// Get planes of A
		Plane planesA[3];
		geomPlanes->getPlane(PlaneId(planeIdsA.getComponent<0>()), planesA[0]);
		geomPlanes->getPlane(PlaneId(planeIdsA.getComponent<1>()), planesA[1]);
		geomPlanes->getPlane(PlaneId(planeIdsA.getComponent<2>()), planesA[2]);

		// Get planes of B
		Plane planesB[3];
		geomPlanes->getPlane(PlaneId(planeIdsB.getComponent<0>()), planesB[0]);
		geomPlanes->getPlane(PlaneId(planeIdsB.getComponent<1>()), planesB[1]);
		geomPlanes->getPlane(PlaneId(planeIdsB.getComponent<2>()), planesB[2]);

		// Point A must be coplanar on all 3 planes of B in order for A == B.
		return hkcdPlanarGeometryPredicates::coplanar(planesA[0], planesA[1], planesA[2], planesB, 3);
	}

	//
	//	Returns true if two lines coincide

	HK_FORCE_INLINE hkBool32 HK_CALL linesCoincide(const PlanesCollection* geomPlanes, hkIntVectorParameter edgeA, hkIntVectorParameter edgeB)
	{
		Plane planesA[2];	geomPlanes->getPlane(PlaneId(edgeA.getComponent<0>()), planesA[0]);		geomPlanes->getPlane(PlaneId(edgeA.getComponent<1>()), planesA[1]);
		Plane planesB[2];	geomPlanes->getPlane(PlaneId(edgeB.getComponent<0>()), planesB[0]);		geomPlanes->getPlane(PlaneId(edgeB.getComponent<1>()), planesB[1]);

		return	hkcdPlanarGeometryPredicates::edgeOnPlane(planesA[0], planesA[1], planesB[0]) &&
				hkcdPlanarGeometryPredicates::edgeOnPlane(planesA[0], planesA[1], planesB[1]);
	}

	//
	//	A (vertex, line) pair. The vertex is on the line.

	struct VertexLinePair
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkcdVertexGeometryImpl::VertexLinePair);

		/// Comparators
		static HK_FORCE_INLINE bool HK_CALL less(const VertexLinePair& vA, const VertexLinePair& vB)	{ return vA.m_lineId < vB.m_lineId; }
		static HK_FORCE_INLINE bool HK_CALL equal(const VertexLinePair& vA, const VertexLinePair& vB)	{ return vA.m_lineId == vB.m_lineId; }

		int m_vtxId;		///< Vertex Id
		LineId m_lineId;	///< Edge Id
	};

	//
	//	Computes the AABBs of the given polygons

	static void HK_CALL computePolygonsAabbs(const hkcdVertexGeometry* geomIn, hkArray<hkcdIntAabb>& polyAabbsOut)
	{
		const hkArray<Vertex>& vertices		= geomIn->getVertices();
		const VPolygonCollection& vPolys	= geomIn->getPolygons();
		
		// Compute polygons' AABBs
		for (VPolygonId vPolyId = vPolys.getFirstPolygonId(); vPolyId.isValid(); vPolyId = vPolys.getNextPolygonId(vPolyId))
		{
			const VPolygon& vPoly	= vPolys.getPolygon(vPolyId);
			const int numVerts		= vPolys.getNumBoundaryEdges(vPolyId);
			hkcdIntAabb aabb;		aabb.setEmpty();

			for (int vi = 0; vi < numVerts; vi++)
			{
				const EdgeId edgeId = vPoly.getBoundaryEdgeId(vi);
				const int vtxId		= geomIn->getEdgeVertex<0>(edgeId);
				const Vertex& vtx	= vertices[vtxId];

				aabb.includePoint(vtx.getPosition());
			}

			// Since our points are approximate, enlarge the AABB by one.
			// Store polygon Id in the .w component
			aabb.expandBy(1);
			aabb.m_min.setComponent<3>(vPolyId.value());
			polyAabbsOut.pushBack(aabb);
		}
	}

	//
	//	Locates the edges containing the given vertex

	static void HK_CALL findEdgeLinesContainingVertex(	const hkcdVertexGeometry* geomIn, const hkArray<hkcdIntAabb>& polyAabbsIn, int vtxId,
														hkArray<VertexLinePair>& pairsOut)
	{
		// Get vertex and its approximate position
		const hkArray<Vertex>& vertices		= geomIn->getVertices();
		const VPolygonCollection& vPolys	= geomIn->getPolygons();
		const Vertex& vtx					= vertices[vtxId];
		const int firstPairOutIdx			= pairsOut.getSize();

		// If the vertex is on an edge, the edge must belong to a polygon, so the vertex should be included in the polygon's AABB!
		const int numPolys = polyAabbsIn.getSize();
		for (int pi = numPolys - 1; pi >= 0; pi--)
		{
			const hkcdIntAabb polyAabb	= polyAabbsIn[pi];
			const VPolygonId vPolyId	(polyAabb.m_min.getComponent<3>());	// We have the polygon Id stored as the .w component of the min.
			const int numEdges			= vPolys.getNumBoundaryEdges(vPolyId);
			const VPolygon& vPoly		= vPolys.getPolygon(vPolyId);
			
			if ( polyAabb.containsPoint(vtx.getPosition()) )
			{
				// Add all poly edges as candidates
				VertexLinePair* newPairs = pairsOut.expandBy(numEdges);

				for (int k = numEdges - 1; k >= 0; k--)
				{
					const EdgeId edgeId		= vPoly.getBoundaryEdgeId(k);
					const Edge& edge		= geomIn->getEdge(edgeId);
					newPairs[k].m_vtxId		= vtxId;
					newPairs[k].m_lineId	= edge.getLineId();
				}
			}
		}

		// Sort all candidates by edge and remove all duplicates!
		const int numCandidates = pairsOut.getSize() - firstPairOutIdx;
		hkSort(&pairsOut[firstPairOutIdx], numCandidates, VertexLinePair::less);
		const int numUniqueCandidates = hkAlgorithm::removeDuplicatesFromSortedList(&pairsOut[firstPairOutIdx], numCandidates, VertexLinePair::equal);
		pairsOut.setSize(firstPairOutIdx + numUniqueCandidates);

		// Test the surviving candidates for coliniarity
		const PlanesCollection* pc	= geomIn->getPlanesCollection();
		const hkIntVector planeIds	= vtx.getPlanes();
		Plane vtxPlaneA;			pc->getPlane(PlaneId(planeIds.getComponent<0>()), vtxPlaneA);
		Plane vtxPlaneB;			pc->getPlane(PlaneId(planeIds.getComponent<1>()), vtxPlaneB);
		Plane vtxPlaneC;			pc->getPlane(PlaneId(planeIds.getComponent<2>()), vtxPlaneC);
		for (int k = pairsOut.getSize() - 1; k >= firstPairOutIdx; k--)
		{
			// Get edge and its planes
			const VertexLinePair& pair	= pairsOut[k];
			const hkIntVector& edge		= geomIn->getLine(pair.m_lineId).getPlanes();
			Plane edgePlanes[2];		pc->getPlane(PlaneId(edge.getComponent<0>()), edgePlanes[0]);
										pc->getPlane(PlaneId(edge.getComponent<1>()), edgePlanes[1]);

			// Testing for colinearity is equivalent to testing for the vertex to be coplanar on both planes defining the line
			if ( !hkcdPlanarGeometryPredicates::coplanar(vtxPlaneA, vtxPlaneB, vtxPlaneC, edgePlanes, 2) )
			{
				pairsOut.removeAt(k);
			}
		}
	}

	//
	//	A (vertex, plane) pair.

	struct VertexPlanePair
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkcdVertexGeometryImpl::VertexPlanePair);

		PlaneId m_planeId;
		int m_vertexIdx;
	};

	//
	//	Comparison functor for VertexPlanePairs

	struct VertexPlanePairLess
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkcdVertexGeometryImpl::VertexPlanePairLess);

		HK_FORCE_INLINE VertexPlanePairLess(const Plane& edgePlane0, const Plane& edgePlane1, const PlanesCollection* geomPlanes);
		HK_FORCE_INLINE VertexPlanePairLess(const VertexPlanePairLess& other);
		HK_FORCE_INLINE hkBool32 operator()(const VertexPlanePair& pairA, const VertexPlanePair& pairB) const;

		const Plane& m_edgePlane0;
		const Plane& m_edgePlane1;
		const PlanesCollection* m_geomPlanes;
	};

	//
	//	Constructors for the VertexPlanePair comparison functor

	HK_FORCE_INLINE VertexPlanePairLess::VertexPlanePairLess(const Plane& edgePlane0, const Plane& edgePlane1, const PlanesCollection* geomPlanes)
	:	m_edgePlane0(edgePlane0)
	,	m_edgePlane1(edgePlane1)
	,	m_geomPlanes(geomPlanes)
	{}

	HK_FORCE_INLINE VertexPlanePairLess::VertexPlanePairLess(const VertexPlanePairLess& other)
	:	m_edgePlane0(other.m_edgePlane0)
	,	m_edgePlane1(other.m_edgePlane1)
	,	m_geomPlanes(other.m_geomPlanes)
	{}

	//
	//	Returns A < B where A and B are VertexPlanePairs

	HK_FORCE_INLINE hkBool32 VertexPlanePairLess::operator()(const VertexPlanePair& pairA, const VertexPlanePair& pairB) const
	{
		// pointA = (edgePlane0, edgePlane1, planeA)
		// pointB = (edgePlane0, edgePlane1, planeB), and plane points in the direction of the edge
		// Evaluate Orientation(pointA, planeB), return true if inside
		Plane planeA;	m_geomPlanes->getPlane(pairA.m_planeId, planeA);
		Plane planeB;	m_geomPlanes->getPlane(pairB.m_planeId, planeB);
		return (hkcdPlanarGeometryPredicates::orientation(m_edgePlane0, m_edgePlane1, planeA, planeB) & hkcdPlanarGeometryPredicates::ORIENTATION_MASK) == hkcdPlanarGeometryPredicates::BEHIND;
	}

	//
	//	Sorts a set of collinear vertices along their edge

	static void HK_CALL sortColinearVertices(	const hkcdVertexGeometry* geomIn,
												VertexLinePair* HK_RESTRICT colinearVerts, int numColinearVerts)
	{
		// Should have at least 2 vertices!
		HK_ASSERT(0x29c6443, numColinearVerts >= 2);
		if ( numColinearVerts == 2 )
		{
			return;	// If the edge has only 2 vertices, we don't have to sort them!
		}

		// Express all vertices as the intersection of the edge planes plus one of their original planes.
		// This should be possible as a vertex = triplet of linearly independent planes, so even if two of the planes
		// coincide with the line planes, there should be a third that's linearly independent. Choose the plane Ids such
		// that they are oriented along the direction of the edge
		hkLocalBuffer<VertexPlanePair> vtxPlanePairs(numColinearVerts);
		const PlanesCollection* geomPlanes	= geomIn->getPlanesCollection();
		const hkArray<Vertex>& verticesIn	= geomIn->getVertices();

		// Get edge planes
		const LineId lineId		= colinearVerts[0].m_lineId;
		const hkIntVector& edge	= geomIn->getLine(lineId).getPlanes();
		Plane edgePlane0;		geomPlanes->getPlane(PlaneId(edge.getComponent<0>()), edgePlane0);
		Plane edgePlane1;		geomPlanes->getPlane(PlaneId(edge.getComponent<1>()), edgePlane1);

		// Compute edge direction
		hkInt64Vector4 iNrm0;		edgePlane0.getExactNormal(iNrm0);
		hkInt64Vector4 iNrm1;		edgePlane1.getExactNormal(iNrm1);
		hkInt128Vector4 vEdgeDir;	vEdgeDir.setCross(iNrm0, iNrm1);

		// For each vertex, get a linearly independent plane
		for (int k = 0; k < numColinearVerts; k++)
		{
			// Get vertex
			const int vtxIdx			= colinearVerts[k].m_vtxId;
			const hkIntVector planeIds	= verticesIn[vtxIdx].getPlanes();
			PlaneId vtxPlaneId;

			// Locate a linearly independent plane
			hkSimdInt<256> dir;
			{
				hkInt64Vector4 iVtxPlaneNrm;
				Plane vtxPlane;

				// Try plane 0
				vtxPlaneId = PlaneId(planeIds.getComponent<0>());
				geomPlanes->getPlane(vtxPlaneId, vtxPlane);
				vtxPlane.getExactNormal(iVtxPlaneNrm);
				dir = vEdgeDir.dot<3>(iVtxPlaneNrm);

				if ( dir.equalZero().anyIsSet() )
				{
					// Plane 0 failed. Try plane 1.
					vtxPlaneId = PlaneId(planeIds.getComponent<1>());
					geomPlanes->getPlane(vtxPlaneId, vtxPlane);
					vtxPlane.getExactNormal(iVtxPlaneNrm);
					dir = vEdgeDir.dot<3>(iVtxPlaneNrm);

					if ( dir.equalZero().anyIsSet() )
					{
						// Plane 1 failed. Plane 2 should work!
						vtxPlaneId = PlaneId(planeIds.getComponent<2>());
						geomPlanes->getPlane(vtxPlaneId, vtxPlane);
						vtxPlane.getExactNormal(iVtxPlaneNrm);
						dir = vEdgeDir.dot<3>(iVtxPlaneNrm);
						HK_ASSERT(0x3a022ba9, !dir.equalZero().anyIsSet());
					}
				}
			}

			// At this point we have a plane.
			vtxPlanePairs[k].m_vertexIdx	= vtxIdx;
			vtxPlanePairs[k].m_planeId		= dir.lessZero().anyIsSet() ? hkcdPlanarGeometryPrimitives::getOppositePlaneId(vtxPlaneId) : vtxPlaneId;
		}

		// Sort vertices along the edge. At this point, sorting is easy, A < B iff B outside the plane of A
		{
			VertexPlanePairLess f(edgePlane0, edgePlane1, geomPlanes);
			hkSort(vtxPlanePairs.begin(), numColinearVerts, f);
		}

		// Copy the sorted vertices back to our array
		for (int k = 0; k < numColinearVerts; k++)
		{
			colinearVerts[k].m_vtxId = (int)vtxPlanePairs[k].m_vertexIdx;
		}
	}
}

//
//	Remaps the vertex indices using the given remap table

void hkcdVertexGeometry::remapVertices(const hkArray<int>& remapTable)
{
	const int numEdges = m_edges.getSize();
	hkBitField flippedEdges(numEdges, hkBitFieldValue::ZERO);

	for (int k = numEdges - 1; k >= 0; k--)
	{
		Edge& e = m_edges[k];
		const int newStartVtx	= remapTable[e.getStartVertex()];
		const int newEndVtx		= remapTable[e.getEndVertex()];

		e.set(newStartVtx, newEndVtx, e.getLineId(), EdgeId(k));

		if ( e.getStartVertex() != newStartVtx )
		{
			flippedEdges.set(k);
		}
	}

	// Apply edge flips
	const int mask = ~FLIPPED_EDGE_FLAG;
	for (VPolygonId vPolyId = m_polys.getFirstPolygonId(); vPolyId.isValid(); vPolyId = m_polys.getNextPolygonId(vPolyId))
	{
		VPolygon& vPoly		= m_polys.accessPolygon(vPolyId);
		const int numBounds	= m_polys.getNumBoundaryEdges(vPolyId);

		for (int k = numBounds - 1; k >= 0; k--)
		{
			const EdgeId edgeId = vPoly.getBoundaryEdgeId(k);
			const int edgeIdx	= edgeId.value() & mask;

			if ( flippedEdges.get(edgeIdx) )
			{
				const int edgeFlip = (edgeId.value() & FLIPPED_EDGE_FLAG) ^ FLIPPED_EDGE_FLAG;
				vPoly.setBoundaryEdgeId(k, EdgeId(edgeIdx | edgeFlip));
			}
		}
	}
}

//
//	Welds vertices with identical plane ids

void hkcdVertexGeometry::weldIdenticalVertices()
{
	// Sort vertices by plane
	const int numSrcVerts = m_vertices.getSize();
	hkSort(m_vertices.begin(), numSrcVerts, hkcdVertexGeometryImpl::vertexLess);

	// Collapse all identical vertices
	hkArray<int> remapTable;
	remapTable.setSize(numSrcVerts);
	if ( numSrcVerts )
	{
		int prev = 0;
		remapTable[m_vertices[prev].getPlanes().getComponent<3>()] = prev;
		hkVector4Comparison maskXYZ;	maskXYZ.set<hkVector4Comparison::MASK_XYZ>();
		for (int crt = 1; crt < numSrcVerts; crt++)
		{
			hkVector4Comparison cmp = m_vertices[crt].getPlanes().compareEqualS32(m_vertices[prev].getPlanes());	// [ax == bx, ay == by, az == bz, *]
			cmp.setAndNot(maskXYZ, cmp);																			// [ax != bx, ay != by, az != bz, 0]

			if ( cmp.anyIsSet() )
			{
				// Vertices not equal, crt survives
				m_vertices[++prev] = m_vertices[crt];
			}

			remapTable[m_vertices[crt].getPlanes().getComponent<3>()] = prev;
		}

		m_vertices.setSize(prev + 1);
	}

	remapVertices(remapTable);

	// Store vertex index in the .w components, for further welding ops
	for (int vi = m_vertices.getSize() - 1; vi >= 0; vi--)
	{
		m_vertices[vi].accessPlanes().setComponent<3>(vi);
	}
}

//
//	Remaps the line indices using the given remap table

void hkcdVertexGeometry::remapLines(const hkArray<int>& remapTable)
{
	for (int k = m_edges.getSize() - 1; k >= 0; k--)
	{
		Edge& edge = m_edges[k];
		const LineId oldId	= edge.getLineId();
		const LineId newId	(remapTable[oldId.value()]);

		edge.setLineId(newId);
	}
}

//
//	Welds all edge lines with identical planes

void hkcdVertexGeometry::weldIdenticalLines()
{
	// Sort the lines by plane Id.
	const int numSrcLines = m_lines.getSize();
	hkSort(m_lines.begin(), numSrcLines, hkcdVertexGeometryImpl::lineLess);

	// Collapse all identical lines
	hkArray<int> remapTable;
	remapTable.setSize(numSrcLines);
	if ( numSrcLines )
	{
		int prev = 0;
		remapTable[m_lines[prev].getPlanes().getComponent<3>()] = prev;
		hkVector4Comparison maskXY;	maskXY.set<hkVector4Comparison::MASK_XY>();

		for (int crt = 1; crt < numSrcLines; crt++)
		{
			hkVector4Comparison cmp = m_lines[crt].getPlanes().compareEqualS32(m_lines[prev].getPlanes());	// [ax == bx, ay == by, *, *]
			cmp.setAndNot(maskXY, cmp);																		// [ax != bx, ay != by, 0, 0]

			if ( cmp.anyIsSet() )
			{
				// Lines not equal, crt survives
				prev++;
				m_lines[prev] = m_lines[crt];
			}

			remapTable[m_lines[crt].getPlanes().getComponent<3>()] = prev;
		}

		m_lines.setSize(prev + 1);
	}

	// Remap line indices
	remapLines(remapTable);

	// Update edge index for further welding ops
	for (int ei = m_lines.getSize() - 1; ei >= 0; ei--)
	{
		m_lines[ei].setId(LineId(ei));
	}
}

//
//	Welds all other vertices

void hkcdVertexGeometry::weldVertices()
{
	// Compute plane intersections
	const int numSrcVerts = m_vertices.getSize();
	hkArray<hkIntVector> intersections;
	intersections.setSize(numSrcVerts + 1);

	for (int k = numSrcVerts - 1; k >= 0; k--)
	{
		// Get planes
		const hkIntVector& srcVtx = m_vertices[k].getPlanes();

		Plane planes[3];
		m_planes->getPlane(PlaneId(srcVtx.getComponent<0>()), planes[0]);
		m_planes->getPlane(PlaneId(srcVtx.getComponent<1>()), planes[1]);
		m_planes->getPlane(PlaneId(srcVtx.getComponent<2>()), planes[2]);

		// Estimate the interval of the exact intersection
		hkIntVector vI;
		hkcdPlanarGeometryPredicates::approximateIntersection(planes, vI);
		vI.setComponent<3>(srcVtx.getComponent<3>());
		intersections[k] = vI;
	}

	// Sort intersections by xyz
	hkSort(intersections.begin(), numSrcVerts, hkcdVertexGeometryImpl::vecLess_XYZ);
	intersections[numSrcVerts].setAll(-1);	// Add the end marker!

	// Attempt to weld only vertices with identical xyz
	hkArray<int> vtxRemap;			vtxRemap.setSize(numSrcVerts, -1);
	hkArray<Vertex> verticesOut;	verticesOut.reserve(numSrcVerts);
	{
		hkIntVector prevVtxGroup;	prevVtxGroup.setAll(-1);
		hkVector4Comparison mXYZ;	mXYZ.set<hkVector4Comparison::MASK_XYZ>();
		int prevGroupStartIdx		= 0;

		for (int k = 0; k <= numSrcVerts; k++)
		{
			const hkIntVector crtVtx	= intersections[k];
			hkVector4Comparison cmp		= prevVtxGroup.compareEqualS32(crtVtx);	// [px == cx, py == cy, pz == cz, *]
			cmp.setAndNot(mXYZ, cmp);											// [px != cx, py != cy, pz != cz, 0]

			if ( cmp.anyIsSet() )
			{
				// A new group has started!
				// Process previous group
				if ( k - prevGroupStartIdx )
				{
					const int startNewVtxIdx = verticesOut.getSize();

					// For each old vertex, try to weld to the new ones
					for (int i = prevGroupStartIdx; i < k; i++)
					{
						const int oldVtxIdx = intersections[i].getComponent<3>();
						hkIntVector vOld	= m_vertices[oldVtxIdx].getPlanes();

						// Compare against all new
						int j = verticesOut.getSize() - 1;
						for (; j >= startNewVtxIdx; j--)
						{
							const hkIntVector vNew = verticesOut[j].getPlanes();
							if ( hkcdVertexGeometryImpl::verticesCoincide(m_planes, vOld, vNew) )
							{
								vtxRemap[oldVtxIdx] = j;
								break;
							}
						}

						// If nothing was found, add new vertex
						if ( j < startNewVtxIdx )
						{
							vtxRemap[oldVtxIdx] = verticesOut.getSize();
							Vertex& nv			= verticesOut.expandOne();
							nv.accessPlanes()	= vOld;
							nv.accessPosition()	= intersections[i];
						}
					}
				}

				// Initialize stuff for the new group
				prevGroupStartIdx	= k;
				prevVtxGroup		= crtVtx;
			}
		}
	}

	// Remap vertices
	m_vertices.swap(verticesOut);
	remapVertices(vtxRemap);

	// Store vertex index in the .w components, for further welding ops
	for (int vi = verticesOut.getSize() - 1; vi >= 0; vi--)
	{
		verticesOut[vi].accessPlanes().setComponent<3>(vi);
	}
}

//
//	Welds all other edge lines

void hkcdVertexGeometry::weldLines()
{
	// Compute edge directions
	const int numSrcLines = m_lines.getSize();
	hkArray<hkIntVector> directions;
	directions.setSize(numSrcLines + 1);
		
	for (int k = numSrcLines - 1; k >= 0; k--)
	{
		// Get planes
		const hkIntVector& line = m_lines[k].getPlanes();
		Plane planeA;	m_planes->getPlane(PlaneId(line.getComponent<0>()), planeA);
		Plane planeB;	m_planes->getPlane(PlaneId(line.getComponent<1>()), planeB);

		// Estimate the interval of the exact intersection
		hkIntVector vI;
		hkcdPlanarGeometryPredicates::approximateEdgeDirection(planeA, planeB, vI);
		vI.setComponent<3>(line.getComponent<3>());
		directions[k] = vI;
	}

	// Sort directions by xyz
	hkSort(directions.begin(), numSrcLines, hkcdVertexGeometryImpl::vecLess_XYZ);
	directions[numSrcLines].setAll(-1);	// Add the end marker!

	// Try to collapse directions in the same group
	hkArray<int> lineRemap;		lineRemap.setSize(numSrcLines, -1);
	hkArray<Line> newLines;		newLines.reserve(numSrcLines);
	{
		hkIntVector prevLineGroup;	prevLineGroup.setAll(-1);
		hkVector4Comparison mXYZ;	mXYZ.set<hkVector4Comparison::MASK_XYZ>();
		int prevGroupStartIdx		= 0;

		for (int k = 0; k <= numSrcLines; k++)
		{
			const hkIntVector crtLine	= directions[k];
			hkVector4Comparison cmp		= prevLineGroup.compareEqualS32(crtLine);	// [px == cx, py == cy, pz == cz, *]
			cmp.setAndNot(mXYZ, cmp);												// [px != cx, py != cy, pz != cz, 0]

			if ( cmp.anyIsSet() )
			{
				// A new group has started!
				// Process previous group
				if ( k - prevGroupStartIdx )
				{
					const int startNewEdgeIdx = newLines.getSize();

					// For each old edge, try to weld to the new ones
					for (int i = prevGroupStartIdx; i < k; i++)
					{
						const int oldLineIdx	= directions[i].getComponent<3>();
						const Line& eOld		= m_lines[oldLineIdx];

						// Compare against all new
						int j = newLines.getSize() - 1;
						for (; j >= startNewEdgeIdx; j--)
						{
							const Line& eNew = newLines[j];
							if ( hkcdVertexGeometryImpl::linesCoincide(m_planes, eOld.getPlanes(), eNew.getPlanes()) )
							{
								lineRemap[oldLineIdx] = j;
								break;
							}
						}

						// If nothing was found, add new edge
						if ( j < startNewEdgeIdx )
						{
							lineRemap[oldLineIdx] = newLines.getSize();
							newLines.pushBack(eOld);
						}
					}
				}

				// Initialize stuff for the new group
				prevGroupStartIdx	= k;
				prevLineGroup		= crtLine;
			}
		}
	}

	// Replace lines with unique ones
	m_lines.swap(newLines);

	// Remap line indices
	remapLines(lineRemap);

	// Store line index in the .w components, for further welding ops
	for (int ei = m_lines.getSize() - 1; ei >= 0; ei--)
	{
		m_lines[ei].setId(LineId(ei));
	}
}

//
//	Repairs the T-junctions by adding additional interior vertices to the polygons

void hkcdVertexGeometry::repairTJunctions()
{
	// Types
	typedef hkcdVertexGeometryImpl::VertexLinePair VertexLinePair;

	// Compute the AABBs of all polygons
	hkArray<hkcdIntAabb> polyAabbs;
	hkcdVertexGeometryImpl::computePolygonsAabbs(this, polyAabbs);

	// For each vertex, find the colinear lines
	hkArray<VertexLinePair> vertexLines;
	{
		const int numVerts = m_vertices.getSize();
		for (int vi = 0; vi < numVerts; vi++)
		{
			hkcdVertexGeometryImpl::findEdgeLinesContainingVertex(this, polyAabbs, vi, vertexLines);
		}

		// Sort by edges
		hkSort(vertexLines.begin(), vertexLines.getSize(), VertexLinePair::less);
	}

	// Compute the offsets of collinear vertex groups
	const int numLines = m_lines.getSize();
	hkArray<int> colinearGroups;
	{
		colinearGroups.setSize(numLines + 1);

		const int numVertexLines	= vertexLines.getSize();
		LineId prevLineId			= LineId::invalid();
		for (int k = 0; k < numVertexLines; k++)
		{
			const VertexLinePair& crtPair = vertexLines[k];
			const LineId crtLineId = crtPair.m_lineId;

			if ( prevLineId != crtLineId )
			{
				// A new group has started!
				HK_ASSERT(0x43d24c79, (int)prevLineId.valueUnchecked() + 1 == (int)crtLineId.valueUnchecked());	// All edges should have at least 2 verts, so they should all be present in the groups!
				colinearGroups[crtLineId.value()] = k;
				prevLineId = crtLineId;
			}
		}
		colinearGroups[numLines] = numVertexLines;
	}

	// For each colinear group, sort the vertices along the edge direction!
	for (int ei = 0; ei < numLines; ei++)
	{
		const int si = colinearGroups[ei];
		const int numVerts = colinearGroups[ei + 1] - si;

		hkcdVertexGeometryImpl::sortColinearVertices(this, &vertexLines[si], numVerts);
	}

	// Finally, we can split the polygons' edges
	{
		VPolygonCollection newPolys;
		hkArray<Edge> newEdges;
		hkArray<EdgeId> newPolyEdges;

		for (VPolygonId srcPolyId = m_polys.getFirstPolygonId(); srcPolyId.isValid(); srcPolyId = m_polys.getNextPolygonId(srcPolyId))
		{
			// Get source polygon
			const VPolygon& srcPoly = m_polys.getPolygon(srcPolyId);
			const int numSrcEdges	= m_polys.getNumBoundaryEdges(srcPolyId);

			// Reset the new poly
			newPolyEdges.setSize(0);

			// Try to split each poly edge
			for (int crt = 0; crt < numSrcEdges; crt++)
			{
				// Get the edge and its vertices
				const EdgeId srcEdgeId	= srcPoly.getBoundaryEdgeId(crt);
				const Edge& srcEdge		= getEdge(srcEdgeId);
				const LineId lineId		= srcEdge.getLineId();
				const int startVtx		= getEdgeVertex<0>(srcEdgeId);
				const int endVtx		= getEdgeVertex<1>(srcEdgeId);

				// Locate the vertices in the colinear group
				int sIdx = colinearGroups[lineId.value()];
				int eIdx = sIdx;
				{
					const int idxMax = colinearGroups[lineId.value() + 1];
					for (; sIdx < idxMax; sIdx++)	{ if ( vertexLines[sIdx].m_vtxId == startVtx )	{ break; } }
					for (; eIdx < idxMax; eIdx++)	{ if ( vertexLines[eIdx].m_vtxId == endVtx )	{ break; } }
					HK_ASSERT(0x1c43c052, (sIdx < idxMax) && (eIdx < idxMax) && (sIdx != eIdx));
				}

				// Add all vertices between sIdx and eIdx to the new poly
				const int dIdx			= (eIdx > sIdx) ? 1 : -1;
				const int numVertsToAdd = dIdx * (eIdx - sIdx);
				for (int i = 0; i < numVertsToAdd; i++)
				{
					const int vtxIdA	= vertexLines[sIdx].m_vtxId;	sIdx += dIdx;
					const int vtxIdB	= vertexLines[sIdx].m_vtxId;
					Edge edgeAB;		edgeAB.set(vtxIdA, vtxIdB, lineId, EdgeId(newEdges.getSize()));
					const EdgeId eid	(newEdges.getSize() | ((edgeAB.getStartVertex() == vtxIdA) ? 0 : FLIPPED_EDGE_FLAG));

					newEdges.pushBack(edgeAB);
					newPolyEdges.pushBack(eid);
				}
			}

			// Add the new poly
			const int numDstEdges		= newPolyEdges.getSize();
			const VPolygonId dstPolyId	= newPolys.alloc(srcPoly.getMaterialId(), srcPoly.getUserData(), numDstEdges);
			VPolygon& dstPoly			= newPolys.accessPolygon(dstPolyId);

			// Set its boundaries
			for (int k = numDstEdges - 1; k >= 0; k--)
			{
				dstPoly.setBoundaryEdgeId(k, newPolyEdges[k]);
			}
		}

		// Swap current and new poly
		m_edges.swap(newEdges);
		m_polys.copy(newPolys);
	}
}

//
//	Remaps the edge indices using the given remap table

void hkcdVertexGeometry::remapEdges(const hkArray<int>& remapTable)
{
	for (VPolygonId vPolyId = m_polys.getFirstPolygonId(); vPolyId.isValid(); vPolyId = m_polys.getNextPolygonId(vPolyId))
	{
		VPolygon& vPoly		= m_polys.accessPolygon(vPolyId);
		const int numEdges	= m_polys.getNumBoundaryEdges(vPolyId);

		for (int ei = numEdges - 1; ei >= 0; ei--)
		{
			const int oldEdgeIdx = vPoly.getBoundaryEdgeId(ei).value();
			const int newEdgeIdx = remapTable[oldEdgeIdx & (~FLIPPED_EDGE_FLAG)] | (oldEdgeIdx & FLIPPED_EDGE_FLAG);

			vPoly.setBoundaryEdgeId(ei, EdgeId(newEdgeIdx));
		}
	}
}

/// Returns all the polygon ids in the collection
void hkcdVertexGeometry::getAllPolygonIds(hkArray<VPolygonId>& vPolyIdsOut)
{
	for (VPolygonId vPolyId = m_polys.getFirstPolygonId(); vPolyId.isValid(); vPolyId = m_polys.getNextPolygonId(vPolyId))
	{
		vPolyIdsOut.pushBack(vPolyId);
	}
}

//
//	Collapses all duplicate edges

void hkcdVertexGeometry::weldIdenticalEdges()
{
	// Sort edges
	const int numSrcEdges = m_edges.getSize();
	hkSort(m_edges.begin(), numSrcEdges, hkcdVertexGeometryImpl::edgeLess);

	// Collapse all identical vertices
	hkArray<int> remapTable;
	remapTable.setSize(numSrcEdges);
	if ( numSrcEdges )
	{
		int prev = 0;
		remapTable[m_edges[prev].getVertices().getComponent<3>()] = prev;
		hkVector4Comparison maskXYZ;	maskXYZ.set<hkVector4Comparison::MASK_XYZ>();
		for (int crt = 1; crt < numSrcEdges; crt++)
		{
			hkVector4Comparison cmp = m_edges[crt].getVertices().compareEqualS32(m_edges[prev].getVertices());	// [ax == bx, ay == by, az == bz, *]
			cmp.setAndNot(maskXYZ, cmp);																		// [ax != bx, ay != by, az != bz, 0]

			if ( cmp.anyIsSet() )
			{
				// Edges not equal, crt survives
				m_edges[++prev] = m_edges[crt];
			}

			remapTable[m_edges[crt].getVertices().getComponent<3>()] = prev;
		}

		m_edges.setSize(prev + 1);
	}

	remapEdges(remapTable);

	// Store edge index in the .w components, for further welding ops
	for (int vi = m_edges.getSize() - 1; vi >= 0; vi--)
	{
		m_edges[vi].accessVertices().setComponent<3>(vi);
	}
}

//
//	Creates a vertex geometry from the given planar geometry

hkcdVertexGeometry* HK_CALL hkcdVertexGeometry::createFromPlanarGeometry(const hkcdPlanarGeometry* srcGeom, const hkArray<PolygonId>& allPolys)
{
	// Create an empty geometry
	hkcdPlanarGeometry* ncGeom	= const_cast<hkcdPlanarGeometry*>(srcGeom);
	hkcdVertexGeometry* vRep	= new hkcdVertexGeometry(ncGeom->accessPlanesCollection(), ncGeom->accessDebugger());

	// Collect the raw vertices and lines (i.e. "infinite" polygon edges)
	const int numPolys = allPolys.getSize();
	for (int k = 0; k < numPolys; k++)
	{
		const PolygonId polyId	= allPolys[k];
		const Polygon& srcPoly	= srcGeom->getPolygon(polyId);
		const int numVerts		= srcGeom->getNumBoundaryPlanes(polyId);

		// Add a vertex polygon
		const VPolygonId vPolyId = vRep->m_polys.alloc(srcPoly.getMaterialId(), polyId.value(), numVerts);
		VPolygon& vPoly = vRep->m_polys.accessPolygon(vPolyId);

		// Add all vertices, edges, and lines
		const int vtxBase	= vRep->m_vertices.getSize();
		Vertex* dstVtx		= vRep->m_vertices.expandBy(numVerts);
		Edge* dstEdge		= vRep->m_edges.expandBy(numVerts);
		Line* dstLine		= vRep->m_lines.expandBy(numVerts);

		const PlaneId supportPlaneId	= srcPoly.getSupportPlaneId();
		PlaneId prevPlaneId				= srcPoly.getBoundaryPlaneId(numVerts - 1);
		for (int crt = 0; crt < numVerts; crt++)
		{
			const PlaneId crtPlaneId	= srcPoly.getBoundaryPlaneId(crt);
			const int vtxId				= vtxBase + crt;
			const LineId lineId			(vtxId);

			dstVtx[crt].setPlanesAndId(supportPlaneId, prevPlaneId, crtPlaneId, vtxId);
			dstLine[crt].set(supportPlaneId, crtPlaneId, lineId);
			dstEdge[crt].set(vtxId, vtxBase + ((crt + 1) % numVerts), lineId, EdgeId(vtxId));

			const EdgeId eid((dstEdge[crt].getStartVertex() == vtxId) ? vtxId : vtxId | FLIPPED_EDGE_FLAG);
			vPoly.setBoundaryEdgeId(crt, eid);

			prevPlaneId = crtPlaneId;
		}
	}

	// Weld identical vertices
	vRep->weldIdenticalVertices();

	// Weld identical lines
	vRep->weldIdenticalLines();

	// Weld all vertices
	vRep->weldVertices();

	// Weld all lines
	vRep->weldLines();

	// Fix T-junctions. After this step, the polygons may have colinear boundary segments!
	vRep->repairTJunctions();

	// Collapse all duplicate edges
	vRep->weldIdenticalEdges();

	// Return geometry
	return vRep;
}

//
//	Creates a vertex geometry from the given subset of polygons in the planar geometry

hkcdVertexGeometry* HK_CALL hkcdVertexGeometry::createFromPlanarGeometry(const hkcdPlanarGeometry* srcGeom)
{
	// Collect all polygons
	hkArray<PolygonId> allPolys;
	srcGeom->getAllPolygons(allPolys);

	// Create V-rep
	return createFromPlanarGeometry(srcGeom, allPolys);
}

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
