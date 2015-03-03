/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/DataStructures/Planar/ConvexCellsTree/hkcdConvexCellsTree2D.h>
#include <Geometry/Collide/DataStructures/Planar/Utils/hkcdPlanarGeometrySimplifier.h>

#include <Common/Base/Algorithm/UnionFind/hkUnionFind.h>

#define ENABLE_SIMPLIFICATION	(1)
#define ENABLE_INCLUSION_TESTS	(0)

//
//	Types

typedef hkcdPlanarGeometryPlanesCollection			PlanesCollection;
typedef hkcdPlanarGeometryPolygonCollection			PPolygonCollection;
typedef hkcdPlanarGeometry::Plane					Plane;
typedef hkcdPlanarGeometry::PlaneId					PlaneId;
typedef hkcdPlanarGeometry::Polygon					PPolygon;
typedef hkcdPlanarGeometry::PolygonId				PPolygonId;

typedef hkcdPlanarSolid::Node						Node;
typedef hkcdPlanarSolid::NodeId						NodeId;
typedef hkcdPlanarSolid::NodeStorage				NodeCollection;

typedef hkcdVertexGeometry::VPolygon				VPolygon;
typedef hkcdVertexGeometry::VPolygonId				VPolygonId;
typedef hkcdVertexGeometry::Edge					VEdge;
typedef hkcdVertexGeometry::EdgeId					VEdgeId;
typedef hkcdVertexGeometry::Line					VLine;
typedef hkcdVertexGeometry::LineId					VLineId;
typedef hkcdVertexGeometry::Vertex					VVertex;
typedef hkArray<VVertex>							VVertexCollection;
typedef hkcdVertexGeometry::VPolygonCollection		VPolygonCollection;

//
//	Utility functions

namespace hkndPlanarSimplificationImpl
{
	//
	//	Pair of polygonId, materialId

	struct PolyMtlPair
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DESTRUCTION, hkndPlanarSimplificationImpl::PolyMtlPair);

		/// Comparator
		static HK_FORCE_INLINE bool HK_CALL less(const PolyMtlPair& pairA, const PolyMtlPair& pairB)
		{
			return	(pairA.m_materialId < pairB.m_materialId) ||
					((pairA.m_materialId == pairB.m_materialId) && (pairA.m_vPolyId < pairB.m_vPolyId));
		}

		hkUint32 m_materialId;	///< Material Id
		VPolygonId m_vPolyId;	///< Polygon Id
	};

	//
	//	Pair of VEdgeId, VPolygonId

	struct EdgePolyPair
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DESTRUCTION, hkndPlanarSimplificationImpl::EdgePolyPair);

		/// Comparator
		static HK_FORCE_INLINE bool HK_CALL polyLess(const EdgePolyPair& pairA, const EdgePolyPair& pairB)
		{
			return pairA.m_vPolyIdx < pairB.m_vPolyIdx;
		}

		/// Comparator
		static HK_FORCE_INLINE bool HK_CALL edgeLess(const EdgePolyPair& pairA, const EdgePolyPair& pairB)
		{
			return (pairA.m_edgeId.valueUnchecked() & (~hkcdVertexGeometry::FLIPPED_EDGE_FLAG)) < (pairB.m_edgeId.valueUnchecked() & (~hkcdVertexGeometry::FLIPPED_EDGE_FLAG));
		}

		/// Comparator
		static HK_FORCE_INLINE bool HK_CALL edgeEqual(const EdgePolyPair& pairA, const EdgePolyPair& pairB)
		{
			return (pairA.m_edgeId.valueUnchecked() & (~hkcdVertexGeometry::FLIPPED_EDGE_FLAG)) == (pairB.m_edgeId.valueUnchecked() & (~hkcdVertexGeometry::FLIPPED_EDGE_FLAG));
		}

		VEdgeId m_edgeId;		/// Edge Id
		hkUint32 m_vPolyIdx;	/// Polygon index
	};

	//
	//	Comparator for plane ids

	static HK_FORCE_INLINE bool HK_CALL planesLess(const PlaneId::Type& pidA, const PlaneId::Type& pidB)
	{
		return (pidA & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG)) < (pidB & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG));
	}

	//
	//	Comparator for plane ids

	static HK_FORCE_INLINE bool HK_CALL planesEqual(const PlaneId& pidA, const PlaneId& pidB)
	{
		return (pidA.valueUnchecked() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG)) == (pidB.valueUnchecked() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG));
	}

	//
	//	The main simplifier

	class Simplifier
	{
		public:

			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DESTRUCTION, hkndPlanarSimplificationImpl::Simplifier);

		public:

			/// Constructor
			Simplifier(hkcdPlanarGeometry* planarGeom, const hkcdVertexGeometry* vertexGeom);

		public:

			/// Runs the simplification
			void execute();

			/// Builds a BSP from the boundary of the surface represented by a group of polygons
			void buildSimplifyingSolidForBoundedGroup(hkcdPlanarSolid& solidOut, const hkArray<VPolygonId>& vPolyIds, hkLocalArray<PPolygonId>& pPolyIds, PlaneId& supportPlaneId, hkUint32 materialId);

		protected:

			/// Simplifies the given set of polygons. They all share the same material Id and thus the same support plane!
			void simplifyCoplanar(const PolyMtlPair* HK_RESTRICT polyMtlPairs, int numPairs, hkUint32 materialId);
			
			/// Simplifies the given group of polygons. They all share the same material, and are enclosed by the same boundary curve
			void simplifyBoundedGroup(const hkArray<VPolygonId>& vPolyIds, hkUint32 materialId);

			/// Collects the boundary edges for the given group of polygons
			void collectBoundaryEdges(const hkArray<VPolygonId>& vPolyIds, hkArray<EdgePolyPair>& boundaryEdgesOut);

			/// Converts the given array of boundary lines to boundary planes
			void extractBoundaryPlanes(PlaneId supportPlaneId, const hkArray<EdgePolyPair>& boundaryEdgesIn, hkArray<PlaneId>& planeIdsOut);

		protected:

			hkcdPlanarGeometry& m_planarGeom;			///< Planar geometry
			const hkcdVertexGeometry& m_vertexGeom;		///< Vertex geometry
			hkArray<EdgePolyPair> m_edgePolyPairs;		///< Temp working buffer
			hkArray<VPolygonId> m_polyGroupIds;			///< Array of all polygon Ids inside the group to be simplified
			hkArray<PlaneId> m_boundaryPlanes;			///< Array of boundary planes

		protected:

			hkcdPlanarGeometry m_workingGeom;			///< Temporary geom, used by the 2D BSP tree
			hkcdPlanarSolid m_workingSolid;				///< Temporary solid, used to simplify the polygons
			hkcdConvexCellsTree2D m_workingRegions;		///< Temporary convex cells 2D tree
			hkPseudoRandomGenerator m_rng;				///< The random number generator
	};

	//
	//	Constructor

	Simplifier::Simplifier(hkcdPlanarGeometry* planarGeom, const hkcdVertexGeometry* vertexGeom)
	:	m_planarGeom(*planarGeom)
	,	m_vertexGeom(*vertexGeom)
	,	m_workingGeom(planarGeom->accessPlanesCollection(), 0, planarGeom->accessDebugger())
	,	m_workingSolid(planarGeom->getPlanesCollection(), 0, planarGeom->accessDebugger())
	,	m_workingRegions(&m_workingGeom, PlaneId::invalid())
	,	m_rng(13)
	{}

	//
	//	Runs the simplification

	void Simplifier::execute()
	{
		const VPolygonCollection& vPolys = m_vertexGeom.getPolygons();

		// Collect all polygons from the vertex geometry
		hkArray<PolyMtlPair> polyMtlPairs;
		for (VPolygonId vPolyId = vPolys.getFirstPolygonId(); vPolyId.isValid(); vPolyId = vPolys.getNextPolygonId(vPolyId))
		{
			const VPolygon& vPoly	= vPolys.getPolygon(vPolyId);
			PolyMtlPair& entry		= polyMtlPairs.expandOne();

			entry.m_materialId	= vPoly.getMaterialId();
			entry.m_vPolyId		= vPolyId;
		}

		// Group them by material Id.
		const int numPairs = polyMtlPairs.getSize();
		hkSort(polyMtlPairs.begin(), numPairs, PolyMtlPair::less);

		// Add an end terminator
		{
			PolyMtlPair& endTerminator	= polyMtlPairs.expandOne();
			endTerminator.m_materialId	= (hkUint32)-1;
			endTerminator.m_vPolyId		= VPolygonId::invalid();
		}

		// Simplify each group of polygons with the same material id
		hkUint32 prevMaterialId = (hkUint32)-1;
		int prevGroupStartIdx	= 0;
		for (int k = 0; k <= numPairs; k++)
		{
			const PolyMtlPair& crtPair = polyMtlPairs[k];
			if ( prevMaterialId != crtPair.m_materialId )
			{
				// New group started, simplify the previous group!
				simplifyCoplanar(&polyMtlPairs[prevGroupStartIdx], k - prevGroupStartIdx, prevMaterialId);
				prevGroupStartIdx	= k;
				prevMaterialId		= crtPair.m_materialId;
			}
		}
	}

	//
	//	Simplifies the given set of polygons. They all share the same material Id and thus the same support plane!

	void Simplifier::simplifyCoplanar(const PolyMtlPair* HK_RESTRICT polyMtlPairs, int numPairs, hkUint32 materialId)
	{
		if ( (numPairs < 2) || (materialId >= hkcdPlanarGeometry::INVALID_MATERIAL_ID) )
		{
			return;	// Nothing to simplify!
		}

		// Get collections
		const VPolygonCollection& vPolys = m_vertexGeom.getPolygons();

		// The pairs are sorted by polygon Id. This makes it easy to assign indices to the polygon Ids in this group!
		// Collect edges
		hkArray<EdgePolyPair>& edgePairs = m_edgePolyPairs;
		edgePairs.setSize(0);

		for (int k = numPairs - 1; k >= 0; k--)
		{
			// Get the polygon
			const PolyMtlPair& pmPair	= polyMtlPairs[k];
			const VPolygonId vPolyId	= pmPair.m_vPolyId;
			const VPolygon& vPoly		= vPolys.getPolygon(vPolyId);
			const int numBounds			= vPolys.getNumBoundaryEdges(vPolyId);

			// Add all its boundary edges
			for (int bi = numBounds - 1; bi >= 0; bi--)
			{
				const VEdgeId edgeId	= vPoly.getBoundaryEdgeId(bi);
				EdgePolyPair& epPair	= edgePairs.expandOne();

				epPair.m_edgeId		= VEdgeId(edgeId.value() & (~hkcdVertexGeometry::FLIPPED_EDGE_FLAG));
				epPair.m_vPolyIdx	= vPolyId.value();
			}
		}

		// Sort edge pairs by polygon Id, so we can perform the reindexing from "true" polygonIds to local indices
		const int numEdgePairs = edgePairs.getSize();
		hkSort(edgePairs.begin(), numEdgePairs, EdgePolyPair::polyLess);

		// Match poly Ids and reindex
		{
			int ia = 0, ib = 0;
			while ( (ia < numPairs) && (ib < numEdgePairs) )
			{
				const PolyMtlPair& pmPair	= polyMtlPairs[ia];
				EdgePolyPair& epPair		= edgePairs[ib];

				if ( pmPair.m_vPolyId.value() < epPair.m_vPolyIdx )			{	ia++;	}							// Poly idex fell behind, catch-up!
				else if ( pmPair.m_vPolyId.value() == epPair.m_vPolyIdx )	{	epPair.m_vPolyIdx = ia;	ib++;	}	// Poly Id match. Replace with local index!
				else														{	HK_ASSERT(0x21efb455,  false);	}			// Should never happen! Missing PolyIds!!!
			}

			// Finish
			HK_ASSERT(0x7d9280e5, (ia == numPairs - 1) && (ib == numEdgePairs));	// Make sure we reindexed everything!
		}

		// Sort by edgeId, to find the internal edges
		hkSort(edgePairs.begin(), numEdgePairs, EdgePolyPair::edgeLess);

		// Add end terminator
		{
			EdgePolyPair& endTerminator = edgePairs.expandOne();
			endTerminator.m_edgeId		= VEdgeId::invalid();
			endTerminator.m_vPolyIdx	= (hkUint32)-1;
		}

		// Compute the graph of polygons
		hkLocalBuffer<int> unionFindBuffer(numPairs);
		hkUnionFind unionFind(unionFindBuffer, numPairs);
		{
			VEdgeId prevEdgeId		= VEdgeId::invalid();
			int prevGroupStartIdx	= 0;

			unionFind.beginAddEdges();
			for (int k = 0; k <= numEdgePairs; k++)
			{
				const EdgePolyPair& crtPair = edgePairs[k];
				if ( prevEdgeId != crtPair.m_edgeId )
				{
					// New edge group started!
					switch ( k - prevGroupStartIdx )
					{
					case 0:		// Null group, ok
					case 1:		// Boundary edge, ok
						break;

					case 2:		// Internal edge, ok
						{
							const EdgePolyPair& ep0 = edgePairs[prevGroupStartIdx];
							const EdgePolyPair& ep1 = edgePairs[prevGroupStartIdx + 1];
							unionFind.addEdge(ep0.m_vPolyIdx, ep1.m_vPolyIdx);
						}
						break;

					default:
						HK_ASSERT(0x1a526929, false);
					}

					prevGroupStartIdx	= k;
					prevEdgeId			= crtPair.m_edgeId;
				}
			}

			unionFind.endAddEdges();
		}

		// Compute the groups of connected polygons
		hkArray<int> numElementsPerGroup;
		unionFind.assignGroups(numElementsPerGroup);
		const int numGroups = numElementsPerGroup.getSize();
		hkArray<int> partitionedPolyIndices;
		unionFind.sortByGroupId(numElementsPerGroup, partitionedPolyIndices);

		// Set-up output
		for (int groupIdx = 0, polyIdx = 0; groupIdx < numGroups; groupIdx++)
		{
			const int groupLen	= numElementsPerGroup[groupIdx];
			const int polyEnd	= polyIdx + groupLen;
			m_polyGroupIds.reserve(groupLen);
			m_polyGroupIds.setSize(0);
			
			for (; polyIdx < polyEnd; polyIdx++)
			{
				const int ppIdx = partitionedPolyIndices[polyIdx];
				const PolyMtlPair& pmPair = polyMtlPairs[ppIdx];
				m_polyGroupIds.pushBack(pmPair.m_vPolyId);
			}

			// Simplify this closed group
			simplifyBoundedGroup(m_polyGroupIds, materialId);
		}
	}

	//
	//	Collects the boundary lines for the given group of polygons

	void Simplifier::collectBoundaryEdges(const hkArray<VPolygonId>& vPolyIds, hkArray<EdgePolyPair>& boundaryEdgesOut)
	{
		// Get collections
		const VPolygonCollection& vPolys = m_vertexGeom.getPolygons();

		// Collect edges
		for (int k = vPolyIds.getSize() - 1; k >= 0; k--)
		{
			// Get the polygon
			const VPolygonId vPolyId	= vPolyIds[k];
			const VPolygon& vPoly		= vPolys.getPolygon(vPolyId);
			const int numBounds			= vPolys.getNumBoundaryEdges(vPolyId);

			// Add all its boundary edges
			VLineId prevLineId	= VLineId::invalid();
			bool isDegenerate	= true;
			for (int bi = numBounds - 1; bi >= 0; bi--)
			{
				const VEdgeId edgeId	= vPoly.getBoundaryEdgeId(bi);
				const VEdge& edge		= m_vertexGeom.getEdge(edgeId);
				EdgePolyPair& epPair	= boundaryEdgesOut.expandOne();

				// Check if the triangle is degenerate
				if		( !prevLineId.isValid() )			{	prevLineId = edge.getLineId();	}
				else if ( prevLineId != edge.getLineId() )	{	isDegenerate = false;			}

				epPair.m_edgeId		= edgeId;
				epPair.m_vPolyIdx	= vPolyId.value();
			}

			if ( isDegenerate )
			{
				boundaryEdgesOut.popBack(numBounds);
			}
		}

		// Sort by edgeId, to find the internal edges
		const int numEdgePairs = boundaryEdgesOut.getSize();
		hkSort(boundaryEdgesOut.begin(), numEdgePairs, EdgePolyPair::edgeLess);

		// Add end terminator
		EdgePolyPair prevEdge;
		{
			prevEdge.m_edgeId	= VEdgeId::invalid();
			prevEdge.m_vPolyIdx	= (hkUint32)-1;
			boundaryEdgesOut.pushBack(prevEdge);
		}

		// Remove internal edges
		{
			int prevGroupStartIdx = 0;
			int wi = 0;

			for (int k = 0; k <= numEdgePairs; k++)
			{
				const EdgePolyPair& crtPair = boundaryEdgesOut[k];
				if ( !EdgePolyPair::edgeEqual(prevEdge, crtPair) )
				{
					// New edge group started!
					switch ( k - prevGroupStartIdx )
					{
					case 0:		// Null group, ok
					case 2:		// Internal edge, ok
						break;

					case 1:		// Boundary edge, ok
						{
							boundaryEdgesOut[wi++] = boundaryEdgesOut[prevGroupStartIdx];
						}
						break;

					default:
						HK_ASSERT(0x514bfd1a, false);
					}

					prevGroupStartIdx	= k;
					prevEdge.m_edgeId	= crtPair.m_edgeId;
				}
			}

			// Resize the array to store just the boundary edges
			boundaryEdgesOut.setSize(wi);
		}
	}

	//
	//	Converts the given array of boundary lines to boundary planes

	void Simplifier::extractBoundaryPlanes(PlaneId supportPlaneId, const hkArray<EdgePolyPair>& boundaryEdgesIn, hkArray<PlaneId>& planeIdsOut)
	{
		// Alloc output
		const int numEdges = boundaryEdgesIn.getSize();
		planeIdsOut.setSize(numEdges);

		// Convert each edge
		for (int k = numEdges - 1; k >= 0; k--)
		{
			const VEdgeId boundaryEdgeId	= boundaryEdgesIn[k].m_edgeId;
			const VEdge& boundaryEdge		= m_vertexGeom.getEdge(boundaryEdgeId);
			const VLineId boundaryLineId	= boundaryEdge.getLineId();
			const VLine& boundaryLine		= m_vertexGeom.getLine(boundaryLineId);

			// Try line's plane 0
			PlaneId planeId = boundaryLine.getPlaneId<0>();
			if ( hkcdPlanarGeometryPrimitives::coplanarPlaneIds(planeId, supportPlaneId) )
			{
				// Plane 0 failed. Try plane 1
				planeId = boundaryLine.getPlaneId<1>();
				if ( hkcdPlanarGeometryPrimitives::coplanarPlaneIds(planeId, supportPlaneId) )
				{
					HK_ASSERT(0x1e86eeb0, false);	// Both planes failed, should not happen!
				}
			}

			// Save result
			planeIdsOut[k] = PlaneId(planeId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG));

			// Sanity check
#if ( ENABLE_INCLUSION_TESTS )
			{
				Plane planeA, planeB, planeC, planeD;
				const PlanesCollection* geomPlanes = m_planarGeom.getPlanesCollection();
				geomPlanes->getPlane(boundaryLine.getPlaneId<0>(), planeA);		geomPlanes->getPlane(boundaryLine.getPlaneId<1>(), planeB);
				geomPlanes->getPlane(supportPlaneId, planeC);					geomPlanes->getPlane(planeId, planeD);

				const hkBool32 linesMatch =	hkcdPlanarGeometryPredicates::edgeOnPlane(planeA, planeB, planeC) &&
											hkcdPlanarGeometryPredicates::edgeOnPlane(planeA, planeB, planeD);
				HK_ASSERT(0x1560de23, linesMatch);
			}
#endif
		}
	}

#if ( ENABLE_INCLUSION_TESTS )	

	//
	//	Classifies a vertex

	static hkcdPlanarGeometryPredicates::Orientation HK_CALL classifyVertex(const hkcdPlanarSolid* solid, NodeId nodeId, const Plane& planeA, const Plane& planeB, const Plane& planeC)
	{
		// Get the node
		const Node& node = solid->getNode(nodeId);
		if ( node.m_typeAndFlags == hkcdPlanarSolid::NODE_TYPE_IN )
		{
			return hkcdPlanarGeometryPredicates::BEHIND;
		}
		if ( node.m_typeAndFlags== hkcdPlanarSolid::NODE_TYPE_OUT )
		{
			return hkcdPlanarGeometryPredicates::IN_FRONT_OF;
		}

		// The node is internal
		Plane planeD;
		solid->getPlanesCollection()->getPlane(node.m_planeId, planeD);
		const hkcdPlanarGeometryPredicates::Orientation o = (hkcdPlanarGeometryPredicates::Orientation)(hkcdPlanarGeometryPredicates::orientation(planeA, planeB, planeC, planeD) & hkcdPlanarGeometryPredicates::ORIENTATION_MASK);

		switch ( o )
		{
		case hkcdPlanarGeometryPredicates::IN_FRONT_OF:
			HK_ASSERT(0x3cf92306, node.m_right.isValid());
			return classifyVertex(solid, node.m_right, planeA, planeB, planeC);

		case hkcdPlanarGeometryPredicates::BEHIND:
			HK_ASSERT(0x1dc02933, node.m_left.isValid());
			return classifyVertex(solid, node.m_left, planeA, planeB, planeC);

		default:
			HK_ASSERT(0x291782ac, o == hkcdPlanarGeometryPredicates::ON_PLANE);
			{
				if ( !node.m_left.isValid() )
				{
					return classifyVertex(solid, node.m_right, planeA, planeB, planeC);
				}
				if ( !node.m_right.isValid() )
				{
					return classifyVertex(solid, node.m_left, planeA, planeB, planeC);
				}

				const hkcdPlanarGeometryPredicates::Orientation oL = classifyVertex(solid, node.m_left, planeA, planeB, planeC);
				const hkcdPlanarGeometryPredicates::Orientation oR = classifyVertex(solid, node.m_right, planeA, planeB, planeC);
				return ( oL == oR ) ? oL : hkcdPlanarGeometryPredicates::ON_PLANE;
			}
		}
	}

	static hkcdPlanarGeometryPredicates::Orientation HK_CALL classifyPolygon(const hkcdPlanarGeometry* geom, PPolygonId pPolyId, const hkcdPlanarSolid* solid)
	{
		const PPolygonCollection& pPolys = geom->getPolygons();
		const hkcdPlanarGeometryPlanesCollection* planeCollection = geom->getPlanesCollection();

		// Classify each vertex of the polygon w.r.t the plane
		hkUint32 numBehind = 0, numInFront = 0, numCoplanar = 0;
		const PPolygon& polygon		= pPolys.getPolygon(pPolyId);
		const hkUint32 numPolyVerts = pPolys.getNumBoundaryPlanes(pPolyId);
		PlaneId prevBoundId			= polygon.getBoundaryPlaneId(numPolyVerts - 1);
		PlaneId polySupportId		= polygon.getSupportPlaneId();
		Plane s;					planeCollection->getPlane(polySupportId, s);
		Plane prevBound;			planeCollection->getPlane(prevBoundId, prevBound);

		for (hkUint32 crtVtx = 0; crtVtx < numPolyVerts; crtVtx++ )
		{
			const PlaneId crtBoundId	= polygon.getBoundaryPlaneId(crtVtx);
			Plane crtBound;				planeCollection->getPlane(crtBoundId, crtBound);

			// Try to get the orientation from cache
			const hkcdPlanarGeometryPredicates::Orientation ori = classifyVertex(solid, solid->getRootNodeId(), s, prevBound, crtBound);
			switch ( ori )
			{
			case hkcdPlanarGeometryPredicates::BEHIND:		numBehind++;	if ( numInFront )	{	return hkcdPlanarGeometryPredicates::INTERSECT;	}	break;
			case hkcdPlanarGeometryPredicates::IN_FRONT_OF:	numInFront++;	if ( numBehind )	{	return hkcdPlanarGeometryPredicates::INTERSECT;	}	break;
			case hkcdPlanarGeometryPredicates::ON_PLANE:	numCoplanar++;	break;
			default:	break;
			}

			prevBoundId = crtBoundId;
			prevBound	= crtBound;
		}

		// Return decision
		if ( numBehind && numInFront )	{	return hkcdPlanarGeometryPredicates::INTERSECT;		}
		if ( numInFront )				{	return hkcdPlanarGeometryPredicates::IN_FRONT_OF;	}
		if ( numBehind )				{	return hkcdPlanarGeometryPredicates::BEHIND;		}

		HK_ASSERT(0x4add6ee2, numCoplanar == numPolyVerts);
		return hkcdPlanarGeometryPredicates::ON_PLANE;
	}

	//
	//	Tests whether a set of polys is contained in the solid

	static bool HK_CALL polysInside(const hkcdPlanarGeometry* geom, const hkArray<PPolygonId>& pPolyIds, const hkcdPlanarSolid* solid)
	{
		for (int k = pPolyIds.getSize() - 1; k >= 0; k--)
		{
			const PPolygonId pPolyId = pPolyIds[k];
			const hkcdPlanarGeometryPredicates::Orientation o = classifyPolygon(geom, pPolyId, solid);

			if ( (o == hkcdPlanarGeometryPredicates::IN_FRONT_OF) || (o == hkcdPlanarGeometryPredicates::INTERSECT) )
			{
				return false;
			}
		}

		return true;
	}

#endif	// ENABLE_INCLUSION_TESTS

	//
	//	Builds a BSP from the boundary of the surface represented by a group of polygons

	void Simplifier::buildSimplifyingSolidForBoundedGroup(hkcdPlanarSolid& solidOut, const hkArray<VPolygonId>& vPolyIds, hkLocalArray<PPolygonId>& pPolyIds, PlaneId& supportPlaneId, hkUint32 materialId)
	{
		const VPolygonCollection& vPolys = m_vertexGeom.getPolygons();
		const PPolygonCollection& pPolys = m_planarGeom.getPolygons();

		const int numPolys = vPolyIds.getSize();

		// Collect boundary edges
		hkArray<EdgePolyPair>& boundaryEdges = m_edgePolyPairs;
		boundaryEdges.setSize(0);
		collectBoundaryEdges(vPolyIds, boundaryEdges);

		// Get collections and alloc temporary data		
		hkLocalArray<PPolygonId> tempPolyIds(numPolys);	tempPolyIds.setSize(numPolys);

		// Collect planar polygon Ids to be simplified		
		for (int k = numPolys - 1; k >= 0; k--)
		{
			const VPolygonId vPolyId	= vPolyIds[k];
			const VPolygon& vPoly		= vPolys.getPolygon(vPolyId);
			const PPolygonId pPolyId	(vPoly.getUserData());
			const PPolygon& pPoly		= pPolys.getPolygon(pPolyId);

			HK_ASSERT(0x370f6a08, !supportPlaneId.isValid() || (supportPlaneId == pPoly.getSupportPlaneId()));
			HK_ASSERT(0x5791d6b5, materialId == pPoly.getMaterialId());
			pPolyIds[k]		= pPolyId;
			supportPlaneId	= pPoly.getSupportPlaneId();
		}

		// Convert boundary lines to boundary planes
		{
			m_boundaryPlanes.setSize(0);
			extractBoundaryPlanes(supportPlaneId, boundaryEdges, m_boundaryPlanes);

			// Sort and remove duplicates
			const int numRawPlanes = m_boundaryPlanes.getSize();
			hkSort(reinterpret_cast<PlaneId::Type*>(m_boundaryPlanes.begin()), numRawPlanes);
			const int numPlanes = hkAlgorithm::removeDuplicatesFromSortedList(m_boundaryPlanes.begin(), numRawPlanes);
			m_boundaryPlanes.setSize(numPlanes);

#if ( ENABLE_INCLUSION_TESTS )
			hkSort(reinterpret_cast<PlaneId::Type*>(m_boundaryPlanes.begin()), numPlanes, planesLess);
			for (int k = 0; k < numPlanes - 1; k++)
			{
				HK_ASSERT(0x1973fd00, !planesEqual(m_boundaryPlanes[k], m_boundaryPlanes[k + 1]));
			}
#endif
			HK_ASSERT(0x7c325706, !m_boundaryPlanes.isEmpty());
		}

		// Convex decompose the boundary
		{
			// Clear temporary geometry data
			m_workingGeom.accessPolygons().clear();
			solidOut.clear();
			m_workingRegions.accessCells()->accessPolygons().clear();

			// Copy the polygons to be simplified
			m_workingGeom.appendGeometryPolygons(m_planarGeom, pPolyIds, false, tempPolyIds);

			// Convex decompose the 2D boundary with a BSP tree
			solidOut.buildTree2D(m_workingGeom, m_rng, m_boundaryPlanes, tempPolyIds, m_workingGeom.accessDebugger());
			solidOut.collapseIdenticalLabels();
		}
	}

	//
	//	Simplifies the given group of polygons. They all share the same material, and are enclosed by the same boundary curve

	void Simplifier::simplifyBoundedGroup(const hkArray<VPolygonId>& vPolyIds, hkUint32 materialId)
	{
		const int numPolys = vPolyIds.getSize();
		if ( numPolys < 2 )
		{
			return;	// Nothing to simplify!
		}

		hkLocalArray<PPolygonId> pPolyIds(numPolys);	pPolyIds.setSize(numPolys);
		PlaneId supportPlaneId = PlaneId::invalid();

		// Convex decompose the boundary
		int numConvexParts = 0;
		{			
			buildSimplifyingSolidForBoundedGroup(m_workingSolid, vPolyIds, pPolyIds, supportPlaneId, materialId);
			
			numConvexParts = m_workingSolid.computeNumNodesWithLabel(hkcdPlanarSolid::NODE_TYPE_IN);
			if ( numPolys <= numConvexParts )
			{
				return;	// The convex decomposition is worse than the original, ignore!
			}
		}

		hkLocalArray<PPolygonId> tempPolyIds(numPolys);	tempPolyIds.setSize(numPolys);

		// We are simplifying the polygons.
		// Create proper polygons from all BSP nodes marked as inside
		{
			m_workingRegions.setSupportPlaneId(supportPlaneId);
			m_workingRegions.buildFromSolid(&m_workingSolid);

			const NodeCollection* bspNodes	= m_workingSolid.getNodes();
			hkcdPlanarGeometry* regionCells	= m_workingRegions.accessCells();

			// Copy the polygons from the working regions into the source geometry
			// We're only interested in nodes marked as IN within the BSP tree
			{
				// Collect the source polygons
				int partIdx = 0;
				tempPolyIds.setSize(numConvexParts);
				for (int ni = bspNodes->getCapacity() - 1; ni >= 0; ni--)
				{
					const Node& bspNode = (*bspNodes)[NodeId(ni)];
					if ( bspNode.isAllocated() && (bspNode.m_typeAndFlags == hkcdPlanarSolid::NODE_TYPE_IN) )
					{
						// Inside node, corresponds to a convex cell (i.e. polygon) of the convex decomposition.
						// Locate its cell
						const PPolygonId srcPolyId	(bspNode.m_data);
						PPolygon& srcPoly			= regionCells->accessPolygon(srcPolyId);

						srcPoly.setMaterialId(materialId);
						tempPolyIds[partIdx++] = srcPolyId;
					}
				}

				// Copy them
				HK_ASSERT(0x969f905, partIdx == numConvexParts);
			}

			// Debug. Make sure the simplified surface is identical to the original surface
#if ( ENABLE_INCLUSION_TESTS )
			{
				// Test whether the original is contained by the simplification
				const bool AinB = polysInside(&m_planarGeom, pPolyIds, &m_workingSolid);
				HK_ASSERT(0x9add26a, AinB);

				// Copy original polys again in our temp geometry
				hkLocalArray<PPolygonId> dbgPolyIds(numPolys);
				dbgPolyIds.setSize(numPolys);
				m_workingGeom.appendGeometryPolygons(m_planarGeom, pPolyIds, false, dbgPolyIds);

				// Get their planes
				hkArray<PlaneId> dbgPlaneIds;
				m_planarGeom.getAllPolygonsPlanes(pPolyIds, dbgPlaneIds, true);
				hkSort(reinterpret_cast<PlaneId::Type*>(dbgPlaneIds.begin()), dbgPlaneIds.getSize(), planesLess);
				dbgPlaneIds.setSize(hkAlgorithm::removeDuplicatesFromSortedList(dbgPlaneIds.begin(), dbgPlaneIds.getSize(), planesEqual));
				{	const int ii = dbgPlaneIds.indexOf(supportPlaneId); if ( ii >= 0 ) dbgPlaneIds.removeAt(ii);	}
				{	const int ii = dbgPlaneIds.indexOf(hkcdPlanarGeometryPrimitives::getOppositePlaneId(supportPlaneId)); if ( ii >= 0 ) dbgPlaneIds.removeAt(ii);	}

 				m_workingSolid.clear();
				m_workingSolid.buildTree2D(m_workingGeom, m_rng, dbgPlaneIds, dbgPolyIds, m_workingGeom.accessDebugger());

				// Test whether the simplification is contained in the original
				bool BinA = polysInside(&regionCells, tempPolyIds, &m_workingSolid);
				HK_ASSERT(0xa7a0c01, BinA);
			}
#endif

			// Perform the change
#if ( ENABLE_SIMPLIFICATION )
			m_planarGeom.removePolygons(pPolyIds);
			m_planarGeom.appendGeometryPolygons(*regionCells, tempPolyIds, false, pPolyIds);
#endif
		}
	}
}

//
//	Performs the simplification. It expects a hkcdVertexGeometry created from
//	the given planar geometry.

void HK_CALL hkcdPlanarGeometrySimplifier::execute(hkcdPlanarGeometry* planarGeometry, const hkcdVertexGeometry* vertexGeometry)
{
	hkndPlanarSimplificationImpl::Simplifier s(planarGeometry, vertexGeometry);
	s.execute();
}

//
//	Builds a BSP from the boundary of the surface represented by a group of polygons

void HK_CALL hkcdPlanarGeometrySimplifier::buildSimplifyingSolidForPolygonGroup(hkcdPlanarGeometry* planarGeometry, const hkcdVertexGeometry* vertexGeometry,
																				const hkArray<VPolygonId>& vPolyIds, hkcdPlanarSolid* solidOut)
{
	hkndPlanarSimplificationImpl::Simplifier s(planarGeometry, vertexGeometry);

	const int numPolys = vPolyIds.getSize();
	hkLocalArray<PPolygonId> pPolyIds(numPolys);	pPolyIds.setSize(numPolys);
	PlaneId supportPlaneId = PlaneId::invalid();

	s.buildSimplifyingSolidForBoundedGroup(*solidOut, vPolyIds, pPolyIds, supportPlaneId, 0);
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
