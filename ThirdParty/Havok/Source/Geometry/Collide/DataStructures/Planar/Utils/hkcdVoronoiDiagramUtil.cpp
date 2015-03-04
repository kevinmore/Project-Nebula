/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/DataStructures/Planar/Geometry/hkcdPlanarGeometry.h>
#include <Geometry/Collide/DataStructures/Planar/ConvexCellsTree/hkcdConvexCellsTree3D.h>
#include <Geometry/Collide/DataStructures/Planar/Utils/hkcdVoronoiDiagramUtil.h>

#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Container/String/hkStringBuf.h>

#define ENABLE_DEBUG_PLANAR_CUTS	(0)

//
//	Utility functions

namespace hkcdVoronoiDiagramImpl
{
	typedef hkcdConvexCellsTree3D::Cell				Cell;
	typedef hkcdConvexCellsTree3D::CellId				CellId;
	typedef hkcdPlanarGeometry::Plane				Plane;
	typedef hkcdPlanarGeometry::PlaneId				PlaneId;
	typedef hkcdPlanarGeometry::Polygon				Polygon;
	typedef hkcdPlanarGeometry::PolygonId			PolygonId;
	typedef hkcdVoronoiDiagramUtil::SitesProvider	SitesProvider;
	typedef hkcdPlanarSolid::Node			Node;
	typedef hkcdPlanarSolid::NodeId			NodeId;

	/// A Voronoi cell
	struct VoronoiCell
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkcdVoronoiDiagramImpl::VoronoiCell);

		/// Distance based comparator
		static HK_FORCE_INLINE bool HK_CALL less(const VoronoiCell& vcA, const VoronoiCell& vcB)	{	return vcA.m_distSq < vcB.m_distSq;	}

		hkIntVector m_pos;	///< The cell's quantized position.
		hkInt64 m_distSq;	///< The cached squared distance to the current Voronoi cell candidate
		CellId m_id;		///< The cell's Id in the convex cells tree
	};

	/// The Voronoi diagram builder
	class VoronoiDiagram : public hkcdPlanarEntity
	{
		public:

			HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY);

			typedef hkcdPlanarCsgOperand	CsgOperand;

		public:

			/// Constructor
			HK_FORCE_INLINE VoronoiDiagram(CsgOperand* operand, hkcdPlanarEntityDebugger* debugger);

			/// Builds the cells
			void buildCells(SitesProvider* voronoiSitesProvider);

			/// Converts the Voronoi cells to geometry
			void extractGeometry(hkGeometry& geomOut) const HK_OVERRIDE;

			/// Removes all intermediary data
			void extractVoronoiCellIds(hkArray<CellId>& voronoiCellIdsOut);

		protected:

			/// Locates a cell by position
			HK_FORCE_INLINE int findCellAt(hkIntVectorParameter iPos) const;

			/// Sorts all cells by distance to the given position
			HK_FORCE_INLINE void sortCells(hkIntVectorParameter iPos);

			/// Computes the sepparating plane between two cell sites. It will also return a unique plane id, i.e. will search for an already existing plane
			HK_FORCE_INLINE PlaneId getSepparatingPlaneId(SitesProvider* vSitesProvider, const int siteIndexA, const int siteIndexB);

			/// Creates a new cell as the union of all given cell pieces
			HK_FORCE_INLINE CellId buildUnionOfCells(const hkArray<CellId>& piecesIn, hkArray<PlaneId>& boundaryPlanesOut);

			/// Tests whether the given point is outside the given cell radius
			hkInt64 computeRadius(hkIntVectorParameter vCenter, const hkArray<CellId>& piecesIn) const;

		protected:

			CsgOperand* m_operand;			///< The geometry that will store the planes
			hkArray<VoronoiCell> m_cells;	///< The Voronoi cells
			CellId m_worldCellId;			///< The world boundary cell
	};

	//
	//	Constructor

	HK_FORCE_INLINE VoronoiDiagram::VoronoiDiagram(CsgOperand* operand, hkcdPlanarEntityDebugger* debugger)
	:	hkcdPlanarEntity(debugger)
	,	m_operand(operand)
	{
		hkcdConvexCellsTree3D* tree = operand->accessRegions();
		m_worldCellId	= tree->createBoundaryCell();
	}

	//
	//	Locates a cell by position

	HK_FORCE_INLINE int VoronoiDiagram::findCellAt(hkIntVectorParameter iPos) const
	{
		for (int i = m_cells.getSize() - 1; i >= 0; i--)
		{
			const hkIntVector v				= m_cells[i].m_pos;
			const hkVector4Comparison cmp	= v.compareEqualS32(iPos).horizontalAnd<3>();
			if ( cmp.anyIsSet() )
			{
				return i;
			}
		}

		return -1;
	}

	//
	//	Sorts all cells by distance to the given position

	HK_FORCE_INLINE void VoronoiDiagram::sortCells(hkIntVectorParameter iPos)
	{
		// Compute distances first
		const int numCells = m_cells.getSize();
		for (int i = numCells - 1; i >= 0; i--)
		{
			VoronoiCell& vCell	= m_cells[i];
			hkIntVector v;		v.setSubS32(vCell.m_pos, iPos);
			vCell.m_distSq		= v.dot<3>(v);
		}

		hkSort(m_cells.begin(), numCells, VoronoiCell::less);
	}

	//
	//	Computes the sepparating plane between two cell sites. It will also return a unique plane id, i.e. will search for an already existing plane

	HK_FORCE_INLINE PlaneId VoronoiDiagram::getSepparatingPlaneId(SitesProvider* vSitesProvider, const int siteIndexA, const int siteIndexB)
	{
		// Compute the plane
		Plane posP;
		vSitesProvider->computePlane(siteIndexA, siteIndexB, posP);

		// Try to weld it to the planes already in the mesh
		Plane negP;	negP.setOpposite(posP);
		hkcdPlanarGeometryPlanesCollection* planesCol = m_operand->accessGeometry()->accessPlanesCollection();
		const int numPlanes = planesCol->getNumPlanes();

		for (int k = numPlanes - 1; k >= 0; k--)
		{
			// As the planes are fully reduced, they must have equal components to match
			const PlaneId planeId	(k);
			Plane candidate;		planesCol->getPlane(planeId, candidate);

			if ( candidate.isEqual(posP) )	{	return PlaneId(k);	}
			if ( candidate.isEqual(negP) )	{	return PlaneId(k | hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);	}
		}

		vSitesProvider->addPlane(siteIndexA, siteIndexB, posP);
		return planesCol->addPlane(posP);
	}

	//
	//	Comparator for boundary planes

	static HK_FORCE_INLINE bool HK_CALL boundsLess(const PlaneId::Type& pidA, const PlaneId::Type& pidB)
	{
		const int pa = pidA & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
		const int pb = pidB & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
		const int fa = pidA & hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG;
		const int fb = pidB & hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG;
		return (pa < pb) || ((pa == pb) && (fa <fb));
	}

	//
	//	Computes the union of cells

	HK_FORCE_INLINE CellId VoronoiDiagram::buildUnionOfCells(const hkArray<CellId>& piecesIn, hkArray<PlaneId>& boundsOut)
	{
		// Early-out for just one cell
		const int numPieces = piecesIn.getSize();
		if ( numPieces == 1 )
		{
			return piecesIn[0];
		}

		// Collect boundary planes from all cells
		boundsOut.setSize(0);
		for (int pi = numPieces - 1; pi >= 0; pi--)
		{
			const CellId cellId		= piecesIn[pi];
			const Cell& cell		= m_operand->getRegions()->getCell(cellId);
			const int numCellBounds	= m_operand->getRegions()->getNumBoundaryPlanes(cellId);

			for (int bi = numCellBounds - 1; bi >= 0; bi--)
			{
				const PolygonId polyId	= cell.getBoundaryPolygonId(bi);
				const Polygon& poly		= m_operand->getGeometry()->getPolygon(polyId);
				boundsOut.pushBack(poly.getSupportPlaneId());
			}
		}

		// Sort boundary planes by Id and remove all duplicates
		int numBounds = boundsOut.getSize();
		hkSort(reinterpret_cast<PlaneId::Type*>(boundsOut.begin()), numBounds, boundsLess);
		numBounds = hkAlgorithm::removeDuplicatesFromSortedList(boundsOut.begin(), numBounds);
		boundsOut.setSize(numBounds);

		// Remove pairs of opposite planes, they are internal
		for (int k = numBounds - 2; k >= 0; k--)
		{
			const int pidA = boundsOut[k + 1].value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
			const int pidB = boundsOut[k].value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
			if ( pidA == pidB )
			{
				boundsOut.removeAt(k + 1);
				boundsOut.removeAt(k);
				k--;
			}
		}

		// Cut using the bounds
		numBounds = boundsOut.getSize();
		CellId retCellId = m_worldCellId;
		for (int k = numBounds - 1; k >= 0; k--)
		{
			CellId insideCellId, outsideCellId;
			m_operand->accessRegions()->splitCell(HK_NULL, retCellId, boundsOut[k], insideCellId, outsideCellId);
			if ( outsideCellId.isValid() )
			{
				// We had a split, continue working on the inside
				retCellId = insideCellId;
			}
		}

		return retCellId;
	}

	//
	//	Compares 2 hkIntVectors by x, y, z in that order

	HK_FORCE_INLINE hkBool32 HK_CALL vectorLess(hkIntVectorParameter vA, hkIntVectorParameter vB)
	{
		const hkVector4Comparison cmpL = vA.compareLessThanS32(vB);
		const hkVector4Comparison cmpE = vA.compareEqualS32(vB);

		const int code = ((cmpL.getMask() << 2) & 0x1C) | (cmpE.getMask() & 3);
		return (0xFAF8FAF0 >> code) & 1;
	}

	HK_FORCE_INLINE hkBool32 HK_CALL vectorEq(hkIntVectorParameter vA, hkIntVectorParameter vB)
	{
		return vA.compareEqualS32(vB).horizontalAnd<3>().anyIsSet();
	}

	//
	//	Computes the vertices of a cell

	static HK_FORCE_INLINE void HK_CALL computeCellsVertices(const hkcdConvexCellsTree3D* cellTree, const CellId* cellIds, int numCells, hkArray<hkIntVector>& cellVertsOut)
	{
		const hkcdPlanarGeometry* geom = cellTree->getGeometry();

		cellVertsOut.setSize(0);
		for (int ci = numCells - 1; ci >= 0; ci--)
		{
			const CellId cellId	= cellIds[ci];
			const Cell& cell	= cellTree->getCell(cellId);
			const int numFaces	= cellTree->getNumBoundaryPlanes(cellId);

			// Get polygons and gather all vertices, we'll have a lot of them shared.
			for (int fi = 0; fi < numFaces; fi++)
			{
				// Get the face
				const PolygonId faceId		= cell.getBoundaryPolygonId(fi);
				const Polygon& face			= geom->getPolygon(faceId);
				const int numEdges			= geom->getNumBoundaryPlanes(faceId);
				const hkUint32 supportId	= face.getSupportPlaneId().value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);

				// Add the vertices
				hkUint32 prevPlaneId	= face.getBoundaryPlaneId(numEdges - 1).value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
				hkIntVector* vbPtr		= cellVertsOut.expandBy(numEdges);
				for (int crt = 0; crt < numEdges; crt++)
				{
					const hkUint32 crtPlaneId	= face.getBoundaryPlaneId(crt).value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
					hkIntVector v;				v.set(supportId, prevPlaneId, crtPlaneId, 0);
					vbPtr[crt].setSortS32<3, HK_SORT_ASCENDING>(v);
					prevPlaneId = crtPlaneId;
				}
			}
		}

		// Sort the vertices by their ids and remove duplicates
		hkSort(cellVertsOut.begin(), cellVertsOut.getSize(), vectorLess);
		const int numVerts = hkAlgorithm::removeDuplicatesFromSortedList(cellVertsOut.begin(), cellVertsOut.getSize(), vectorEq);
		cellVertsOut.setSize(numVerts);
	}

	//
	//	Tests whether the given point is outside the given cell radius

	hkInt64 VoronoiDiagram::computeRadius(hkIntVectorParameter vCenter, const hkArray<CellId>& piecesIn) const
	{
		hkInplaceArray<hkIntVector, 128> vtxIds;
		computeCellsVertices(m_operand->getRegions(), piecesIn.begin(), piecesIn.getSize(), vtxIds);

		// Approximate the vertices' position
		Plane planes[3];
		hkInt64 maxRadiusSq = 0;
		const hkcdPlanarGeometryPlanesCollection* planesCol = m_operand->getGeometry()->getPlanesCollection();
		for (int vi = vtxIds.getSize() - 1; vi >= 0; vi--)
		{
			// Estimate vertex
			hkIntVector vtx = vtxIds[vi];
			planesCol->getPlane(PlaneId(vtx.getComponent<0>()), planes[0]);
			planesCol->getPlane(PlaneId(vtx.getComponent<1>()), planes[1]);
			planesCol->getPlane(PlaneId(vtx.getComponent<2>()), planes[2]);
			hkcdPlanarGeometryPredicates::approximateIntersection(planes, vtx);

			// Overestimate the radius
			vtx.setSubS32(vtx, vCenter);
			const hkInt64 rMinSq	= vtx.dot<3>(vtx);
			const hkInt64 rMaxSq	= rMinSq + (hkInt64)(vtx.horizontalAddS32<3>() + 3);
			const hkInt64 rSq		= hkMath::max2(rMinSq, rMaxSq);
			maxRadiusSq	= hkMath::max2(maxRadiusSq, rSq);
		}

		// Return the maximum radius
		return maxRadiusSq;
	}

	//
	//	Builds the cells

	void VoronoiDiagram::buildCells(SitesProvider* vSitesProvider)
	{
		const hkcdPlanarGeometryPlanesCollection* planesCol = m_operand->getGeometry()->getPlanesCollection();
		const int numVerts = vSitesProvider->getNumSites();
		if ( !numVerts )
		{
			return;
		}

		// Add the first vertex and the entire world cell
		{
			VoronoiCell& vCell	= m_cells.expandOne();
			hkVector4 fPos;		vSitesProvider->getSitePosition(0, fPos);

			vCell.m_id = m_worldCellId;
			planesCol->convertWorldPosition(fPos, vCell.m_pos);
			vCell.m_pos.setComponent<3>(0);
		}

		// Add one vertex at a time, updating the existing cells
		hkArray<CellId> newCellPieces;
		hkArray<PlaneId> newCellBounds;
		for (int vi = 1; vi < numVerts; vi++)
		{
			// Create a new empty cell for this vertex
			VoronoiCell newCell;
			{
				hkVector4 fPos;	vSitesProvider->getSitePosition(vi, fPos);

				newCell.m_id = CellId::invalid();
				planesCol->convertWorldPosition(fPos, newCell.m_pos);
				newCell.m_pos.setComponent<3>(vi);
			}

			// Weld cell with others, ignore if it has the same position
			if ( findCellAt(newCell.m_pos) >= 0 )
			{
				// We already have this cell, ignore!
				continue;
			}

			// Sort all cells by distance to the new cell
			sortCells(newCell.m_pos);

			// For each existing cell, try to split with the sepparating plane
			const int numCells = m_cells.getSize();
			newCellPieces.setSize(0);
			hkInt64 newCellRadiusSq = -1L;
			hkArray<PlaneId> sepPlaneIds;
			for (int ci = 0; ci < numCells; ci++)
			{
				VoronoiCell& vCell			= m_cells[ci];
				const PlaneId splitPlaneId	= getSepparatingPlaneId(vSitesProvider, vi, vCell.m_pos.getComponent<3>());
				sepPlaneIds.pushBack(splitPlaneId);

				// Try to split the cell with the split plane
				CellId insideCellId		= CellId::invalid();
				CellId outsideCellId	= CellId::invalid();
				m_operand->accessRegions()->splitCell(HK_NULL, vCell.m_id, splitPlaneId, insideCellId, outsideCellId);

				// See what to do
				if ( outsideCellId.isValid() )
				{
					// The outside cell is non-empty
					if ( insideCellId.isValid() )
					{
						// The cell ci had to be split. Associate the outside bit with ci and the inside bit with the new candidate
						vCell.m_id		= outsideCellId;
						newCellRadiusSq = -1L;
						newCellPieces.pushBack(insideCellId);
					}
					else
					{
						// The original cell is completely outside the sepparating plane, test for early exit!
						if ( newCellRadiusSq < 0 )
						{
							newCellRadiusSq = computeRadius(newCell.m_pos, newCellPieces);
						}
						if ( vCell.m_distSq > (newCellRadiusSq << 2L) )
						{
							// Can stop, all other cells are outside the bounding sphere of the new cell!
							break;
						}
					}
				}
				else
				{
					// There's no outside cell, cell ci is fully contained inside the sepparating plane.
					// This should not happen, do not add the new site!
					HK_ASSERT(0x30837253, newCellPieces.getSize() == 0);
					break;
				}
			}

			// At this point we can add the new cell
			if ( newCellPieces.getSize() )
			{
				newCell.m_id = buildUnionOfCells(newCellPieces, newCellBounds);
				m_cells.pushBack(newCell);
			}
		}
	}

	//
	//	Converts the Voronoi cells to geometry

	void VoronoiDiagram::extractGeometry(hkGeometry& geomOut) const
	{
		const hkcdConvexCellsCollection* treeCells = m_operand->getRegions()->getCells();
		for (CellId cellId = treeCells->getFirstCellId(); cellId.isValid(); cellId = treeCells->getNextCellId(cellId))
		{
			hkGeometry g;
			m_operand->getRegions()->extractCellGeometry(cellId, g);
			geomOut.appendGeometry(g);
		}
	}

	//
	//	Removes all intermediary data

	void VoronoiDiagram::extractVoronoiCellIds(hkArray<CellId>& voronoiCellIdsOut)
	{
		// Remove all cells that are not among the Voronoi cells
		hkcdConvexCellsCollection* treeCells = m_operand->accessRegions()->accessCells();
		const int numSites	= m_cells.getSize();
		CellId cellId		= treeCells->getFirstCellId();

		while ( cellId.isValid() )
		{
			// See if we've got a site with this Id
			int k = numSites - 1;
			for (; k >= 0; k--)
			{
				const VoronoiCell& vCell = m_cells[k];
				if ( vCell.m_id == cellId )
				{
					break;
				}
			}

			const CellId nextCellId = treeCells->getNextCellId(cellId);
			if ( k < 0 )
			{
				// This is a temporary cell, delete!
				treeCells->freeCell(cellId);
			}
			else
			{
				// This is a Voronoi cell, store in the output array
				const VoronoiCell& vCell = m_cells[k];
				voronoiCellIdsOut[vCell.m_pos.getComponent<3>()] = cellId;
			}
			cellId = nextCellId;
		}
	}
	
}

//
//	Utility functions

namespace hkcdVoronoiDiagramImpl
{
	// Type shortcuts
	typedef hkcdPlanarGeometry::PlaneId			PlaneId;
	typedef hkcdPlanarGeometry::PolygonId		PolygonId;
	typedef hkcdPlanarSolid::NodeId		NodeId;
	typedef hkcdPlanarSolid::Node		Node;

	/// Stack entry
	struct StackEntry
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkcdVoronoiDiagramImpl::StackEntry);

		PlaneId* m_planeIds;		///< The array of plane Ids that can be used to classify the cells
		int* m_cellIds;				///< The array of cell Ids that still need to be classified
		int m_numPlaneIds;			///< The number of plane Ids
		int m_numCellIds;			///< The number of cell Ids
		int m_isLeftChild;			///< True if this child is the left child of the parent node
		NodeId m_parentNodeId;		///< The parent node that pushed this entry
	};

	/// The stack
	class Stack
	{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkcdVoronoiDiagramImpl::Stack);

		enum { LEFT_CHILD_FLAG	= 0x80000000, };

	public:

		HK_FORCE_INLINE void pop(StackEntry& entryOut)
		{
			const int n = m_storage.getSize();
			entryOut.m_isLeftChild		= m_storage[n - 2] & LEFT_CHILD_FLAG;
			entryOut.m_parentNodeId		= NodeId(m_storage[n - 1]);
			entryOut.m_numCellIds		= m_storage[n - 2] & (~LEFT_CHILD_FLAG);
			entryOut.m_numPlaneIds		= m_storage[n - 3];
			entryOut.m_cellIds			= &m_storage[n - 3 - entryOut.m_numCellIds];
			entryOut.m_planeIds			= reinterpret_cast<PlaneId*>(&m_storage[n - 3 - entryOut.m_numCellIds - entryOut.m_numPlaneIds]);
			m_storage.setSize(n - 3 - entryOut.m_numCellIds - entryOut.m_numPlaneIds);
		}

		HK_FORCE_INLINE void push(const StackEntry& entryIn)
		{
			const int n = m_storage.getSize() + entryIn.m_numPlaneIds + entryIn.m_numCellIds + 3;
			m_storage.setSize(n);
			hkString::memCpy4(&m_storage[n - 3 - entryIn.m_numCellIds - entryIn.m_numPlaneIds], entryIn.m_planeIds, entryIn.m_numPlaneIds);
			hkString::memCpy4(&m_storage[n - 3 - entryIn.m_numCellIds], entryIn.m_cellIds, entryIn.m_numCellIds);
			m_storage[n - 3]	= entryIn.m_numPlaneIds;
			m_storage[n - 2]	= entryIn.m_numCellIds | (entryIn.m_isLeftChild ? (int)LEFT_CHILD_FLAG : 0);
			m_storage[n - 1]	= entryIn.m_parentNodeId.valueUnchecked();
		}

		HK_FORCE_INLINE bool isEmpty() const
		{
			return !m_storage.getSize();
		}

	protected:

		hkArray<int> m_storage;	///< The stack storage
	};

	//
	//	A cell as a collection of vertices

	struct VtxCell
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkcdVoronoiDiagramImpl::VtxCell);

		hkArray<hkIntVector> m_verts;
		CellId m_id;
		CellId m_origId;
	};

	//
	//	Classifies a cell w.r.t. a plane

	static hkcdPlanarGeometryPredicates::Orientation HK_CALL approxClassify(const hkcdPlanarGeometry* geom, const VtxCell& vtxCell, PlaneId planeId)
	{
		// Get polygon and splitting plane
		const hkcdPlanarGeometryPlanesCollection* geomPlanes = geom->getPlanesCollection();
		Plane splitPlane;		geomPlanes->getPlane(planeId, splitPlane);

		// Classify each vertex of the polygon w.r.t the plane
		hkUint32 numBehind = 0, numInFront = 0, numCoplanar = 0;
		const int numVerts = vtxCell.m_verts.getSize();

		Plane planes[3];
		for (int i = 0; i < numVerts; i++)
		{
			const hkIntVector iv = vtxCell.m_verts[i];
			geomPlanes->getPlane(PlaneId(iv.getComponent<0>()), planes[0]);
			geomPlanes->getPlane(PlaneId(iv.getComponent<1>()), planes[1]);
			geomPlanes->getPlane(PlaneId(iv.getComponent<2>()), planes[2]);

			const hkcdPlanarGeometryPredicates::Orientation ori = hkcdPlanarGeometryPredicates::approximateOrientation(planes[0], planes[1], planes[2], splitPlane);
			switch ( ori )
			{
			case hkcdPlanarGeometryPredicates::BEHIND:		numBehind++;	if ( numInFront )	{	return hkcdPlanarGeometryPredicates::INTERSECT;	}	break;
			case hkcdPlanarGeometryPredicates::IN_FRONT_OF:	numInFront++;	if ( numBehind )	{	return hkcdPlanarGeometryPredicates::INTERSECT;	}	break;
			case hkcdPlanarGeometryPredicates::ON_PLANE:	numCoplanar++;	break;
			default:	break;
			}
		}

		// Return decision
		if ( numBehind && numInFront )	{	return hkcdPlanarGeometryPredicates::INTERSECT;		}
		if ( numInFront )				{	return hkcdPlanarGeometryPredicates::IN_FRONT_OF;	}
		if ( numBehind )				{	return hkcdPlanarGeometryPredicates::BEHIND;		}

		// Should never get here, unless the cell is degenerate!
		HK_ASSERT(0x566cc7e1, false);
		return hkcdPlanarGeometryPredicates::ON_PLANE;
	}

	//
	//	Selects a splitting plane from the given list of polygons

	static int HK_CALL pickSplittingPlane(	const hkcdPlanarGeometry* geom, const hkArray<VtxCell>& vtxCells,
											const PlaneId* HK_RESTRICT planeIds, int numPlanes,
											const int* HK_RESTRICT cellIds, int numCells)
	{
		// Initialize our best estimate for the splitting plane
		int bestCost		= -0x7FFFFFFF;
		int bestPlaneIdx	= -1;

		// Try a fixed number of random planes
		for (int crtTry = 0; crtTry < numPlanes; crtTry++)
		{
			const PlaneId splitPlaneId	= planeIds[crtTry];

			// Clear statistics
			int numInFront	= 0;
			int numBehind	= 0;
			int numSplit	= 0;

			// Test all other polygons against the current splitting plane
			for (int ci = 0; ci < numCells; ci++)
			{
				const int cellIdx		= cellIds[ci];
				const VtxCell& vCell	= vtxCells[cellIdx];
				const hkcdPlanarGeometryPredicates::Orientation ori = approxClassify(geom, vCell, splitPlaneId);

				switch ( ori )
				{
				case hkcdPlanarGeometryPredicates::BEHIND:		numBehind++;	break;
				case hkcdPlanarGeometryPredicates::IN_FRONT_OF:	numInFront++;	break;
				case hkcdPlanarGeometryPredicates::INTERSECT:	numSplit++;		break;
				default:	break;
				}
			}

			// Compute heuristic: h(splitPlane) = front - wSplit * split, wSplit = 8
			const int heuristic = numInFront - (numSplit << 3);
			if ( heuristic > bestCost )
			{
				bestCost = heuristic;
				bestPlaneIdx = crtTry;
			}
		}

		// Return our best estimate
		HK_ASSERT(0x31536947, bestPlaneIdx >= 0);
		return bestPlaneIdx;
	}

	//
	//	Collects all the planes of the given cells

	static void HK_CALL getAllCellsPlanes(const hkcdConvexCellsTree3D* cellsTree, const hkArray<int>& cellIndicesIn, const hkArray<VtxCell>& vCellsIn, hkArray<PlaneId>& planesOut)
	{
		// Reset planes
		planesOut.setSize(0);
		const hkcdPlanarGeometry* mesh = cellsTree->getGeometry();

		const int numCells = cellIndicesIn.getSize();
		for (int k = numCells - 1; k >= 0; k--)
		{
			const VtxCell& vCell	= vCellsIn[cellIndicesIn[k]];
			const CellId cellId		= vCell.m_id;
			const Cell& cell		= cellsTree->getCell(cellId);
			const int numBounds		= cellsTree->getNumBoundaryPlanes(cellId);
			PlaneId* ptr			= planesOut.expandBy(numBounds);

			for (int bi = numBounds - 1; bi >= 0; bi--)
			{
				const PolygonId polyId	= cell.getBoundaryPolygonId(bi);
				const Polygon& poly		= mesh->getPolygon(polyId);
				const PlaneId planeId	= poly.getSupportPlaneId();

				ptr[bi] = PlaneId(planeId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG));
			}
		}

		// Sort all plane Ids and remove duplicates
		const int numPlanes = planesOut.getSize();
		hkSort(reinterpret_cast<PlaneId::Type*>(planesOut.begin()), numPlanes);
		const int numUniquePlanes = hkAlgorithm::removeDuplicatesFromSortedList(planesOut.begin(), numPlanes);
		planesOut.setSize(numUniquePlanes);
	}
}

//
//	Builds the diagram from the given set of points. Returns a set of uniquely determined sepparating planes,
//	a set of Voronoi cells that match one-to-one with the given input points, and their boundary plane Ids.
//	The boundary plane of a cell is considered to be oriented from m_pointIdxA to m_pointIdxB iif the plane index is positive.
//	The plane has opposite orientation otherwise.

void HK_CALL hkcdVoronoiDiagramUtil::build(	const hkAabb& coordinateConversionAabb, SitesProvider* sitesProvider,
											hkRefPtr<CsgOperand>& operandOut, hkArray<CellId>& cellIdsOut,
											hkcdPlanarEntityDebugger* debugger)
{
	// Allocate output
	operandOut.setAndDontIncrementRefCount(new CsgOperand());

	// Create geometry
	hkcdPlanarGeometryPlanesCollection* planesCol = new hkcdPlanarGeometryPlanesCollection(coordinateConversionAabb);
	{
		hkcdPlanarGeometry* geom = new hkcdPlanarGeometry(planesCol, 0, debugger);
		operandOut->setGeometry(geom);
		geom->removeReference();
		planesCol->removeReference();
	}

	// Create convex cell tree
	{
		hkcdConvexCellsTree3D* regions = new hkcdConvexCellsTree3D(operandOut->accessGeometry());
		operandOut->setRegions(regions);
		regions->removeReference();
	}

	// Build the cells
	hkcdVoronoiDiagramImpl::VoronoiDiagram diag(operandOut, debugger);
	diag.buildCells(sitesProvider);

	// Build output
	const int numSites = sitesProvider->getNumSites();
	cellIdsOut.setSize(numSites, CellId::invalid());
	diag.extractVoronoiCellIds(cellIdsOut);

	// Remove invalid cells
	for (int s = 0 ; s < cellIdsOut.getSize() ; s++)
	{
		if ( !cellIdsOut[s].isValid() )
		{
			cellIdsOut.removeAt(s);
			s--;
		}
	}
}

//
//	Builds the diagram from the given set of points.Returns the result as a BSP tree of planar cuts, with the convex cell Ids stored in the
//	terminal nodes

void HK_CALL hkcdVoronoiDiagramUtil::buildPlanarCuts(CsgOperand* voronoiTree, const hkArray<CellId>& cellIdsIn, hkArray<PlaneId>& usedPlaneIds)
{
	// Type shortcuts
	typedef hkcdVoronoiDiagramImpl::Stack		Stack;
	typedef hkcdVoronoiDiagramImpl::StackEntry	StackEntry;
	typedef hkcdVoronoiDiagramImpl::VtxCell		VtxCell;

	hkcdConvexCellsTree3D* cellTree = voronoiTree->accessRegions();

	// Gather the non-null cells
	int numSrcCells				= cellIdsIn.getSize();
	hkArray<int> frontCellIds;	frontCellIds.reserve(numSrcCells);
	hkArray<int> backCellIds;	backCellIds.reserve(numSrcCells);
	hkArray<VtxCell> vtxCells;
	for (int k = numSrcCells - 1; k >= 0; k--)
	{
		if ( cellIdsIn[k].isValid() )
		{
			// Create a new vertex cell and cache its vertices
			frontCellIds.pushBack(vtxCells.getSize());

			VtxCell& vCell = vtxCells.expandOne();
			vCell.m_id		= cellIdsIn[k];
			vCell.m_origId	= vCell.m_id;
			hkcdVoronoiDiagramImpl::computeCellsVertices(cellTree, &vCell.m_id, 1, vCell.m_verts);
		}
	}

	// Gather the used planes
	hkArray<PlaneId> frontPlaneIds;	hkcdVoronoiDiagramImpl::getAllCellsPlanes(cellTree, frontCellIds, vtxCells, frontPlaneIds);
	int numSrcPlanes				= frontPlaneIds.getSize();
	hkArray<PlaneId> backPlaneIds;
	hkArray<PlaneId> tempPlaneIds;

	// Remove boundary planes
	frontPlaneIds.removeAtAndCopy(0, hkcdPlanarGeometryPlanesCollection::NUM_BOUNDS);
	usedPlaneIds.append(frontPlaneIds);			// Outputs used plane ids
	// Preallocate the rest of the working memory
	numSrcPlanes = frontPlaneIds.getSize();
	backPlaneIds.reserve(numSrcPlanes);
	tempPlaneIds.reserve(numSrcPlanes);
	
	// Push all original data on the stack
	Stack stack;
	{
		StackEntry entry;
		entry.m_planeIds		= const_cast<PlaneId*>(frontPlaneIds.begin());
		entry.m_cellIds			= const_cast<int*>(frontCellIds.begin());
		entry.m_numPlaneIds		= numSrcPlanes;
		entry.m_numCellIds		= numSrcCells;
		entry.m_parentNodeId	= NodeId::invalid();
		entry.m_isLeftChild		= false;
		stack.push(entry);
	}

	// Allocate the output
	hkcdPlanarGeometry* geom = voronoiTree->accessGeometry();
	hkcdPlanarSolid* cutsOut = new hkcdPlanarSolid(geom->getPlanesCollection(), 0, geom->accessDebugger());
	voronoiTree->setSolid(cutsOut);
	cutsOut->removeReference();
	
	// While the stack is not empty, pop and process each entry
	while ( !stack.isEmpty() )
	{
		// Pop entry
		StackEntry entry;
		stack.pop(entry);
		HK_ASSERT(0x75217846, entry.m_numCellIds);

		if ( !entry.m_numPlaneIds )
		{
			// Leaf node!
			HK_ASSERT(0xa5642ed, entry.m_numCellIds == 1);

			// Create a node that will store the cell Id.
			NodeId nodeId	= cutsOut->createNode(PlaneId::invalid(), NodeId::invalid(), NodeId::invalid());
			Node& node		= cutsOut->accessNode(nodeId);
			node.m_parent	= entry.m_parentNodeId;
			node.m_data		= vtxCells[entry.m_cellIds[0]].m_origId.value();

#if ( ENABLE_DEBUG_PLANAR_CUTS )
			{
				hkGeometry gg;		cellTree->extractCellGeometry(vtxCells[entry.m_cellIds[0]].m_id, gg);
				hkStringBuf strb;	strb.printf("Added leaf node %d for cellId %d. geom = %d", nodeId.value(), node.m_data, cellTree->accessDebugger()->addGeometry("", &gg));
				HK_REPORT(strb);
			}
#endif

			if ( entry.m_parentNodeId.isValid() )
			{
				Node& parent = cutsOut->accessNode(entry.m_parentNodeId);
				if ( entry.m_isLeftChild )	{	parent.m_left	= nodeId;	}
				else						{	parent.m_right	= nodeId;	}
			}
			else
			{ 
				cutsOut->setRootNode(nodeId);
			}

			continue;
		}

		// Pick a splitting plane
		const int splitPlaneIdx		= hkcdVoronoiDiagramImpl::pickSplittingPlane(geom, vtxCells, entry.m_planeIds, entry.m_numPlaneIds, entry.m_cellIds, entry.m_numCellIds);
		const PlaneId splitPlaneId	= entry.m_planeIds[splitPlaneIdx];
		for (int k = splitPlaneIdx + 1; k < entry.m_numPlaneIds; k++)
		{
			entry.m_planeIds[k - 1] = entry.m_planeIds[k];
		}
		entry.m_numPlaneIds--;
		HK_ASSERT(0x57c5b1ea, splitPlaneId.isValid());

		// Classify polygons w.r.t. the splitting plane
		frontCellIds.setSize(0);	backCellIds.setSize(0);
		for (int k = 0; k < entry.m_numCellIds; k++)
		{
			const int cellId			= entry.m_cellIds[k];
			const VtxCell& vCell		= vtxCells[cellId];
			const CellId vCellId		= vCell.m_id;
			const CellId vCellOrigId	= vCell.m_origId;

			// Split cell
			CellId insideCellId, outsideCellId;
			cellTree->splitCell(HK_NULL, vCellId, splitPlaneId, insideCellId, outsideCellId);
			
			if ( insideCellId.isValid() && outsideCellId.isValid() )
			{
				// Create splits
				const int insideCellIdx		= vtxCells.getSize();		backCellIds.pushBack(insideCellIdx);
				const int outsideCellIdx	= insideCellIdx + 1;		frontCellIds.pushBack(outsideCellIdx);

				// Compute their vertices
				VtxCell* splits = vtxCells.expandBy(2);
				splits[0].m_id	= insideCellId;		splits[0].m_origId	= vCellOrigId;		hkcdVoronoiDiagramImpl::computeCellsVertices(cellTree, &insideCellId, 1, splits[0].m_verts);
				splits[1].m_id	= outsideCellId;	splits[1].m_origId	= vCellOrigId;		hkcdVoronoiDiagramImpl::computeCellsVertices(cellTree, &outsideCellId, 1, splits[1].m_verts);
			}
			else if ( insideCellId.isValid() )	{	HK_ASSERT(0x764f876d, !outsideCellId.isValid() && (insideCellId == vCellId));	backCellIds.pushBack(cellId);	}
			else								{	HK_ASSERT(0x598af846, !insideCellId.isValid() && outsideCellId == vCellId);		frontCellIds.pushBack(cellId);	}
		}

		// Allocate a new node
		NodeId nodeId = cutsOut->createNode(splitPlaneId, NodeId::invalid(), NodeId::invalid());
		{
			Node& node		= cutsOut->accessNode(nodeId);
			node.m_parent	= entry.m_parentNodeId;
			node.m_data		= CellId::InvalidValue;

			if ( entry.m_parentNodeId.isValid() )
			{
				Node& parent = cutsOut->accessNode(entry.m_parentNodeId);
				if ( entry.m_isLeftChild )	{	parent.m_left	= nodeId;	}
				else						{	parent.m_right	= nodeId;	}
			}
			else
			{ 
				cutsOut->setRootNode(nodeId);
			}
		}

#if ( ENABLE_DEBUG_PLANAR_CUTS )
		{
			hkGeometry gg;
			for (int k = 0; k < entry.m_numCellIds; k++)
			{
				hkGeometry ggg;
				cellTree->extractCellGeometry(vtxCells[entry.m_cellIds[k]].m_id, ggg);
				gg.appendGeometry(ggg);
			}

			hkStringBuf strb;
			strb.printf("Added node %d, parent %d, splitting plane %d. geom = %d", nodeId.value(), entry.m_parentNodeId.valueUnchecked(),
						splitPlaneId.value(), cellTree->accessDebugger()->addGeometry("", &gg));
			HK_REPORT(strb);
		}
#endif

		// Gather the left & right planes
		if ( backCellIds.getSize() )
		{
			tempPlaneIds.setSize(0);				hkcdVoronoiDiagramImpl::getAllCellsPlanes(cellTree, backCellIds, vtxCells, tempPlaneIds);
			backPlaneIds.setSize(numSrcPlanes);		backPlaneIds.setSize(hkAlgorithm::intersectionOfSortedLists(entry.m_planeIds, entry.m_numPlaneIds, tempPlaneIds.begin(), tempPlaneIds.getSize(), backPlaneIds.begin()));
		}
		if ( frontCellIds.getSize() )
		{
			tempPlaneIds.setSize(0);				hkcdVoronoiDiagramImpl::getAllCellsPlanes(cellTree, frontCellIds, vtxCells, tempPlaneIds);
			frontPlaneIds.setSize(numSrcPlanes);	frontPlaneIds.setSize(hkAlgorithm::intersectionOfSortedLists(entry.m_planeIds, entry.m_numPlaneIds, tempPlaneIds.begin(), tempPlaneIds.getSize(), frontPlaneIds.begin()));
		}

		// Recurse on left
		if ( backCellIds.getSize() )
		{
			StackEntry leftEntry;
			leftEntry.m_isLeftChild		= true;
			leftEntry.m_cellIds			= backCellIds.begin();
			leftEntry.m_numCellIds		= backCellIds.getSize();
			leftEntry.m_planeIds		= backPlaneIds.begin();
			leftEntry.m_numPlaneIds		= backPlaneIds.getSize();
			leftEntry.m_parentNodeId	= nodeId;
			stack.push(leftEntry);			
		}

		// Recurse on right
		if ( frontCellIds.getSize() )
		{
			StackEntry rightEntry;
			rightEntry.m_isLeftChild	= false;
			rightEntry.m_cellIds		= frontCellIds.begin();
			rightEntry.m_numCellIds		= frontCellIds.getSize();
			rightEntry.m_planeIds		= frontPlaneIds.begin();
			rightEntry.m_numPlaneIds	= frontPlaneIds.getSize();
			rightEntry.m_parentNodeId	= nodeId;
			stack.push(rightEntry);
		}
	}
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
