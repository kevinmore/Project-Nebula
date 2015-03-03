/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Geometry/Collide/hkcdCollide.h>
static const char s_libraryName[] = "hkcdCollide";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkcdCollideRegister() {}

#include <Geometry/Collide/DataStructures/IntAabb/hkcdIntAabb.h>


// hkcdIntAabb ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkcdIntAabb, s_libraryName)

#include <Geometry/Collide/DataStructures/Planar/CSG/hkcdPlanarCsgOperand.h>


// hkcdPlanarCsgOperand ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdPlanarCsgOperand)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GeomExtraInfos)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GeomSource)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdPlanarCsgOperand)
    HK_TRACKER_MEMBER(hkcdPlanarCsgOperand, m_geometry, 0, "hkcdPlanarGeometry *") // class hkcdPlanarGeometry *
    HK_TRACKER_MEMBER(hkcdPlanarCsgOperand, m_geomInfos, 0, "hkcdPlanarCsgOperand::GeomExtraInfos *") // struct hkcdPlanarCsgOperand::GeomExtraInfos *
    HK_TRACKER_MEMBER(hkcdPlanarCsgOperand, m_solid, 0, "hkcdPlanarSolid *") // class hkcdPlanarSolid *
    HK_TRACKER_MEMBER(hkcdPlanarCsgOperand, m_regions, 0, "hkcdConvexCellsTree3D *") // class hkcdConvexCellsTree3D *
    HK_TRACKER_MEMBER(hkcdPlanarCsgOperand, m_geomSources, 0, "hkArray<hkcdPlanarCsgOperand::GeomSource, hkContainerHeapAllocator>") // hkArray< struct hkcdPlanarCsgOperand::GeomSource, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdPlanarCsgOperand, s_libraryName, hkReferencedObject)


// GeomExtraInfos hkcdPlanarCsgOperand

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdPlanarCsgOperand::GeomExtraInfos)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdPlanarCsgOperand::GeomExtraInfos)
    HK_TRACKER_MEMBER(hkcdPlanarCsgOperand::GeomExtraInfos, m_danglingPolyIds, 0, "hkArray<hkHandle<hkUint32, 0, hkcdPlanarGeometryPolygonCollection::PolygonIdDiscriminant>, hkContainerHeapAllocator>") // hkArray< struct hkHandle< hkUint32, 0, struct hkcdPlanarGeometryPolygonCollection::PolygonIdDiscriminant >, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdPlanarCsgOperand::GeomExtraInfos, s_libraryName, hkReferencedObject)


// GeomSource hkcdPlanarCsgOperand

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdPlanarCsgOperand::GeomSource)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdPlanarCsgOperand::GeomSource)
    HK_TRACKER_MEMBER(hkcdPlanarCsgOperand::GeomSource, m_geometry, 0, "hkcdPlanarGeometry *") // class hkcdPlanarGeometry *
    HK_TRACKER_MEMBER(hkcdPlanarCsgOperand::GeomSource, m_geomInfos, 0, "hkcdPlanarCsgOperand::GeomExtraInfos *") // struct hkcdPlanarCsgOperand::GeomExtraInfos *
    HK_TRACKER_MEMBER(hkcdPlanarCsgOperand::GeomSource, m_cutoutSolid, 0, "hkcdPlanarSolid *") // class hkcdPlanarSolid *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdPlanarCsgOperand::GeomSource, s_libraryName, hkReferencedObject)

#include <Geometry/Collide/DataStructures/Planar/CSG/hkcdPlanarGeometryBooleanUtil.h>


// hkcdPlanarGeometryBooleanUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdPlanarGeometryBooleanUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BooleanState)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Operation)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkcdPlanarGeometryBooleanUtil, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdPlanarGeometryBooleanUtil, Operation, s_libraryName)


// BooleanState hkcdPlanarGeometryBooleanUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdPlanarGeometryBooleanUtil, BooleanState, s_libraryName)

#include <Geometry/Collide/DataStructures/Planar/ConvexCellsTree/hkcdConvexCellsTree2D.h>


// hkcdConvexCellsTree2D ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdConvexCellsTree2D)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdConvexCellsTree2D)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdConvexCellsTree2D, s_libraryName, hkcdConvexCellsTreehkcdPlanarGeometryPolygonCollectionPolygonhkHandleunsignedint0hkcdPlanarGeometryPolygonCollectionPolygonIdDiscriminanthkcdPlanarGeometry)

#include <Geometry/Collide/DataStructures/Planar/ConvexCellsTree/hkcdConvexCellsTree3D.h>


// hkcdConvexCellsTree3D ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdConvexCellsTree3D)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PolygonSurfaceType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdConvexCellsTree3D)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdConvexCellsTree3D, s_libraryName, hkcdConvexCellsTreehkcdConvexCellsCollectionCellhkHandleunsignedint0hkcdConvexCellsCollectionCellIdDiscriminanthkcdConvexCellsCollection)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdConvexCellsTree3D, PolygonSurfaceType, s_libraryName)

#include <Geometry/Collide/DataStructures/Planar/Geometry/hkcdPlanarGeometry.h>


// hkcdPlanarGeometry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdPlanarGeometry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdPlanarGeometry)
    HK_TRACKER_MEMBER(hkcdPlanarGeometry, m_planes, 0, "hkcdPlanarGeometryPlanesCollection *") // class hkcdPlanarGeometryPlanesCollection *
    HK_TRACKER_MEMBER(hkcdPlanarGeometry, m_polys, 0, "hkcdPlanarGeometryPolygonCollection *") // class hkcdPlanarGeometryPolygonCollection *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdPlanarGeometry, s_libraryName, hkcdPlanarEntity)

#include <Geometry/Collide/DataStructures/Planar/Geometry/hkcdVertexGeometry.h>


// hkcdVertexGeometry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdVertexGeometry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Vertex)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LineIdDiscriminant)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Line)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(EdgeIdDiscriminant)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Edge)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VPolygonCollectionBase)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VPolygon)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VPolygonCollection)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(EdgeIdConstants)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdVertexGeometry)
    HK_TRACKER_MEMBER(hkcdVertexGeometry, m_planes, 0, "hkcdPlanarGeometryPlanesCollection *") // class hkcdPlanarGeometryPlanesCollection *
    HK_TRACKER_MEMBER(hkcdVertexGeometry, m_vertices, 0, "hkArray<hkcdVertexGeometry::Vertex, hkContainerHeapAllocator>") // hkArray< struct hkcdVertexGeometry::Vertex, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkcdVertexGeometry, m_lines, 0, "hkArray<hkcdVertexGeometry::Line, hkContainerHeapAllocator>") // hkArray< struct hkcdVertexGeometry::Line, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkcdVertexGeometry, m_edges, 0, "hkArray<hkcdVertexGeometry::Edge, hkContainerHeapAllocator>") // hkArray< struct hkcdVertexGeometry::Edge, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkcdVertexGeometry, m_polys, 0, "hkcdVertexGeometry::VPolygonCollection") // struct hkcdVertexGeometry::VPolygonCollection
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdVertexGeometry, s_libraryName, hkcdPlanarEntity)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdVertexGeometry, EdgeIdConstants, s_libraryName)


// Vertex hkcdVertexGeometry
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdVertexGeometry, Vertex, s_libraryName)


// LineIdDiscriminant hkcdVertexGeometry
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdVertexGeometry, LineIdDiscriminant, s_libraryName)


// Line hkcdVertexGeometry
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdVertexGeometry, Line, s_libraryName)


// EdgeIdDiscriminant hkcdVertexGeometry
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdVertexGeometry, EdgeIdDiscriminant, s_libraryName)


// Edge hkcdVertexGeometry
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdVertexGeometry, Edge, s_libraryName)


// VPolygonCollectionBase hkcdVertexGeometry

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdVertexGeometry::VPolygonCollectionBase)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VPolygonIdDiscriminant)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdVertexGeometry::VPolygonCollectionBase)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdVertexGeometry::VPolygonCollectionBase, s_libraryName, hkcdPlanarGeometryPrimitivesCollection28)


// VPolygonIdDiscriminant hkcdVertexGeometry::VPolygonCollectionBase
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdVertexGeometry::VPolygonCollectionBase, VPolygonIdDiscriminant, s_libraryName)


// VPolygon hkcdVertexGeometry
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdVertexGeometry, VPolygon, s_libraryName)


// VPolygonCollection hkcdVertexGeometry

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdVertexGeometry::VPolygonCollection)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdVertexGeometry::VPolygonCollection)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdVertexGeometry::VPolygonCollection, s_libraryName, hkcdVertexGeometry::VPolygonCollectionBase)

#include <Geometry/Collide/DataStructures/Planar/Memory/hkcdConvexCellsCollection.h>


// hkcdConvexCellsCollection ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdConvexCellsCollection)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CellIdDiscriminant)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Cell)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Labels)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdConvexCellsCollection)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdConvexCellsCollection, s_libraryName, hkcdPlanarGeometryPrimitivesCollection28)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdConvexCellsCollection, Labels, s_libraryName)


// CellIdDiscriminant hkcdConvexCellsCollection
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdConvexCellsCollection, CellIdDiscriminant, s_libraryName)


// Cell hkcdConvexCellsCollection
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdConvexCellsCollection, Cell, s_libraryName)

#include <Geometry/Collide/DataStructures/Planar/Memory/hkcdPlanarGeometryPlanesCollection.h>


// hkcdPlanarGeometryPlanesCollection ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdPlanarGeometryPlanesCollection)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Bounds)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdPlanarGeometryPlanesCollection)
    HK_TRACKER_MEMBER(hkcdPlanarGeometryPlanesCollection, m_planes, 0, "hkArray<hkcdPlanarGeometryPrimitives::Plane, hkContainerHeapAllocator>") // hkArray< struct hkcdPlanarGeometryPrimitives::Plane, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkcdPlanarGeometryPlanesCollection, m_cache, 0, "hkcdPlanarGeometryPredicates::OrientationCacheBase<12>*") // struct hkcdPlanarGeometryPredicates::OrientationCacheBase< 12 >*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdPlanarGeometryPlanesCollection, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdPlanarGeometryPlanesCollection, Bounds, s_libraryName)

#include <Geometry/Collide/DataStructures/Planar/Memory/hkcdPlanarGeometryPolygonCollection.h>


// hkcdPlanarGeometryPolygonCollection ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdPlanarGeometryPolygonCollection)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PolygonIdDiscriminant)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Polygon)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdPlanarGeometryPolygonCollection)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdPlanarGeometryPolygonCollection, s_libraryName, hkcdPlanarGeometryPrimitivesCollection28)


// PolygonIdDiscriminant hkcdPlanarGeometryPolygonCollection
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdPlanarGeometryPolygonCollection, PolygonIdDiscriminant, s_libraryName)


// Polygon hkcdPlanarGeometryPolygonCollection
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdPlanarGeometryPolygonCollection, Polygon, s_libraryName)

#include <Geometry/Collide/DataStructures/Planar/Predicates/hkcdPlanarGeometryPredicates.h>


// hkcdPlanarGeometryPredicates ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkcdPlanarGeometryPredicates, s_libraryName)

#include <Geometry/Collide/DataStructures/Planar/Primitives/hkcdPlanarGeometryPrimitives.h>


// NumBitsMantissa hkcdPlanarGeometryPrimitives
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkcdPlanarGeometryPrimitives::NumBitsMantissa, s_libraryName, hkcdPlanarGeometryPrimitives_NumBitsMantissa)


// NumBitsMantissaD hkcdPlanarGeometryPrimitives
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkcdPlanarGeometryPrimitives::NumBitsMantissaD, s_libraryName, hkcdPlanarGeometryPrimitives_NumBitsMantissaD)


// NumBitsVertex hkcdPlanarGeometryPrimitives
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkcdPlanarGeometryPrimitives::NumBitsVertex, s_libraryName, hkcdPlanarGeometryPrimitives_NumBitsVertex)


// NumBitsEdge hkcdPlanarGeometryPrimitives
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkcdPlanarGeometryPrimitives::NumBitsEdge, s_libraryName, hkcdPlanarGeometryPrimitives_NumBitsEdge)


// NumBitsPlaneNormal hkcdPlanarGeometryPrimitives
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkcdPlanarGeometryPrimitives::NumBitsPlaneNormal, s_libraryName, hkcdPlanarGeometryPrimitives_NumBitsPlaneNormal)


// NumBitsPlaneOffset hkcdPlanarGeometryPrimitives
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkcdPlanarGeometryPrimitives::NumBitsPlaneOffset, s_libraryName, hkcdPlanarGeometryPrimitives_NumBitsPlaneOffset)


// NumBitsPlanesIntersectionEdgeDir hkcdPlanarGeometryPrimitives
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkcdPlanarGeometryPrimitives::NumBitsPlanesIntersectionEdgeDir, s_libraryName, hkcdPlanarGeometryPrimitives_NumBitsPlanesIntersectionEdgeDir)


// NumBitsPlanesDet3 hkcdPlanarGeometryPrimitives
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkcdPlanarGeometryPrimitives::NumBitsPlanesDet3, s_libraryName, hkcdPlanarGeometryPrimitives_NumBitsPlanesDet3)


// NumBitsPlaneMulOffset hkcdPlanarGeometryPrimitives
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkcdPlanarGeometryPrimitives::NumBitsPlaneMulOffset, s_libraryName, hkcdPlanarGeometryPrimitives_NumBitsPlaneMulOffset)


// NumBitsPlaneCrossOffset hkcdPlanarGeometryPrimitives
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkcdPlanarGeometryPrimitives::NumBitsPlaneCrossOffset, s_libraryName, hkcdPlanarGeometryPrimitives_NumBitsPlaneCrossOffset)


// NumBitsPlanesDet4 hkcdPlanarGeometryPrimitives
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkcdPlanarGeometryPrimitives::NumBitsPlanesDet4, s_libraryName, hkcdPlanarGeometryPrimitives_NumBitsPlanesDet4)


// Plane hkcdPlanarGeometryPrimitives
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkcdPlanarGeometryPrimitives::Plane, s_libraryName, hkcdPlanarGeometryPrimitives_Plane)


// Vertex hkcdPlanarGeometryPrimitives
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkcdPlanarGeometryPrimitives::Vertex, s_libraryName, hkcdPlanarGeometryPrimitives_Vertex)


// PlaneIdDiscriminant hkcdPlanarGeometryPrimitives
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkcdPlanarGeometryPrimitives::PlaneIdDiscriminant, s_libraryName, hkcdPlanarGeometryPrimitives_PlaneIdDiscriminant)

// hkcdPlanarGeometryPrimitives PlaneIdConstants
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkcdPlanarGeometryPrimitives::PlaneIdConstants, s_libraryName, hkcdPlanarGeometryPrimitives_PlaneIdConstants)
#include <Geometry/Collide/DataStructures/Planar/Solid/hkcdPlanarSolid.h>


// hkcdPlanarSolid ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdPlanarSolid)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(NodeIdDiscriminant)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(NodeStorage)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Node)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ArraySlot)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ArrayIdDiscriminant)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ArrayMgr)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(NodeTypes)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdPlanarSolid)
    HK_TRACKER_MEMBER(hkcdPlanarSolid, m_nodes, 0, "hkcdPlanarSolid::NodeStorage *") // struct hkcdPlanarSolid::NodeStorage *
    HK_TRACKER_MEMBER(hkcdPlanarSolid, m_planes, 0, "hkcdPlanarGeometryPlanesCollection *") // const class hkcdPlanarGeometryPlanesCollection *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdPlanarSolid, s_libraryName, hkcdPlanarEntity)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdPlanarSolid, NodeTypes, s_libraryName)


// NodeIdDiscriminant hkcdPlanarSolid
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdPlanarSolid, NodeIdDiscriminant, s_libraryName)


// NodeStorage hkcdPlanarSolid

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdPlanarSolid::NodeStorage)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdPlanarSolid::NodeStorage)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdPlanarSolid::NodeStorage, s_libraryName, hkReferencedObject)


// Node hkcdPlanarSolid
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdPlanarSolid, Node, s_libraryName)


// ArraySlot hkcdPlanarSolid
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdPlanarSolid, ArraySlot, s_libraryName)


// ArrayIdDiscriminant hkcdPlanarSolid
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdPlanarSolid, ArrayIdDiscriminant, s_libraryName)


// ArrayMgr hkcdPlanarSolid

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdPlanarSolid::ArrayMgr)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdPlanarSolid::ArrayMgr)
    HK_TRACKER_MEMBER(hkcdPlanarSolid::ArrayMgr, m_arrays, 0, "hkFreeListArray<hkcdPlanarSolid::ArraySlot, hkHandle<hkUint32, 4294967295, hkcdPlanarSolid::ArrayIdDiscriminant>, 128, hkDefaultFreeListArrayOperations<hkcdPlanarSolid::ArraySlot> >") // struct hkFreeListArray< struct hkcdPlanarSolid::ArraySlot, struct hkHandle< hkUint32, 4294967295, struct hkcdPlanarSolid::ArrayIdDiscriminant >, 128, struct hkDefaultFreeListArrayOperations< struct hkcdPlanarSolid::ArraySlot > >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdPlanarSolid::ArrayMgr, s_libraryName, hkcdPlanarGeometryPrimitivesCollection28)

#include <Geometry/Collide/DataStructures/Planar/Utils/hkcdPlanarGeometryConvexHullUtil.h>


// hkcdPlanarGeometryConvexHullUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkcdPlanarGeometryConvexHullUtil, s_libraryName)

#include <Geometry/Collide/DataStructures/Planar/Utils/hkcdPlanarGeometrySimplifier.h>


// hkcdPlanarGeometrySimplifier ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkcdPlanarGeometrySimplifier, s_libraryName)

#include <Geometry/Collide/DataStructures/Planar/Utils/hkcdPlanarGeometryWeldUtil.h>


// hkcdPlanarGeometryWeldUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkcdPlanarGeometryWeldUtil, s_libraryName)

#include <Geometry/Collide/DataStructures/Planar/Utils/hkcdVoronoiDiagramUtil.h>


// hkcdVoronoiDiagramUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdVoronoiDiagramUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SitesProvider)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkcdVoronoiDiagramUtil, s_libraryName)


// SitesProvider hkcdVoronoiDiagramUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdVoronoiDiagramUtil::SitesProvider)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdVoronoiDiagramUtil::SitesProvider)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkcdVoronoiDiagramUtil::SitesProvider, s_libraryName)

#include <Geometry/Collide/DataStructures/Planar/hkcdPlanarEntity.h>


// hkcdPlanarEntityDebugger ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdPlanarEntityDebugger)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdPlanarEntityDebugger)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdPlanarEntityDebugger, s_libraryName, hkReferencedObject)


// hkcdPlanarEntity ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdPlanarEntity)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdPlanarEntity)
    HK_TRACKER_MEMBER(hkcdPlanarEntity, m_debugger, 0, "hkcdPlanarEntityDebugger *") // class hkcdPlanarEntityDebugger *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdPlanarEntity, s_libraryName, hkReferencedObject)

#include <Geometry/Collide/Shapes/hkcdShape.h>


// hkcdShapeType ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkcdShapeType, s_libraryName)


// hkcdShapeDispatchType ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkcdShapeDispatchType, s_libraryName)


// hkcdShapeInfoCodecType ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkcdShapeInfoCodecType, s_libraryName)


// hkcdShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdShape, s_libraryName, hkReferencedObject)

#include <Geometry/Collide/Util/ShapeVirtualTable/hkcdShapeVirtualTableUtil.h>


// hkcdShapeVirtualTableUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkcdShapeVirtualTableUtil, s_libraryName)

#include <Geometry/Collide/World/hkcdWorld.h>


// hkcdWorld ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdWorld)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(HObjectDiscriminant)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(HGroupDiscriminant)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Object)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Group)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GroupSet)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ICodec)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IUnaryProcessor)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ILinearCastProcessor)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IBinaryProcessor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdWorld)
    HK_TRACKER_MEMBER(hkcdWorld, m_objects, 0, "hkFreeListArray<hkcdWorld::Object, hkHandle<hkUint32, 0, hkcdWorld::HObjectDiscriminant>, 32, hkDefaultFreeListArrayOperations<hkcdWorld::Object> >") // struct hkFreeListArray< struct hkcdWorld::Object, struct hkHandle< hkUint32, 0, struct hkcdWorld::HObjectDiscriminant >, 32, struct hkDefaultFreeListArrayOperations< struct hkcdWorld::Object > >
    HK_TRACKER_MEMBER(hkcdWorld, m_groups, 0, "hkFreeListArray<hkcdWorld::Group, hkHandle<hkUint32, 0, hkcdWorld::HGroupDiscriminant>, 8, hkDefaultFreeListArrayOperations<hkcdWorld::Group> >") // struct hkFreeListArray< struct hkcdWorld::Group, struct hkHandle< hkUint32, 0, struct hkcdWorld::HGroupDiscriminant >, 8, struct hkDefaultFreeListArrayOperations< struct hkcdWorld::Group > >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdWorld, s_libraryName, hkReferencedObject)


// HObjectDiscriminant hkcdWorld
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdWorld, HObjectDiscriminant, s_libraryName)


// HGroupDiscriminant hkcdWorld
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdWorld, HGroupDiscriminant, s_libraryName)


// Object hkcdWorld
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdWorld, Object, s_libraryName)


// Group hkcdWorld

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdWorld::Group)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AabbType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdWorld::Group)
    HK_TRACKER_MEMBER(hkcdWorld::Group, m_name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkcdWorld::Group, m_codec, 0, "hkcdWorld::ICodec*") // struct hkcdWorld::ICodec*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkcdWorld::Group, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdWorld::Group, AabbType, s_libraryName)


// GroupSet hkcdWorld
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdWorld, GroupSet, s_libraryName)


// ICodec hkcdWorld

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdWorld::ICodec)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdWorld::ICodec)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkcdWorld::ICodec, s_libraryName)


// IUnaryProcessor hkcdWorld

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdWorld::IUnaryProcessor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdWorld::IUnaryProcessor)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkcdWorld::IUnaryProcessor, s_libraryName)


// ILinearCastProcessor hkcdWorld

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdWorld::ILinearCastProcessor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdWorld::ILinearCastProcessor)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkcdWorld::ILinearCastProcessor, s_libraryName)


// IBinaryProcessor hkcdWorld

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdWorld::IBinaryProcessor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdWorld::IBinaryProcessor)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkcdWorld::IBinaryProcessor, s_libraryName)

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
