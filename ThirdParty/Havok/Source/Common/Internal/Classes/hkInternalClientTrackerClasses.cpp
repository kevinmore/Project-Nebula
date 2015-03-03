/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Common/Internal/hkInternal.h>
static const char s_libraryName[] = "hkInternalClient";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkInternalClientRegister() {}

#include <Common/Internal/ConvexHull/Deprecated/hkGeomConvexHullBuilder.h>


// hkpGeomConvexHullConfig ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpGeomConvexHullConfig, s_libraryName)


// hkGeomConvexHullBuilder ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGeomConvexHullBuilder)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WeightedLine)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WrappingLine)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WeightedNeighbour)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PlaneAndPoints)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VisitedEdgeInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkGeomConvexHullBuilder, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkGeomConvexHullBuilder, VisitedEdgeInfo, s_libraryName)


// WeightedLine hkGeomConvexHullBuilder

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGeomConvexHullBuilder::WeightedLine)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkGeomConvexHullBuilder::WeightedLine)
    HK_TRACKER_MEMBER(hkGeomConvexHullBuilder::WeightedLine, m_leftEdge, 0, "hkGeomEdge*") // class hkGeomEdge*
    HK_TRACKER_MEMBER(hkGeomConvexHullBuilder::WeightedLine, m_rightEdge, 0, "hkGeomEdge*") // class hkGeomEdge*
    HK_TRACKER_MEMBER(hkGeomConvexHullBuilder::WeightedLine, m_source, 0, "hkGeomConvexHullBuilder::WeightedLine*") // struct hkGeomConvexHullBuilder::WeightedLine*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkGeomConvexHullBuilder::WeightedLine, s_libraryName)


// WrappingLine hkGeomConvexHullBuilder

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGeomConvexHullBuilder::WrappingLine)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkGeomConvexHullBuilder::WrappingLine)
    HK_TRACKER_MEMBER(hkGeomConvexHullBuilder::WrappingLine, m_leftEdge, 0, "hkGeomEdge*") // class hkGeomEdge*
    HK_TRACKER_MEMBER(hkGeomConvexHullBuilder::WrappingLine, m_rightEdge, 0, "hkGeomEdge*") // class hkGeomEdge*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkGeomConvexHullBuilder::WrappingLine, s_libraryName)


// WeightedNeighbour hkGeomConvexHullBuilder

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGeomConvexHullBuilder::WeightedNeighbour)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkGeomConvexHullBuilder::WeightedNeighbour)
    HK_TRACKER_MEMBER(hkGeomConvexHullBuilder::WeightedNeighbour, m_edge, 0, "hkGeomEdge*") // class hkGeomEdge*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkGeomConvexHullBuilder::WeightedNeighbour, s_libraryName)


// PlaneAndPoints hkGeomConvexHullBuilder

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGeomConvexHullBuilder::PlaneAndPoints)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkGeomConvexHullBuilder::PlaneAndPoints)
    HK_TRACKER_MEMBER(hkGeomConvexHullBuilder::PlaneAndPoints, m_v0, 0, "hkGeomEdge*") // class hkGeomEdge*
    HK_TRACKER_MEMBER(hkGeomConvexHullBuilder::PlaneAndPoints, m_v1, 0, "hkGeomEdge*") // class hkGeomEdge*
    HK_TRACKER_MEMBER(hkGeomConvexHullBuilder::PlaneAndPoints, m_v2, 0, "hkGeomEdge*") // class hkGeomEdge*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkGeomConvexHullBuilder::PlaneAndPoints, s_libraryName)

#include <Common/Internal/ConvexHull/Deprecated/hkGeomConvexHullTester.h>


// hkGeomConvexHullTester ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkGeomConvexHullTester, s_libraryName)

#include <Common/Internal/ConvexHull/Deprecated/hkGeomConvexHullTolerances.h>


// hkGeomConvexHullTolerances ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkGeomConvexHullTolerances, s_libraryName)

#include <Common/Internal/ConvexHull/Deprecated/hkGeomEdge.h>


// hkGeomEdge ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkGeomEdge, s_libraryName)

#include <Common/Internal/ConvexHull/Deprecated/hkGeomHull.h>


// hkGeomHull ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGeomHull)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WrappingEdge)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkGeomHull)
    HK_TRACKER_MEMBER(hkGeomHull, m_vertexBase, 0, "hkVector4*") // hkVector4*
    HK_TRACKER_MEMBER(hkGeomHull, m_edges, 0, "hkInplaceArray<hkGeomEdge, 128, hkContainerHeapAllocator>") // class hkInplaceArray< class hkGeomEdge, 128, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkGeomHull, s_libraryName)


// WrappingEdge hkGeomHull
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkGeomHull, WrappingEdge, s_libraryName)

// None neighbourDirection
HK_TRACKER_IMPLEMENT_SIMPLE(neighbourDirection, s_libraryName)
#include <Common/Internal/GeometryProcessing/Boolean/hkgpBoolean.h>


// hkgpBoolean ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpBoolean)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Operator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Operand)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ErrorType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Error)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Config)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Infos)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Evaluator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkgpBoolean, s_libraryName)


// Operator hkgpBoolean
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpBoolean, Operator, s_libraryName)


// Operand hkgpBoolean

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpBoolean::Operand)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(eType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpBoolean::Operand)
    HK_TRACKER_MEMBER(hkgpBoolean::Operand, m_geometry, 0, "hkGeometry*") // const struct hkGeometry*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpBoolean::Operand, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpBoolean::Operand, eType, s_libraryName)


// ErrorType hkgpBoolean
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpBoolean, ErrorType, s_libraryName)


// Error hkgpBoolean
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpBoolean, Error, s_libraryName)


// Config hkgpBoolean
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpBoolean, Config, s_libraryName)


// Infos hkgpBoolean

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpBoolean::Infos)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpBoolean::Infos)
    HK_TRACKER_MEMBER(hkgpBoolean::Infos, m_errors, 0, "hkArray<hkgpBoolean::Error, hkContainerHeapAllocator>") // hkArray< struct hkgpBoolean::Error, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpBoolean::Infos, s_libraryName)


// Evaluator hkgpBoolean

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpBoolean::Evaluator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpBoolean::Evaluator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkgpBoolean::Evaluator, s_libraryName)

#include <Common/Internal/GeometryProcessing/CollisionGeometryOptimizer/hkgpCgo.h>


// hkgpCgo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpCgo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Tracker)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Config)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IProgress)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClusterData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkgpCgo, s_libraryName)


// Tracker hkgpCgo

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpCgo::Tracker)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpCgo::Tracker)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkgpCgo::Tracker, s_libraryName)


// Config hkgpCgo

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpCgo::Config)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VertexSemantic)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VertexCombinator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SolverAccuracy)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpCgo::Config)
    HK_TRACKER_MEMBER(hkgpCgo::Config, m_tracker, 0, "hkgpCgo::Tracker*") // struct hkgpCgo::Tracker*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpCgo::Config, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpCgo::Config, VertexSemantic, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpCgo::Config, VertexCombinator, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpCgo::Config, SolverAccuracy, s_libraryName)


// IProgress hkgpCgo

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpCgo::IProgress)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpCgo::IProgress)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkgpCgo::IProgress, s_libraryName)


// ClusterData hkgpCgo
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpCgo, ClusterData, s_libraryName)

#include <Common/Internal/GeometryProcessing/IndexedMesh/hkgpIndexedMesh.h>


// hkgpIndexedMeshDefinitions ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpIndexedMeshDefinitions)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Vertex)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Triangle)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Edge)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkgpIndexedMeshDefinitions, s_libraryName)

// hk.MemoryTracker ignore VertexBase

// Vertex hkgpIndexedMeshDefinitions
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpIndexedMeshDefinitions, Vertex, s_libraryName)

// hk.MemoryTracker ignore BaseTriangle

// Triangle hkgpIndexedMeshDefinitions
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpIndexedMeshDefinitions, Triangle, s_libraryName)


// Edge hkgpIndexedMeshDefinitions
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpIndexedMeshDefinitions, Edge, s_libraryName)


// hkgpIndexedMesh ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpIndexedMesh)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Flags)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(EdgeMatch)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(StripConfig)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SetInfos)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(EdgeBarrier)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IVertexRemoval)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ITriangleRemoval)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IEdgeCollapse)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpIndexedMesh)
    HK_TRACKER_MEMBER(hkgpIndexedMesh, m_vMap, 0, "hkPointerMap<hkUlong, hkgpIndexedMeshDefinitions::Vertex*, hkContainerHeapAllocator>") // class hkPointerMap< hkUlong, struct hkgpIndexedMeshDefinitions::Vertex*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkgpIndexedMesh, m_eMap, 0, "hkGeometryProcessing::HashTable<hkgpIndexedMesh::EdgeMatch, hkContainerHeapAllocator>") // class hkGeometryProcessing::HashTable< struct hkgpIndexedMesh::EdgeMatch, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkgpIndexedMesh, m_invalidTriangles, 0, "hkArray<hkUlong, hkContainerHeapAllocator>") // hkArray< hkUlong, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkgpIndexedMesh, m_sets, 0, "hkArray<hkgpIndexedMesh::SetInfos, hkContainerHeapAllocator>") // hkArray< struct hkgpIndexedMesh::SetInfos, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkgpIndexedMesh, s_libraryName, hkgpAbstractMeshhkgpIndexedMeshDefinitionsEdgehkgpIndexedMeshDefinitionsVertexhkgpIndexedMeshDefinitionsTrianglehkContainerHeapAllocator)


// Flags hkgpIndexedMesh
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpIndexedMesh, Flags, s_libraryName)


// EdgeMatch hkgpIndexedMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpIndexedMesh::EdgeMatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpIndexedMesh::EdgeMatch)
    HK_TRACKER_MEMBER(hkgpIndexedMesh::EdgeMatch, m_edge, 0, "hkgpIndexedMeshDefinitions::Edge") // struct hkgpIndexedMeshDefinitions::Edge
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpIndexedMesh::EdgeMatch, s_libraryName)

// hk.MemoryTracker ignore SortByAscendingReferences

// StripConfig hkgpIndexedMesh
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpIndexedMesh, StripConfig, s_libraryName)


// SetInfos hkgpIndexedMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpIndexedMesh::SetInfos)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpIndexedMesh::SetInfos)
    HK_TRACKER_MEMBER(hkgpIndexedMesh::SetInfos, m_links, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpIndexedMesh::SetInfos, s_libraryName)


// EdgeBarrier hkgpIndexedMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpIndexedMesh::EdgeBarrier)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpIndexedMesh::EdgeBarrier)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpIndexedMesh::EdgeBarrier, s_libraryName)


// IVertexRemoval hkgpIndexedMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpIndexedMesh::IVertexRemoval)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpIndexedMesh::IVertexRemoval)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkgpIndexedMesh::IVertexRemoval, s_libraryName)


// ITriangleRemoval hkgpIndexedMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpIndexedMesh::ITriangleRemoval)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpIndexedMesh::ITriangleRemoval)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkgpIndexedMesh::ITriangleRemoval, s_libraryName, hkgpIndexedMesh::IVertexRemoval)


// IEdgeCollapse hkgpIndexedMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpIndexedMesh::IEdgeCollapse)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpIndexedMesh::IEdgeCollapse)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkgpIndexedMesh::IEdgeCollapse, s_libraryName, hkgpIndexedMesh::ITriangleRemoval)

#include <Common/Internal/GeometryProcessing/Mesh/hkgpMesh.h>


// hkgpMeshBase ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpMeshBase)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Vertex)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Triangle)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Edge)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkgpMeshBase, s_libraryName)

// hk.MemoryTracker ignore BaseVertex

// Vertex hkgpMeshBase
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpMeshBase, Vertex, s_libraryName)

// hk.MemoryTracker ignore BaseTriangle

// Triangle hkgpMeshBase
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpMeshBase, Triangle, s_libraryName)


// Edge hkgpMeshBase
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpMeshBase, Edge, s_libraryName)


// hkgpMesh ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpMesh)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TrianglePair)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SingleEdge)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FloodPolicy)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FloodFillDetachedPartsPolicy)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FloodFillDetachedOrMaterialBoundariesPartsPolicy)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CollapseEdgePolicy)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FanEdgeCollector)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CollapseMetric)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SimplifyConfig)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SurfaceSamplingConfig)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(HoleFillingConfig)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PointShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LineShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ExtrudeShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TriangleShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ExternShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConvexHullShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SortByArea)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Location)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(eHollows)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpMesh)
    HK_TRACKER_MEMBER(hkgpMesh, m_iconvexoverlap, 0, "hkgpMesh::IConvexOverlap*") // const class hkgpMesh::IConvexOverlap*
    HK_TRACKER_MEMBER(hkgpMesh, m_trianglesTree, 0, "void*") // void*
    HK_TRACKER_MEMBER(hkgpMesh, m_randSamples, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkgpMesh, m_planes, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkgpMesh, m_planeRoots, 0, "hkArray<hkgpMeshBase::Triangle*, hkContainerHeapAllocator>") // hkArray< struct hkgpMeshBase::Triangle*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkgpMesh, m_convexHull, 0, "hkgpConvexHull*") // class hkgpConvexHull*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkgpMesh, s_libraryName, hkgpAbstractMeshhkgpMeshBaseEdgehkgpMeshBaseVertexhkgpMeshBaseTrianglehkContainerHeapAllocator)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpMesh, eHollows, s_libraryName)


// TrianglePair hkgpMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpMesh::TrianglePair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpMesh::TrianglePair)
    HK_TRACKER_MEMBER(hkgpMesh::TrianglePair, m_a, 0, "hkgpMeshBase::Triangle*") // struct hkgpMeshBase::Triangle*
    HK_TRACKER_MEMBER(hkgpMesh::TrianglePair, m_b, 0, "hkgpMeshBase::Triangle*") // struct hkgpMeshBase::Triangle*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpMesh::TrianglePair, s_libraryName)


// SingleEdge hkgpMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpMesh::SingleEdge)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpMesh::SingleEdge)
    HK_TRACKER_MEMBER(hkgpMesh::SingleEdge, m_a, 0, "hkgpMeshBase::Vertex*") // const struct hkgpMeshBase::Vertex*
    HK_TRACKER_MEMBER(hkgpMesh::SingleEdge, m_b, 0, "hkgpMeshBase::Vertex*") // const struct hkgpMeshBase::Vertex*
    HK_TRACKER_MEMBER(hkgpMesh::SingleEdge, m_e, 0, "hkgpMeshBase::Edge") // struct hkgpMeshBase::Edge
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpMesh::SingleEdge, s_libraryName)


// FloodPolicy hkgpMesh
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpMesh, FloodPolicy, s_libraryName)


// FloodFillDetachedPartsPolicy hkgpMesh
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpMesh, FloodFillDetachedPartsPolicy, s_libraryName)


// FloodFillDetachedOrMaterialBoundariesPartsPolicy hkgpMesh
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpMesh, FloodFillDetachedOrMaterialBoundariesPartsPolicy, s_libraryName)


// CollapseEdgePolicy hkgpMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpMesh::CollapseEdgePolicy)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpMesh::CollapseEdgePolicy)
    HK_TRACKER_MEMBER(hkgpMesh::CollapseEdgePolicy, m_mesh, 0, "hkgpMesh*") // class hkgpMesh*
    HK_TRACKER_MEMBER(hkgpMesh::CollapseEdgePolicy, m_newVertex, 0, "hkgpMeshBase::Vertex*") // struct hkgpMeshBase::Vertex*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpMesh::CollapseEdgePolicy, s_libraryName)


// FanEdgeCollector hkgpMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpMesh::FanEdgeCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpMesh::FanEdgeCollector)
    HK_TRACKER_MEMBER(hkgpMesh::FanEdgeCollector, m_edges, 0, "hkInplaceArray<hkgpMeshBase::Edge, 16, hkContainerHeapAllocator>") // class hkInplaceArray< struct hkgpMeshBase::Edge, 16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpMesh::FanEdgeCollector, s_libraryName)


// CollapseMetric hkgpMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpMesh::CollapseMetric)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpMesh::CollapseMetric)
    HK_TRACKER_MEMBER(hkgpMesh::CollapseMetric, m_mesh, 0, "hkgpMesh*") // class hkgpMesh*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpMesh::CollapseMetric, s_libraryName)


// SimplifyConfig hkgpMesh
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpMesh, SimplifyConfig, s_libraryName)


// SurfaceSamplingConfig hkgpMesh
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpMesh, SurfaceSamplingConfig, s_libraryName)


// HoleFillingConfig hkgpMesh
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpMesh, HoleFillingConfig, s_libraryName)

// hk.MemoryTracker ignore IConvexOverlap

// PointShape hkgpMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpMesh::PointShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpMesh::PointShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkgpMesh::PointShape, s_libraryName, hkgpMesh::IConvexOverlap::IConvexShape)


// LineShape hkgpMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpMesh::LineShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpMesh::LineShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkgpMesh::LineShape, s_libraryName, hkgpMesh::IConvexOverlap::IConvexShape)


// ExtrudeShape hkgpMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpMesh::ExtrudeShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpMesh::ExtrudeShape)
    HK_TRACKER_MEMBER(hkgpMesh::ExtrudeShape, m_vertices, 0, "hkInplaceArray<hkFourTransposedPointsf, 2, hkContainerHeapAllocator>") // class hkInplaceArray< class hkFourTransposedPointsf, 2, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkgpMesh::ExtrudeShape, s_libraryName, hkgpMesh::IConvexOverlap::IConvexShape)


// TriangleShape hkgpMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpMesh::TriangleShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpMesh::TriangleShape)
    HK_TRACKER_MEMBER(hkgpMesh::TriangleShape, m_triangle, 0, "hkgpMeshBase::Triangle*") // struct hkgpMeshBase::Triangle*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkgpMesh::TriangleShape, s_libraryName, hkgpMesh::IConvexOverlap::IConvexShape)


// ExternShape hkgpMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpMesh::ExternShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpMesh::ExternShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkgpMesh::ExternShape, s_libraryName, hkgpMesh::IConvexOverlap::IConvexShape)


// ConvexHullShape hkgpMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpMesh::ConvexHullShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpMesh::ConvexHullShape)
    HK_TRACKER_MEMBER(hkgpMesh::ConvexHullShape, m_hull, 0, "hkgpConvexHull*") // const class hkgpConvexHull*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkgpMesh::ConvexHullShape, s_libraryName, hkgpMesh::IConvexOverlap::IConvexShape)


// SortByArea hkgpMesh
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpMesh, SortByArea, s_libraryName)


// Location hkgpMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpMesh::Location)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Region)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpMesh::Location)
    HK_TRACKER_MEMBER(hkgpMesh::Location, m_region, 0, "hkgpMesh::Location::Region") // struct hkgpMesh::Location::Region
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpMesh::Location, s_libraryName)


// Region hkgpMesh::Location

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpMesh::Location::Region)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(eType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpMesh::Location::Region)
    HK_TRACKER_MEMBER(hkgpMesh::Location::Region, m_feature, 0, "hkgpMeshBase::Edge") // struct hkgpMeshBase::Edge
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpMesh::Location::Region, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpMesh::Location::Region, eType, s_libraryName)

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
