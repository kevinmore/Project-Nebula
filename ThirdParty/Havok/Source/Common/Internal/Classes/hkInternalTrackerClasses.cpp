/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Common/Internal/hkInternal.h>
static const char s_libraryName[] = "hkInternal";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkInternalRegister() {}

#include <Common/Internal/ConvexHull/hkGeometryUtility.h>


// hkGeometryUtility ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkGeometryUtility, s_libraryName)

// None hkGeomConvexHullMode
HK_TRACKER_IMPLEMENT_SIMPLE(hkGeomConvexHullMode, s_libraryName)
// None hkGeomObbMode
HK_TRACKER_IMPLEMENT_SIMPLE(hkGeomObbMode, s_libraryName)
#include <Common/Internal/ConvexHull/hkPlaneEquationUtil.h>


// hkPlaneEquationUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkPlaneEquationUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IntersectionPoint)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VisitedEdge)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpPlaneEqnIndexPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkPlaneEquationUtil, s_libraryName)


// IntersectionPoint hkPlaneEquationUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkPlaneEquationUtil, IntersectionPoint, s_libraryName)


// VisitedEdge hkPlaneEquationUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkPlaneEquationUtil::VisitedEdge)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkPlaneEquationUtil::VisitedEdge)
    HK_TRACKER_MEMBER(hkPlaneEquationUtil::VisitedEdge, m_fromPoint, 0, "hkPlaneEquationUtil::IntersectionPoint*") // struct hkPlaneEquationUtil::IntersectionPoint*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkPlaneEquationUtil::VisitedEdge, s_libraryName)


// hkpPlaneEqnIndexPair hkPlaneEquationUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkPlaneEquationUtil, hkpPlaneEqnIndexPair, s_libraryName)

#include <Common/Internal/GeometryProcessing/AbstractMesh/hkgpAbstractMesh.h>

// hk.MemoryTracker ignore hkgpAbstractMeshDefinitions
#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>


// hkgpConvexHull ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpConvexHull)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BuildConfig)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SimplifyConfig)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AbsoluteScaleConfig)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Inputs)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Side)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpConvexHull)
    HK_TRACKER_MEMBER(hkgpConvexHull, m_data, 0, "hkgpConvexHullImpl*") // class hkgpConvexHullImpl*
    HK_TRACKER_MEMBER(hkgpConvexHull, m_userData, 0, "void*") // void*
    HK_TRACKER_MEMBER(hkgpConvexHull, m_userObject, 0, "hkgpConvexHull::IUserObject*") // struct hkgpConvexHull::IUserObject*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkgpConvexHull, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpConvexHull, Inputs, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpConvexHull, Side, s_libraryName)


// BuildConfig hkgpConvexHull
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpConvexHull, BuildConfig, s_libraryName)


// SimplifyConfig hkgpConvexHull
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpConvexHull, SimplifyConfig, s_libraryName)


// AbsoluteScaleConfig hkgpConvexHull
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpConvexHull, AbsoluteScaleConfig, s_libraryName)

// hk.MemoryTracker ignore IUserObject
// hk.MemoryTracker ignore IBooleanFunction
// hk.MemoryTracker ignore Vertex
// hk.MemoryTracker ignore Triangle
#include <Common/Internal/GeometryProcessing/JobQueue/hkgpJobQueue.h>


// hkgpJobQueue ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpJobQueue)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpJobQueue)
    HK_TRACKER_MEMBER(hkgpJobQueue, m_threads, 0, "hkInplaceArray<hkThread*, 8, hkContainerHeapAllocator>") // class hkInplaceArray< class hkThread*, 8, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkgpJobQueue, m_jobsLock, 0, "hkCriticalSection*") // class hkCriticalSection*
    HK_TRACKER_MEMBER(hkgpJobQueue, m_newJobEvent, 0, "hkSemaphore*") // class hkSemaphore*
    HK_TRACKER_MEMBER(hkgpJobQueue, m_endJobEvent, 0, "hkSemaphore*") // class hkSemaphore*
    HK_TRACKER_MEMBER(hkgpJobQueue, m_endThreadEvent, 0, "hkSemaphore*") // class hkSemaphore*
    HK_TRACKER_MEMBER(hkgpJobQueue, m_jobs, 0, "hkArray<hkgpJobQueue::IJob*, hkContainerHeapAllocator>") // hkArray< struct hkgpJobQueue::IJob*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpJobQueue, s_libraryName)

// hk.MemoryTracker ignore IJob
#include <Common/Internal/GeometryProcessing/Topology/hkgpTopology.h>


// hkgpTopology ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkgpTopology, s_libraryName)

#include <Common/Internal/GeometryProcessing/Topology/hkgpVertexTriangleTopology.h>


// hkgpVertexTriangleTopologyBase ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpVertexTriangleTopologyBase)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Triangle)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Edge)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CheckFlag)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpVertexTriangleTopologyBase)
    HK_TRACKER_MEMBER(hkgpVertexTriangleTopologyBase, m_triangleFreeList, 0, "hkFreeList") // class hkFreeList
    HK_TRACKER_MEMBER(hkgpVertexTriangleTopologyBase, m_triangles, 0, "hkArray<hkgpVertexTriangleTopologyBase::Triangle*, hkContainerHeapAllocator>") // hkArray< struct hkgpVertexTriangleTopologyBase::Triangle*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkgpVertexTriangleTopologyBase, m_vertexEdgeMap, 0, "hkPointerMultiMap<hkInt32, hkUint32>") // class hkPointerMultiMap< hkInt32, hkUint32 >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpVertexTriangleTopologyBase, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpVertexTriangleTopologyBase, CheckFlag, s_libraryName)


// Triangle hkgpVertexTriangleTopologyBase
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpVertexTriangleTopologyBase, Triangle, s_libraryName)


// Edge hkgpVertexTriangleTopologyBase

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpVertexTriangleTopologyBase::Edge)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpVertexTriangleTopologyBase::Edge)
    HK_TRACKER_MEMBER(hkgpVertexTriangleTopologyBase::Edge, m_triangle, 0, "hkgpVertexTriangleTopologyBase::Triangle*") // struct hkgpVertexTriangleTopologyBase::Triangle*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpVertexTriangleTopologyBase::Edge, s_libraryName)

#include <Common/Internal/MeshSimplifier/hkMeshSimplifier.h>


// hkQemMutableMesh ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkQemMutableMesh)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Face)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Vertex)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkQemMutableMesh)
    HK_TRACKER_MEMBER(hkQemMutableMesh, m_neighorhoods, 0, "hkArray<hkArray<hkInt32, hkContainerHeapAllocator>, hkContainerHeapAllocator>") // hkArray< hkArray< hkInt32, struct hkContainerHeapAllocator >, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkQemMutableMesh, m_vertices, 0, "hkArray<hkQemMutableMesh::Vertex, hkContainerHeapAllocator>") // hkArray< struct hkQemMutableMesh::Vertex, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkQemMutableMesh, m_faces, 0, "hkArray<hkQemMutableMesh::Face, hkContainerHeapAllocator>") // hkArray< struct hkQemMutableMesh::Face, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkQemMutableMesh, m_vertexMap, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkQemMutableMesh, s_libraryName)


// Face hkQemMutableMesh
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkQemMutableMesh, Face, s_libraryName)


// Vertex hkQemMutableMesh
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkQemMutableMesh, Vertex, s_libraryName)


// hkQemPairContraction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkQemPairContraction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkQemPairContraction)
    HK_TRACKER_MEMBER(hkQemPairContraction, m_deadFaces, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkQemPairContraction, m_deltaFaces, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkQemPairContraction, s_libraryName)


// hkQemVertexPair ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkQemVertexPair, s_libraryName)


// hkQemQuadric ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkQemQuadric, s_libraryName)


// hkQemMeshSimplifier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkQemMeshSimplifier)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Parameters)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Contraction)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(HeapEntry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SimplificationStatus)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkQemMeshSimplifier)
    HK_TRACKER_MEMBER(hkQemMeshSimplifier, m_heap, 0, "hkArray<hkQemMeshSimplifier::HeapEntry, hkContainerHeapAllocator>") // hkArray< struct hkQemMeshSimplifier::HeapEntry, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkQemMeshSimplifier, m_mesh, 0, "hkQemMutableMesh*") // class hkQemMutableMesh*
    HK_TRACKER_MEMBER(hkQemMeshSimplifier, m_quadrics, 0, "hkArray<hkQemQuadric, hkContainerHeapAllocator>") // hkArray< struct hkQemQuadric, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkQemMeshSimplifier, m_edgeMapping, 0, "hkArray<hkArray<hkInt32, hkContainerHeapAllocator>, hkContainerHeapAllocator>") // hkArray< hkArray< hkInt32, struct hkContainerHeapAllocator >, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkQemMeshSimplifier, m_pairs, 0, "hkArray<hkQemVertexPair, hkContainerHeapAllocator>") // hkArray< struct hkQemVertexPair, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkQemMeshSimplifier, m_contractionHistory, 0, "hkArray<hkQemMeshSimplifier::Contraction, hkContainerHeapAllocator>") // hkArray< struct hkQemMeshSimplifier::Contraction, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkQemMeshSimplifier, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkQemMeshSimplifier, SimplificationStatus, s_libraryName)


// Parameters hkQemMeshSimplifier
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkQemMeshSimplifier, Parameters, s_libraryName)


// Contraction hkQemMeshSimplifier
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkQemMeshSimplifier, Contraction, s_libraryName)


// HeapEntry hkQemMeshSimplifier
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkQemMeshSimplifier, HeapEntry, s_libraryName)

#include <Common/Internal/Misc/hkSystemDate.h>


// hkSystemDate ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkSystemDate, s_libraryName)

#include <Common/Internal/SimplexSolver/hkSimplexSolver.h>


// hkSurfaceConstraintInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkSurfaceConstraintInfo, s_libraryName)


// hkSimplexSolverInput ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkSimplexSolverInput, s_libraryName)


// hkSurfaceConstraintInteraction ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkSurfaceConstraintInteraction, s_libraryName)


// hkSimplexSolverOutput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSimplexSolverOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSimplexSolverOutput)
    HK_TRACKER_MEMBER(hkSimplexSolverOutput, m_planeInteractions, 0, "hkSurfaceConstraintInteraction*") // struct hkSurfaceConstraintInteraction*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSimplexSolverOutput, s_libraryName)

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
