/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Physics2012/Internal/hkpInternal.h>
static const char s_libraryName[] = "hkpInternalClient";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpInternalClientRegister() {}

#include <Physics2012/Internal/Collide/Gjk/Continuous/hkpContinuousGsk.h>


// hkp4dGskVertexCollidePointsInput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkp4dGskVertexCollidePointsInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkp4dGskVertexCollidePointsInput)
    HK_TRACKER_MEMBER(hkp4dGskVertexCollidePointsInput, m_motionA, 0, "hkPadSpu<hkMotionState*>") // class hkPadSpu< const class hkMotionState* >
    HK_TRACKER_MEMBER(hkp4dGskVertexCollidePointsInput, m_motionB, 0, "hkPadSpu<hkMotionState*>") // class hkPadSpu< const class hkMotionState* >
    HK_TRACKER_MEMBER(hkp4dGskVertexCollidePointsInput, m_verticesA, 0, "hkPadSpu<hkVector4f*>") // class hkPadSpu< hkVector4f* >
    HK_TRACKER_MEMBER(hkp4dGskVertexCollidePointsInput, m_stepInfo, 0, "hkPadSpu<hkStepInfo*>") // class hkPadSpu< const class hkStepInfo* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkp4dGskVertexCollidePointsInput, s_libraryName)


// hkp4dGskVertexCollidePointsOutput ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp4dGskVertexCollidePointsOutput, s_libraryName)


// hkp4dGskTolerances ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp4dGskTolerances, s_libraryName)

#include <Physics2012/Internal/Collide/Gjk/GskManifold/hkpGskManifoldUtil.h>


// hkpGskManifoldWork ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpGskManifoldWork, s_libraryName)

// None hkpGskManifoldAddStatus
HK_TRACKER_IMPLEMENT_SIMPLE(hkpGskManifoldAddStatus, s_libraryName)
// None hkpGskManifoldUtilMgrHandling
HK_TRACKER_IMPLEMENT_SIMPLE(hkpGskManifoldUtilMgrHandling, s_libraryName)
// None hkpGskManifoldPointExistsFlags
HK_TRACKER_IMPLEMENT_SIMPLE(hkpGskManifoldPointExistsFlags, s_libraryName)
#include <Physics2012/Internal/Collide/Gjk/hkpGsk.h>


// hkpGskOut ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpGskOut, s_libraryName)


// hkpExtendedGskOut ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpExtendedGskOut, s_libraryName)


// hkpGsk ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpGsk)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GetClosesetPointInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(NextCase)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SupportTypes)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SupportState)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ReduceDimensionResult)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpGsk, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpGsk, NextCase, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpGsk, SupportTypes, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpGsk, SupportState, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpGsk, ReduceDimensionResult, s_libraryName)


// GetClosesetPointInput hkpGsk

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpGsk::GetClosesetPointInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpGsk::GetClosesetPointInput)
    HK_TRACKER_MEMBER(hkpGsk::GetClosesetPointInput, m_aTb, 0, "hkPadSpu<hkTransformf*>") // class hkPadSpu< const hkTransformf* >
    HK_TRACKER_MEMBER(hkpGsk::GetClosesetPointInput, m_transformA, 0, "hkPadSpu<hkTransformf*>") // class hkPadSpu< const hkTransformf* >
    HK_TRACKER_MEMBER(hkpGsk::GetClosesetPointInput, m_shapeA, 0, "hkPadSpu<hkpConvexShape*>") // class hkPadSpu< const class hkpConvexShape* >
    HK_TRACKER_MEMBER(hkpGsk::GetClosesetPointInput, m_shapeB, 0, "hkPadSpu<hkpConvexShape*>") // class hkPadSpu< const class hkpConvexShape* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpGsk::GetClosesetPointInput, s_libraryName)

// None hkpGskStatus
HK_TRACKER_IMPLEMENT_SIMPLE(hkpGskStatus, s_libraryName)
#include <Physics2012/Internal/Collide/Mopp/Builder/Assembler/hkpMoppAssembler.h>


// hkpMoppAssembler ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppAssembler)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpMoppAssemblerParams)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppAssembler)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpMoppAssembler, s_libraryName, hkReferencedObject)


// hkpMoppAssemblerParams hkpMoppAssembler
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppAssembler, hkpMoppAssemblerParams, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Builder/Compiler/hkpMoppCompiler.h>


// hkpMoppCompilerChunkInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppCompilerChunkInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Chunk)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppCompilerChunkInfo)
    HK_TRACKER_MEMBER(hkpMoppCompilerChunkInfo, m_reindexInfo, 0, "hkArray<hkpMoppCodeReindexedTerminal, hkContainerHeapAllocator>") // hkArray< struct hkpMoppCodeReindexedTerminal, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpMoppCompilerChunkInfo, m_chunks, 0, "hkArray<hkpMoppCompilerChunkInfo::Chunk, hkContainerHeapAllocator>") // hkArray< struct hkpMoppCompilerChunkInfo::Chunk, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpMoppCompilerChunkInfo, m_compressor, 0, "hkpPrimitiveCompressor*") // class hkpPrimitiveCompressor*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpMoppCompilerChunkInfo, s_libraryName)


// Chunk hkpMoppCompilerChunkInfo

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppCompilerChunkInfo::Chunk)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppCompilerChunkInfo::Chunk)
    HK_TRACKER_MEMBER(hkpMoppCompilerChunkInfo::Chunk, m_code, 0, "hkpMoppCodeGenerator*") // class hkpMoppCodeGenerator*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpMoppCompilerChunkInfo::Chunk, s_libraryName)


// hkpMoppCompiler ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppCompiler)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppCompiler)
    HK_TRACKER_MEMBER(hkpMoppCompiler, m_debugRootNode, 0, "hkpMoppTreeNode*") // class hkpMoppTreeNode*
    HK_TRACKER_MEMBER(hkpMoppCompiler, m_chunkInfo, 0, "hkpMoppCompilerChunkInfo*") // struct hkpMoppCompilerChunkInfo*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpMoppCompiler, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Builder/Compiler/hkpPrimitiveCompressor.h>

// hk.MemoryTracker ignore hkpPrimitiveCompressor
#include <Physics2012/Internal/Collide/Mopp/Builder/Mediator/CachedShape/hkpMoppCachedShapeMediator.h>


// hkpMoppCachedShapeMediator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppCachedShapeMediator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpConvexShapeData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppCachedShapeMediator)
    HK_TRACKER_MEMBER(hkpMoppCachedShapeMediator, m_arrayConvexShapeData, 0, "hkArray<hkpMoppCachedShapeMediator::hkpConvexShapeData, hkContainerHeapAllocator>") // hkArray< struct hkpMoppCachedShapeMediator::hkpConvexShapeData, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpMoppCachedShapeMediator, m_shapeCollection, 0, "hkpShapeContainer*") // const class hkpShapeContainer*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppCachedShapeMediator, s_libraryName, hkpMoppMediator)


// hkpConvexShapeData hkpMoppCachedShapeMediator
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppCachedShapeMediator, hkpConvexShapeData, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Builder/Mediator/Shape/hkpMoppShapeMediator.h>


// hkpMoppShapeMediator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppShapeMediator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppShapeMediator)
    HK_TRACKER_MEMBER(hkpMoppShapeMediator, m_shape, 0, "hkpShapeContainer*") // const class hkpShapeContainer*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppShapeMediator, s_libraryName, hkpMoppMediator)

#include <Physics2012/Internal/Collide/Mopp/Builder/Mediator/hkpMoppMediator.h>


// hkpMoppMediator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppMediator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppMediator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpMoppMediator, s_libraryName, hkReferencedObject)

#include <Physics2012/Internal/Collide/Mopp/Builder/Splitter/hkpMoppCostFunction.h>


// hkpMoppCostFunction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppCostFunction)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpMoppSplitCostParams)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpPlaneRightParams)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpPlanesParams)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppCostFunction)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppCostFunction, s_libraryName, hkReferencedObject)


// hkpMoppSplitCostParams hkpMoppCostFunction
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppCostFunction, hkpMoppSplitCostParams, s_libraryName)


// hkpPlaneRightParams hkpMoppCostFunction

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppCostFunction::hkpPlaneRightParams)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppCostFunction::hkpPlaneRightParams)
    HK_TRACKER_MEMBER(hkpMoppCostFunction::hkpPlaneRightParams, m_plane, 0, "hkpMoppSplittingPlaneDirection*") // const struct hkpMoppSplittingPlaneDirection*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpMoppCostFunction::hkpPlaneRightParams, s_libraryName)


// hkpPlanesParams hkpMoppCostFunction
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppCostFunction, hkpPlanesParams, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Builder/Splitter/hkpMoppSplitTypes.h>


// hkpMoppExtent ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppExtent, s_libraryName)


// hkpMoppCompilerPrimitive ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppCompilerPrimitive, s_libraryName)


// hkpMoppSplittingPlaneDirection ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppSplittingPlaneDirection, s_libraryName)


// hkpMoppAssemblerData ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppAssemblerData, s_libraryName)


// hkpMoppTreeNode ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppTreeNode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpMopp3DOPExtents)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppTreeNode)
    HK_TRACKER_MEMBER(hkpMoppTreeNode, m_parent, 0, "hkpMoppTreeInternalNode*") // class hkpMoppTreeInternalNode*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpMoppTreeNode, s_libraryName)


// hkpMopp3DOPExtents hkpMoppTreeNode
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppTreeNode, hkpMopp3DOPExtents, s_libraryName)


// hkpMoppTreeTerminal ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppTreeTerminal)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppTreeTerminal)
    HK_TRACKER_MEMBER(hkpMoppTreeTerminal, m_primitive, 0, "hkpMoppCompilerPrimitive*") // struct hkpMoppCompilerPrimitive*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppTreeTerminal, s_libraryName, hkpMoppTreeNode)


// hkpMoppBasicNode ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppBasicNode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpMoppCostInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppBasicNode)
    HK_TRACKER_MEMBER(hkpMoppBasicNode, m_plane, 0, "hkpMoppSplittingPlaneDirection*") // const struct hkpMoppSplittingPlaneDirection*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppBasicNode, s_libraryName, hkpMoppTreeNode)


// hkpMoppCostInfo hkpMoppBasicNode
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppBasicNode, hkpMoppCostInfo, s_libraryName)


// hkpMoppTreeInternalNode ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppTreeInternalNode)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppTreeInternalNode)
    HK_TRACKER_MEMBER(hkpMoppTreeInternalNode, m_leftBranch, 0, "hkpMoppTreeNode*") // class hkpMoppTreeNode*
    HK_TRACKER_MEMBER(hkpMoppTreeInternalNode, m_rightBranch, 0, "hkpMoppTreeNode*") // class hkpMoppTreeNode*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppTreeInternalNode, s_libraryName, hkpMoppBasicNode)

// None hkpMoppMeshType
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppMeshType, s_libraryName)
#include <Physics2012/Internal/Collide/Mopp/Builder/Splitter/hkpMoppSplitter.h>


// hkpMoppNodeMgr ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppNodeMgr)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppNodeMgr)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpMoppNodeMgr, s_libraryName, hkReferencedObject)


// hkpMoppSplitter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppSplitter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpMoppScratchArea)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpMoppSplitParams)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppSplitter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpMoppSplitter, s_libraryName, hkpMoppNodeMgr)


// hkpMoppScratchArea hkpMoppSplitter

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppSplitter::hkpMoppScratchArea)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppSplitter::hkpMoppScratchArea)
    HK_TRACKER_MEMBER(hkpMoppSplitter::hkpMoppScratchArea, m_primitives, 0, "hkpMoppCompilerPrimitive*") // struct hkpMoppCompilerPrimitive*
    HK_TRACKER_MEMBER(hkpMoppSplitter::hkpMoppScratchArea, m_nodes, 0, "hkpMoppTreeInternalNode*") // class hkpMoppTreeInternalNode*
    HK_TRACKER_MEMBER(hkpMoppSplitter::hkpMoppScratchArea, m_terminals, 0, "hkpMoppTreeTerminal*") // class hkpMoppTreeTerminal*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpMoppSplitter::hkpMoppScratchArea, s_libraryName)


// hkpMoppSplitParams hkpMoppSplitter
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppSplitter, hkpMoppSplitParams, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCommands.h>

// None HK_MOPP_SPLIT_DIRECTIONS
HK_TRACKER_IMPLEMENT_SIMPLE(HK_MOPP_SPLIT_DIRECTIONS, s_libraryName)
// None hkpMoppCommands
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppCommands, s_libraryName)
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppAabbCastVirtualMachine.h>


// hkpMoppAabbCastVirtualMachineQueryInt ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppAabbCastVirtualMachineQueryInt, s_libraryName)


// hkpMoppAabbCastVirtualMachineQueryFloat ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppAabbCastVirtualMachineQueryFloat, s_libraryName)


// hkpMoppAabbCastVirtualMachine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppAabbCastVirtualMachine)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpAabbCastInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppAabbCastVirtualMachine)
    HK_TRACKER_MEMBER(hkpMoppAabbCastVirtualMachine, m_code, 0, "hkpMoppCode*") // const class hkpMoppCode*
    HK_TRACKER_MEMBER(hkpMoppAabbCastVirtualMachine, m_castCollector, 0, "hkpCdPointCollector*") // class hkpCdPointCollector*
    HK_TRACKER_MEMBER(hkpMoppAabbCastVirtualMachine, m_startPointCollector, 0, "hkpCdPointCollector*") // class hkpCdPointCollector*
    HK_TRACKER_MEMBER(hkpMoppAabbCastVirtualMachine, m_input, 0, "hkpMoppAabbCastVirtualMachine::hkpAabbCastInput*") // const struct hkpMoppAabbCastVirtualMachine::hkpAabbCastInput*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppAabbCastVirtualMachine, s_libraryName, hkpMoppVirtualMachine)


// hkpAabbCastInput hkpMoppAabbCastVirtualMachine

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppAabbCastVirtualMachine::hkpAabbCastInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppAabbCastVirtualMachine::hkpAabbCastInput)
    HK_TRACKER_MEMBER(hkpMoppAabbCastVirtualMachine::hkpAabbCastInput, m_collisionInput, 0, "hkpLinearCastCollisionInput*") // const struct hkpLinearCastCollisionInput*
    HK_TRACKER_MEMBER(hkpMoppAabbCastVirtualMachine::hkpAabbCastInput, m_castBody, 0, "hkpCdBody*") // const class hkpCdBody*
    HK_TRACKER_MEMBER(hkpMoppAabbCastVirtualMachine::hkpAabbCastInput, m_moppBody, 0, "hkpCdBody*") // const class hkpCdBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpMoppAabbCastVirtualMachine::hkpAabbCastInput, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppEarlyExitObbVirtualMachine.h>


// hkpMoppEarlyExitObbVirtualMachineQuery ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppEarlyExitObbVirtualMachineQuery, s_libraryName)


// hkpMoppEarlyExitObbVirtualMachine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppEarlyExitObbVirtualMachine)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppEarlyExitObbVirtualMachine)
    HK_TRACKER_MEMBER(hkpMoppEarlyExitObbVirtualMachine, m_code, 0, "hkpMoppCode*") // const class hkpMoppCode*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppEarlyExitObbVirtualMachine, s_libraryName, hkpMoppVirtualMachine)

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppFindAllVirtualMachine.h>


// hkpMoppFindAllVirtualMachine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppFindAllVirtualMachine)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpMoppFindAllVirtualMachineQuery)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppFindAllVirtualMachine)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppFindAllVirtualMachine, s_libraryName, hkpMoppVirtualMachine)


// hkpMoppFindAllVirtualMachineQuery hkpMoppFindAllVirtualMachine
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppFindAllVirtualMachine, hkpMoppFindAllVirtualMachineQuery, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppIntAabbVirtualMachine.h>


// hkpMoppIntAabbVirtualMachineQuery ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppIntAabbVirtualMachineQuery, s_libraryName)


// hkpMoppIntAabbVirtualMachine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppIntAabbVirtualMachine)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppIntAabbVirtualMachine)
    HK_TRACKER_MEMBER(hkpMoppIntAabbVirtualMachine, m_code, 0, "hkpMoppCode*") // const class hkpMoppCode*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppIntAabbVirtualMachine, s_libraryName, hkpMoppVirtualMachine)

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppKDopGeometriesVirtualMachine.h>


// hkpMoppKDopGeometriesVirtualMachine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppKDopGeometriesVirtualMachine)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpMoppKDopGeometriesVirtualMachineQuery)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppKDopGeometriesVirtualMachine)
    HK_TRACKER_MEMBER(hkpMoppKDopGeometriesVirtualMachine, m_visitedTerminals, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpMoppKDopGeometriesVirtualMachine, m_kDopGeometries, 0, "hkpMoppInfo*") // struct hkpMoppInfo*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppKDopGeometriesVirtualMachine, s_libraryName, hkpMoppVirtualMachine)


// hkpMoppKDopGeometriesVirtualMachineQuery hkpMoppKDopGeometriesVirtualMachine
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppKDopGeometriesVirtualMachine, hkpMoppKDopGeometriesVirtualMachineQuery, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppLongRayVirtualMachine.h>


// hkpMoppLongRayVirtualMachine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppLongRayVirtualMachine)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppLongRayVirtualMachine)
    HK_TRACKER_MEMBER(hkpMoppLongRayVirtualMachine, m_code, 0, "hkpMoppCode*") // const class hkpMoppCode*
    HK_TRACKER_MEMBER(hkpMoppLongRayVirtualMachine, m_ray, 0, "hkpShapeRayCastInput") // struct hkpShapeRayCastInput
    HK_TRACKER_MEMBER(hkpMoppLongRayVirtualMachine, m_rayResultPtr, 0, "hkpShapeRayCastOutput*") // struct hkpShapeRayCastOutput*
    HK_TRACKER_MEMBER(hkpMoppLongRayVirtualMachine, m_collector, 0, "hkpRayHitCollector*") // class hkpRayHitCollector*
    HK_TRACKER_MEMBER(hkpMoppLongRayVirtualMachine, m_body, 0, "hkpCdBody*") // const class hkpCdBody*
    HK_TRACKER_MEMBER(hkpMoppLongRayVirtualMachine, m_collection, 0, "hkpShapeContainer*") // const class hkpShapeContainer*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppLongRayVirtualMachine, s_libraryName, hkpMoppVirtualMachine)

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppModifyVirtualMachine.h>


// hkpMoppModifyVirtualMachine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppModifyVirtualMachine)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppModifyVirtualMachine)
    HK_TRACKER_MEMBER(hkpMoppModifyVirtualMachine, m_modifier, 0, "hkpMoppModifier*") // class hkpMoppModifier*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppModifyVirtualMachine, s_libraryName, hkpMoppObbVirtualMachine)

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppObbVirtualMachine.h>


// hkpMoppObbVirtualMachineQuery ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppObbVirtualMachineQuery, s_libraryName)


// hkpMoppObbVirtualMachine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppObbVirtualMachine)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppObbVirtualMachine)
    HK_TRACKER_MEMBER(hkpMoppObbVirtualMachine, m_code, 0, "hkPadSpu<hkpMoppCode*>") // class hkPadSpu< const class hkpMoppCode* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppObbVirtualMachine, s_libraryName, hkpMoppVirtualMachine)

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppRayBundleVirtualMachine.h>


// RayPointBundle ::
HK_TRACKER_IMPLEMENT_SIMPLE(RayPointBundle, s_libraryName)


// hkpMoppRayBundleVirtualMachine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppRayBundleVirtualMachine)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppRayBundleVirtualMachine)
    HK_TRACKER_MEMBER(hkpMoppRayBundleVirtualMachine, m_code, 0, "hkpMoppCode*") // const class hkpMoppCode*
    HK_TRACKER_MEMBER(hkpMoppRayBundleVirtualMachine, m_rays, 0, "hkpShapeRayBundleCastInput*") // const struct hkpShapeRayBundleCastInput*
    HK_TRACKER_MEMBER(hkpMoppRayBundleVirtualMachine, m_rayResultPtr, 0, "hkpShapeRayBundleCastOutput*") // struct hkpShapeRayBundleCastOutput*
    HK_TRACKER_MEMBER(hkpMoppRayBundleVirtualMachine, m_collector, 0, "hkpRayHitCollector*") // class hkpRayHitCollector*
    HK_TRACKER_MEMBER(hkpMoppRayBundleVirtualMachine, m_body, 0, "hkpCdBody*") // const class hkpCdBody*
    HK_TRACKER_MEMBER(hkpMoppRayBundleVirtualMachine, m_collection, 0, "hkpShapeContainer*") // const class hkpShapeContainer*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppRayBundleVirtualMachine, s_libraryName, hkpMoppVirtualMachine)

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppRayVirtualMachine.h>


// hkpMoppRayVirtualMachineQuery ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppRayVirtualMachineQuery, s_libraryName)


// hkpMoppRayVirtualMachine ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppRayVirtualMachine, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppSphereVirtualMachine.h>


// hkpMoppSphereVirtualMachine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppSphereVirtualMachine)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpMoppSphereVirtualMachineQuery)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppSphereVirtualMachine)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppSphereVirtualMachine, s_libraryName, hkpMoppVirtualMachine)


// hkpMoppSphereVirtualMachineQuery hkpMoppSphereVirtualMachine
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppSphereVirtualMachine, hkpMoppSphereVirtualMachineQuery, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppStatisticsVirtualMachine.h>


// hkpMoppStatisticsVirtualMachine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppStatisticsVirtualMachine)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Entry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpMoppStatisticsVirtualMachineQuery)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppStatisticsVirtualMachine)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppStatisticsVirtualMachine, s_libraryName, hkpMoppVirtualMachine)


// Entry hkpMoppStatisticsVirtualMachine
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppStatisticsVirtualMachine, Entry, s_libraryName)


// hkpMoppStatisticsVirtualMachineQuery hkpMoppStatisticsVirtualMachine
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppStatisticsVirtualMachine, hkpMoppStatisticsVirtualMachineQuery, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppVirtualMachine.h>


// hkpMoppPrimitiveInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppPrimitiveInfo, s_libraryName)


// hkpMoppVirtualMachine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppVirtualMachine)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppVirtualMachine)
    HK_TRACKER_MEMBER(hkpMoppVirtualMachine, m_primitives_out, 0, "hkArray<hkpMoppPrimitiveInfo, hkContainerHeapAllocator>*") // hkArray< class hkpMoppPrimitiveInfo, struct hkContainerHeapAllocator >*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpMoppVirtualMachine, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Utility/hkpMoppDebugger.h>


// hkpMoppDebugger ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppDebugger)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpMoppPath)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpDbgQuery)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppDebugger)
    HK_TRACKER_MEMBER(hkpMoppDebugger, m_paths, 0, "hkArray<hkpMoppDebugger::hkpMoppPath, hkContainerHeapAllocator>") // hkArray< class hkpMoppDebugger::hkpMoppPath, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpMoppDebugger, m_searchPath, 0, "char*") // char*
    HK_TRACKER_MEMBER(hkpMoppDebugger, m_moppCode, 0, "hkpMoppCode*") // const class hkpMoppCode*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppDebugger, s_libraryName, hkReferencedObject)


// hkpMoppPath hkpMoppDebugger
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppDebugger, hkpMoppPath, s_libraryName)


// hkpDbgQuery hkpMoppDebugger
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppDebugger, hkpDbgQuery, s_libraryName)

#include <Physics2012/Internal/Dynamics/World/Simulation/Continuous/ToiResourceMgr/Default/hkpDefaultToiResourceMgr.h>


// hkpDefaultToiResourceMgr ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDefaultToiResourceMgr)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDefaultToiResourceMgr)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDefaultToiResourceMgr, s_libraryName, hkpToiResourceMgr)

#include <Physics2012/Internal/Dynamics/World/Simulation/Continuous/ToiResourceMgr/hkpToiResourceMgr.h>


// hkpToiResources ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpToiResources)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpToiResources)
    HK_TRACKER_MEMBER(hkpToiResources, m_scratchpad, 0, "char*") // char*
    HK_TRACKER_MEMBER(hkpToiResources, m_priorityClassMap, 0, "hkUint8*") // const hkUint8*
    HK_TRACKER_MEMBER(hkpToiResources, m_priorityClassRatios, 0, "float*") // const float*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpToiResources, s_libraryName)


// hkpToiResourceMgr ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpToiResourceMgr)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConstraintViolationInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpToiResourceMgr)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpToiResourceMgr, s_libraryName, hkReferencedObject)


// ConstraintViolationInfo hkpToiResourceMgr

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpToiResourceMgr::ConstraintViolationInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpToiResourceMgr::ConstraintViolationInfo)
    HK_TRACKER_MEMBER(hkpToiResourceMgr::ConstraintViolationInfo, m_constraint, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
    HK_TRACKER_MEMBER(hkpToiResourceMgr::ConstraintViolationInfo, m_contactPoint, 0, "hkContactPoint*") // const class hkContactPoint*
    HK_TRACKER_MEMBER(hkpToiResourceMgr::ConstraintViolationInfo, m_contactPointProperties, 0, "hkpContactPointProperties*") // const class hkpContactPointProperties*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpToiResourceMgr::ConstraintViolationInfo, s_libraryName)

// None hkpToiResourceMgrResponse
HK_TRACKER_IMPLEMENT_SIMPLE(hkpToiResourceMgrResponse, s_libraryName)
#include <Physics2012/Internal/Dynamics/World/Simulation/Continuous/hkpToiEvent.h>


// hkpToiEvent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpToiEvent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpToiEvent)
    HK_TRACKER_MEMBER(hkpToiEvent, m_entities, 0, "hkpEntity* [2]") // class hkpEntity* [2]
    HK_TRACKER_MEMBER(hkpToiEvent, m_contactMgr, 0, "hkpDynamicsContactMgr*") // class hkpDynamicsContactMgr*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpToiEvent, s_libraryName)

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
