/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Physics/Internal/hknpInternal.h>
static const char s_libraryName[] = "hknpInternal";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hknpInternalRegister() {}

#include <Physics/Internal/Collide/BroadPhase/Hybrid/hknpHybridBroadPhase.h>


// hknpHybridBroadPhase ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpHybridBroadPhase)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpHybridBroadPhase)
    HK_TRACKER_MEMBER(hknpHybridBroadPhase, m_broadPhaseConfig, 0, "hknpBroadPhaseConfig *") // const class hknpBroadPhaseConfig *
    HK_TRACKER_MEMBER(hknpHybridBroadPhase, m_trees, 0, "hknpHybridAabbTree<hkUint32, hkUint32>* [8]") // class hknpHybridAabbTree< hkUint32, hkUint32 >* [8]
    HK_TRACKER_MEMBER(hknpHybridBroadPhase, m_collideTreePairs, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpHybridBroadPhase, m_dirtyBodies, 0, "hkBitField") // class hkBitField
    HK_TRACKER_MEMBER(hknpHybridBroadPhase, m_taskContext, 0, "hknpHybridBroadPhaseTaskContext*") // class hknpHybridBroadPhaseTaskContext*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpHybridBroadPhase, s_libraryName, hknpBroadPhase)

#include <Physics/Internal/Collide/BroadPhase/Hybrid/hknpHybridBroadPhaseTasks.h>


// hknpHybridBroadPhaseTaskContext ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpHybridBroadPhaseTaskContext)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpHybridBroadPhaseTaskContext)
    HK_TRACKER_MEMBER(hknpHybridBroadPhaseTaskContext, m_simulationContext, 0, "hknpSimulationContext*") // class hknpSimulationContext*
    HK_TRACKER_MEMBER(hknpHybridBroadPhaseTaskContext, m_broadPhase, 0, "hknpHybridBroadPhase*") // class hknpHybridBroadPhase*
    HK_TRACKER_MEMBER(hknpHybridBroadPhaseTaskContext, m_bodies, 0, "hknpBody*") // class hknpBody*
    HK_TRACKER_MEMBER(hknpHybridBroadPhaseTaskContext, m_previousAabbs, 0, "hkAabb16*") // const struct hkAabb16*
    HK_TRACKER_MEMBER(hknpHybridBroadPhaseTaskContext, m_bodyManager, 0, "hknpBodyManager*") // class hknpBodyManager*
    HK_TRACKER_MEMBER(hknpHybridBroadPhaseTaskContext, m_updateDirtyBodiesTask, 0, "hknpHBPUpdateDirtyBodiesTask*") // class hknpHBPUpdateDirtyBodiesTask*
    HK_TRACKER_MEMBER(hknpHybridBroadPhaseTaskContext, m_updateTreeTaskPool, 0, "hknpHBPUpdateTreeTask* [8]") // class hknpHBPUpdateTreeTask* [8]
    HK_TRACKER_MEMBER(hknpHybridBroadPhaseTaskContext, m_treeVsTreeTaskPool, 0, "hknpHBPTreeVsTreeTask* [16]") // class hknpHBPTreeVsTreeTask* [16]
    HK_TRACKER_MEMBER(hknpHybridBroadPhaseTaskContext, m_finishUpdateTask, 0, "hknpHBPFinishUpdateTask*") // class hknpHBPFinishUpdateTask*
    HK_TRACKER_MEMBER(hknpHybridBroadPhaseTaskContext, m_newPairsStream, 0, "hkBlockStream<hknpBodyIdPair>*") // class hkBlockStream< struct hknpBodyIdPair >*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpHybridBroadPhaseTaskContext, s_libraryName)

#include <Physics/Internal/Collide/Gsk/hknpGskUtil.h>


// hknpGskUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpGskUtil, s_libraryName)

#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>


// hknpDeactivatedIsland ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDeactivatedIsland)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ActivationInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDeactivatedIsland)
    HK_TRACKER_MEMBER(hknpDeactivatedIsland, m_cdCaches, 0, "hknpCdCacheRange") // class hknpCdCacheRange
    HK_TRACKER_MEMBER(hknpDeactivatedIsland, m_cdChildCaches, 0, "hknpCdCacheRange") // class hknpCdCacheRange
    HK_TRACKER_MEMBER(hknpDeactivatedIsland, m_bodyIds, 0, "hkArray<hknpBodyId, hkContainerHeapAllocator>") // hkArray< struct hknpBodyId, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpDeactivatedIsland, m_activationListeners, 0, "hkArray<hknpDeactivatedIsland::ActivationInfo, hkContainerHeapAllocator>") // hkArray< struct hknpDeactivatedIsland::ActivationInfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpDeactivatedIsland, m_deletedCaches, 0, "hkArray<hknpBodyIdPair, hkContainerHeapAllocator>") // hkArray< struct hknpBodyIdPair, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpDeactivatedIsland, s_libraryName)


// ActivationInfo hknpDeactivatedIsland

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDeactivatedIsland::ActivationInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDeactivatedIsland::ActivationInfo)
    HK_TRACKER_MEMBER(hknpDeactivatedIsland::ActivationInfo, m_activationListener, 0, "hknpActivationListener*") // class hknpActivationListener*
    HK_TRACKER_MEMBER(hknpDeactivatedIsland::ActivationInfo, m_userData, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpDeactivatedIsland::ActivationInfo, s_libraryName)


// hknpDeactivationManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDeactivationManager)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDeactivationManager)
    HK_TRACKER_MEMBER(hknpDeactivationManager, m_world, 0, "hknpWorld*") // class hknpWorld*
    HK_TRACKER_MEMBER(hknpDeactivationManager, m_deactivationStates, 0, "hkArray<hknpDeactivationState, hkContainerHeapAllocator>") // hkArray< struct hknpDeactivationState, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpDeactivationManager, m_deactivatedIslands, 0, "hkArray<hknpDeactivatedIsland*, hkContainerHeapAllocator>") // hkArray< class hknpDeactivatedIsland*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpDeactivationManager, m_freeIslandIds, 0, "hkArray<hkHandle<hkUint16, 65535, hknpIslandIdDiscriminant>, hkContainerHeapAllocator>") // hkArray< struct hkHandle< hkUint16, 65535, struct hknpIslandIdDiscriminant >, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpDeactivationManager, m_bodyLinks, 0, "hkPointerMap<hkUint64, hkInt32, hkContainerHeapAllocator>") // class hkPointerMap< hkUint64, hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpDeactivationManager, m_bodiesMarkedForDeactivation, 0, "hkArray<hknpBodyId, hkContainerHeapAllocator>") // hkArray< struct hknpBodyId, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpDeactivationManager, m_islandsMarkedForActivation, 0, "hkArray<hkHandle<hkUint16, 65535, hknpIslandIdDiscriminant>, hkContainerHeapAllocator>") // hkArray< struct hkHandle< hkUint16, 65535, struct hknpIslandIdDiscriminant >, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpDeactivationManager, m_newlyDeactivatedIsland, 0, "hknpDeactivatedIsland*") // class hknpDeactivatedIsland*
    HK_TRACKER_MEMBER(hknpDeactivationManager, m_inactiveIslands, 0, "hkArray<hkHandle<hkUint16, 65535, hknpIslandIdDiscriminant>, hkContainerHeapAllocator>") // hkArray< struct hkHandle< hkUint16, 65535, struct hknpIslandIdDiscriminant >, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpDeactivationManager, m_newActivatedPairs, 0, "hkArray<hknpBodyIdPair, hkContainerHeapAllocator>") // hkArray< struct hknpBodyIdPair, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpDeactivationManager, m_newActivatedCdCacheRanges, 0, "hkArray<hknpCdCacheRange, hkContainerHeapAllocator>") // hkArray< class hknpCdCacheRange, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpDeactivationManager, m_newActivatedBodyIds, 0, "hkArray<hknpBodyId, hkContainerHeapAllocator>") // hkArray< struct hknpBodyId, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpDeactivationManager, s_libraryName)


// hknpDeactivationThreadData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDeactivationThreadData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDeactivationThreadData)
    HK_TRACKER_MEMBER(hknpDeactivationThreadData, m_solverVelOkToDeactivate, 0, "hkBitField") // class hkBitField
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpDeactivationThreadData, s_libraryName)


// hknpDeactivationStepInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDeactivationStepInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDeactivationStepInfo)
    HK_TRACKER_MEMBER(hknpDeactivationStepInfo, m_unionFind, 0, "hkUnionFind*") // class hkUnionFind*
    HK_TRACKER_MEMBER(hknpDeactivationStepInfo, m_buffer, 0, "hkFixedArray<hkInt32>*") // class hkFixedArray< hkInt32 >*
    HK_TRACKER_MEMBER(hknpDeactivationStepInfo, m_data, 0, "hknpDeactivationThreadData [12]") // struct hknpDeactivationThreadData [12]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpDeactivationStepInfo, s_libraryName)

#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationState.h>


// hknpDeactivationState ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpDeactivationState, s_libraryName)

#include <Physics/Internal/Dynamics/World/SpaceSplitter/hknpSpaceSplitter.h>


// hknpSpaceSplitter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpSpaceSplitter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Link)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SpaceSplitterType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpSpaceSplitter)
    HK_TRACKER_MEMBER(hknpSpaceSplitter, m_linksSortedForMultithreadedSolving, 0, "hkArray<hknpSpaceSplitter::Link, hkContainerHeapAllocator>") // hkArray< struct hknpSpaceSplitter::Link, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpSpaceSplitter, s_libraryName, hkBaseObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpSpaceSplitter, SpaceSplitterType, s_libraryName)


// Link hknpSpaceSplitter
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpSpaceSplitter, Link, s_libraryName)


// hknpSpaceSplitterData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpSpaceSplitterData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Int64Vector4)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hknpSpaceSplitterData, s_libraryName)


// Int64Vector4 hknpSpaceSplitterData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpSpaceSplitterData, Int64Vector4, s_libraryName)


// hknpSingleCellSpaceSplitter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpSingleCellSpaceSplitter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpSingleCellSpaceSplitter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpSingleCellSpaceSplitter, s_libraryName, hknpSpaceSplitter)


// hknpGridSpaceSplitter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpGridSpaceSplitter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpGridSpaceSplitter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpGridSpaceSplitter, s_libraryName, hknpSpaceSplitter)


// hknpDynamicSpaceSplitter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDynamicSpaceSplitter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDynamicSpaceSplitter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpDynamicSpaceSplitter, s_libraryName, hknpSpaceSplitter)

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
