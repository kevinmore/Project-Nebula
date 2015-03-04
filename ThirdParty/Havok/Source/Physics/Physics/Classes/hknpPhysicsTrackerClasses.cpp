/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Physics/Physics/hknpPhysics.h>
static const char s_libraryName[] = "hknpPhysics";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hknpPhysicsRegister() {}

#include <Common/Base/Container/FreeListArray/hkFreeListArray.h>

// hk.MemoryTracker ignore hkFreeListArrayhknpMotionPropertieshknpMotionPropertiesId8hknpMotionPropertiesFreeListArrayOperations
// hk.MemoryTracker ignore hkFreeListArrayhknpMaterialhknpMaterialId8hknpMaterialFreeListArrayOperations
// hk.MemoryTracker ignore hkFreeListArrayhknpShapeInstancehkHandleshort32767hknpShapeInstanceIdDiscriminant8hknpShapeInstance
#include <Physics/Physics/Collide/BroadPhase/BruteForce/hknpBruteForceBroadPhase.h>


// hknpBruteForceBroadPhase ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpBruteForceBroadPhase)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpBruteForceBroadPhase)
    HK_TRACKER_MEMBER(hknpBruteForceBroadPhase, m_bodies, 0, "hkArray<hknpBodyId, hkContainerHeapAllocator>") // hkArray< struct hknpBodyId, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpBruteForceBroadPhase, m_freeList, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpBruteForceBroadPhase, s_libraryName, hknpBroadPhase)

#include <Physics/Physics/Collide/BroadPhase/hknpBroadPhase.h>


// hknpBroadPhase ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpBroadPhase)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(UpdateMode)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpBroadPhase)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hknpBroadPhase, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpBroadPhase, UpdateMode, s_libraryName)

#include <Physics/Physics/Collide/BroadPhase/hknpBroadPhaseConfig.h>


// hknpBroadPhaseConfig ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpBroadPhaseConfig)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Layer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpBroadPhaseConfig)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpBroadPhaseConfig, s_libraryName, hkReferencedObject)


// Layer hknpBroadPhaseConfig
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpBroadPhaseConfig, Layer, s_libraryName)


// hknpDefaultBroadPhaseConfig ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDefaultBroadPhaseConfig)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDefaultBroadPhaseConfig)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpDefaultBroadPhaseConfig, s_libraryName, hknpBroadPhaseConfig)

#include <Physics/Physics/Collide/Dispatcher/hknpCollisionDispatcher.h>


// hknpCollisionDispatcher ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCollisionDispatcher, s_libraryName)

#include <Physics/Physics/Collide/Filter/AlwaysHit/hknpAlwaysHitCollisionFilter.h>


// hknpAlwaysHitCollisionFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpAlwaysHitCollisionFilter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpAlwaysHitCollisionFilter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpAlwaysHitCollisionFilter, s_libraryName, hknpCollisionFilter)

#include <Physics/Physics/Collide/Filter/Constraint/hknpConstraintCollisionFilter.h>


// hknpConstraintCollisionFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpConstraintCollisionFilter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpConstraintCollisionFilter)
    HK_TRACKER_MEMBER(hknpConstraintCollisionFilter, m_subscribedWorld, 0, "hknpWorld*") // class hknpWorld*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpConstraintCollisionFilter, s_libraryName, hknpPairCollisionFilter)

#include <Physics/Physics/Collide/Filter/DisableCollision/hknpDisableCollisionFilter.h>


// hknpDisableCollisionFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDisableCollisionFilter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDisableCollisionFilter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpDisableCollisionFilter, s_libraryName, hknpCollisionFilter)

#include <Physics/Physics/Collide/Filter/Group/hknpGroupCollisionFilter.h>


// hknpGroupCollisionFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpGroupCollisionFilter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpGroupCollisionFilter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpGroupCollisionFilter, s_libraryName, hknpCollisionFilter)

#include <Physics/Physics/Collide/Filter/Pair/hknpPairCollisionFilter.h>


// hknpPairCollisionFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpPairCollisionFilter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Key)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MapOperations)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MapPairFilterKeyOverrideType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpPairCollisionFilter)
    HK_TRACKER_MEMBER(hknpPairCollisionFilter, m_disabledPairs, 0, "hkMap<hknpPairCollisionFilter::Key, hkUint32, hknpPairCollisionFilter::MapOperations, hkContainerHeapAllocator>") // class hkMap< struct hknpPairCollisionFilter::Key, hkUint32, struct hknpPairCollisionFilter::MapOperations, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpPairCollisionFilter, m_childFilter, 0, "hknpCollisionFilter*") // const class hknpCollisionFilter*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpPairCollisionFilter, s_libraryName, hknpCollisionFilter)


// Key hknpPairCollisionFilter
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpPairCollisionFilter, Key, s_libraryName)


// MapOperations hknpPairCollisionFilter
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpPairCollisionFilter, MapOperations, s_libraryName)


// MapPairFilterKeyOverrideType hknpPairCollisionFilter

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpPairCollisionFilter::MapPairFilterKeyOverrideType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpPairCollisionFilter::MapPairFilterKeyOverrideType)
    HK_TRACKER_MEMBER(hknpPairCollisionFilter::MapPairFilterKeyOverrideType, m_elem, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpPairCollisionFilter::MapPairFilterKeyOverrideType, s_libraryName)

#include <Physics/Physics/Collide/Filter/hknpCollisionFilter.h>


// hknpCollisionFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCollisionFilter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FilterInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FilterType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCollisionFilter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpCollisionFilter, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpCollisionFilter, FilterType, s_libraryName)


// FilterInput hknpCollisionFilter

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCollisionFilter::FilterInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCollisionFilter::FilterInput)
    HK_TRACKER_MEMBER(hknpCollisionFilter::FilterInput, m_body, 0, "hknpBody*") // const class hknpBody*
    HK_TRACKER_MEMBER(hknpCollisionFilter::FilterInput, m_rootShape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpCollisionFilter::FilterInput, m_parentShape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpCollisionFilter::FilterInput, m_shape, 0, "hknpShape*") // const class hknpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpCollisionFilter::FilterInput, s_libraryName)

#include <Physics/Physics/Collide/Modifier/ManifoldEventCreator/hknpManifoldEventCreator.h>


// hknpManifoldEventCreator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpManifoldEventCreator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpManifoldEventCreator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpManifoldEventCreator, s_libraryName, hknpModifier)

#include <Physics/Physics/Collide/Modifier/Welding/hknpWeldingModifier.h>


// hknpTriangleWeldingModifier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpTriangleWeldingModifier)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpTriangleWeldingModifier)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpTriangleWeldingModifier, s_libraryName, hknpWeldingModifier)


// hknpNeighborWeldingModifier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpNeighborWeldingModifier)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpNeighborWeldingModifier)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpNeighborWeldingModifier, s_libraryName, hknpWeldingModifier)


// hknpMotionWeldingModifier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMotionWeldingModifier)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpMotionWeldingModifier)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpMotionWeldingModifier, s_libraryName, hknpWeldingModifier)

#include <Physics/Physics/Collide/NarrowPhase/Detector/ConvexComposite/hknpConvexCompositeCollisionDetector.h>


// hknpConvexCompositeCollisionDetector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpConvexCompositeCollisionDetector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpConvexCompositeCollisionDetector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpConvexCompositeCollisionDetector, s_libraryName, hknpCompositeCollisionDetector)

#include <Physics/Physics/Collide/NarrowPhase/hknpCollidePipeline.h>


// hknpCollidePipeline ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCollidePipeline, s_libraryName)

#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>


// hknpManifoldBase ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpManifoldBase)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ManifoldType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpManifoldBase)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpManifoldBase, s_libraryName, hkcdManifold4)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpManifoldBase, ManifoldType, s_libraryName)


// hknpManifold ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpManifold, s_libraryName)

#include <Physics/Physics/Collide/Query/Collector/hknpAllHitsCollector.h>


// hknpAllHitsCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpAllHitsCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpAllHitsCollector)
    HK_TRACKER_MEMBER(hknpAllHitsCollector, m_hits, 0, "hkInplaceArray<hknpCollisionResult, 10, hkContainerHeapAllocator>") // class hkInplaceArray< struct hknpCollisionResult, 10, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpAllHitsCollector, s_libraryName, hknpCollisionQueryCollector)

#include <Physics/Physics/Collide/Query/Collector/hknpAnyHitCollector.h>


// hknpAnyHitCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpAnyHitCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpAnyHitCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpAnyHitCollector, s_libraryName, hknpClosestHitCollector)

#include <Physics/Physics/Collide/Query/Collector/hknpClosestHitCollector.h>


// hknpClosestHitCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpClosestHitCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpClosestHitCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpClosestHitCollector, s_libraryName, hknpCollisionQueryCollector)

#include <Physics/Physics/Collide/Query/Collector/hknpCollisionQueryCollector.h>


// hknpCollisionQueryCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCollisionQueryCollector)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Hints)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCollisionQueryCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hknpCollisionQueryCollector, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpCollisionQueryCollector, Hints, s_libraryName)

#include <Physics/Physics/Collide/Query/Collector/hknpFlippedGetClosestPointsQueryCollector.h>


// hknpFlippedGetClosestPointsQueryCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpFlippedGetClosestPointsQueryCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpFlippedGetClosestPointsQueryCollector)
    HK_TRACKER_MEMBER(hknpFlippedGetClosestPointsQueryCollector, m_childCollector, 0, "hknpCollisionQueryCollector*") // class hknpCollisionQueryCollector*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpFlippedGetClosestPointsQueryCollector, s_libraryName, hknpCollisionQueryCollector)

#include <Physics/Physics/Collide/Query/Collector/hknpFlippedShapeCastQueryCollector.h>


// hknpFlippedShapeCastQueryCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpFlippedShapeCastQueryCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpFlippedShapeCastQueryCollector)
    HK_TRACKER_MEMBER(hknpFlippedShapeCastQueryCollector, m_childCollector, 0, "hknpCollisionQueryCollector*") // class hknpCollisionQueryCollector*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpFlippedShapeCastQueryCollector, s_libraryName, hknpCollisionQueryCollector)

#include <Physics/Physics/Collide/Query/Multithreaded/hknpCollisionQuerySubTask.h>


// hknpAabbQuerySubTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpAabbQuerySubTask)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpAabbQuerySubTask)
    HK_TRACKER_MEMBER(hknpAabbQuerySubTask, m_pShape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpAabbQuerySubTask, m_resultArray, 0, "hkPadSpu<hknpCollisionResult*>") // class hkPadSpu< struct hknpCollisionResult* >
    HK_TRACKER_MEMBER(hknpAabbQuerySubTask, m_filter, 0, "hkPadSpu<hknpCollisionFilter*>") // class hkPadSpu< class hknpCollisionFilter* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpAabbQuerySubTask, s_libraryName)


// hknpPairGetClosestPointsSubTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpPairGetClosestPointsSubTask)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpPairGetClosestPointsSubTask)
    HK_TRACKER_MEMBER(hknpPairGetClosestPointsSubTask, m_queryShape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpPairGetClosestPointsSubTask, m_targetShape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpPairGetClosestPointsSubTask, m_resultArray, 0, "hkPadSpu<hknpCollisionResult*>") // class hkPadSpu< struct hknpCollisionResult* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpPairGetClosestPointsSubTask, s_libraryName)


// hknpWorldGetClosestPointsSubTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpWorldGetClosestPointsSubTask)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpWorldGetClosestPointsSubTask)
    HK_TRACKER_MEMBER(hknpWorldGetClosestPointsSubTask, m_pWorld, 0, "hknpWorld*") // class hknpWorld*
    HK_TRACKER_MEMBER(hknpWorldGetClosestPointsSubTask, m_resultArray, 0, "hkPadSpu<hknpCollisionResult*>") // class hkPadSpu< struct hknpCollisionResult* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpWorldGetClosestPointsSubTask, s_libraryName)


// hknpPairShapeCastSubTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpPairShapeCastSubTask)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpPairShapeCastSubTask)
    HK_TRACKER_MEMBER(hknpPairShapeCastSubTask, m_targetShape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpPairShapeCastSubTask, m_resultArray, 0, "hkPadSpu<hknpCollisionResult*>") // class hkPadSpu< struct hknpCollisionResult* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpPairShapeCastSubTask, s_libraryName)


// hknpWorldShapeCastSubTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpWorldShapeCastSubTask)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpWorldShapeCastSubTask)
    HK_TRACKER_MEMBER(hknpWorldShapeCastSubTask, m_pWorld, 0, "hknpWorld*") // class hknpWorld*
    HK_TRACKER_MEMBER(hknpWorldShapeCastSubTask, m_queryShape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpWorldShapeCastSubTask, m_resultArray, 0, "hkPadSpu<hknpCollisionResult*>") // class hkPadSpu< struct hknpCollisionResult* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpWorldShapeCastSubTask, s_libraryName)


// hknpCollisionQuerySubTask ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCollisionQuerySubTask, s_libraryName)

#include <Physics/Physics/Collide/Query/Multithreaded/hknpCollisionQueryTask.h>


// hknpCollisionQueryTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCollisionQueryTask)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCollisionQueryTask)
    HK_TRACKER_MEMBER(hknpCollisionQueryTask, m_querySharedData, 0, "hknpQuerySharedData") // class hknpQuerySharedData
    HK_TRACKER_MEMBER(hknpCollisionQueryTask, m_subTasks, 0, "hkArray<hknpCollisionQuerySubTask, hkContainerHeapAllocator>") // hkArray< struct hknpCollisionQuerySubTask, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCollisionQueryTask, s_libraryName, hkTask)

#include <Physics/Physics/Collide/Query/Multithreaded/hknpQuerySharedData.h>


// hknpQuerySharedData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpQuerySharedData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpQuerySharedData)
    HK_TRACKER_MEMBER(hknpQuerySharedData, m_collisionFilter, 0, "hkPadSpu<hknpCollisionFilter*>") // class hkPadSpu< class hknpCollisionFilter* >
    HK_TRACKER_MEMBER(hknpQuerySharedData, m_shapeTagCodec, 0, "hkPadSpu<hknpShapeTagCodec*>") // class hkPadSpu< const class hknpShapeTagCodec* >
    HK_TRACKER_MEMBER(hknpQuerySharedData, m_pCollisionQueryDispatcher, 0, "hknpCollisionQueryDispatcherBase*") // class hknpCollisionQueryDispatcherBase*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpQuerySharedData, s_libraryName)

#include <Physics/Physics/Collide/Query/Multithreaded/hknpRaycastTask.h>


// hknpShapeRaycastSubTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeRaycastSubTask)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShapeRaycastSubTask)
    HK_TRACKER_MEMBER(hknpShapeRaycastSubTask, m_targetShape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpShapeRaycastSubTask, m_resultArray, 0, "hkPadSpu<hknpCollisionResult*>") // class hkPadSpu< struct hknpCollisionResult* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpShapeRaycastSubTask, s_libraryName)


// hknpWorldRaycastSubTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpWorldRaycastSubTask)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpWorldRaycastSubTask)
    HK_TRACKER_MEMBER(hknpWorldRaycastSubTask, m_pWorld, 0, "hknpWorld*") // class hknpWorld*
    HK_TRACKER_MEMBER(hknpWorldRaycastSubTask, m_shapeTagCodec, 0, "hknpShapeTagCodec*") // const class hknpShapeTagCodec*
    HK_TRACKER_MEMBER(hknpWorldRaycastSubTask, m_resultArray, 0, "hkPadSpu<hknpCollisionResult*>") // class hkPadSpu< struct hknpCollisionResult* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpWorldRaycastSubTask, s_libraryName)


// hknpRaycastTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpRaycastTask)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SubTask)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SubTaskType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpRaycastTask)
    HK_TRACKER_MEMBER(hknpRaycastTask, m_querySharedData, 0, "hknpQuerySharedData") // class hknpQuerySharedData
    HK_TRACKER_MEMBER(hknpRaycastTask, m_subTasks, 0, "hkArray<hknpRaycastTask::SubTask, hkContainerHeapAllocator>") // hkArray< struct hknpRaycastTask::SubTask, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpRaycastTask, s_libraryName, hkTask)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpRaycastTask, SubTaskType, s_libraryName)


// SubTask hknpRaycastTask
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpRaycastTask, SubTask, s_libraryName)

#include <Physics/Physics/Collide/Query/hknpCollisionQuery.h>


// hknpCollisionQueryType ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCollisionQueryType, s_libraryName)


// hknpShapeTagPathEntry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeTagPathEntry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShapeTagPathEntry)
    HK_TRACKER_MEMBER(hknpShapeTagPathEntry, m_parentShape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpShapeTagPathEntry, m_shape, 0, "hknpShape*") // const class hknpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpShapeTagPathEntry, s_libraryName)


// hknpCollisionQueryContext ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCollisionQueryContext)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCollisionQueryContext)
    HK_TRACKER_MEMBER(hknpCollisionQueryContext, m_dispatcher, 0, "hknpCollisionQueryDispatcherBase*") // const class hknpCollisionQueryDispatcherBase*
    HK_TRACKER_MEMBER(hknpCollisionQueryContext, m_shapeTagCodec, 0, "hknpShapeTagCodec*") // const class hknpShapeTagCodec*
    HK_TRACKER_MEMBER(hknpCollisionQueryContext, m_queryTriangle, 0, "hknpTriangleShape*") // class hknpTriangleShape*
    HK_TRACKER_MEMBER(hknpCollisionQueryContext, m_targetTriangle, 0, "hknpTriangleShape*") // class hknpTriangleShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpCollisionQueryContext, s_libraryName)


// hknpQueryFilterData ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpQueryFilterData, s_libraryName)


// hknpShapeQueryInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeQueryInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShapeQueryInfo)
    HK_TRACKER_MEMBER(hknpShapeQueryInfo, m_body, 0, "hkPadSpu<hknpBody*>") // class hkPadSpu< const class hknpBody* >
    HK_TRACKER_MEMBER(hknpShapeQueryInfo, m_rootShape, 0, "hkPadSpu<hknpShape*>") // class hkPadSpu< const class hknpShape* >
    HK_TRACKER_MEMBER(hknpShapeQueryInfo, m_parentShape, 0, "hkPadSpu<hknpShape*>") // class hkPadSpu< const class hknpShape* >
    HK_TRACKER_MEMBER(hknpShapeQueryInfo, m_shapeToWorld, 0, "hkPadSpu<hkTransformf*>") // class hkPadSpu< const hkTransformf* >
    HK_TRACKER_MEMBER(hknpShapeQueryInfo, m_shapeKeyMask, 0, "hkPadSpu<hknpShapeKeyMask*>") // class hkPadSpu< const struct hknpShapeKeyMask* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpShapeQueryInfo, s_libraryName)


// hknpRayCastQuery ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpRayCastQuery)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpRayCastQuery)
    HK_TRACKER_MEMBER(hknpRayCastQuery, m_filter, 0, "hkPadSpu<hknpCollisionFilter*>") // class hkPadSpu< class hknpCollisionFilter* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpRayCastQuery, s_libraryName, hkcdRay)


// hknpShapeCastQuery ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeCastQuery)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShapeCastQuery)
    HK_TRACKER_MEMBER(hknpShapeCastQuery, m_shape, 0, "hkPadSpu<hknpShape*>") // class hkPadSpu< const class hknpShape* >
    HK_TRACKER_MEMBER(hknpShapeCastQuery, m_body, 0, "hkPadSpu<hknpBody*>") // class hkPadSpu< const class hknpBody* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpShapeCastQuery, s_libraryName, hknpRayCastQuery)


// hknpClosestPointsQuery ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpClosestPointsQuery)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpClosestPointsQuery)
    HK_TRACKER_MEMBER(hknpClosestPointsQuery, m_shape, 0, "hkPadSpu<hknpShape*>") // class hkPadSpu< const class hknpShape* >
    HK_TRACKER_MEMBER(hknpClosestPointsQuery, m_body, 0, "hkPadSpu<hknpBody*>") // class hkPadSpu< const class hknpBody* >
    HK_TRACKER_MEMBER(hknpClosestPointsQuery, m_filter, 0, "hkPadSpu<hknpCollisionFilter*>") // class hkPadSpu< class hknpCollisionFilter* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpClosestPointsQuery, s_libraryName)


// hknpAabbQuery ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpAabbQuery)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpAabbQuery)
    HK_TRACKER_MEMBER(hknpAabbQuery, m_filter, 0, "hkPadSpu<hknpCollisionFilter*>") // class hkPadSpu< class hknpCollisionFilter* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpAabbQuery, s_libraryName)


// hknpCollisionResult ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCollisionResult)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BodyInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hknpCollisionResult, s_libraryName)


// BodyInfo hknpCollisionResult
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpCollisionResult, BodyInfo, s_libraryName)


// hknpAabbQueryUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpAabbQueryUtil, s_libraryName)

#include <Physics/Physics/Collide/Query/hknpCollisionQueryDispatcher.h>


// hknpCollisionQueryDispatcherBase ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCollisionQueryDispatcherBase, s_libraryName)


// hknpCollisionQueryDispatcher ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCollisionQueryDispatcher, s_libraryName)


// hknpShapeCastQueryDispatcher ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpShapeCastQueryDispatcher, s_libraryName)


// hknpGetClosestPointsQueryDispatcher ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpGetClosestPointsQueryDispatcher, s_libraryName)

#include <Physics/Physics/Collide/Query/hknpCollisionResultAccessors.h>


// hknpRayCastQueryResult ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpRayCastQueryResult, s_libraryName)


// hknpShapeCastQueryResult ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpShapeCastQueryResult, s_libraryName)


// hknpClosestPointsQueryResult ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpClosestPointsQueryResult, s_libraryName)


// hknpAabbQueryResult ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpAabbQueryResult, s_libraryName)

#include <Physics/Physics/Collide/Query/hknpQueryAabbNmpUtil.h>


// hknpQueryAabbNmpUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpQueryAabbNmpUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Buffer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hknpQueryAabbNmpUtil, s_libraryName)


// Buffer hknpQueryAabbNmpUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpQueryAabbNmpUtil, Buffer, s_libraryName)

#include <Physics/Physics/Collide/Shape/Composite/Compound/Dynamic/hknpDynamicCompoundShape.h>


// hknpDynamicCompoundShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDynamicCompoundShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDynamicCompoundShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpDynamicCompoundShape, s_libraryName, hknpCompoundShape)

#include <Physics/Physics/Collide/Shape/Composite/Compound/Static/hknpStaticCompoundShape.h>


// hknpStaticCompoundShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpStaticCompoundShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpStaticCompoundShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpStaticCompoundShape, s_libraryName, hknpCompoundShape)

#include <Physics/Physics/Collide/Shape/Composite/Compound/hknpCompoundShape.h>


// hknpCompoundShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCompoundShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCompoundShape)
    HK_TRACKER_MEMBER(hknpCompoundShape, m_instances, 0, "hkFreeListArray<hknpShapeInstance, hkHandle<hkInt16, 32767, hknpShapeInstanceIdDiscriminant>, 8, hknpShapeInstance>") // struct hkFreeListArray< struct hknpShapeInstance, struct hkHandle< hkInt16, 32767, struct hknpShapeInstanceIdDiscriminant >, 8, struct hknpShapeInstance >
    HK_TRACKER_MEMBER(hknpCompoundShape, m_mutationSignals, 0, "hknpShape::MutationSignals") // struct hknpShape::MutationSignals
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpCompoundShape, s_libraryName, hknpCompositeShape)

#include <Physics/Physics/Collide/Shape/Composite/Compound/hknpShapeInstance.h>


// hknpShapeInstance ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeInstance)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Flags)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShapeInstance)
    HK_TRACKER_MEMBER(hknpShapeInstance, m_shape, 0, "hknpShape *") // const class hknpShape *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpShapeInstance, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpShapeInstance, Flags, s_libraryName)


// hknpShapeInstanceIdDiscriminant ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpShapeInstanceIdDiscriminant, s_libraryName)

#include <Physics/Physics/Collide/Shape/Composite/HeightField/Compressed/hknpCompressedHeightFieldShape.h>


// hknpCompressedHeightFieldShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCompressedHeightFieldShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCompressedHeightFieldShape)
    HK_TRACKER_MEMBER(hknpCompressedHeightFieldShape, m_storage, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpCompressedHeightFieldShape, m_shapeTags, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCompressedHeightFieldShape, s_libraryName, hknpHeightFieldShape)

#include <Physics/Physics/Collide/Shape/Composite/HeightField/hknpHeightFieldShape.h>


// hknpHeightFieldShapeCinfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpHeightFieldShapeCinfo, s_libraryName)


// hknpHeightFieldShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpHeightFieldShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpHeightFieldShape)
    HK_TRACKER_MEMBER(hknpHeightFieldShape, m_minMaxTree, 0, "hknpMinMaxQuadTree") // struct hknpMinMaxQuadTree
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpHeightFieldShape, s_libraryName, hknpCompositeShape)

#include <Physics/Physics/Collide/Shape/Composite/HeightField/hknpHeightFieldShapeUtils.h>


// QuadTreeWalkerStackElement hknpHeightFieldShapeUtils
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hknpHeightFieldShapeUtils::QuadTreeWalkerStackElement, s_libraryName, hknpHeightFieldShapeUtils_QuadTreeWalkerStackElement)


// NoCacheT hknpHeightFieldShapeUtils
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hknpHeightFieldShapeUtils::NoCacheT, s_libraryName, hknpHeightFieldShapeUtils_NoCacheT)

#include <Physics/Physics/Collide/Shape/Composite/HeightField/hknpMinMaxQuadTree.h>


// hknpMinMaxQuadTree ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMinMaxQuadTree)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MinMaxLevel)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpMinMaxQuadTree)
    HK_TRACKER_MEMBER(hknpMinMaxQuadTree, m_coarseTreeData, 0, "hkArray<hknpMinMaxQuadTree::MinMaxLevel, hkContainerHeapAllocator>") // hkArray< struct hknpMinMaxQuadTree::MinMaxLevel, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpMinMaxQuadTree, s_libraryName)


// MinMaxLevel hknpMinMaxQuadTree

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMinMaxQuadTree::MinMaxLevel)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpMinMaxQuadTree::MinMaxLevel)
    HK_TRACKER_MEMBER(hknpMinMaxQuadTree::MinMaxLevel, m_minMaxData, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpMinMaxQuadTree::MinMaxLevel, s_libraryName)

#include <Physics/Physics/Collide/Shape/Composite/Masked/hknpMaskedCompositeShape.h>


// hknpMaskedCompositeShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMaskedCompositeShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MaskWrapper)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpMaskedCompositeShape)
    HK_TRACKER_MEMBER(hknpMaskedCompositeShape, m_shape, 0, "hknpCompositeShape *") // const class hknpCompositeShape *
    HK_TRACKER_MEMBER(hknpMaskedCompositeShape, m_mask, 0, "hknpShapeKeyMask*") // struct hknpShapeKeyMask*
    HK_TRACKER_MEMBER(hknpMaskedCompositeShape, m_maskWrapper, 0, "hknpMaskedCompositeShape::MaskWrapper") // struct hknpMaskedCompositeShape::MaskWrapper
    HK_TRACKER_MEMBER(hknpMaskedCompositeShape, m_mutationSignals, 0, "hknpShape::MutationSignals") // struct hknpShape::MutationSignals
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpMaskedCompositeShape, s_libraryName, hknpCompositeShape)


// MaskWrapper hknpMaskedCompositeShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMaskedCompositeShape::MaskWrapper)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpMaskedCompositeShape::MaskWrapper)
    HK_TRACKER_MEMBER(hknpMaskedCompositeShape::MaskWrapper, m_maskedShape, 0, "hknpMaskedCompositeShape*") // class hknpMaskedCompositeShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpMaskedCompositeShape::MaskWrapper, s_libraryName, hknpShapeKeyMask)

#include <Physics/Physics/Collide/Shape/Composite/Mesh/Compressed/hknpCompressedMeshShape.h>


// hknpCompressedMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCompressedMeshShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCompressedMeshShape)
    HK_TRACKER_MEMBER(hknpCompressedMeshShape, m_quadIsFlat, 0, "hkBitField") // class hkBitField
    HK_TRACKER_MEMBER(hknpCompressedMeshShape, m_triangleIsInternal, 0, "hkBitField") // class hkBitField
    HK_TRACKER_MEMBER(hknpCompressedMeshShape, m_extendedData, 0, "hknpCompressedMeshShapeTreeExt*") // struct hknpCompressedMeshShapeTreeExt*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCompressedMeshShape, s_libraryName, hknpCompositeShape)

#include <Physics/Physics/Collide/Shape/Composite/Mesh/Compressed/hknpCompressedMeshShapeCinfo.h>


// hknpCompressedMeshShapeCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCompressedMeshShapeCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCompressedMeshShapeCinfo)
    HK_TRACKER_MEMBER(hknpCompressedMeshShapeCinfo, m_triangleIndexToShapeKeyMap, 0, "hkArray<hkUint32, hkContainerHeapAllocator>*") // hkArray< hkUint32, struct hkContainerHeapAllocator >*
    HK_TRACKER_MEMBER(hknpCompressedMeshShapeCinfo, m_triangleIndexToVertexOrderMap, 0, "hkArray<hkUint8, hkContainerHeapAllocator>*") // hkArray< hkUint8, struct hkContainerHeapAllocator >*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hknpCompressedMeshShapeCinfo, s_libraryName)


// hknpDefaultCompressedMeshShapeCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDefaultCompressedMeshShapeCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDefaultCompressedMeshShapeCinfo)
    HK_TRACKER_MEMBER(hknpDefaultCompressedMeshShapeCinfo, m_geometry, 0, "hkGeometry*") // const struct hkGeometry*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpDefaultCompressedMeshShapeCinfo, s_libraryName, hknpCompressedMeshShapeCinfo)

#include <Physics/Physics/Collide/Shape/Composite/Mesh/Compressed/hknpCompressedMeshShapeUtil.h>


// hknpCompressedMeshShapeUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCompressedMeshShapeUtil, s_libraryName)

#include <Physics/Physics/Collide/Shape/Composite/Mesh/Extern/hknpDefaultExternMeshShapeMesh.h>


// hknpDefaultExternMeshShapeMesh ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDefaultExternMeshShapeMesh)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDefaultExternMeshShapeMesh)
    HK_TRACKER_MEMBER(hknpDefaultExternMeshShapeMesh, m_geometry, 0, "hkGeometry *") // struct hkGeometry *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpDefaultExternMeshShapeMesh, s_libraryName, hknpExternMeshShape::Mesh)

#include <Physics/Physics/Collide/Shape/Composite/Mesh/Extern/hknpExternMeshShape.h>


// hknpExternMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpExternMeshShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Mesh)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpExternMeshShape)
    HK_TRACKER_MEMBER(hknpExternMeshShape, m_mesh, 0, "hknpExternMeshShape::Mesh*") // const class hknpExternMeshShape::Mesh*
    HK_TRACKER_MEMBER(hknpExternMeshShape, m_tree, 0, "hknpExternMeshShapeTree*") // const struct hknpExternMeshShapeTree*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpExternMeshShape, s_libraryName, hknpCompositeShape)


// Mesh hknpExternMeshShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpExternMeshShape::Mesh)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpExternMeshShape::Mesh)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpExternMeshShape::Mesh, s_libraryName, hkReferencedObject)

#include <Physics/Physics/Collide/Shape/Composite/Mesh/Extern/hknpExternMeshShapeUtil.h>


// hknpExternMeshShapeUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpExternMeshShapeUtil, s_libraryName)

#include <Physics/Physics/Collide/Shape/Composite/hknpCompositeShape.h>


// hknpCompositeShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCompositeShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(EdgeWeld)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCompositeShape)
    HK_TRACKER_MEMBER(hknpCompositeShape, m_edgeWeldingMap, 0, "hknpSparseCompactMap<hkUint16>") // class hknpSparseCompactMap< hkUint16 >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpCompositeShape, s_libraryName, hknpShape)


// EdgeWeld hknpCompositeShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpCompositeShape, EdgeWeld, s_libraryName)

#include <Physics/Physics/Collide/Shape/Composite/hknpSparseCompactMap.h>


// Entry hknpSparseCompactMapUtil
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hknpSparseCompactMapUtil::Entry, s_libraryName, hknpSparseCompactMapUtil_Entry)

// hk.MemoryTracker ignore hknpSparseCompactMapunsignedshort
#include <Physics/Physics/Collide/Shape/Convex/Capsule/hknpCapsuleShape.h>


// hknpCapsuleShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCapsuleShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCapsuleShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCapsuleShape, s_libraryName, hknpConvexPolytopeShape)

#include <Physics/Physics/Collide/Shape/Convex/Polytope/hknpConvexPolytopeShape.h>


// hknpConvexPolytopeShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpConvexPolytopeShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Face)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpConvexPolytopeShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpConvexPolytopeShape, s_libraryName, hknpConvexShape)


// Face hknpConvexPolytopeShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpConvexPolytopeShape, Face, s_libraryName)

#include <Physics/Physics/Collide/Shape/Convex/Scaled/hknpScaledConvexShape.h>


// hknpScaledConvexShapeBase ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpScaledConvexShapeBase)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpScaledConvexShapeBase)
    HK_TRACKER_MEMBER(hknpScaledConvexShapeBase, m_childShape, 0, "hknpConvexShape*") // const class hknpConvexShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpScaledConvexShapeBase, s_libraryName, hknpShape)


// hknpScaledConvexShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpScaledConvexShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpScaledConvexShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpScaledConvexShape, s_libraryName, hknpScaledConvexShapeBase)

#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.h>


// hknpSphereShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpSphereShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpSphereShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpSphereShape, s_libraryName, hknpConvexShape)

#include <Physics/Physics/Collide/Shape/Convex/Triangle/hknpTriangleShape.h>


// hknpTriangleShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpTriangleShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpTriangleShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpTriangleShape, s_libraryName, hknpConvexPolytopeShape)


// hknpInplaceTriangleShape ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpInplaceTriangleShape, s_libraryName)

#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>


// hknpConvexShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpConvexShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BuildConfig)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpConvexShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpConvexShape, s_libraryName, hknpShape)


// BuildConfig hknpConvexShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpConvexShape::BuildConfig)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpConvexShape::BuildConfig)
    HK_TRACKER_MEMBER(hknpConvexShape::BuildConfig, m_extraTransform, 0, "hkTransform*") // hkTransform*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpConvexShape::BuildConfig, s_libraryName)

#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShapeUtil.h>


// hknpConvexShapeUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpConvexShapeUtil, s_libraryName)

#include <Physics/Physics/Collide/Shape/TagCodec/MaterialPalette/hknpMaterialPaletteShapeTagCodec.h>


// hknpMaterialPaletteShapeTagCodec ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMaterialPaletteShapeTagCodec)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PaletteInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpMaterialPaletteShapeTagCodec)
    HK_TRACKER_MEMBER(hknpMaterialPaletteShapeTagCodec, m_paletteMaterials, 0, "hkArray<hknpMaterialId, hkContainerHeapAllocator>") // hkArray< struct hknpMaterialId, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpMaterialPaletteShapeTagCodec, m_materialLibrary, 0, "hknpMaterialLibrary *") // class hknpMaterialLibrary *
    HK_TRACKER_MEMBER(hknpMaterialPaletteShapeTagCodec, m_paletteMap, 0, "hkMap<hknpMaterialPalette*, hknpMaterialPaletteShapeTagCodec::PaletteInfo, hkMapOperations<hknpMaterialPalette*>, hkContainerHeapAllocator>") // class hkMap< const class hknpMaterialPalette*, struct hknpMaterialPaletteShapeTagCodec::PaletteInfo, struct hkMapOperations< const class hknpMaterialPalette* >, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpMaterialPaletteShapeTagCodec, m_shapeMap, 0, "hkPointerMap<hknpCompositeShape*, hknpMaterialPalette*, hkContainerHeapAllocator>") // class hkPointerMap< const class hknpCompositeShape*, const class hknpMaterialPalette*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpMaterialPaletteShapeTagCodec, s_libraryName, hknpShapeTagCodec)


// PaletteInfo hknpMaterialPaletteShapeTagCodec
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMaterialPaletteShapeTagCodec, PaletteInfo, s_libraryName)

#include <Physics/Physics/Collide/Shape/TagCodec/Null/hknpNullShapeTagCodec.h>


// hknpNullShapeTagCodec ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpNullShapeTagCodec)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpNullShapeTagCodec)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpNullShapeTagCodec, s_libraryName, hknpShapeTagCodec)

#include <Physics/Physics/Collide/Shape/TagCodec/hknpShapeTagCodec.h>


// hknpShapeTagCodec ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeTagCodec)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Context)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CodecType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShapeTagCodec)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpShapeTagCodec, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpShapeTagCodec, CodecType, s_libraryName)


// Context hknpShapeTagCodec

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeTagCodec::Context)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShapeTagCodec::Context)
    HK_TRACKER_MEMBER(hknpShapeTagCodec::Context, m_body, 0, "hknpBody*") // const class hknpBody*
    HK_TRACKER_MEMBER(hknpShapeTagCodec::Context, m_rootShape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpShapeTagCodec::Context, m_parentShape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpShapeTagCodec::Context, m_shape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpShapeTagCodec::Context, m_partnerBody, 0, "hknpBody*") // const class hknpBody*
    HK_TRACKER_MEMBER(hknpShapeTagCodec::Context, m_partnerRootShape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpShapeTagCodec::Context, m_partnerShape, 0, "hknpShape*") // const class hknpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpShapeTagCodec::Context, s_libraryName)

#include <Physics/Physics/Collide/Shape/hknpShape.h>


// hknpShapeKeyIterator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeKeyIterator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShapeKeyIterator)
    HK_TRACKER_MEMBER(hknpShapeKeyIterator, m_shape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpShapeKeyIterator, m_mask, 0, "hknpShapeKeyMask*") // const struct hknpShapeKeyMask*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpShapeKeyIterator, s_libraryName, hkReferencedObject)


// hknpShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SdfQuery)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SdfContactPoint)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MassConfig)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BuildSurfaceGeometryConfig)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MutationSignals)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FlagsEnum)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MutationFlagsEnum)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ScaleMode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConvexRadiusDisplayMode)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShape)
    HK_TRACKER_MEMBER(hknpShape, m_properties, 0, "hkRefCountedProperties*") // class hkRefCountedProperties*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpShape, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpShape, FlagsEnum, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpShape, MutationFlagsEnum, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpShape, ScaleMode, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpShape, ConvexRadiusDisplayMode, s_libraryName)


// SdfQuery hknpShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShape::SdfQuery)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShape::SdfQuery)
    HK_TRACKER_MEMBER(hknpShape::SdfQuery, m_sphereCenters, 0, "hkVector4*") // const hkVector4*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpShape::SdfQuery, s_libraryName)


// SdfContactPoint hknpShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpShape, SdfContactPoint, s_libraryName)


// MassConfig hknpShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpShape, MassConfig, s_libraryName)


// BuildSurfaceGeometryConfig hknpShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpShape, BuildSurfaceGeometryConfig, s_libraryName)


// MutationSignals hknpShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShape::MutationSignals)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ShapeMutatedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ShapeDestroyedSignal)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShape::MutationSignals)
    HK_TRACKER_MEMBER(hknpShape::MutationSignals, m_shapeDestroyed, 0, "hknpShape::MutationSignals::ShapeDestroyedSignal") // struct hknpShape::MutationSignals::ShapeDestroyedSignal
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpShape::MutationSignals, s_libraryName)


// ShapeMutatedSignal hknpShape::MutationSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpShape::MutationSignals, ShapeMutatedSignal, s_libraryName)


// ShapeDestroyedSignal hknpShape::MutationSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpShape::MutationSignals, ShapeDestroyedSignal, s_libraryName)


// hknpShapeKeyMask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeKeyMask)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShapeKeyMask)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hknpShapeKeyMask, s_libraryName)


// hknpShapeMassProperties ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeMassProperties)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShapeMassProperties)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpShapeMassProperties, s_libraryName, hkReferencedObject)

#include <Physics/Physics/Collide/Shape/hknpShapeCollector.h>


// hknpShapeCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShapeCollector)
    HK_TRACKER_MEMBER(hknpShapeCollector, m_shapeOut, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpShapeCollector, m_triangleShapePrototype, 0, "hknpTriangleShape*") // class hknpTriangleShape*
    HK_TRACKER_MEMBER(hknpShapeCollector, m_parentShape, 0, "hknpShape*") // const class hknpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpShapeCollector, s_libraryName)


// hknpShapeCollectorWithInplaceTriangle ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpShapeCollectorWithInplaceTriangle, s_libraryName)

#include <Physics/Physics/Collide/Shape/hknpShapeQueryInterface.h>


// hknpShapeQueryInterface ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpShapeQueryInterface, s_libraryName)

#include <Physics/Physics/Collide/Shape/hknpShapeType.h>


// hknpShapeType ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpShapeType, s_libraryName)


// hknpCollisionDispatchType ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCollisionDispatchType, s_libraryName)

#include <Physics/Physics/Collide/Shape/hknpShapeUtil.h>


// hknpShapeUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpShapeUtil, s_libraryName)

#include <Physics/Physics/Collide/hknpCollideSharedData.h>


// hknpModifierSharedData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpModifierSharedData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpModifierSharedData)
    HK_TRACKER_MEMBER(hknpModifierSharedData, m_spaceSplitter, 0, "hkPadSpu<hknpSpaceSplitter*>") // class hkPadSpu< class hknpSpaceSplitter* >
    HK_TRACKER_MEMBER(hknpModifierSharedData, m_solverInfo, 0, "hkPadSpu<hknpSolverInfo*>") // class hkPadSpu< const struct hknpSolverInfo* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpModifierSharedData, s_libraryName)


// hknpInternalCollideSharedData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpInternalCollideSharedData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpInternalCollideSharedData)
    HK_TRACKER_MEMBER(hknpInternalCollideSharedData, m_bodies, 0, "hkPadSpu<hknpBody*>") // class hkPadSpu< class hknpBody* >
    HK_TRACKER_MEMBER(hknpInternalCollideSharedData, m_motions, 0, "hkPadSpu<hknpMotion*>") // class hkPadSpu< class hknpMotion* >
    HK_TRACKER_MEMBER(hknpInternalCollideSharedData, m_intSpaceUtil, 0, "hkPadSpu<hkIntSpaceUtil*>") // class hkPadSpu< class hkIntSpaceUtil* >
    HK_TRACKER_MEMBER(hknpInternalCollideSharedData, m_cdCacheStreamInOnPpu, 0, "hkPadSpu<hknpCdCacheStream*>") // class hkPadSpu< class hknpCdCacheStream* >
    HK_TRACKER_MEMBER(hknpInternalCollideSharedData, m_cdCacheStreamIn2OnPpu, 0, "hkPadSpu<hknpCdCacheStream*>") // class hkPadSpu< class hknpCdCacheStream* >
    HK_TRACKER_MEMBER(hknpInternalCollideSharedData, m_childCdCacheStreamInOnPpu, 0, "hkPadSpu<hknpCdCacheStream*>") // class hkPadSpu< class hknpCdCacheStream* >
    HK_TRACKER_MEMBER(hknpInternalCollideSharedData, m_childCdCacheCacheStreamIn2OnPpu, 0, "hkPadSpu<hknpCdCacheStream*>") // class hkPadSpu< class hknpCdCacheStream* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpInternalCollideSharedData, s_libraryName, hknpModifierSharedData)


// hknpCollideSharedData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCollideSharedData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCollideSharedData)
    HK_TRACKER_MEMBER(hknpCollideSharedData, m_qualities, 0, "hkPadSpu<hknpBodyQuality*>") // class hkPadSpu< const class hknpBodyQuality* >
    HK_TRACKER_MEMBER(hknpCollideSharedData, m_materials, 0, "hkPadSpu<hknpMaterial*>") // class hkPadSpu< const class hknpMaterial* >
    HK_TRACKER_MEMBER(hknpCollideSharedData, m_heapAllocator, 0, "hkPadSpu<hkBlockStreamAllocator*>") // class hkPadSpu< class hkBlockStreamAllocator* >
    HK_TRACKER_MEMBER(hknpCollideSharedData, m_tempAllocator, 0, "hkPadSpu<hkBlockStreamAllocator*>") // class hkPadSpu< class hkBlockStreamAllocator* >
    HK_TRACKER_MEMBER(hknpCollideSharedData, m_simulationContext, 0, "hkPadSpu<hknpSimulationContext*>") // class hkPadSpu< class hknpSimulationContext* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCollideSharedData, s_libraryName, hknpInternalCollideSharedData)

#include <Physics/Physics/Dynamics/Action/CentrifugalForce/hknpCentrifugalForceAction.h>


// hknpCentrifugalForceAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCentrifugalForceAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCentrifugalForceAction)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCentrifugalForceAction, s_libraryName, hknpUnaryAction)

#include <Physics/Physics/Dynamics/Action/Manager/hknpActionManager.h>


// hknpActionManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpActionManager)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpActionManager)
    HK_TRACKER_MEMBER(hknpActionManager, m_activeActions, 0, "hkArray<hknpAction*, hkContainerHeapAllocator>") // hkArray< class hknpAction*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpActionManager, s_libraryName, hknpActivationListener)

#include <Physics/Physics/Dynamics/Action/Spring/hknpSpringAction.h>


// hknpSpringAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpSpringAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpSpringAction)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpSpringAction, s_libraryName, hknpBinaryAction)

#include <Physics/Physics/Dynamics/Action/hknpAction.h>


// hknpAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpAction)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ApplyActionResult)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpAction)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpAction, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpAction, ApplyActionResult, s_libraryName)


// hknpUnaryAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpUnaryAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpUnaryAction)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpUnaryAction, s_libraryName, hknpAction)


// hknpBinaryAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpBinaryAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpBinaryAction)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpBinaryAction, s_libraryName, hknpAction)

#include <Physics/Physics/Dynamics/Body/hknpBody.h>


// hknpBody ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpBody)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FlagsEnum)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SpuFlagsEnum)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpBody)
    HK_TRACKER_MEMBER(hknpBody, m_shape, 0, "hknpShape*") // const class hknpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpBody, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpBody, FlagsEnum, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpBody, SpuFlagsEnum, s_libraryName)


// hknpBodyCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpBodyCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpBodyCinfo)
    HK_TRACKER_MEMBER(hknpBodyCinfo, m_shape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpBodyCinfo, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hknpBodyCinfo, m_localFrame, 0, "hkLocalFrame *") // class hkLocalFrame *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpBodyCinfo, s_libraryName)

#include <Physics/Physics/Dynamics/Body/hknpBodyId.h>


// hknpBodyIdBaseDiscriminant ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpBodyIdBaseDiscriminant, s_libraryName)


// hknpBodyId ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpBodyId, s_libraryName)


// hknpBodyReference ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpBodyReference)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpBodyReference)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpBodyReference, s_libraryName, hkReferencedObject)


// hknpBodyIdPair ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpBodyIdPair, s_libraryName)

#include <Physics/Physics/Dynamics/Body/hknpBodyManager.h>


// hknpBodyManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpBodyManager)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BodyIterator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ScheduledBodyChange)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PropertyBuffer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ScheduledBodyFlagsEnum)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpBodyManager)
    HK_TRACKER_MEMBER(hknpBodyManager, m_world, 0, "hknpWorld*") // class hknpWorld*
    HK_TRACKER_MEMBER(hknpBodyManager, m_motionManager, 0, "hknpMotionManager*") // class hknpMotionManager*
    HK_TRACKER_MEMBER(hknpBodyManager, m_bodies, 0, "hkArray<hknpBody, hkContainerHeapAllocator>") // hkArray< class hknpBody, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpBodyManager, m_previousAabbs, 0, "hkArray<hkAabb16, hkContainerHeapAllocator>") // hkArray< struct hkAabb16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpBodyManager, m_bodyNames, 0, "hkArray<hkStringPtr, hkContainerHeapAllocator>") // hkArray< hkStringPtr, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpBodyManager, m_propertyMap, 0, "hkMap<hkUint16, hknpBodyManager::PropertyBuffer*, hkMapOperations<hkUint16>, hkContainerHeapAllocator>") // class hkMap< hkUint16, struct hknpBodyManager::PropertyBuffer*, struct hkMapOperations< hkUint16 >, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpBodyManager, m_activeBodyIds, 0, "hkArray<hknpBodyId, hkContainerHeapAllocator>") // hkArray< struct hknpBodyId, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpBodyManager, m_bodyIdToCellIndexMap, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpBodyManager, m_scheduledBodyChanges, 0, "hkArray<hknpBodyManager::ScheduledBodyChange, hkContainerHeapAllocator>") // hkArray< struct hknpBodyManager::ScheduledBodyChange, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpBodyManager, m_scheduledBodyChangeIndices, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpBodyManager, m_bodiesToAddAsActive, 0, "hkArray<hknpBodyId, hkContainerHeapAllocator>") // hkArray< struct hknpBodyId, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpBodyManager, m_bodiesToAddAsInactive, 0, "hkArray<hknpBodyId, hkContainerHeapAllocator>") // hkArray< struct hknpBodyId, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpBodyManager, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpBodyManager, ScheduledBodyFlagsEnum, s_libraryName)


// BodyIterator hknpBodyManager
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpBodyManager, BodyIterator, s_libraryName)


// ScheduledBodyChange hknpBodyManager
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpBodyManager, ScheduledBodyChange, s_libraryName)


// PropertyBuffer hknpBodyManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpBodyManager::PropertyBuffer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpBodyManager::PropertyBuffer)
    HK_TRACKER_MEMBER(hknpBodyManager::PropertyBuffer, m_occupancy, 0, "hkBitField") // class hkBitField
    HK_TRACKER_MEMBER(hknpBodyManager::PropertyBuffer, m_properties, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpBodyManager::PropertyBuffer, s_libraryName)

#include <Physics/Physics/Dynamics/Body/hknpBodyQuality.h>


// hknpBodyQuality ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpBodyQuality, s_libraryName)

#include <Physics/Physics/Dynamics/Body/hknpBodyQualityId.h>


// hknpBodyQualityIdBaseDiscriminant ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpBodyQualityIdBaseDiscriminant, s_libraryName)


// hknpBodyQualityId ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpBodyQualityId, s_libraryName)

#include <Physics/Physics/Dynamics/Body/hknpBodyQualityLibrary.h>


// hknpBodyQualityLibraryCinfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpBodyQualityLibraryCinfo, s_libraryName)


// hknpBodyQualityLibrary ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpBodyQualityLibrary)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(QualityModifiedSignal)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpBodyQualityLibrary)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpBodyQualityLibrary, s_libraryName, hkReferencedObject)


// QualityModifiedSignal hknpBodyQualityLibrary
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpBodyQualityLibrary, QualityModifiedSignal, s_libraryName)

#include <Physics/Physics/Dynamics/Constraint/hknpConstraint.h>


// hknpConstraint ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpConstraint)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FlagBits)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpConstraint)
    HK_TRACKER_MEMBER(hknpConstraint, m_data, 0, "hkpConstraintData *") // class hkpConstraintData *
    HK_TRACKER_MEMBER(hknpConstraint, m_runtime, 0, "hkpConstraintRuntime*") // struct hkpConstraintRuntime*
    HK_TRACKER_MEMBER(hknpConstraint, m_atoms, 0, "hkpConstraintAtom*") // const struct hkpConstraintAtom*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpConstraint, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpConstraint, FlagBits, s_libraryName)

#include <Physics/Physics/Dynamics/Constraint/hknpConstraintCinfo.h>


// hknpConstraintCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpConstraintCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpConstraintCinfo)
    HK_TRACKER_MEMBER(hknpConstraintCinfo, m_constraintData, 0, "hkpConstraintData *") // class hkpConstraintData *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpConstraintCinfo, s_libraryName)

#include <Physics/Physics/Dynamics/Material/hknpMaterial.h>


// hknpMaterial ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMaterial)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FreeListArrayOperations)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FlagsEnum)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CombinePolicy)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TriggerVolumeType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MassChangerCategory)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpMaterial)
    HK_TRACKER_MEMBER(hknpMaterial, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hknpMaterial, m_surfaceVelocity, 0, "hknpSurfaceVelocity*") // class hknpSurfaceVelocity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpMaterial, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMaterial, FlagsEnum, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMaterial, CombinePolicy, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMaterial, TriggerVolumeType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMaterial, MassChangerCategory, s_libraryName)


// FreeListArrayOperations hknpMaterial
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMaterial, FreeListArrayOperations, s_libraryName)


// hknpRefMaterial ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpRefMaterial)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpRefMaterial)
    HK_TRACKER_MEMBER(hknpRefMaterial, m_material, 0, "hknpMaterial") // class hknpMaterial
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpRefMaterial, s_libraryName, hkReferencedObject)


// hknpMaterialDescriptor ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMaterialDescriptor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpMaterialDescriptor)
    HK_TRACKER_MEMBER(hknpMaterialDescriptor, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hknpMaterialDescriptor, m_material, 0, "hknpRefMaterial *") // class hknpRefMaterial *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpMaterialDescriptor, s_libraryName)

#include <Physics/Physics/Dynamics/Material/hknpMaterialId.h>


// hknpMaterialIdBaseDiscriminant ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpMaterialIdBaseDiscriminant, s_libraryName)


// hknpMaterialId ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpMaterialId, s_libraryName)

#include <Physics/Physics/Dynamics/Material/hknpMaterialLibrary.h>


// hknpMaterialLibrary ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMaterialLibrary)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MaterialAddedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MaterialModifiedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MaterialRemovedSignal)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpMaterialLibrary)
    HK_TRACKER_MEMBER(hknpMaterialLibrary, m_entries, 0, "hkFreeListArray<hknpMaterial, hknpMaterialId, 8, hknpMaterial::FreeListArrayOperations>") // struct hkFreeListArray< class hknpMaterial, struct hknpMaterialId, 8, struct hknpMaterial::FreeListArrayOperations >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpMaterialLibrary, s_libraryName, hkReferencedObject)


// MaterialAddedSignal hknpMaterialLibrary
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMaterialLibrary, MaterialAddedSignal, s_libraryName)


// MaterialModifiedSignal hknpMaterialLibrary
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMaterialLibrary, MaterialModifiedSignal, s_libraryName)


// MaterialRemovedSignal hknpMaterialLibrary
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMaterialLibrary, MaterialRemovedSignal, s_libraryName)

#include <Physics/Physics/Dynamics/Material/hknpMaterialPalette.h>


// hknpMaterialPaletteEntryIdDiscriminant ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpMaterialPaletteEntryIdDiscriminant, s_libraryName)


// hknpMaterialPalette ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMaterialPalette)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpMaterialPalette)
    HK_TRACKER_MEMBER(hknpMaterialPalette, m_entries, 0, "hkArray<hknpMaterialDescriptor, hkContainerHeapAllocator>") // hkArray< struct hknpMaterialDescriptor, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpMaterialPalette, s_libraryName, hkReferencedObject)

#include <Physics/Physics/Dynamics/Modifier/DefaultModifierSet/hknpDefaultModifierSet.h>


// hknpDefaultModifierSet ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDefaultModifierSet)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDefaultModifierSet)
    HK_TRACKER_MEMBER(hknpDefaultModifierSet, m_manifoldEventCreator, 0, "hknpManifoldEventCreator") // class hknpManifoldEventCreator
    HK_TRACKER_MEMBER(hknpDefaultModifierSet, m_contactImpulseEventCreator, 0, "hknpContactImpulseEventCreator") // class hknpContactImpulseEventCreator
    HK_TRACKER_MEMBER(hknpDefaultModifierSet, m_clippedImpulseEventCreator, 0, "hknpContactImpulseClippedEventCreator") // class hknpContactImpulseClippedEventCreator
    HK_TRACKER_MEMBER(hknpDefaultModifierSet, m_constraintForceEventCreator, 0, "hknpConstraintForceEventCreator") // class hknpConstraintForceEventCreator
    HK_TRACKER_MEMBER(hknpDefaultModifierSet, m_constraintForceExceededEventCreator, 0, "hknpConstraintForceExceededEventCreator") // class hknpConstraintForceExceededEventCreator
    HK_TRACKER_MEMBER(hknpDefaultModifierSet, m_restitutionModifier, 0, "hknpRestitutionModifier") // class hknpRestitutionModifier
    HK_TRACKER_MEMBER(hknpDefaultModifierSet, m_softContactModifier, 0, "hknpSoftContactModifier") // class hknpSoftContactModifier
    HK_TRACKER_MEMBER(hknpDefaultModifierSet, m_triggerVolumeModifier, 0, "hknpTriggerVolumeModifier") // class hknpTriggerVolumeModifier
    HK_TRACKER_MEMBER(hknpDefaultModifierSet, m_surfaceVelocityModifier, 0, "hknpSurfaceVelocityModifier") // class hknpSurfaceVelocityModifier
    HK_TRACKER_MEMBER(hknpDefaultModifierSet, m_massChangerModifier, 0, "hknpMassChangerModifier") // class hknpMassChangerModifier
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpDefaultModifierSet, s_libraryName, hkReferencedObject)

#include <Physics/Physics/Dynamics/Modifier/EventCreators/ConstraintForce/hknpConstraintForceEventCreator.h>


// hknpConstraintForceEventCreator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpConstraintForceEventCreator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpConstraintForceEventCreator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpConstraintForceEventCreator, s_libraryName, hknpModifier)

#include <Physics/Physics/Dynamics/Modifier/EventCreators/ConstraintForceExceeded/hknpConstraintForceExceededEventCreator.h>


// hknpConstraintForceExceededEventCreator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpConstraintForceExceededEventCreator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpConstraintForceExceededEventCreator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpConstraintForceExceededEventCreator, s_libraryName, hknpModifier)

#include <Physics/Physics/Dynamics/Modifier/EventCreators/ContactImpulse/hknpContactImpulseEventCreator.h>


// hknpContactImpulseEventCreator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpContactImpulseEventCreator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpContactImpulseEventCreator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpContactImpulseEventCreator, s_libraryName, hknpModifier)

#include <Physics/Physics/Dynamics/Modifier/EventCreators/ContactImpulseClipped/hknpContactImpulseClippedEventCreator.h>


// hknpContactImpulseClippedEventCreator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpContactImpulseClippedEventCreator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpContactImpulseClippedEventCreator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpContactImpulseClippedEventCreator, s_libraryName, hknpModifier)

#include <Physics/Physics/Dynamics/Modifier/MassChanger/hknpMassChangerModifier.h>


// hknpMassChangerModifier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMassChangerModifier)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpMassChangerModifier)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpMassChangerModifier, s_libraryName, hknpModifier)

#include <Physics/Physics/Dynamics/Modifier/Restitution/hknpRestitutionModifier.h>


// hknpRestitutionModifier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpRestitutionModifier)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpRestitutionModifier)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpRestitutionModifier, s_libraryName, hknpModifier)

#include <Physics/Physics/Dynamics/Modifier/SoftContact/hknpSoftContactModifier.h>


// hknpSoftContactModifier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpSoftContactModifier)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpSoftContactModifier)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpSoftContactModifier, s_libraryName, hknpModifier)

#include <Physics/Physics/Dynamics/Modifier/SurfaceVelocity/CircularSurfaceVelocity/hknpCircularSurfaceVelocity.h>


// hknpCircularSurfaceVelocity ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCircularSurfaceVelocity)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCircularSurfaceVelocity)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCircularSurfaceVelocity, s_libraryName, hknpSurfaceVelocity)

#include <Physics/Physics/Dynamics/Modifier/SurfaceVelocity/LinearSurfaceVelocity/hknpLinearSurfaceVelocity.h>


// hknpLinearSurfaceVelocity ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpLinearSurfaceVelocity)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ProjectMethod)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpLinearSurfaceVelocity)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpLinearSurfaceVelocity, s_libraryName, hknpSurfaceVelocity)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpLinearSurfaceVelocity, ProjectMethod, s_libraryName)

#include <Physics/Physics/Dynamics/Modifier/SurfaceVelocity/hknpSurfaceVelocity.h>


// hknpSurfaceVelocity ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpSurfaceVelocity)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Space)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpSurfaceVelocity)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpSurfaceVelocity, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpSurfaceVelocity, Space, s_libraryName)

#include <Physics/Physics/Dynamics/Modifier/SurfaceVelocity/hknpSurfaceVelocityModifier.h>


// hknpSurfaceVelocityModifier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpSurfaceVelocityModifier)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpSurfaceVelocityModifier)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpSurfaceVelocityModifier, s_libraryName, hknpModifier)

#include <Physics/Physics/Dynamics/Modifier/TriggerVolume/hknpTriggerVolumeModifier.h>


// hknpTriggerVolumeModifier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpTriggerVolumeModifier)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpTriggerVolumeModifier)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpTriggerVolumeModifier, s_libraryName, hknpModifier)

#include <Physics/Physics/Dynamics/Modifier/hknpModifier.h>


// hknpModifier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpModifier)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ManifoldCreatedCallbackInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SolverCallbackInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FunctionType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpModifier)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hknpModifier, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpModifier, FunctionType, s_libraryName)


// ManifoldCreatedCallbackInput hknpModifier

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpModifier::ManifoldCreatedCallbackInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpModifier::ManifoldCreatedCallbackInput)
    HK_TRACKER_MEMBER(hknpModifier::ManifoldCreatedCallbackInput, m_collisionCache, 0, "hknpManifoldCollisionCache*") // struct hknpManifoldCollisionCache*
    HK_TRACKER_MEMBER(hknpModifier::ManifoldCreatedCallbackInput, m_collisionCacheInMainMemory, 0, "hknpManifoldCollisionCache*") // struct hknpManifoldCollisionCache*
    HK_TRACKER_MEMBER(hknpModifier::ManifoldCreatedCallbackInput, m_manifold, 0, "hknpManifold*") // struct hknpManifold*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpModifier::ManifoldCreatedCallbackInput, s_libraryName)


// SolverCallbackInput hknpModifier

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpModifier::SolverCallbackInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpModifier::SolverCallbackInput)
    HK_TRACKER_MEMBER(hknpModifier::SolverCallbackInput, m_contactJacobian, 0, "hknpContactJacobian<1>*") // const struct hknpContactJacobian< 1 >*
    HK_TRACKER_MEMBER(hknpModifier::SolverCallbackInput, m_contactJacobianInMainMemory, 0, "hknpContactJacobian<1>*") // const struct hknpContactJacobian< 1 >*
    HK_TRACKER_MEMBER(hknpModifier::SolverCallbackInput, m_collisionCache, 0, "hknpManifoldCollisionCache*") // const struct hknpManifoldCollisionCache*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpModifier::SolverCallbackInput, s_libraryName)


// hknpWeldingModifier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpWeldingModifier)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WeldingInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpWeldingModifier)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hknpWeldingModifier, s_libraryName)


// WeldingInfo hknpWeldingModifier
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWeldingModifier, WeldingInfo, s_libraryName)

#include <Physics/Physics/Dynamics/Motion/hknpMotion.h>


// hknpMotion ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpMotion, s_libraryName)

#include <Physics/Physics/Dynamics/Motion/hknpMotionCinfo.h>


// hknpMotionCinfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpMotionCinfo, s_libraryName)

#include <Physics/Physics/Dynamics/Motion/hknpMotionId.h>


// hknpMotionIdBaseDiscriminant ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpMotionIdBaseDiscriminant, s_libraryName)


// hknpMotionId ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpMotionId, s_libraryName)

#include <Physics/Physics/Dynamics/Motion/hknpMotionManager.h>


// hknpMotionManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMotionManager)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MotionIterator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CellData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpMotionManager)
    HK_TRACKER_MEMBER(hknpMotionManager, m_bodyManager, 0, "hknpBodyManager*") // class hknpBodyManager*
    HK_TRACKER_MEMBER(hknpMotionManager, m_motions, 0, "hkArray<hknpMotion, hkContainerHeapAllocator>") // hkArray< class hknpMotion, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpMotionManager, m_activeMotionGrid, 0, "hkArray<hknpMotionManager::CellData, hkContainerHeapAllocator>") // hkArray< struct hknpMotionManager::CellData, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpMotionManager, s_libraryName)


// MotionIterator hknpMotionManager
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMotionManager, MotionIterator, s_libraryName)


// CellData hknpMotionManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMotionManager::CellData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpMotionManager::CellData)
    HK_TRACKER_MEMBER(hknpMotionManager::CellData, m_solverIdToMotionId, 0, "hkArray<hknpMotionId, hkContainerHeapAllocator>") // hkArray< struct hknpMotionId, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpMotionManager::CellData, s_libraryName)

#include <Physics/Physics/Dynamics/Motion/hknpMotionProperties.h>


// hknpMotionProperties ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMotionProperties)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FreeListArrayOperations)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FlagsEnum)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SolverStabilizationType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DeactivationStrategy)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hknpMotionProperties, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMotionProperties, FlagsEnum, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMotionProperties, SolverStabilizationType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMotionProperties, DeactivationStrategy, s_libraryName)


// FreeListArrayOperations hknpMotionProperties
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMotionProperties, FreeListArrayOperations, s_libraryName)

#include <Physics/Physics/Dynamics/Motion/hknpMotionPropertiesId.h>


// hknpMotionPropertiesIdBaseDiscriminant ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpMotionPropertiesIdBaseDiscriminant, s_libraryName)


// hknpMotionPropertiesId ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpMotionPropertiesId, s_libraryName)

#include <Physics/Physics/Dynamics/Motion/hknpMotionPropertiesLibrary.h>


// hknpMotionPropertiesLibrary ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMotionPropertiesLibrary)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MotionPropertiesAddedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MotionPropertiesModifiedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MotionPropertiesRemovedSignal)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpMotionPropertiesLibrary)
    HK_TRACKER_MEMBER(hknpMotionPropertiesLibrary, m_entries, 0, "hkFreeListArray<hknpMotionProperties, hknpMotionPropertiesId, 8, hknpMotionProperties::FreeListArrayOperations>") // struct hkFreeListArray< class hknpMotionProperties, struct hknpMotionPropertiesId, 8, struct hknpMotionProperties::FreeListArrayOperations >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpMotionPropertiesLibrary, s_libraryName, hkReferencedObject)


// MotionPropertiesAddedSignal hknpMotionPropertiesLibrary
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMotionPropertiesLibrary, MotionPropertiesAddedSignal, s_libraryName)


// MotionPropertiesModifiedSignal hknpMotionPropertiesLibrary
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMotionPropertiesLibrary, MotionPropertiesModifiedSignal, s_libraryName)


// MotionPropertiesRemovedSignal hknpMotionPropertiesLibrary
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMotionPropertiesLibrary, MotionPropertiesRemovedSignal, s_libraryName)

#include <Physics/Physics/Dynamics/Simulation/Multithreaded/hknpConstraintSetupTask.h>


// hknpGatherConstraintsTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpGatherConstraintsTask)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpGatherConstraintsTask)
    HK_TRACKER_MEMBER(hknpGatherConstraintsTask, m_constraintStates, 0, "hknpConstraintAtomSolverSetup::ConstraintStates") // struct hknpConstraintAtomSolverSetup::ConstraintStates
    HK_TRACKER_MEMBER(hknpGatherConstraintsTask, m_subTasks, 0, "hknpConstraintAtomSolverSetup::SubTasks") // struct hknpConstraintAtomSolverSetup::SubTasks
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpGatherConstraintsTask, s_libraryName, hkTask)


// hknpConstraintSetupTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpConstraintSetupTask)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpConstraintSetupTask)
    HK_TRACKER_MEMBER(hknpConstraintSetupTask, m_constraintStates, 0, "hknpConstraintAtomSolverSetup::ConstraintStates*") // const struct hknpConstraintAtomSolverSetup::ConstraintStates*
    HK_TRACKER_MEMBER(hknpConstraintSetupTask, m_subTasks, 0, "hknpConstraintAtomSolverSetup::SubTasks*") // const struct hknpConstraintAtomSolverSetup::SubTasks*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpConstraintSetupTask, s_libraryName, hkTask)

#include <Physics/Physics/Dynamics/Simulation/Multithreaded/hknpMultithreadedSimulation.h>


// hknpMultithreadedSimulation ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpMultithreadedSimulation)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Stage)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpMultithreadedSimulation)
    HK_TRACKER_MEMBER(hknpMultithreadedSimulation, m_inactiveCdCacheGrid, 0, "hknpGrid<hkBlockStreamBase::Range>") // class hknpGrid< class hkBlockStreamBase::Range >
    HK_TRACKER_MEMBER(hknpMultithreadedSimulation, m_crossGridCdCacheGrid, 0, "hknpGrid<hkBlockStreamBase::Range>") // class hknpGrid< class hkBlockStreamBase::Range >
    HK_TRACKER_MEMBER(hknpMultithreadedSimulation, m_newPairsStream, 0, "hkBlockStream<hknpBodyIdPair>") // class hkBlockStream< struct hknpBodyIdPair >
    HK_TRACKER_MEMBER(hknpMultithreadedSimulation, m_narrowPhaseTask, 0, "hknpNarrowPhaseTask *") // class hknpNarrowPhaseTask *
    HK_TRACKER_MEMBER(hknpMultithreadedSimulation, m_gatherConstraintsTask, 0, "hknpGatherConstraintsTask *") // class hknpGatherConstraintsTask *
    HK_TRACKER_MEMBER(hknpMultithreadedSimulation, m_constraintSetupTask, 0, "hknpConstraintSetupTask *") // class hknpConstraintSetupTask *
    HK_TRACKER_MEMBER(hknpMultithreadedSimulation, m_solverTask, 0, "hknpSolverTask *") // class hknpSolverTask *
    HK_TRACKER_MEMBER(hknpMultithreadedSimulation, m_solverData, 0, "hknpSolverData*") // class hknpSolverData*
    HK_TRACKER_MEMBER(hknpMultithreadedSimulation, m_solverTaskQueue, 0, "hkTaskQueue") // class hkTaskQueue
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpMultithreadedSimulation, s_libraryName, hknpSimulation)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpMultithreadedSimulation, Stage, s_libraryName)

#include <Physics/Physics/Dynamics/Simulation/Multithreaded/hknpNarrowPhaseTask.h>


// hknpNarrowPhaseTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpNarrowPhaseTask)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ThreadOutput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SubTask)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpNarrowPhaseTask)
    HK_TRACKER_MEMBER(hknpNarrowPhaseTask, m_currentSubTaskIndexPpu, 0, "hkPadSpu<hkUint32*>") // class hkPadSpu< hkUint32* >
    HK_TRACKER_MEMBER(hknpNarrowPhaseTask, m_subTasks, 0, "hkArray<hknpNarrowPhaseTask::SubTask, hkContainerHeapAllocator>") // hkArray< struct hknpNarrowPhaseTask::SubTask, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpNarrowPhaseTask, m_solverData, 0, "hkPadSpu<hknpSolverData*>") // class hkPadSpu< class hknpSolverData* >
    HK_TRACKER_MEMBER(hknpNarrowPhaseTask, m_cdCacheStreamsOut, 0, "hkArray<hknpNarrowPhaseTask::ThreadOutput, hkContainerHeapAllocator>") // hkArray< struct hknpNarrowPhaseTask::ThreadOutput, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpNarrowPhaseTask, m_cdCacheStreamIn, 0, "hkPadSpu<hknpCdCacheStream*>") // class hkPadSpu< class hknpCdCacheStream* >
    HK_TRACKER_MEMBER(hknpNarrowPhaseTask, m_childCdCacheStreamIn, 0, "hkPadSpu<hknpCdCacheStream*>") // class hkPadSpu< class hknpCdCacheStream* >
    HK_TRACKER_MEMBER(hknpNarrowPhaseTask, m_cdCacheStreamIn2, 0, "hkPadSpu<hknpCdCacheStream*>") // class hkPadSpu< class hknpCdCacheStream* >
    HK_TRACKER_MEMBER(hknpNarrowPhaseTask, m_childCdCacheStreamIn2, 0, "hkPadSpu<hknpCdCacheStream*>") // class hkPadSpu< class hknpCdCacheStream* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpNarrowPhaseTask, s_libraryName, hkTask)


// ThreadOutput hknpNarrowPhaseTask

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpNarrowPhaseTask::ThreadOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpNarrowPhaseTask::ThreadOutput)
    HK_TRACKER_MEMBER(hknpNarrowPhaseTask::ThreadOutput, m_cdCacheStream, 0, "hknpCdCacheStream") // class hknpCdCacheStream
    HK_TRACKER_MEMBER(hknpNarrowPhaseTask::ThreadOutput, m_inactiveCdCacheStream, 0, "hknpCdCacheStream") // class hknpCdCacheStream
    HK_TRACKER_MEMBER(hknpNarrowPhaseTask::ThreadOutput, m_crossGridCdCacheStream, 0, "hknpCdCacheStream") // class hknpCdCacheStream
    HK_TRACKER_MEMBER(hknpNarrowPhaseTask::ThreadOutput, m_childCdCacheStream, 0, "hknpCdCacheStream") // class hknpCdCacheStream
    HK_TRACKER_MEMBER(hknpNarrowPhaseTask::ThreadOutput, m_inactiveChildCdCacheStream, 0, "hknpCdCacheStream") // class hknpCdCacheStream
    HK_TRACKER_MEMBER(hknpNarrowPhaseTask::ThreadOutput, m_crossChildCdCacheStream, 0, "hknpCdCacheStream") // class hknpCdCacheStream
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpNarrowPhaseTask::ThreadOutput, s_libraryName)


// SubTask hknpNarrowPhaseTask
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpNarrowPhaseTask, SubTask, s_libraryName)

#include <Physics/Physics/Dynamics/Simulation/Multithreaded/hknpSolverTask.h>


// hknpSolverTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpSolverTask)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpSolverTask)
    HK_TRACKER_MEMBER(hknpSolverTask, m_syncBuffer, 0, "hkUint32*") // hkUint32*
    HK_TRACKER_MEMBER(hknpSolverTask, m_cellIdxToGlobalSolverId, 0, "hknpGrid<hknpIdxRange>") // class hknpGrid< struct hknpIdxRange >
    HK_TRACKER_MEMBER(hknpSolverTask, m_deactivationStepInfo, 0, "hknpDeactivationStepInfo*") // class hknpDeactivationStepInfo*
    HK_TRACKER_MEMBER(hknpSolverTask, m_spaceSplitterData, 0, "hknpSpaceSplitterData*") // struct hknpSpaceSplitterData*
    HK_TRACKER_MEMBER(hknpSolverTask, m_jacobianGrids, 0, "hknpConstraintSolverJacobianGrid* [3]") // class hknpConstraintSolverJacobianGrid* [3]
    HK_TRACKER_MEMBER(hknpSolverTask, m_taskBuilder, 0, "hkSimpleSchedulerTaskBuilder*") // class hkSimpleSchedulerTaskBuilder*
    HK_TRACKER_MEMBER(hknpSolverTask, m_taskGraph, 0, "hkDefaultTaskGraph") // struct hkDefaultTaskGraph
    HK_TRACKER_MEMBER(hknpSolverTask, m_subTasks, 0, "hkArray<hknpSolverSchedulerTask, hkContainerHeapAllocator>") // hkArray< struct hknpSolverSchedulerTask, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpSolverTask, m_solverStepInfo, 0, "hknpSolverStepInfo*") // class hknpSolverStepInfo*
    HK_TRACKER_MEMBER(hknpSolverTask, m_taskQueue, 0, "hkTaskQueue*") // class hkTaskQueue*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpSolverTask, s_libraryName, hkTask)

#include <Physics/Physics/Dynamics/Simulation/SingleThreaded/hknpSingleThreadedSimulation.h>


// hknpSingleThreadedSimulation ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpSingleThreadedSimulation)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpSingleThreadedSimulation)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpSingleThreadedSimulation, s_libraryName, hknpSimulation)

#include <Physics/Physics/Dynamics/Simulation/Utils/hknpCacheSorter.h>


// hknpCacheSorter ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCacheSorter, s_libraryName)

#include <Physics/Physics/Dynamics/Simulation/Utils/hknpSimulationDeterminismUtil.h>


// hknpSimulationDeterminismUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSimulationDeterminismUtil, s_libraryName)

#include <Physics/Physics/Dynamics/Simulation/hknpSimulation.h>


// hknpSimulation ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpSimulation)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CheckConsistencyFlags)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpSimulation)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hknpSimulation, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpSimulation, CheckConsistencyFlags, s_libraryName)

#include <Physics/Physics/Dynamics/Simulation/hknpSimulationContext.h>


// hknpSimulationContext ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpSimulationContext)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpSimulationContext)
    HK_TRACKER_MEMBER(hknpSimulationContext, m_taskGraph, 0, "hkTaskGraph*") // struct hkTaskGraph*
    HK_TRACKER_MEMBER(hknpSimulationContext, m_threadContexts, 0, "hknpSimulationThreadContext [12]") // class hknpSimulationThreadContext [12]
    HK_TRACKER_MEMBER(hknpSimulationContext, m_commandGrid, 0, "hknpCommandGrid") // class hknpCommandGrid
    HK_TRACKER_MEMBER(hknpSimulationContext, m_deferredCommandStream, 0, "hkBlockStream<hkCommand>") // class hkBlockStream< class hkCommand >
    HK_TRACKER_MEMBER(hknpSimulationContext, m_deferredCommandStreamAllocator, 0, "hkThreadLocalBlockStreamAllocator*") // class hkThreadLocalBlockStreamAllocator*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpSimulationContext, s_libraryName)

#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>


// hknpCommandGrid ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCommandGrid, s_libraryName)


// hknpSimulationThreadContext ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpSimulationThreadContext)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpSimulationThreadContext)
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_world, 0, "hkPadSpu<hknpWorld*>") // class hkPadSpu< class hknpWorld* >
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_qualities, 0, "hkPadSpu<hknpBodyQuality*>") // class hkPadSpu< const class hknpBodyQuality* >
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_materials, 0, "hkPadSpu<hknpMaterial*>") // class hkPadSpu< const class hknpMaterial* >
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_modifierManager, 0, "hkPadSpu<hknpModifierManager*>") // class hkPadSpu< class hknpModifierManager* >
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_shapeTagCodec, 0, "hkPadSpu<hknpShapeTagCodec*>") // class hkPadSpu< const class hknpShapeTagCodec* >
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_triangleShapePrototypes, 0, "hkPadSpu<hknpTriangleShape*> [2]") // class hkPadSpu< class hknpTriangleShape* > [2]
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_deactivationData, 0, "hkPadSpu<hknpDeactivationThreadData*>") // class hkPadSpu< struct hknpDeactivationThreadData* >
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_solverStepInfo, 0, "hkPadSpu<hknpSolverStepInfo*>") // class hkPadSpu< class hknpSolverStepInfo* >
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_deactivationStepInfo, 0, "hkPadSpu<hknpDeactivationStepInfo*>") // class hkPadSpu< class hknpDeactivationStepInfo* >
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_cellIdxToGlobalSolverId, 0, "hkPadSpu<hknpGrid<hknpIdxRange>*>") // class hkPadSpu< class hknpGrid< struct hknpIdxRange >* >
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_spaceSplitterData, 0, "hkPadSpu<hknpSpaceSplitterData*>") // class hkPadSpu< struct hknpSpaceSplitterData* >
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_tempAllocator, 0, "hkPadSpu<hkThreadLocalBlockStreamAllocator*>") // class hkPadSpu< class hkThreadLocalBlockStreamAllocator* >
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_heapAllocator, 0, "hkPadSpu<hkThreadLocalBlockStreamAllocator*>") // class hkPadSpu< class hkThreadLocalBlockStreamAllocator* >
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_commandBlockStream, 0, "hkBlockStream<hkCommand>") // class hkBlockStream< class hkCommand >
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_commandWriter, 0, "hkPadSpu<hkBlockStreamCommandWriter*>") // class hkPadSpu< class hkBlockStreamCommandWriter* >
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_commandGrid, 0, "hkPadSpu<hknpCommandGrid*>") // class hkPadSpu< class hknpCommandGrid* >
    HK_TRACKER_MEMBER(hknpSimulationThreadContext, m_currentGridEntryRange, 0, "hkBlockStreamBase::LinkedRange") // class hkBlockStreamBase::LinkedRange
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpSimulationThreadContext, s_libraryName)

#include <Physics/Physics/Dynamics/World/CacheManager/hknpCollisionCacheManager.h>


// hknpCollisionCacheManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCollisionCacheManager)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCollisionCacheManager)
    HK_TRACKER_MEMBER(hknpCollisionCacheManager, m_cdCacheStream, 0, "hknpCdCacheStream") // class hknpCdCacheStream
    HK_TRACKER_MEMBER(hknpCollisionCacheManager, m_childCdCacheStream, 0, "hknpCdCacheStream") // class hknpCdCacheStream
    HK_TRACKER_MEMBER(hknpCollisionCacheManager, m_newCdCacheStream, 0, "hknpCdCacheStream") // class hknpCdCacheStream
    HK_TRACKER_MEMBER(hknpCollisionCacheManager, m_newChildCdCacheStream, 0, "hknpCdCacheStream") // class hknpCdCacheStream
    HK_TRACKER_MEMBER(hknpCollisionCacheManager, m_inactiveCdCacheStream, 0, "hknpCdCacheStream") // class hknpCdCacheStream
    HK_TRACKER_MEMBER(hknpCollisionCacheManager, m_inactiveChildCdCacheStream, 0, "hknpCdCacheStream") // class hknpCdCacheStream
    HK_TRACKER_MEMBER(hknpCollisionCacheManager, m_newUserCdCacheStream, 0, "hknpCdCacheStream") // class hknpCdCacheStream
    HK_TRACKER_MEMBER(hknpCollisionCacheManager, m_newUserChildCdCacheStream, 0, "hknpCdCacheStream") // class hknpCdCacheStream
    HK_TRACKER_MEMBER(hknpCollisionCacheManager, m_newUserCollisionPairs, 0, "hkArray<hknpBodyIdPair, hkContainerHeapAllocator>") // hkArray< struct hknpBodyIdPair, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpCollisionCacheManager, m_cdCacheGrid, 0, "hknpGrid<hkBlockStreamBase::Range>") // class hknpGrid< class hkBlockStreamBase::Range >
    HK_TRACKER_MEMBER(hknpCollisionCacheManager, m_newCdCacheGrid, 0, "hknpGrid<hkBlockStreamBase::Range>") // class hknpGrid< class hkBlockStreamBase::Range >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpCollisionCacheManager, s_libraryName)

#include <Physics/Physics/Dynamics/World/Commands/hknpCommands.h>


// hknpApiCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpApiCommand, s_libraryName)


// hknpEmptyCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpEmptyCommand, s_libraryName)


// hknpApiCommandProcessor ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpApiCommandProcessor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpApiCommandProcessor)
    HK_TRACKER_MEMBER(hknpApiCommandProcessor, m_world, 0, "hknpWorld*") // class hknpWorld*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpApiCommandProcessor, s_libraryName, hkSecondaryCommandDispatcher)


// hknpCreateBodyCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCreateBodyCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 0 > as it is a template


// hknpDestroyBodyCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpDestroyBodyCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 2 > as it is a template


// hknpAddBodyCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpAddBodyCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 1 > as it is a template


// hknpRemoveBodyCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpRemoveBodyCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 3 > as it is a template


// hknpDetachBodyCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpDetachBodyCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 5 > as it is a template


// hknpAttachBodyCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpAttachBodyCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 4 > as it is a template


// hknpSetBodyTransformCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetBodyTransformCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 6 > as it is a template


// hknpSetBodyPositionCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetBodyPositionCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 7 > as it is a template


// hknpSetBodyOrientationCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetBodyOrientationCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 8 > as it is a template


// hknpSetBodyVelocityCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetBodyVelocityCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 9 > as it is a template


// hknpSetBodyLinearVelocityCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetBodyLinearVelocityCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 10 > as it is a template


// hknpSetBodyAngularVelocityCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetBodyAngularVelocityCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 11 > as it is a template


// hknpReintegrateBodyCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpReintegrateBodyCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 16 > as it is a template


// hknpApplyLinearImpulseCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpApplyLinearImpulseCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 12 > as it is a template


// hknpApplyAngularImpulseCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpApplyAngularImpulseCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 13 > as it is a template


// hknpApplyPointImpulseCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpApplyPointImpulseCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 14 > as it is a template


// hknpSetPointVelocityCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetPointVelocityCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 15 > as it is a template


// hknpSetBodyMassCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetBodyMassCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 18 > as it is a template


// hknpSetBodyMotionCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetBodyMotionCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 17 > as it is a template


// hknpSetBodyCenterOfMassCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetBodyCenterOfMassCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 19 > as it is a template


// hknpSetBodyShapeCommand ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpSetBodyShapeCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpSetBodyShapeCommand)
    HK_TRACKER_MEMBER(hknpSetBodyShapeCommand, m_shape, 0, "hknpShape*") // const class hknpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpSetBodyShapeCommand, s_libraryName, hknpApiCommand)

 // Skipping Class hkCommandTypeDiscriminator< 20 > as it is a template


// hknpSetBodyMotionPropertiesCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetBodyMotionPropertiesCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 21 > as it is a template


// hknpSetBodyMaterialCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetBodyMaterialCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 22 > as it is a template


// hknpSetBodyQualityCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetBodyQualityCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 23 > as it is a template


// hknpActivateBodyCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpActivateBodyCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 24 > as it is a template


// hknpSetBodyCollisionFilterInfoCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetBodyCollisionFilterInfoCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 25 > as it is a template


// hknpRebuildBodyCollisionCachesCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpRebuildBodyCollisionCachesCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 26 > as it is a template


// hknpSetWorldGravityCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetWorldGravityCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 37 > as it is a template


// hknpSetMaterialFrictionCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetMaterialFrictionCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 38 > as it is a template


// hknpSetMaterialRestitutionCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetMaterialRestitutionCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 39 > as it is a template


// hknpSetBodyCollisionLookAheadDistanceCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSetBodyCollisionLookAheadDistanceCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 27 > as it is a template


// hknpReserved1VelocityCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpReserved1VelocityCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 28 > as it is a template


// hknpReserved2VelocityCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpReserved2VelocityCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 29 > as it is a template


// hknpReserved3VelocityCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpReserved3VelocityCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 30 > as it is a template


// hknpReserved4VelocityCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpReserved4VelocityCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 31 > as it is a template


// hknpReserved5VelocityCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpReserved5VelocityCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 32 > as it is a template


// hknpReserved6VelocityCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpReserved6VelocityCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 33 > as it is a template


// hknpReserved7VelocityCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpReserved7VelocityCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 34 > as it is a template


// hknpReserved8VelocityCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpReserved8VelocityCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 35 > as it is a template


// hknpReserved9VelocityCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpReserved9VelocityCommand, s_libraryName)

 // Skipping Class hkCommandTypeDiscriminator< 36 > as it is a template

#include <Physics/Physics/Dynamics/World/Commands/hknpInternalCommands.h>


// hknpInternalCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpInternalCommand, s_libraryName)


// hknpEmptyInternalCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpEmptyInternalCommand, s_libraryName)


// hknpCellIndexModifiedCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCellIndexModifiedCommand, s_libraryName)

 // Skipping Class hknpInternalCommandTypeDiscriminator< 0 > as it is a template


// hknpAtomSolverForceClippedCommand ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpAtomSolverForceClippedCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpAtomSolverForceClippedCommand)
    HK_TRACKER_MEMBER(hknpAtomSolverForceClippedCommand, m_constraint, 0, "hknpConstraint*") // class hknpConstraint*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpAtomSolverForceClippedCommand, s_libraryName, hknpInternalCommand)

 // Skipping Class hknpInternalCommandTypeDiscriminator< 1 > as it is a template


// hknpValidateTriggerVolumeEventCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpValidateTriggerVolumeEventCommand, s_libraryName)

 // Skipping Class hknpInternalCommandTypeDiscriminator< 2 > as it is a template


// hknpMotionWeldTOICommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpMotionWeldTOICommand, s_libraryName)

 // Skipping Class hknpInternalCommandTypeDiscriminator< 3 > as it is a template


// hknpAddConstraintRangeCommand ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpAddConstraintRangeCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpAddConstraintRangeCommand)
    HK_TRACKER_MEMBER(hknpAddConstraintRangeCommand, m_range, 0, "hknpConstraintSolverJacobianRange2") // struct hknpConstraintSolverJacobianRange2
    HK_TRACKER_MEMBER(hknpAddConstraintRangeCommand, m_grid, 0, "hknpConstraintSolverJacobianGrid*") // class hknpConstraintSolverJacobianGrid*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpAddConstraintRangeCommand, s_libraryName, hknpInternalCommand)

 // Skipping Class hknpInternalCommandTypeDiscriminator< 4 > as it is a template


// hknpInternalCommandProcessor ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpInternalCommandProcessor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpInternalCommandProcessor)
    HK_TRACKER_MEMBER(hknpInternalCommandProcessor, m_world, 0, "hknpWorld*") // class hknpWorld*
    HK_TRACKER_MEMBER(hknpInternalCommandProcessor, m_solverData, 0, "hknpSolverData*") // class hknpSolverData*
    HK_TRACKER_MEMBER(hknpInternalCommandProcessor, m_simulationThreadContext, 0, "hknpSimulationThreadContext*") // class hknpSimulationThreadContext*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpInternalCommandProcessor, s_libraryName, hkSecondaryCommandDispatcher)

#include <Physics/Physics/Dynamics/World/Deactivation/CdCacheFilter/hknpDeactiveCdCacheFilter.h>

// hk.MemoryTracker ignore hknpDeactiveCdCacheFilter
#include <Physics/Physics/Dynamics/World/Deactivation/Util/hknpDeactivationStateUtil.h>


// hknpDeactivationStateUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpDeactivationStateUtil, s_libraryName)

#include <Physics/Physics/Dynamics/World/Deactivation/hknpActivationListener.h>


// hknpActivationListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpActivationListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpActivationListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hknpActivationListener, s_libraryName)

#include <Physics/Physics/Dynamics/World/Deactivation/hknpCollisionPair.h>


// hknpCollisionPair ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCollisionPair, s_libraryName)

// Skipping Class hknpCdPairWriter as it is derived from a template


// hknpCdPairStream ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCdPairStream, s_libraryName)

#include <Physics/Physics/Dynamics/World/Events/Dispatchers/hknpEventMergeAndDispatcher.h>


// hknpEventMergeAndDispatcher ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpEventMergeAndDispatcher)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TriggerVolumeEventWithCount)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpEventMergeAndDispatcher)
    HK_TRACKER_MEMBER(hknpEventMergeAndDispatcher, m_triggerVolumeEvents, 0, "hkArray<hknpEventMergeAndDispatcher::TriggerVolumeEventWithCount, hkContainerHeapAllocator>") // hkArray< struct hknpEventMergeAndDispatcher::TriggerVolumeEventWithCount, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpEventMergeAndDispatcher, s_libraryName, hknpEventDispatcher)


// TriggerVolumeEventWithCount hknpEventMergeAndDispatcher
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpEventMergeAndDispatcher, TriggerVolumeEventWithCount, s_libraryName)

#include <Physics/Physics/Dynamics/World/Events/hknpEventDispatcher.h>


// hknpEventHandlerInput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpEventHandlerInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpEventHandlerInput)
    HK_TRACKER_MEMBER(hknpEventHandlerInput, m_world, 0, "hknpWorld*") // class hknpWorld*
    HK_TRACKER_MEMBER(hknpEventHandlerInput, m_solverData, 0, "hknpSolverData*") // class hknpSolverData*
    HK_TRACKER_MEMBER(hknpEventHandlerInput, m_commandWriter, 0, "hkBlockStream<hkCommand>::Writer*") // class hkBlockStream<hkCommand>::Writer*
    HK_TRACKER_MEMBER(hknpEventHandlerInput, m_simulationThreadContext, 0, "hknpSimulationThreadContext*") // class hknpSimulationThreadContext*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpEventHandlerInput, s_libraryName)


// hknpEventSignal ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpEventSignal, s_libraryName)


// hknpEventDispatcher ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpEventDispatcher)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Entry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpEventDispatcher)
    HK_TRACKER_MEMBER(hknpEventDispatcher, m_world, 0, "hknpWorld*") // class hknpWorld*
    HK_TRACKER_MEMBER(hknpEventDispatcher, m_solverData, 0, "hknpSolverData*") // class hknpSolverData*
    HK_TRACKER_MEMBER(hknpEventDispatcher, m_commandWriter, 0, "hkBlockStream<hkCommand>::Writer*") // class hkBlockStream<hkCommand>::Writer*
    HK_TRACKER_MEMBER(hknpEventDispatcher, m_simulationThreadContext, 0, "hknpSimulationThreadContext*") // class hknpSimulationThreadContext*
    HK_TRACKER_MEMBER(hknpEventDispatcher, m_entryPool, 0, "hkArray<hknpEventDispatcher::Entry, hkContainerHeapAllocator>") // hkArray< struct hknpEventDispatcher::Entry, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpEventDispatcher, m_bodyToEntryMap, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpEventDispatcher, s_libraryName, hkSecondaryCommandDispatcher)


// Entry hknpEventDispatcher
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpEventDispatcher, Entry, s_libraryName)

#include <Physics/Physics/Dynamics/World/Events/hknpEventType.h>


// hknpEventType ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpEventType, s_libraryName)

#include <Physics/Physics/Dynamics/World/Events/hknpEvents.h>


// hknpEvent ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpEvent, s_libraryName)


// hknpEmptyEvent ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpEmptyEvent, s_libraryName)


// hknpUnaryBodyEvent ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpUnaryBodyEvent, s_libraryName)


// hknpBinaryBodyEvent ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpBinaryBodyEvent, s_libraryName)


// hknpBodyActivationEvent ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpBodyActivationEvent, s_libraryName)

 // Skipping Class hknpEventTypeDiscriminator< 12 > as it is a template


// hknpBodyExitedBroadPhaseEvent ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpBodyExitedBroadPhaseEvent, s_libraryName)

 // Skipping Class hknpEventTypeDiscriminator< 13 > as it is a template


// hknpReserved0Event ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpReserved0Event, s_libraryName)

 // Skipping Class hknpEventTypeDiscriminator< 20 > as it is a template


// hknpReserved1Event ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpReserved1Event, s_libraryName)

 // Skipping Class hknpEventTypeDiscriminator< 21 > as it is a template


// hknpReserved2Event ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpReserved2Event, s_libraryName)

 // Skipping Class hknpEventTypeDiscriminator< 22 > as it is a template


// hknpReserved3Event ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpReserved3Event, s_libraryName)

 // Skipping Class hknpEventTypeDiscriminator< 23 > as it is a template

#include <Physics/Physics/Dynamics/World/Grid/hknpGrid.h>


// hknpIdxRange ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpIdxRange, s_libraryName)

#include <Physics/Physics/Dynamics/World/ModifierManager/hknpModifierManager.h>


// hknpModifierManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpModifierManager)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ModifierEntry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ModifierEntries)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Priority)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpModifierManager)
    HK_TRACKER_MEMBER(hknpModifierManager, m_modifiersPerFunction, 0, "hknpModifierManager::ModifierEntries [8]") // struct hknpModifierManager::ModifierEntries [8]
    HK_TRACKER_MEMBER(hknpModifierManager, m_neighborWeldingModifier, 0, "hknpWeldingModifier*") // class hknpWeldingModifier*
    HK_TRACKER_MEMBER(hknpModifierManager, m_motionWeldingModifier, 0, "hknpWeldingModifier*") // class hknpWeldingModifier*
    HK_TRACKER_MEMBER(hknpModifierManager, m_triangleWeldingModifier, 0, "hknpWeldingModifier*") // class hknpWeldingModifier*
    HK_TRACKER_MEMBER(hknpModifierManager, m_constraintSolvers, 0, "hknpConstraintSolver* [8]") // class hknpConstraintSolver* [8]
    HK_TRACKER_MEMBER(hknpModifierManager, m_collisionDetectors, 0, "hknpCollisionDetector* [7]") // class hknpCollisionDetector* [7]
    HK_TRACKER_MEMBER(hknpModifierManager, m_collisionFilter, 0, "hknpCollisionFilter *") // class hknpCollisionFilter *
    HK_TRACKER_MEMBER(hknpModifierManager, m_collisionQueryFilter, 0, "hknpCollisionFilter *") // class hknpCollisionFilter *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpModifierManager, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpModifierManager, Priority, s_libraryName)


// ModifierEntry hknpModifierManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpModifierManager::ModifierEntry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpModifierManager::ModifierEntry)
    HK_TRACKER_MEMBER(hknpModifierManager::ModifierEntry, m_modifier, 0, "hknpModifier*") // class hknpModifier*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpModifierManager::ModifierEntry, s_libraryName)


// ModifierEntries hknpModifierManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpModifierManager::ModifierEntries)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpModifierManager::ModifierEntries)
    HK_TRACKER_MEMBER(hknpModifierManager::ModifierEntries, m_entries, 0, "hknpModifierManager::ModifierEntry [8]") // struct hknpModifierManager::ModifierEntry [8]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpModifierManager::ModifierEntries, s_libraryName)

#include <Physics/Physics/Dynamics/World/ShapeManager/hknpShapeManager.h>


// hknpShapeManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeManager)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MutableShapeInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShapeManager)
    HK_TRACKER_MEMBER(hknpShapeManager, m_mutableShapeInfos, 0, "hkArray<hknpShapeManager::MutableShapeInfo*, hkContainerHeapAllocator>") // hkArray< struct hknpShapeManager::MutableShapeInfo*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpShapeManager, m_freeMutableShapeInfos, 0, "hkArray<hknpShapeManager::MutableShapeInfo*, hkContainerHeapAllocator>") // hkArray< struct hknpShapeManager::MutableShapeInfo*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpShapeManager, s_libraryName)


// MutableShapeInfo hknpShapeManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeManager::MutableShapeInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShapeManager::MutableShapeInfo)
    HK_TRACKER_MEMBER(hknpShapeManager::MutableShapeInfo, m_shapeManager, 0, "hknpShapeManager*") // class hknpShapeManager*
    HK_TRACKER_MEMBER(hknpShapeManager::MutableShapeInfo, m_shape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpShapeManager::MutableShapeInfo, m_bodyIds, 0, "hkArray<hknpBodyId, hkContainerHeapAllocator>") // hkArray< struct hknpBodyId, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpShapeManager::MutableShapeInfo, s_libraryName)

#include <Physics/Physics/Dynamics/World/hknpStepInput.h>

// hk.MemoryTracker ignore hknpStepInput
#include <Physics/Physics/Dynamics/World/hknpWorld.h>


// hknpWorld ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpWorld)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AdditionFlagsEnum)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PivotLocation)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(UpdateMassPropertiesMode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(RebuildCachesMode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(RebuildMassPropertiesMode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SimulationStage)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpWorld)
    HK_TRACKER_MEMBER(hknpWorld, m_bodyManager, 0, "hknpBodyManager") // class hknpBodyManager
    HK_TRACKER_MEMBER(hknpWorld, m_motionManager, 0, "hknpMotionManager") // class hknpMotionManager
    HK_TRACKER_MEMBER(hknpWorld, m_modifierManager, 0, "hknpModifierManager*") // class hknpModifierManager*
    HK_TRACKER_MEMBER(hknpWorld, m_actionManager, 0, "hknpActionManager*") // class hknpActionManager*
    HK_TRACKER_MEMBER(hknpWorld, m_persistentStreamAllocator, 0, "hkBlockStreamAllocator *") // class hkBlockStreamAllocator *
    HK_TRACKER_MEMBER(hknpWorld, m_stepLocalStreamAllocator, 0, "hkBlockStreamAllocator*") // class hkBlockStreamAllocator*
    HK_TRACKER_MEMBER(hknpWorld, m_broadPhase, 0, "hknpBroadPhase*") // class hknpBroadPhase*
    HK_TRACKER_MEMBER(hknpWorld, m_collisionCacheManager, 0, "hknpCollisionCacheManager*") // class hknpCollisionCacheManager*
    HK_TRACKER_MEMBER(hknpWorld, m_collisionDispatcher, 0, "hknpCollisionDispatcher*") // class hknpCollisionDispatcher*
    HK_TRACKER_MEMBER(hknpWorld, m_collisionQueryDispatcher, 0, "hknpCollisionQueryDispatcherBase*") // class hknpCollisionQueryDispatcherBase*
    HK_TRACKER_MEMBER(hknpWorld, m_contactSolver, 0, "hknpContactSolver*") // class hknpContactSolver*
    HK_TRACKER_MEMBER(hknpWorld, m_constraintAtomSolver, 0, "hknpConstraintAtomSolver*") // class hknpConstraintAtomSolver*
    HK_TRACKER_MEMBER(hknpWorld, m_solverVelocities, 0, "hkArray<hknpSolverVelocity, hkContainerHeapAllocator>") // hkArray< class hknpSolverVelocity, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpWorld, m_solverSumVelocities, 0, "hkArray<hknpSolverSumVelocity, hkContainerHeapAllocator>") // hkArray< class hknpSolverSumVelocity, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpWorld, m_deactivationManager, 0, "hknpDeactivationManager*") // class hknpDeactivationManager*
    HK_TRACKER_MEMBER(hknpWorld, m_deactiveCdCacheFilter, 0, "hknpDeactiveCdCacheFilter *") // class hknpDeactiveCdCacheFilter *
    HK_TRACKER_MEMBER(hknpWorld, m_simulation, 0, "hknpSimulation*") // class hknpSimulation*
    HK_TRACKER_MEMBER(hknpWorld, m_spaceSplitter, 0, "hknpSpaceSplitter*") // class hknpSpaceSplitter*
    HK_TRACKER_MEMBER(hknpWorld, m_simulationContext, 0, "hknpSimulationContext*") // class hknpSimulationContext*
    HK_TRACKER_MEMBER(hknpWorld, m_commandDispatcher, 0, "hkPrimaryCommandDispatcher*") // class hkPrimaryCommandDispatcher*
    HK_TRACKER_MEMBER(hknpWorld, m_traceDispatcher, 0, "hkSecondaryCommandDispatcher *") // class hkSecondaryCommandDispatcher *
    HK_TRACKER_MEMBER(hknpWorld, m_userReferencedObjects, 0, "hkArray<hkReferencedObject *, hkContainerHeapAllocator>") // hkArray< const class hkReferencedObject *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpWorld, m_defaultModifierSet, 0, "hkReferencedObject *") // class hkReferencedObject *
    HK_TRACKER_MEMBER(hknpWorld, m_materialLibrary, 0, "hknpMaterialLibrary *") // class hknpMaterialLibrary *
    HK_TRACKER_MEMBER(hknpWorld, m_motionPropertiesLibrary, 0, "hknpMotionPropertiesLibrary *") // class hknpMotionPropertiesLibrary *
    HK_TRACKER_MEMBER(hknpWorld, m_qualityLibrary, 0, "hknpBodyQualityLibrary *") // class hknpBodyQualityLibrary *
    HK_TRACKER_MEMBER(hknpWorld, m_dirtyMaterials, 0, "hkBitField") // class hkBitField
    HK_TRACKER_MEMBER(hknpWorld, m_dirtyQualities, 0, "hkBitField") // class hkBitField
    HK_TRACKER_MEMBER(hknpWorld, m_shapeTagCodec, 0, "hknpShapeTagCodec *") // class hknpShapeTagCodec *
    HK_TRACKER_MEMBER(hknpWorld, m_nullShapeTagCodec, 0, "hknpNullShapeTagCodec") // class hknpNullShapeTagCodec
    HK_TRACKER_MEMBER(hknpWorld, m_eventDispatcher, 0, "hknpEventDispatcher *") // class hknpEventDispatcher *
    HK_TRACKER_MEMBER(hknpWorld, m_internalPhysicsCommandDispatcher, 0, "hknpInternalCommandProcessor *") // class hknpInternalCommandProcessor *
    HK_TRACKER_MEMBER(hknpWorld, m_shapeManager, 0, "hknpShapeManager") // class hknpShapeManager
    HK_TRACKER_MEMBER(hknpWorld, m_preCollideTask, 0, "hknpPreCollideTask*") // class hknpPreCollideTask*
    HK_TRACKER_MEMBER(hknpWorld, m_postCollideTask, 0, "hknpPostCollideTask*") // class hknpPostCollideTask*
    HK_TRACKER_MEMBER(hknpWorld, m_postSolveTask, 0, "hknpPostSolveTask*") // class hknpPostSolveTask*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpWorld, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorld, AdditionFlagsEnum, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorld, PivotLocation, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorld, UpdateMassPropertiesMode, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorld, RebuildCachesMode, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorld, RebuildMassPropertiesMode, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorld, SimulationStage, s_libraryName)

#include <Physics/Physics/Dynamics/World/hknpWorldCinfo.h>


// hknpWorldCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpWorldCinfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SimulationType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SolverType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LeavingBroadPhaseBehavior)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpWorldCinfo)
    HK_TRACKER_MEMBER(hknpWorldCinfo, m_userBodyBuffer, 0, "hknpBody*") // class hknpBody*
    HK_TRACKER_MEMBER(hknpWorldCinfo, m_userMotionBuffer, 0, "hknpMotion*") // class hknpMotion*
    HK_TRACKER_MEMBER(hknpWorldCinfo, m_materialLibrary, 0, "hknpMaterialLibrary *") // class hknpMaterialLibrary *
    HK_TRACKER_MEMBER(hknpWorldCinfo, m_motionPropertiesLibrary, 0, "hknpMotionPropertiesLibrary *") // class hknpMotionPropertiesLibrary *
    HK_TRACKER_MEMBER(hknpWorldCinfo, m_qualityLibrary, 0, "hknpBodyQualityLibrary *") // class hknpBodyQualityLibrary *
    HK_TRACKER_MEMBER(hknpWorldCinfo, m_persistentStreamAllocator, 0, "hkBlockStreamAllocator*") // class hkBlockStreamAllocator*
    HK_TRACKER_MEMBER(hknpWorldCinfo, m_broadPhaseConfig, 0, "hknpBroadPhaseConfig *") // class hknpBroadPhaseConfig *
    HK_TRACKER_MEMBER(hknpWorldCinfo, m_collisionFilter, 0, "hknpCollisionFilter *") // class hknpCollisionFilter *
    HK_TRACKER_MEMBER(hknpWorldCinfo, m_collisionQueryFilter, 0, "hknpCollisionFilter *") // class hknpCollisionFilter *
    HK_TRACKER_MEMBER(hknpWorldCinfo, m_shapeTagCodec, 0, "hknpShapeTagCodec *") // class hknpShapeTagCodec *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpWorldCinfo, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldCinfo, SimulationType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldCinfo, SolverType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldCinfo, LeavingBroadPhaseBehavior, s_libraryName)

#include <Physics/Physics/Dynamics/World/hknpWorldShiftUtil.h>


// hknpWorldShiftUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpWorldShiftUtil, s_libraryName)

#include <Physics/Physics/Dynamics/World/hknpWorldSignals.h>


// hknpWorldSignals ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpWorldSignals)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WorldDestroyedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WorldShiftedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BodyBufferFullSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BodyBufferChangedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BodyCreatedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BodyAddedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BodyRemovedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BodyDestroyedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MotionBufferFullSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MotionBufferChangedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MotionCreatedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MotionDestroyedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(StaticBodyMovedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BodySwitchStaticDynamicSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BodyAttachToCompoundSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BodyDetachToCompoundSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BodyShapeSetSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BodyChangedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConstraintAddedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConstraintRemovedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ImmediateConstraintAddedSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PreCollideSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PostCollideSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PreSolveSignal)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PostSolveSignal)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hknpWorldSignals, s_libraryName)


// WorldDestroyedSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, WorldDestroyedSignal, s_libraryName)


// WorldShiftedSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, WorldShiftedSignal, s_libraryName)


// BodyBufferFullSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, BodyBufferFullSignal, s_libraryName)


// BodyBufferChangedSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, BodyBufferChangedSignal, s_libraryName)


// BodyCreatedSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, BodyCreatedSignal, s_libraryName)


// BodyAddedSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, BodyAddedSignal, s_libraryName)


// BodyRemovedSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, BodyRemovedSignal, s_libraryName)


// BodyDestroyedSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, BodyDestroyedSignal, s_libraryName)


// MotionBufferFullSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, MotionBufferFullSignal, s_libraryName)


// MotionBufferChangedSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, MotionBufferChangedSignal, s_libraryName)


// MotionCreatedSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, MotionCreatedSignal, s_libraryName)


// MotionDestroyedSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, MotionDestroyedSignal, s_libraryName)


// StaticBodyMovedSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, StaticBodyMovedSignal, s_libraryName)


// BodySwitchStaticDynamicSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, BodySwitchStaticDynamicSignal, s_libraryName)


// BodyAttachToCompoundSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, BodyAttachToCompoundSignal, s_libraryName)


// BodyDetachToCompoundSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, BodyDetachToCompoundSignal, s_libraryName)


// BodyShapeSetSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, BodyShapeSetSignal, s_libraryName)


// BodyChangedSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, BodyChangedSignal, s_libraryName)


// ConstraintAddedSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, ConstraintAddedSignal, s_libraryName)


// ConstraintRemovedSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, ConstraintRemovedSignal, s_libraryName)


// ImmediateConstraintAddedSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, ImmediateConstraintAddedSignal, s_libraryName)


// PreCollideSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, PreCollideSignal, s_libraryName)


// PostCollideSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, PostCollideSignal, s_libraryName)


// PreSolveSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, PreSolveSignal, s_libraryName)


// PostSolveSignal hknpWorldSignals
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpWorldSignals, PostSolveSignal, s_libraryName)

#include <Physics/Physics/Extensions/ActiveBodySet/hknpActiveBodySet.h>


// hknpActiveBodySet ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpActiveBodySet)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpActiveBodySet)
    HK_TRACKER_MEMBER(hknpActiveBodySet, m_activeBodies, 0, "hkArray<hknpBodyId, hkContainerHeapAllocator>") // hkArray< struct hknpBodyId, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpActiveBodySet, m_inactiveBodies, 0, "hkArray<hknpBodyId, hkContainerHeapAllocator>") // hkArray< struct hknpBodyId, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpActiveBodySet, m_world, 0, "hknpWorld*") // class hknpWorld*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpActiveBodySet, s_libraryName, hkReferencedObject)


// hknpTriggerVolumeFilteredBodySet ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpTriggerVolumeFilteredBodySet)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpTriggerVolumeFilteredBodySet)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpTriggerVolumeFilteredBodySet, s_libraryName, hknpActiveBodySet)

#include <Physics/Physics/Extensions/AutoLookAheadDistance/hknpAutoLookAheadDistanceUtil.h>


// hknpAutoLookAheadDistanceUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpAutoLookAheadDistanceUtil)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpAutoLookAheadDistanceUtil)
    HK_TRACKER_MEMBER(hknpAutoLookAheadDistanceUtil, m_world, 0, "hknpWorld*") // class hknpWorld*
    HK_TRACKER_MEMBER(hknpAutoLookAheadDistanceUtil, m_registeredBodies, 0, "hkArray<hknpBodyId, hkContainerHeapAllocator>") // hkArray< struct hknpBodyId, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpAutoLookAheadDistanceUtil, s_libraryName, hkReferencedObject)

#include <Physics/Physics/Extensions/CharacterControl/Proxy/hknpCharacterProxy.h>


// hknpCharacterProxyProperty ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterProxyProperty)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterProxyProperty)
    HK_TRACKER_MEMBER(hknpCharacterProxyProperty, m_proxyController, 0, "hknpCharacterProxy*") // class hknpCharacterProxy*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpCharacterProxyProperty, s_libraryName)


// hknpCharacterProxy ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterProxy)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TriggerVolume)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TriggerVolumeHit)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ShapeInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterProxy)
    HK_TRACKER_MEMBER(hknpCharacterProxy, m_manifold, 0, "hkArray<hknpCollisionResult, hkContainerHeapAllocator>") // hkArray< struct hknpCollisionResult, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpCharacterProxy, m_shape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpCharacterProxy, m_world, 0, "hknpWorld*") // class hknpWorld*
    HK_TRACKER_MEMBER(hknpCharacterProxy, m_listeners, 0, "hkArray<hknpCharacterProxyListener*, hkContainerHeapAllocator>") // hkArray< class hknpCharacterProxyListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpCharacterProxy, m_overlappingTriggers, 0, "hkArray<hknpCharacterProxy::TriggerVolume, hkContainerHeapAllocator>") // hkArray< struct hknpCharacterProxy::TriggerVolume, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCharacterProxy, s_libraryName, hkReferencedObject)


// TriggerVolume hknpCharacterProxy
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpCharacterProxy, TriggerVolume, s_libraryName)


// TriggerVolumeHit hknpCharacterProxy
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpCharacterProxy, TriggerVolumeHit, s_libraryName)


// ShapeInfo hknpCharacterProxy

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterProxy::ShapeInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterProxy::ShapeInfo)
    HK_TRACKER_MEMBER(hknpCharacterProxy::ShapeInfo, m_body, 0, "hknpBody*") // const class hknpBody*
    HK_TRACKER_MEMBER(hknpCharacterProxy::ShapeInfo, m_shape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpCharacterProxy::ShapeInfo, m_transform, 0, "hkTransform*") // const hkTransform*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpCharacterProxy::ShapeInfo, s_libraryName)

#include <Physics/Physics/Extensions/CharacterControl/Proxy/hknpCharacterProxyCinfo.h>


// hknpCharacterProxyCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterProxyCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterProxyCinfo)
    HK_TRACKER_MEMBER(hknpCharacterProxyCinfo, m_shape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpCharacterProxyCinfo, m_world, 0, "hknpWorld*") // class hknpWorld*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCharacterProxyCinfo, s_libraryName, hkReferencedObject)

#include <Physics/Physics/Extensions/CharacterControl/Proxy/hknpCharacterProxyListener.h>


// hknpCharacterObjectInteractionEvent ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCharacterObjectInteractionEvent, s_libraryName)


// hknpCharacterObjectInteractionResult ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCharacterObjectInteractionResult, s_libraryName)


// hknpCharacterProxyListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterProxyListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterProxyListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCharacterProxyListener, s_libraryName, hkReferencedObject)

#include <Physics/Physics/Extensions/CharacterControl/RigidBody/hknpCharacterRigidBody.h>


// hknpCharacterRigidBody ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterRigidBody)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SupportInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ContactPointInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ContactType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterRigidBody)
    HK_TRACKER_MEMBER(hknpCharacterRigidBody, m_shape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpCharacterRigidBody, m_world, 0, "hknpWorld*") // class hknpWorld*
    HK_TRACKER_MEMBER(hknpCharacterRigidBody, m_listeners, 0, "hkArray<hknpCharacterRigidBodyListener*, hkContainerHeapAllocator>") // hkArray< class hknpCharacterRigidBodyListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpCharacterRigidBody, m_filteredContactPoints, 0, "hkArray<hknpCharacterRigidBody::ContactPointInfo, hkContainerHeapAllocator>") // hkArray< struct hknpCharacterRigidBody::ContactPointInfo, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCharacterRigidBody, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpCharacterRigidBody, ContactType, s_libraryName)


// SupportInfo hknpCharacterRigidBody
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpCharacterRigidBody, SupportInfo, s_libraryName)


// ContactPointInfo hknpCharacterRigidBody
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpCharacterRigidBody, ContactPointInfo, s_libraryName)

#include <Physics/Physics/Extensions/CharacterControl/RigidBody/hknpCharacterRigidBodyCinfo.h>


// hknpCharacterRigidBodyCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterRigidBodyCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterRigidBodyCinfo)
    HK_TRACKER_MEMBER(hknpCharacterRigidBodyCinfo, m_shape, 0, "hknpShape*") // const class hknpShape*
    HK_TRACKER_MEMBER(hknpCharacterRigidBodyCinfo, m_world, 0, "hknpWorld*") // class hknpWorld*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCharacterRigidBodyCinfo, s_libraryName, hkReferencedObject)

#include <Physics/Physics/Extensions/CharacterControl/RigidBody/hknpCharacterRigidBodyListener.h>


// hknpCharacterRigidBodyListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterRigidBodyListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterRigidBodyListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCharacterRigidBodyListener, s_libraryName, hkReferencedObject)

#include <Physics/Physics/Extensions/CharacterControl/StateMachine/States/hknpCharacterStateClimbing.h>


// hknpCharacterStateClimbing ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterStateClimbing)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterStateClimbing)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCharacterStateClimbing, s_libraryName, hknpCharacterState)

#include <Physics/Physics/Extensions/CharacterControl/StateMachine/States/hknpCharacterStateFlying.h>


// hknpCharacterStateFlying ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterStateFlying)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterStateFlying)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCharacterStateFlying, s_libraryName, hknpCharacterState)

#include <Physics/Physics/Extensions/CharacterControl/StateMachine/States/hknpCharacterStateInAir.h>


// hknpCharacterStateInAir ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterStateInAir)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterStateInAir)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCharacterStateInAir, s_libraryName, hknpCharacterState)

#include <Physics/Physics/Extensions/CharacterControl/StateMachine/States/hknpCharacterStateJumping.h>


// hknpCharacterStateJumping ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterStateJumping)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterStateJumping)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCharacterStateJumping, s_libraryName, hknpCharacterState)

#include <Physics/Physics/Extensions/CharacterControl/StateMachine/States/hknpCharacterStateOnGround.h>


// hknpCharacterStateOnGround ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterStateOnGround)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterStateOnGround)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCharacterStateOnGround, s_libraryName, hknpCharacterState)

#include <Physics/Physics/Extensions/CharacterControl/StateMachine/Utils/hknpCharacterMovementUtil.h>


// hknpCharacterMovementUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterMovementUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hknpMovementUtilInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hknpCharacterMovementUtil, s_libraryName)


// hknpMovementUtilInput hknpCharacterMovementUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpCharacterMovementUtil, hknpMovementUtilInput, s_libraryName)

#include <Physics/Physics/Extensions/CharacterControl/StateMachine/hknpCharacterContext.h>


// hknpCharacterInput ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCharacterInput, s_libraryName)


// hknpCharacterOutput ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCharacterOutput, s_libraryName)


// hknpCharacterContext ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterContext)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CharacterType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterContext)
    HK_TRACKER_MEMBER(hknpCharacterContext, m_stateManager, 0, "hknpCharacterStateManager*") // const class hknpCharacterStateManager*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCharacterContext, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpCharacterContext, CharacterType, s_libraryName)

#include <Physics/Physics/Extensions/CharacterControl/StateMachine/hknpCharacterState.h>


// hknpCharacterState ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterState)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hknpCharacterStateType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterState)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpCharacterState, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpCharacterState, hknpCharacterStateType, s_libraryName)

#include <Physics/Physics/Extensions/CharacterControl/StateMachine/hknpCharacterStateManager.h>


// hknpCharacterStateManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpCharacterStateManager)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpCharacterStateManager)
    HK_TRACKER_MEMBER(hknpCharacterStateManager, m_states, 0, "hknpCharacterState* [11]") // class hknpCharacterState* [11]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpCharacterStateManager, s_libraryName, hkReferencedObject)

#include <Physics/Physics/Extensions/CharacterControl/hknpCharacterSurfaceInfo.h>


// hknpCharacterSurfaceInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCharacterSurfaceInfo, s_libraryName)

#include <Physics/Physics/Extensions/Destruction/BreakOffModifier/hknpDestructionBreakOffModifier.h>


// hknpDestructionBreakOffModifier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDestructionBreakOffModifier)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Manifold)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ContactEvent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDestructionBreakOffModifier)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpDestructionBreakOffModifier, s_libraryName, hknpModifier)


// Manifold hknpDestructionBreakOffModifier

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDestructionBreakOffModifier::Manifold)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDestructionBreakOffModifier::Manifold)
    HK_TRACKER_MEMBER(hknpDestructionBreakOffModifier::Manifold, m_materialA, 0, "hknpMaterial*") // const class hknpMaterial*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpDestructionBreakOffModifier::Manifold, s_libraryName, hknpPaddedManifold0)


// ContactEvent hknpDestructionBreakOffModifier

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDestructionBreakOffModifier::ContactEvent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDestructionBreakOffModifier::ContactEvent)
    HK_TRACKER_MEMBER(hknpDestructionBreakOffModifier::ContactEvent, m_manifold, 0, "hknpDestructionBreakOffModifier::Manifold") // struct hknpDestructionBreakOffModifier::Manifold
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpDestructionBreakOffModifier::ContactEvent, s_libraryName, hkCommand)

#include <Physics/Physics/Extensions/Destruction/Serialization/hknpDestructionShapeProperties.h>


// hknpDestructionShapeProperties ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpDestructionShapeProperties)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpDestructionShapeProperties)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpDestructionShapeProperties, s_libraryName, hkReferencedObject)

#include <Physics/Physics/Extensions/LeafShapeSensor/hknpLeafShapeSensor.h>


// hknpLeafShapeSensor ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpLeafShapeSensor)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LeafShapeId)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Context)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MapOpsBodyId)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CallbackType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpLeafShapeSensor)
    HK_TRACKER_MEMBER(hknpLeafShapeSensor, m_world, 0, "hknpWorld*") // class hknpWorld*
    HK_TRACKER_MEMBER(hknpLeafShapeSensor, m_bodyContexts, 0, "hkMap<hknpBodyId, hknpLeafShapeSensor::Data<hknpLeafShapeSensor::Context*>, hknpLeafShapeSensor::MapOpsBodyId, hkContainerHeapAllocator>") // class hkMap< struct hknpBodyId, struct hknpLeafShapeSensor::Data< struct hknpLeafShapeSensor::Context* >, struct hknpLeafShapeSensor::MapOpsBodyId, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpLeafShapeSensor, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpLeafShapeSensor, CallbackType, s_libraryName)


// LeafShapeId hknpLeafShapeSensor
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpLeafShapeSensor, LeafShapeId, s_libraryName)


// Context hknpLeafShapeSensor

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpLeafShapeSensor::Context)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpLeafShapeSensor::Context)
    HK_TRACKER_MEMBER(hknpLeafShapeSensor::Context, m_shapeKeys, 0, "hkMap<hkUint32, hknpLeafShapeSensor::Data<hkUlong>, hkMapOperations<hkUint32>, hkContainerHeapAllocator>*") // class hkMap< hkUint32, struct hknpLeafShapeSensor::Data< hkUlong >, struct hkMapOperations< hkUint32 >, struct hkContainerHeapAllocator >*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpLeafShapeSensor::Context, s_libraryName)


// MapOpsBodyId hknpLeafShapeSensor
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpLeafShapeSensor, MapOpsBodyId, s_libraryName)

#include <Physics/Physics/Extensions/PhysicsSystem/hknpPhysicsSceneData.h>


// hknpRefWorldCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpRefWorldCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpRefWorldCinfo)
    HK_TRACKER_MEMBER(hknpRefWorldCinfo, m_info, 0, "hknpWorldCinfo") // struct hknpWorldCinfo
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpRefWorldCinfo, s_libraryName, hkReferencedObject)


// hknpPhysicsSceneData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpPhysicsSceneData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpPhysicsSceneData)
    HK_TRACKER_MEMBER(hknpPhysicsSceneData, m_systemDatas, 0, "hkArray<hknpPhysicsSystemData *, hkContainerHeapAllocator>") // hkArray< class hknpPhysicsSystemData *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpPhysicsSceneData, m_worldCinfo, 0, "hknpRefWorldCinfo*") // class hknpRefWorldCinfo*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpPhysicsSceneData, s_libraryName, hkReferencedObject)

#include <Physics/Physics/Extensions/PhysicsSystem/hknpPhysicsSystem.h>


// hknpPhysicsSystemData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpPhysicsSystemData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpPhysicsSystemData)
    HK_TRACKER_MEMBER(hknpPhysicsSystemData, m_materials, 0, "hkArray<hknpMaterial, hkContainerHeapAllocator>") // hkArray< class hknpMaterial, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpPhysicsSystemData, m_motionProperties, 0, "hkArray<hknpMotionProperties, hkContainerHeapAllocator>") // hkArray< class hknpMotionProperties, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpPhysicsSystemData, m_motionCinfos, 0, "hkArray<hknpMotionCinfo, hkContainerHeapAllocator>") // hkArray< struct hknpMotionCinfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpPhysicsSystemData, m_bodyCinfos, 0, "hkArray<hknpBodyCinfo, hkContainerHeapAllocator>") // hkArray< struct hknpBodyCinfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpPhysicsSystemData, m_constraintCinfos, 0, "hkArray<hknpConstraintCinfo, hkContainerHeapAllocator>") // hkArray< class hknpConstraintCinfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpPhysicsSystemData, m_referencedObjects, 0, "hkArray<hkReferencedObject *, hkContainerHeapAllocator>") // hkArray< const class hkReferencedObject *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpPhysicsSystemData, m_name, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpPhysicsSystemData, s_libraryName, hkReferencedObject)


// hknpPhysicsSystem ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpPhysicsSystem)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Flags)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpPhysicsSystem)
    HK_TRACKER_MEMBER(hknpPhysicsSystem, m_data, 0, "hknpPhysicsSystemData *") // const class hknpPhysicsSystemData *
    HK_TRACKER_MEMBER(hknpPhysicsSystem, m_world, 0, "hknpWorld *") // class hknpWorld *
    HK_TRACKER_MEMBER(hknpPhysicsSystem, m_bodyIds, 0, "hkArray<hknpBodyId, hkContainerHeapAllocator>") // hkArray< struct hknpBodyId, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpPhysicsSystem, m_constraints, 0, "hkArray<hknpConstraint*, hkContainerHeapAllocator>") // hkArray< class hknpConstraint*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpPhysicsSystem, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpPhysicsSystem, Flags, s_libraryName)

#include <Physics/Physics/Extensions/PhysicsSystem/hknpPhysicsSystemDataUtil.h>


// hknpPhysicsSystemDataUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpPhysicsSystemDataUtil, s_libraryName)

#include <Physics/Physics/Extensions/PhysicsSystem/hknpPhysicsToSceneDataBridge.h>


// hknpPhysicsToSceneDataBridge ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpPhysicsToSceneDataBridge)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(RootLevelContainer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpPhysicsToSceneDataBridge)
    HK_TRACKER_MEMBER(hknpPhysicsToSceneDataBridge, m_sceneDataContext, 0, "hkxSceneDataContext *") // class hkxSceneDataContext *
    HK_TRACKER_MEMBER(hknpPhysicsToSceneDataBridge, m_physicsWorld, 0, "hknpWorld *") // class hknpWorld *
    HK_TRACKER_MEMBER(hknpPhysicsToSceneDataBridge, m_debugger, 0, "hkVisualDebugger *") // class hkVisualDebugger *
    HK_TRACKER_MEMBER(hknpPhysicsToSceneDataBridge, m_loadedContainers, 0, "hkArray<hknpPhysicsToSceneDataBridge::RootLevelContainer, hkContainerHeapAllocator>") // hkArray< struct hknpPhysicsToSceneDataBridge::RootLevelContainer, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpPhysicsToSceneDataBridge, m_rigidBodyNameToIdMap, 0, "hkStringMap<hkUint32, hkContainerHeapAllocator>") // class hkStringMap< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpPhysicsToSceneDataBridge, m_rigidBodyToMeshIdMap, 0, "hkPointerMap<hkUint32, hkUlong, hkContainerHeapAllocator>") // class hkPointerMap< hkUint32, hkUlong, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpPhysicsToSceneDataBridge, m_rigidBodyToTransformMap, 0, "hkPointerMap<hkUint32, hkTransformf*, hkContainerHeapAllocator>") // class hkPointerMap< hkUint32, hkTransformf*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpPhysicsToSceneDataBridge, m_meshIdToScaleAndSkewTransformMap, 0, "hkPointerMap<hkUlong, hkMatrix4f*, hkContainerHeapAllocator>") // class hkPointerMap< hkUlong, hkMatrix4f*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpPhysicsToSceneDataBridge, s_libraryName, hkReferencedObject)


// RootLevelContainer hknpPhysicsToSceneDataBridge

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpPhysicsToSceneDataBridge::RootLevelContainer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpPhysicsToSceneDataBridge::RootLevelContainer)
    HK_TRACKER_MEMBER(hknpPhysicsToSceneDataBridge::RootLevelContainer, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hknpPhysicsToSceneDataBridge::RootLevelContainer, m_container, 0, "hkRootLevelContainer") // class hkRootLevelContainer
    HK_TRACKER_MEMBER(hknpPhysicsToSceneDataBridge::RootLevelContainer, m_systems, 0, "hkArray<hknpPhysicsSystem*, hkContainerHeapAllocator>") // hkArray< class hknpPhysicsSystem*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpPhysicsToSceneDataBridge::RootLevelContainer, s_libraryName)

#include <Physics/Physics/Extensions/ShapeProcessing/ShapeScaling/hknpShapeScalingUtil.h>


// hknpShapeScalingUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeScalingUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ShapePair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hknpShapeScalingUtil, s_libraryName)


// ShapePair hknpShapeScalingUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeScalingUtil::ShapePair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShapeScalingUtil::ShapePair)
    HK_TRACKER_MEMBER(hknpShapeScalingUtil::ShapePair, m_originalShape, 0, "hknpShape *") // const class hknpShape *
    HK_TRACKER_MEMBER(hknpShapeScalingUtil::ShapePair, m_newShape, 0, "hknpShape *") // class hknpShape *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpShapeScalingUtil::ShapePair, s_libraryName)

#include <Physics/Physics/Extensions/ShapeProcessing/ShapeSkinning/hknpShapeSkinningUtil.h>


// hknpShapeSkinningUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeSkinningUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Input)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hknpShapeSkinningUtil, s_libraryName)


// Input hknpShapeSkinningUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpShapeSkinningUtil::Input)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpShapeSkinningUtil::Input)
    HK_TRACKER_MEMBER(hknpShapeSkinningUtil::Input, m_shapes, 0, "restrict hknpShape**") // restrict const const class hknpShape**
    HK_TRACKER_MEMBER(hknpShapeSkinningUtil::Input, m_transforms, 0, "restrict hkTransform*") // restrict const hkTransform*
    HK_TRACKER_MEMBER(hknpShapeSkinningUtil::Input, m_vertexPositions, 0, "restrict hkVector4*") // restrict hkVector4*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpShapeSkinningUtil::Input, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/AeroDynamics/Default/hknpVehicleDefaultAerodynamics.h>


// hknpVehicleDefaultAerodynamics ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleDefaultAerodynamics)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleDefaultAerodynamics)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpVehicleDefaultAerodynamics, s_libraryName, hknpVehicleAerodynamics)

#include <Physics/Physics/Extensions/Vehicle/AeroDynamics/hknpVehicleAerodynamics.h>


// hknpVehicleAerodynamics ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleAerodynamics)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AerodynamicsDragOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleAerodynamics)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpVehicleAerodynamics, s_libraryName, hkReferencedObject)


// AerodynamicsDragOutput hknpVehicleAerodynamics
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpVehicleAerodynamics, AerodynamicsDragOutput, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/Brake/Default/hknpVehicleDefaultBrake.h>


// hknpVehicleDefaultBrake ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleDefaultBrake)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WheelBrakingProperties)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleDefaultBrake)
    HK_TRACKER_MEMBER(hknpVehicleDefaultBrake, m_wheelBrakingProperties, 0, "hkArray<hknpVehicleDefaultBrake::WheelBrakingProperties, hkContainerHeapAllocator>") // hkArray< struct hknpVehicleDefaultBrake::WheelBrakingProperties, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpVehicleDefaultBrake, s_libraryName, hknpVehicleBrake)


// WheelBrakingProperties hknpVehicleDefaultBrake
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpVehicleDefaultBrake, WheelBrakingProperties, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/Brake/hknpVehicleBrake.h>


// hknpVehicleBrake ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleBrake)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WheelBreakingOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleBrake)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpVehicleBrake, s_libraryName, hkReferencedObject)


// WheelBreakingOutput hknpVehicleBrake

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleBrake::WheelBreakingOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleBrake::WheelBreakingOutput)
    HK_TRACKER_MEMBER(hknpVehicleBrake::WheelBreakingOutput, m_brakingTorque, 0, "hkInplaceArray<float, 32, hkContainerHeapAllocator>") // class hkInplaceArray< float, 32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpVehicleBrake::WheelBreakingOutput, m_isFixed, 0, "hkInplaceArray<hkBool, 32, hkContainerHeapAllocator>") // class hkInplaceArray< hkBool, 32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpVehicleBrake::WheelBreakingOutput, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/Camera/hknp1dAngularFollowCam.h>


// hknp1dAngularFollowCam ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknp1dAngularFollowCam)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CameraInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CameraOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknp1dAngularFollowCam)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknp1dAngularFollowCam, s_libraryName, hkReferencedObject)


// CameraInput hknp1dAngularFollowCam
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknp1dAngularFollowCam, CameraInput, s_libraryName)


// CameraOutput hknp1dAngularFollowCam
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknp1dAngularFollowCam, CameraOutput, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/Camera/hknp1dAngularFollowCamCinfo.h>


// hknp1dAngularFollowCamCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknp1dAngularFollowCamCinfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CameraSet)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hknp1dAngularFollowCamCinfo, s_libraryName)


// CameraSet hknp1dAngularFollowCamCinfo
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknp1dAngularFollowCamCinfo, CameraSet, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/DriverInput/Default/hknpVehicleDefaultAnalogDriverInput.h>


// hknpVehicleDriverInputAnalogStatus ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleDriverInputAnalogStatus)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleDriverInputAnalogStatus)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpVehicleDriverInputAnalogStatus, s_libraryName, hknpVehicleDriverInputStatus)


// hknpVehicleDefaultAnalogDriverInput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleDefaultAnalogDriverInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleDefaultAnalogDriverInput)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpVehicleDefaultAnalogDriverInput, s_libraryName, hknpVehicleDriverInput)

#include <Physics/Physics/Extensions/Vehicle/DriverInput/hknpVehicleDriverInput.h>


// hknpVehicleDriverInputStatus ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleDriverInputStatus)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleDriverInputStatus)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpVehicleDriverInputStatus, s_libraryName, hkReferencedObject)


// hknpVehicleDriverInput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleDriverInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FilteredDriverInputOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleDriverInput)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpVehicleDriverInput, s_libraryName, hkReferencedObject)


// FilteredDriverInputOutput hknpVehicleDriverInput
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpVehicleDriverInput, FilteredDriverInputOutput, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/Engine/Default/hknpVehicleDefaultEngine.h>


// hknpVehicleDefaultEngine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleDefaultEngine)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleDefaultEngine)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpVehicleDefaultEngine, s_libraryName, hknpVehicleEngine)

#include <Physics/Physics/Extensions/Vehicle/Engine/hknpVehicleEngine.h>


// hknpVehicleEngine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleEngine)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(EngineOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleEngine)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpVehicleEngine, s_libraryName, hkReferencedObject)


// EngineOutput hknpVehicleEngine
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpVehicleEngine, EngineOutput, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/Friction/hknpVehicleFriction.h>


// hknpVehicleStepInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpVehicleStepInfo, s_libraryName)


// hknpVehicleFrictionDescription ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleFrictionDescription)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Cinfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AxisDescription)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hknpVehicleFrictionDescription, s_libraryName)


// Cinfo hknpVehicleFrictionDescription
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpVehicleFrictionDescription, Cinfo, s_libraryName)


// AxisDescription hknpVehicleFrictionDescription
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpVehicleFrictionDescription, AxisDescription, s_libraryName)


// hknpVehicleFrictionStatus ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleFrictionStatus)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AxisStatus)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hknpVehicleFrictionStatus, s_libraryName)


// AxisStatus hknpVehicleFrictionStatus
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpVehicleFrictionStatus, AxisStatus, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/Steering/Default/hknpVehicleDefaultSteering.h>


// hknpVehicleDefaultSteering ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleDefaultSteering)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleDefaultSteering)
    HK_TRACKER_MEMBER(hknpVehicleDefaultSteering, m_doesWheelSteer, 0, "hkArray<hkBool, hkContainerHeapAllocator>") // hkArray< hkBool, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpVehicleDefaultSteering, s_libraryName, hknpVehicleSteering)

#include <Physics/Physics/Extensions/Vehicle/Steering/hknpVehicleSteering.h>


// hknpVehicleSteering ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleSteering)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SteeringAnglesOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleSteering)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpVehicleSteering, s_libraryName, hkReferencedObject)


// SteeringAnglesOutput hknpVehicleSteering

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleSteering::SteeringAnglesOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleSteering::SteeringAnglesOutput)
    HK_TRACKER_MEMBER(hknpVehicleSteering::SteeringAnglesOutput, m_wheelsSteeringAngle, 0, "hkInplaceArray<float, 32, hkContainerHeapAllocator>") // class hkInplaceArray< float, 32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpVehicleSteering::SteeringAnglesOutput, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/Suspension/Default/hknpVehicleDefaultSuspension.h>


// hknpVehicleDefaultSuspension ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleDefaultSuspension)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WheelSpringSuspensionParameters)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleDefaultSuspension)
    HK_TRACKER_MEMBER(hknpVehicleDefaultSuspension, m_wheelSpringParams, 0, "hkArray<hknpVehicleDefaultSuspension::WheelSpringSuspensionParameters, hkContainerHeapAllocator>") // hkArray< struct hknpVehicleDefaultSuspension::WheelSpringSuspensionParameters, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpVehicleDefaultSuspension, s_libraryName, hknpVehicleSuspension)


// WheelSpringSuspensionParameters hknpVehicleDefaultSuspension
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpVehicleDefaultSuspension, WheelSpringSuspensionParameters, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/Suspension/hknpVehicleSuspension.h>


// hknpVehicleSuspension ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleSuspension)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SuspensionWheelParameters)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleSuspension)
    HK_TRACKER_MEMBER(hknpVehicleSuspension, m_wheelParams, 0, "hkArray<hknpVehicleSuspension::SuspensionWheelParameters, hkContainerHeapAllocator>") // hkArray< struct hknpVehicleSuspension::SuspensionWheelParameters, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpVehicleSuspension, s_libraryName, hkReferencedObject)


// SuspensionWheelParameters hknpVehicleSuspension
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpVehicleSuspension, SuspensionWheelParameters, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/Transmission/Default/hknpVehicleDefaultTransmission.h>


// hknpVehicleDefaultTransmission ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleDefaultTransmission)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleDefaultTransmission)
    HK_TRACKER_MEMBER(hknpVehicleDefaultTransmission, m_gearsRatio, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpVehicleDefaultTransmission, m_wheelsTorqueRatio, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpVehicleDefaultTransmission, s_libraryName, hknpVehicleTransmission)

#include <Physics/Physics/Extensions/Vehicle/Transmission/hknpVehicleTransmission.h>


// hknpVehicleTransmission ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleTransmission)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TransmissionOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleTransmission)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpVehicleTransmission, s_libraryName, hkReferencedObject)


// TransmissionOutput hknpVehicleTransmission

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleTransmission::TransmissionOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleTransmission::TransmissionOutput)
    HK_TRACKER_MEMBER(hknpVehicleTransmission::TransmissionOutput, m_wheelsTransmittedTorque, 0, "float*") // float*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpVehicleTransmission::TransmissionOutput, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/TyreMarks/hknpTyremarksInfo.h>


// hknpTyremarkPoint ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpTyremarkPoint, s_libraryName)


// hknpTyremarksWheel ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpTyremarksWheel)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpTyremarksWheel)
    HK_TRACKER_MEMBER(hknpTyremarksWheel, m_tyremarkPoints, 0, "hkArray<hknpTyremarkPoint, hkContainerHeapAllocator>") // hkArray< struct hknpTyremarkPoint, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpTyremarksWheel, s_libraryName, hkReferencedObject)


// hknpTyremarksInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpTyremarksInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpTyremarksInfo)
    HK_TRACKER_MEMBER(hknpTyremarksInfo, m_tyremarksWheel, 0, "hkArray<hknpTyremarksWheel*, hkContainerHeapAllocator>") // hkArray< class hknpTyremarksWheel*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpTyremarksInfo, s_libraryName, hkReferencedObject)

#include <Physics/Physics/Extensions/Vehicle/VelocityDamper/Default/hknpVehicleDefaultVelocityDamper.h>


// hknpVehicleDefaultVelocityDamper ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleDefaultVelocityDamper)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleDefaultVelocityDamper)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpVehicleDefaultVelocityDamper, s_libraryName, hknpVehicleVelocityDamper)

#include <Physics/Physics/Extensions/Vehicle/VelocityDamper/hknpVehicleVelocityDamper.h>


// hknpVehicleVelocityDamper ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleVelocityDamper)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleVelocityDamper)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpVehicleVelocityDamper, s_libraryName, hkReferencedObject)

#include <Physics/Physics/Extensions/Vehicle/WheelCollide/LinearCast/hknpVehicleLinearCastWheelCollide.h>


// hknpVehicleLinearCastWheelCollide ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleLinearCastWheelCollide)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WheelState)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleLinearCastWheelCollide)
    HK_TRACKER_MEMBER(hknpVehicleLinearCastWheelCollide, m_wheelStates, 0, "hkArray<hknpVehicleLinearCastWheelCollide::WheelState, hkContainerHeapAllocator>") // hkArray< struct hknpVehicleLinearCastWheelCollide::WheelState, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpVehicleLinearCastWheelCollide, s_libraryName, hknpVehicleWheelCollide)


// WheelState hknpVehicleLinearCastWheelCollide

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleLinearCastWheelCollide::WheelState)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleLinearCastWheelCollide::WheelState)
    HK_TRACKER_MEMBER(hknpVehicleLinearCastWheelCollide::WheelState, m_shape, 0, "hknpShape*") // const class hknpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpVehicleLinearCastWheelCollide::WheelState, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/WheelCollide/RayCast/hknpVehicleRayCastWheelCollide.h>


// hknpVehicleRayCastWheelCollide ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleRayCastWheelCollide)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleRayCastWheelCollide)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpVehicleRayCastWheelCollide, s_libraryName, hknpVehicleWheelCollide)

#include <Physics/Physics/Extensions/Vehicle/WheelCollide/hknpVehicleWheelCollide.h>


// hknpVehicleWheelCollide ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleWheelCollide)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CollisionDetectionWheelOutput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WheelCollideType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleWheelCollide)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hknpVehicleWheelCollide, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpVehicleWheelCollide, WheelCollideType, s_libraryName)


// CollisionDetectionWheelOutput hknpVehicleWheelCollide
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpVehicleWheelCollide, CollisionDetectionWheelOutput, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/hknpVehicleData.h>


// hknpVehicleData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WheelComponentParams)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleData)
    HK_TRACKER_MEMBER(hknpVehicleData, m_wheelParams, 0, "hkArray<hknpVehicleData::WheelComponentParams, hkContainerHeapAllocator>") // hkArray< struct hknpVehicleData::WheelComponentParams, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpVehicleData, m_numWheelsPerAxle, 0, "hkArray<hkInt8, hkContainerHeapAllocator>") // hkArray< hkInt8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpVehicleData, s_libraryName, hkReferencedObject)


// WheelComponentParams hknpVehicleData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpVehicleData, WheelComponentParams, s_libraryName)

#include <Physics/Physics/Extensions/Vehicle/hknpVehicleInstance.h>


// hknpVehicleJobResults ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpVehicleJobResults, s_libraryName)


// hknpVehicleInstance ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpVehicleInstance)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WheelInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpVehicleInstance)
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_data, 0, "hknpVehicleData*") // class hknpVehicleData*
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_driverInput, 0, "hknpVehicleDriverInput*") // class hknpVehicleDriverInput*
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_steering, 0, "hknpVehicleSteering*") // class hknpVehicleSteering*
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_engine, 0, "hknpVehicleEngine*") // class hknpVehicleEngine*
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_transmission, 0, "hknpVehicleTransmission*") // class hknpVehicleTransmission*
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_brake, 0, "hknpVehicleBrake*") // class hknpVehicleBrake*
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_suspension, 0, "hknpVehicleSuspension*") // class hknpVehicleSuspension*
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_aerodynamics, 0, "hknpVehicleAerodynamics*") // class hknpVehicleAerodynamics*
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_wheelCollide, 0, "hknpVehicleWheelCollide*") // class hknpVehicleWheelCollide*
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_tyreMarks, 0, "hknpTyremarksInfo*") // class hknpTyremarksInfo*
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_velocityDamper, 0, "hknpVehicleVelocityDamper*") // class hknpVehicleVelocityDamper*
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_wheelsInfo, 0, "hkArray<hknpVehicleInstance::WheelInfo, hkContainerHeapAllocator>") // hkArray< struct hknpVehicleInstance::WheelInfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_deviceStatus, 0, "hknpVehicleDriverInputStatus*") // class hknpVehicleDriverInputStatus*
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_isFixed, 0, "hkArray<hkBool, hkContainerHeapAllocator>") // hkArray< hkBool, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_wheelsSteeringAngle, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpVehicleInstance, m_world, 0, "hknpWorld*") // const class hknpWorld*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpVehicleInstance, s_libraryName, hknpUnaryAction)


// WheelInfo hknpVehicleInstance
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpVehicleInstance, WheelInfo, s_libraryName)

#include <Physics/Physics/Extensions/WorldSnapshot/hknpWorldSnapshot.h>


// hknpWorldSnapshot ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpWorldSnapshot)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpWorldSnapshot)
    HK_TRACKER_MEMBER(hknpWorldSnapshot, m_worldCinfo, 0, "hknpWorldCinfo") // struct hknpWorldCinfo
    HK_TRACKER_MEMBER(hknpWorldSnapshot, m_bodies, 0, "hkArray<hknpBody, hkContainerHeapAllocator>") // hkArray< class hknpBody, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpWorldSnapshot, m_bodyNames, 0, "hkArray<hkStringPtr, hkContainerHeapAllocator>") // hkArray< hkStringPtr, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpWorldSnapshot, m_motions, 0, "hkArray<hknpMotion, hkContainerHeapAllocator>") // hkArray< class hknpMotion, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hknpWorldSnapshot, m_constraints, 0, "hkArray<hknpConstraintCinfo, hkContainerHeapAllocator>") // hkArray< class hknpConstraintCinfo, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hknpWorldSnapshot, s_libraryName, hkReferencedObject)

#include <Physics/Physics/hknpTypes.h>


// hknpCdCacheRange ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCdCacheRange, s_libraryName)


// hknpShapeKeyPath ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpShapeKeyPath, s_libraryName)


// hknpImmediateConstraintIdDiscriminant ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpImmediateConstraintIdDiscriminant, s_libraryName)


// hknpBodyPropertyKeys ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpBodyPropertyKeys, s_libraryName)


// hknpShapePropertyKeys ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpShapePropertyKeys, s_libraryName)


// hknpSolverIdDiscriminant ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpSolverIdDiscriminant, s_libraryName)


// hknpIslandIdDiscriminant ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpIslandIdDiscriminant, s_libraryName)


// hknpActivationMode ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpActivationMode, s_libraryName)


// hknpActivationBehavior ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpActivationBehavior, s_libraryName)


// hknpCollisionCacheType ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCollisionCacheType, s_libraryName)


// hknpCdCacheDestructReason ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCdCacheDestructReason, s_libraryName)


// hknpJacobianGridType ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpJacobianGridType, s_libraryName)


// hknpConstraintSolverType ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpConstraintSolverType, s_libraryName)


// hknpCommandDispatchType ::
HK_TRACKER_IMPLEMENT_SIMPLE(hknpCommandDispatchType, s_libraryName)


// hknpManifoldSolverInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hknpManifoldSolverInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Flags)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hknpManifoldSolverInfo)
    HK_TRACKER_MEMBER(hknpManifoldSolverInfo, m_contactJacobian, 0, "hknpContactJacobian<1>*") // struct hknpContactJacobian< 1 >*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hknpManifoldSolverInfo, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hknpManifoldSolverInfo, Flags, s_libraryName)

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
