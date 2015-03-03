/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Physics2012/Collide/hkpCollide.h>
static const char s_libraryName[] = "hkpCollide";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpCollideRegister() {}

#include <Physics2012/Collide/Agent/Collidable/hkpCdBody.h>


// hkpCdBody ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCdBody)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCdBody)
    HK_TRACKER_MEMBER(hkpCdBody, m_shape, 0, "hkpShape*") // const class hkpShape*
    HK_TRACKER_MEMBER(hkpCdBody, m_motion, 0, "void*") // const void*
    HK_TRACKER_MEMBER(hkpCdBody, m_parent, 0, "hkpCdBody*") // const class hkpCdBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCdBody, s_libraryName)

#include <Physics2012/Collide/Agent/Collidable/hkpCdPoint.h>


// hkpCdPoint ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCdPoint, s_libraryName)

#include <Physics2012/Collide/Agent/Collidable/hkpCollidable.h>


// hkpCollidable ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCollidable)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BoundingVolumeData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ForceCollideOntoPpuReasons)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCollidable)
    HK_TRACKER_MEMBER(hkpCollidable, m_boundingVolumeData, 0, "hkpCollidable::BoundingVolumeData") // struct hkpCollidable::BoundingVolumeData
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCollidable, s_libraryName, hkpCdBody)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollidable, ForceCollideOntoPpuReasons, s_libraryName)


// BoundingVolumeData hkpCollidable

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCollidable::BoundingVolumeData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCollidable::BoundingVolumeData)
    HK_TRACKER_MEMBER(hkpCollidable::BoundingVolumeData, m_childShapeAabbs, 0, "hkAabbUint32*") // struct hkAabbUint32*
    HK_TRACKER_MEMBER(hkpCollidable::BoundingVolumeData, m_childShapeKeys, 0, "hkUint32*") // hkUint32*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCollidable::BoundingVolumeData, s_libraryName)

#include <Physics2012/Collide/Agent/Collidable/hkpCollidableQualityType.h>

// None hkpCollidableQualityType
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCollidableQualityType, s_libraryName)
#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpBvCompressedMeshAgent.h>


// hkpBvCompressedMeshAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBvCompressedMeshAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBvCompressedMeshAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBvCompressedMeshAgent, s_libraryName, hkpBvTreeAgent)

#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpBvTreeAgent.h>


// hkpBvTreeAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBvTreeAgent)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpBvAgentEntryInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LinearCastAabbCastCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBvTreeAgent)
    HK_TRACKER_MEMBER(hkpBvTreeAgent, m_collisionPartners, 0, "hkArray<hkpBvTreeAgent::hkpBvAgentEntryInfo, hkContainerHeapAllocator>") // hkArray< struct hkpBvTreeAgent::hkpBvAgentEntryInfo, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBvTreeAgent, s_libraryName, hkpCollisionAgent)


// hkpBvAgentEntryInfo hkpBvTreeAgent

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBvTreeAgent::hkpBvAgentEntryInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBvTreeAgent::hkpBvAgentEntryInfo)
    HK_TRACKER_MEMBER(hkpBvTreeAgent::hkpBvAgentEntryInfo, m_collisionAgent, 0, "hkpCollisionAgent*") // class hkpCollisionAgent*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpBvTreeAgent::hkpBvAgentEntryInfo, s_libraryName)


// LinearCastAabbCastCollector hkpBvTreeAgent

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBvTreeAgent::LinearCastAabbCastCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBvTreeAgent::LinearCastAabbCastCollector)
    HK_TRACKER_MEMBER(hkpBvTreeAgent::LinearCastAabbCastCollector, m_startCollector, 0, "hkpCdPointCollector*") // class hkpCdPointCollector*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBvTreeAgent::LinearCastAabbCastCollector, s_libraryName, hkpAabbCastCollector)

#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpMoppAgent.h>


// hkpMoppAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppAgent, s_libraryName, hkpBvTreeAgent)

#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpStaticCompoundAgent.h>


// hkpStaticCompoundAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpStaticCompoundAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpStaticCompoundAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpStaticCompoundAgent, s_libraryName, hkpBvTreeAgent)

#include <Physics2012/Collide/Agent/CompoundAgent/BvTreeStream/hkpBvTreeStreamAgent.h>


// hkpBvTreeStreamAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBvTreeStreamAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBvTreeStreamAgent)
    HK_TRACKER_MEMBER(hkpBvTreeStreamAgent, m_dispatcher, 0, "hkpCollisionDispatcher*") // class hkpCollisionDispatcher*
    HK_TRACKER_MEMBER(hkpBvTreeStreamAgent, m_agentTrack, 0, "hkpAgent1nTrack") // struct hkpAgent1nTrack
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBvTreeStreamAgent, s_libraryName, hkpCollisionAgent)

#include <Physics2012/Collide/Agent/CompoundAgent/BvTreeStream/hkpMoppBvTreeStreamAgent.h>


// hkpMoppBvTreeStreamAgent ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppBvTreeStreamAgent, s_libraryName)

#include <Physics2012/Collide/Agent/CompoundAgent/List/hkpListAgent.h>


// hkpListAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpListAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpListAgent)
    HK_TRACKER_MEMBER(hkpListAgent, m_dispatcher, 0, "hkpCollisionDispatcher*") // class hkpCollisionDispatcher*
    HK_TRACKER_MEMBER(hkpListAgent, m_agentTrack, 0, "hkpAgent1nTrack") // struct hkpAgent1nTrack
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpListAgent, s_libraryName, hkpCollisionAgent)

#include <Physics2012/Collide/Agent/CompoundAgent/ShapeCollection/hkpShapeCollectionAgent.h>


// hkpShapeCollectionAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeCollectionAgent)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(KeyAgentPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeCollectionAgent)
    HK_TRACKER_MEMBER(hkpShapeCollectionAgent, m_agents, 0, "hkInplaceArray<hkpShapeCollectionAgent::KeyAgentPair, 4, hkContainerHeapAllocator>") // class hkInplaceArray< struct hkpShapeCollectionAgent::KeyAgentPair, 4, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpShapeCollectionAgent, s_libraryName, hkpCollisionAgent)


// KeyAgentPair hkpShapeCollectionAgent

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeCollectionAgent::KeyAgentPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeCollectionAgent::KeyAgentPair)
    HK_TRACKER_MEMBER(hkpShapeCollectionAgent::KeyAgentPair, m_agent, 0, "hkpCollisionAgent*") // class hkpCollisionAgent*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpShapeCollectionAgent::KeyAgentPair, s_libraryName)

#include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgr.h>


// hkpContactMgr ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpContactMgr)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Type)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ToiAccept)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpContactMgr)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpContactMgr, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpContactMgr, Type, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpContactMgr, ToiAccept, s_libraryName)

#include <Physics2012/Collide/Agent/ConvexAgent/BoxBox/hkpBoxBoxAgent.h>


// hkpBoxBoxAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBoxBoxAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBoxBoxAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBoxBoxAgent, s_libraryName, hkpIterativeLinearCastAgent)

#include <Physics2012/Collide/Agent/ConvexAgent/CapsuleCapsule/hkpCapsuleCapsuleAgent.h>


// hkpCapsuleCapsuleAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCapsuleCapsuleAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCapsuleCapsuleAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCapsuleCapsuleAgent, s_libraryName, hkpIterativeLinearCastAgent)

#include <Physics2012/Collide/Agent/ConvexAgent/CapsuleTriangle/hkpCapsuleTriangleAgent.h>


// hkpCapsuleTriangleAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCapsuleTriangleAgent)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClosestPointResult)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCapsuleTriangleAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCapsuleTriangleAgent, s_libraryName, hkpIterativeLinearCastAgent)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCapsuleTriangleAgent, ClosestPointResult, s_libraryName)

#include <Physics2012/Collide/Agent/ConvexAgent/Gjk/hkpClosestPointManifold.h>


// hkpClosestPointManifold ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpClosestPointManifold)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpAgentContactPoint)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpClosestPointManifold, s_libraryName)


// hkpAgentContactPoint hkpClosestPointManifold
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpClosestPointManifold, hkpAgentContactPoint, s_libraryName)

#include <Physics2012/Collide/Agent/ConvexAgent/Gjk/hkpGskBaseAgent.h>


// hkpGskBaseAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpGskBaseAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpGskBaseAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpGskBaseAgent, s_libraryName, hkpIterativeLinearCastAgent)

#include <Physics2012/Collide/Agent/ConvexAgent/Gjk/hkpGskConvexConvexAgent.h>


// hkpGskConvexConvexAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpGskConvexConvexAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpGskConvexConvexAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpGskConvexConvexAgent, s_libraryName, hkpGskBaseAgent)

#include <Physics2012/Collide/Agent/ConvexAgent/Gjk/hkpGskfAgent.h>


// hkpGskfAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpGskfAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpGskfAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpGskfAgent, s_libraryName, hkpGskBaseAgent)

#include <Physics2012/Collide/Agent/ConvexAgent/Gjk/hkpPredGskfAgent.h>


// hkpPredGskfAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPredGskfAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPredGskfAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPredGskfAgent, s_libraryName, hkpGskfAgent)

#include <Physics2012/Collide/Agent/ConvexAgent/SphereBox/hkpSphereBoxAgent.h>


// hkpSphereBoxAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSphereBoxAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSphereBoxAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSphereBoxAgent, s_libraryName, hkpIterativeLinearCastAgent)

#include <Physics2012/Collide/Agent/ConvexAgent/SphereCapsule/hkpSphereCapsuleAgent.h>


// hkpSphereCapsuleAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSphereCapsuleAgent)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClosestPointResult)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSphereCapsuleAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSphereCapsuleAgent, s_libraryName, hkpIterativeLinearCastAgent)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSphereCapsuleAgent, ClosestPointResult, s_libraryName)

#include <Physics2012/Collide/Agent/ConvexAgent/SphereSphere/hkpSphereSphereAgent.h>


// hkpSphereSphereAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSphereSphereAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSphereSphereAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSphereSphereAgent, s_libraryName, hkpIterativeLinearCastAgent)

#include <Physics2012/Collide/Agent/ConvexAgent/SphereTriangle/hkpSphereTriangleAgent.h>


// hkpSphereTriangleAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSphereTriangleAgent)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClosestPointResult)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSphereTriangleAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSphereTriangleAgent, s_libraryName, hkpIterativeLinearCastAgent)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSphereTriangleAgent, ClosestPointResult, s_libraryName)

#include <Physics2012/Collide/Agent/Deprecated/ConvexList/hkpConvexListAgent.h>


// hkpConvexListAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexListAgent)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(StreamData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexListAgent)
    HK_TRACKER_MEMBER(hkpConvexListAgent, m_dispatcher, 0, "hkpCollisionDispatcher*") // class hkpCollisionDispatcher*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConvexListAgent, s_libraryName, hkpPredGskfAgent)


// StreamData hkpConvexListAgent

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexListAgent::StreamData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexListAgent::StreamData)
    HK_TRACKER_MEMBER(hkpConvexListAgent::StreamData, m_agentTrack, 0, "hkpAgent1nTrack") // struct hkpAgent1nTrack
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpConvexListAgent::StreamData, s_libraryName)

#include <Physics2012/Collide/Agent/Deprecated/ConvexList/hkpConvexListUtils.h>


// hkpProcessCollisionOutputBackup ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpProcessCollisionOutputBackup)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpProcessCollisionOutputBackup)
    HK_TRACKER_MEMBER(hkpProcessCollisionOutputBackup, m_firstPoint, 0, "hkpProcessCdPoint*") // struct hkpProcessCdPoint*
    HK_TRACKER_MEMBER(hkpProcessCollisionOutputBackup, m_weldingInformation, 0, "hkpProcessCollisionOutput::PotentialInfo") // struct hkpProcessCollisionOutput::PotentialInfo
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpProcessCollisionOutputBackup, s_libraryName)


// hkpMapPointsToSubShapeContactMgr ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMapPointsToSubShapeContactMgr)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMapPointsToSubShapeContactMgr)
    HK_TRACKER_MEMBER(hkpMapPointsToSubShapeContactMgr, m_contactMgr, 0, "hkpContactMgr*") // class hkpContactMgr*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMapPointsToSubShapeContactMgr, s_libraryName, hkpContactMgr)

#include <Physics2012/Collide/Agent/Deprecated/MultiSphere/hkpMultiSphereAgent.h>


// hkpMultiSphereAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMultiSphereAgent)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(KeyAgentPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMultiSphereAgent)
    HK_TRACKER_MEMBER(hkpMultiSphereAgent, m_agents, 0, "hkInplaceArray<hkpMultiSphereAgent::KeyAgentPair, 4, hkContainerHeapAllocator>") // class hkInplaceArray< struct hkpMultiSphereAgent::KeyAgentPair, 4, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMultiSphereAgent, s_libraryName, hkpCollisionAgent)


// KeyAgentPair hkpMultiSphereAgent

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMultiSphereAgent::KeyAgentPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMultiSphereAgent::KeyAgentPair)
    HK_TRACKER_MEMBER(hkpMultiSphereAgent::KeyAgentPair, m_agent, 0, "hkpCollisionAgent*") // class hkpCollisionAgent*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpMultiSphereAgent::KeyAgentPair, s_libraryName)

#include <Physics2012/Collide/Agent/Deprecated/MultiSphereTriangle/hkpMultiSphereTriangleAgent.h>


// hkpMultiSphereTriangleAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMultiSphereTriangleAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMultiSphereTriangleAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMultiSphereTriangleAgent, s_libraryName, hkpIterativeLinearCastAgent)

#include <Physics2012/Collide/Agent/FullManifoldAgent/hkpFullManifoldAgent.h>


// hkpFullManifoldAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpFullManifoldAgent)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ContactPoint)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpFullManifoldAgent)
    HK_TRACKER_MEMBER(hkpFullManifoldAgent, m_contactPointIds, 0, "hkArray<hkpFullManifoldAgent::ContactPoint, hkContainerHeapAllocator>") // hkArray< struct hkpFullManifoldAgent::ContactPoint, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpFullManifoldAgent, s_libraryName, hkpCollisionAgent)


// ContactPoint hkpFullManifoldAgent
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpFullManifoldAgent, ContactPoint, s_libraryName)

#include <Physics2012/Collide/Agent/HeightFieldAgent/hkpHeightFieldAgent.h>


// hkpHeightFieldAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpHeightFieldAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpHeightFieldAgent)
    HK_TRACKER_MEMBER(hkpHeightFieldAgent, m_contactPointId, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpHeightFieldAgent, s_libraryName, hkpCollisionAgent)

#include <Physics2012/Collide/Agent/MiscAgent/Bv/hkpBvAgent.h>


// hkpBvAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBvAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBvAgent)
    HK_TRACKER_MEMBER(hkpBvAgent, m_boundingVolumeAgent, 0, "hkpCollisionAgent*") // class hkpCollisionAgent*
    HK_TRACKER_MEMBER(hkpBvAgent, m_childAgent, 0, "hkpCollisionAgent*") // class hkpCollisionAgent*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBvAgent, s_libraryName, hkpCollisionAgent)

#include <Physics2012/Collide/Agent/MiscAgent/MultirayConvex/hkpMultiRayConvexAgent.h>


// hkpMultiRayConvexAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMultiRayConvexAgent)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ContactPointInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMultiRayConvexAgent)
    HK_TRACKER_MEMBER(hkpMultiRayConvexAgent, m_contactInfo, 0, "hkInplaceArray<hkpMultiRayConvexAgent::ContactPointInfo, 4, hkContainerHeapAllocator>") // class hkInplaceArray< struct hkpMultiRayConvexAgent::ContactPointInfo, 4, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMultiRayConvexAgent, s_libraryName, hkpIterativeLinearCastAgent)


// ContactPointInfo hkpMultiRayConvexAgent
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMultiRayConvexAgent, ContactPointInfo, s_libraryName)

#include <Physics2012/Collide/Agent/MiscAgent/Phantom/hkpPhantomAgent.h>


// hkpPhantomAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPhantomAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPhantomAgent)
    HK_TRACKER_MEMBER(hkpPhantomAgent, m_collidableA, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpPhantomAgent, m_collidableB, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpPhantomAgent, m_shapeA, 0, "hkpPhantomCallbackShape*") // class hkpPhantomCallbackShape*
    HK_TRACKER_MEMBER(hkpPhantomAgent, m_shapeB, 0, "hkpPhantomCallbackShape*") // class hkpPhantomCallbackShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPhantomAgent, s_libraryName, hkpCollisionAgent)

#include <Physics2012/Collide/Agent/MiscAgent/Transform/hkpTransformAgent.h>


// hkpTransformAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpTransformAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpTransformAgent)
    HK_TRACKER_MEMBER(hkpTransformAgent, m_childAgent, 0, "hkpCollisionAgent*") // class hkpCollisionAgent*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpTransformAgent, s_libraryName, hkpCollisionAgent)

#include <Physics2012/Collide/Agent/Query/hkpCdBodyPairCollector.h>


// hkpCdBodyPairCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCdBodyPairCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCdBodyPairCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkpCdBodyPairCollector, s_libraryName)

#include <Physics2012/Collide/Agent/Query/hkpCdPointCollector.h>


// hkpCdPointCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCdPointCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCdPointCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkpCdPointCollector, s_libraryName)

#include <Physics2012/Collide/Agent/Query/hkpLinearCastCollisionInput.h>


// hkpLinearCastCollisionInput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpLinearCastCollisionInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpLinearCastCollisionInput)
    HK_TRACKER_MEMBER(hkpLinearCastCollisionInput, m_config, 0, "hkpCollisionAgentConfig*") // struct hkpCollisionAgentConfig*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpLinearCastCollisionInput, s_libraryName, hkpCollisionInput)

#include <Physics2012/Collide/Agent/Util/LinearCast/hkpIterativeLinearCastAgent.h>


// hkpIterativeLinearCastAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpIterativeLinearCastAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpIterativeLinearCastAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpIterativeLinearCastAgent, s_libraryName, hkpCollisionAgent)

#include <Physics2012/Collide/Agent/Util/Null/hkpNullAgent.h>


// hkpNullAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpNullAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpNullAgent)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpNullAgent, s_libraryName, hkpCollisionAgent)

#include <Physics2012/Collide/Agent/Util/Symmetric/hkpSymmetricAgentLinearCast.h>


// hkpSymmetricAgentFlipCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSymmetricAgentFlipCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSymmetricAgentFlipCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSymmetricAgentFlipCollector, s_libraryName, hkpCdPointCollector)


// hkpSymmetricAgentFlipCastCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSymmetricAgentFlipCastCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSymmetricAgentFlipCastCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSymmetricAgentFlipCastCollector, s_libraryName, hkpCdPointCollector)


// hkpSymmetricAgentFlipBodyCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSymmetricAgentFlipBodyCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSymmetricAgentFlipBodyCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSymmetricAgentFlipBodyCollector, s_libraryName, hkpCdBodyPairCollector)

#include <Physics2012/Collide/Agent/hkpCollisionAgent.h>


// hkpCollisionAgent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCollisionAgent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCollisionAgent)
    HK_TRACKER_MEMBER(hkpCollisionAgent, m_contactMgr, 0, "hkpContactMgr*") // class hkpContactMgr*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpCollisionAgent, s_libraryName, hkReferencedObject)

#include <Physics2012/Collide/Agent/hkpCollisionAgentConfig.h>


// hkpCollisionAgentConfig ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCollisionAgentConfig, s_libraryName)

#include <Physics2012/Collide/Agent/hkpCollisionInput.h>


// hkpCollisionInput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCollisionInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Aabb32Info)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCollisionInput)
    HK_TRACKER_MEMBER(hkpCollisionInput, m_dispatcher, 0, "hkPadSpu<hkpCollisionDispatcher*>") // class hkPadSpu< class hkpCollisionDispatcher* >
    HK_TRACKER_MEMBER(hkpCollisionInput, m_filter, 0, "hkPadSpu<hkpCollisionFilter*>") // class hkPadSpu< const class hkpCollisionFilter* >
    HK_TRACKER_MEMBER(hkpCollisionInput, m_convexListFilter, 0, "hkPadSpu<hkpConvexListFilter*>") // class hkPadSpu< const class hkpConvexListFilter* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCollisionInput, s_libraryName)


// Aabb32Info hkpCollisionInput
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollisionInput, Aabb32Info, s_libraryName)

#include <Physics2012/Collide/Agent/hkpCollisionQualityInfo.h>


// hkpCollisionQualityInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCollisionQualityInfo, s_libraryName)

#include <Physics2012/Collide/Agent/hkpProcessCdPoint.h>


// hkpProcessCdPoint ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpProcessCdPoint, s_libraryName)

#include <Physics2012/Collide/Agent/hkpProcessCollisionData.h>


// hkpProcessCollisionData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpProcessCollisionData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ToiInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpProcessCollisionData)
    HK_TRACKER_MEMBER(hkpProcessCollisionData, m_firstFreeContactPoint, 0, "hkPadSpu<hkpProcessCdPoint*>") // class hkPadSpu< struct hkpProcessCdPoint* >
    HK_TRACKER_MEMBER(hkpProcessCollisionData, m_constraintOwner, 0, "hkPadSpu<hkpConstraintOwner*>") // class hkPadSpu< class hkpConstraintOwner* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpProcessCollisionData, s_libraryName)


// ToiInfo hkpProcessCollisionData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpProcessCollisionData, ToiInfo, s_libraryName)

#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>


// hkpProcessCollisionInput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpProcessCollisionInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpProcessCollisionInput)
    HK_TRACKER_MEMBER(hkpProcessCollisionInput, m_collisionQualityInfo, 0, "hkPadSpu<hkpCollisionQualityInfo*>") // class hkPadSpu< struct hkpCollisionQualityInfo* >
    HK_TRACKER_MEMBER(hkpProcessCollisionInput, m_spareAgentSector, 0, "hkpAgent1nSector*") // struct hkpAgent1nSector*
    HK_TRACKER_MEMBER(hkpProcessCollisionInput, m_dynamicsInfo, 0, "void*") // void*
    HK_TRACKER_MEMBER(hkpProcessCollisionInput, m_config, 0, "hkpCollisionAgentConfig*") // struct hkpCollisionAgentConfig*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpProcessCollisionInput, s_libraryName, hkpCollisionInput)

#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>


// hkpProcessCollisionOutput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpProcessCollisionOutput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ContactRef)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PotentialInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpProcessCollisionOutput)
    HK_TRACKER_MEMBER(hkpProcessCollisionOutput, m_potentialContacts, 0, "hkPadSpu<hkpProcessCollisionOutput::PotentialInfo*>") // class hkPadSpu< struct hkpProcessCollisionOutput::PotentialInfo* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpProcessCollisionOutput, s_libraryName, hkpProcessCollisionData)


// ContactRef hkpProcessCollisionOutput

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpProcessCollisionOutput::ContactRef)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpProcessCollisionOutput::ContactRef)
    HK_TRACKER_MEMBER(hkpProcessCollisionOutput::ContactRef, m_contactPoint, 0, "hkpProcessCdPoint*") // struct hkpProcessCdPoint*
    HK_TRACKER_MEMBER(hkpProcessCollisionOutput::ContactRef, m_agentEntry, 0, "hkpAgentEntry*") // struct hkpAgentEntry*
    HK_TRACKER_MEMBER(hkpProcessCollisionOutput::ContactRef, m_agentData, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpProcessCollisionOutput::ContactRef, s_libraryName)


// PotentialInfo hkpProcessCollisionOutput

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpProcessCollisionOutput::PotentialInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpProcessCollisionOutput::PotentialInfo)
    HK_TRACKER_MEMBER(hkpProcessCollisionOutput::PotentialInfo, m_firstFreePotentialContact, 0, "hkpProcessCollisionOutput::ContactRef*") // struct hkpProcessCollisionOutput::ContactRef*
    HK_TRACKER_MEMBER(hkpProcessCollisionOutput::PotentialInfo, m_firstFreeRepresentativeContact, 0, "hkpProcessCdPoint**") // struct hkpProcessCdPoint**
    HK_TRACKER_MEMBER(hkpProcessCollisionOutput::PotentialInfo, m_representativeContacts, 0, "hkpProcessCdPoint* [256]") // struct hkpProcessCdPoint* [256]
    HK_TRACKER_MEMBER(hkpProcessCollisionOutput::PotentialInfo, m_potentialContacts, 0, "hkpProcessCollisionOutput::ContactRef [256]") // struct hkpProcessCollisionOutput::ContactRef [256]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpProcessCollisionOutput::PotentialInfo, s_libraryName)

#include <Physics2012/Collide/Agent3/CapsuleTriangle/hkpCapsuleTriangleAgent3.h>


// hkpCapsuleTriangleCache3 ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCapsuleTriangleCache3, s_libraryName)

#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachine.h>


// hkpAgent1nMachineEntry ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpAgent1nMachineEntry, s_libraryName)


// hkpAgent1nMachinePaddedEntry ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpAgent1nMachinePaddedEntry, s_libraryName)


// hkpAgent1nMachineTimEntry ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpAgent1nMachineTimEntry, s_libraryName)


// hkpAgent1nMachine_VisitorInput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAgent1nMachine_VisitorInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpAgent1nMachine_VisitorInput)
    HK_TRACKER_MEMBER(hkpAgent1nMachine_VisitorInput, m_bodyA, 0, "hkpCdBody*") // const class hkpCdBody*
    HK_TRACKER_MEMBER(hkpAgent1nMachine_VisitorInput, m_collectionBodyB, 0, "hkpCdBody*") // const class hkpCdBody*
    HK_TRACKER_MEMBER(hkpAgent1nMachine_VisitorInput, m_containerShapeA, 0, "hkpShapeContainer*") // const class hkpShapeContainer*
    HK_TRACKER_MEMBER(hkpAgent1nMachine_VisitorInput, m_containerShapeB, 0, "hkpShapeContainer*") // const class hkpShapeContainer*
    HK_TRACKER_MEMBER(hkpAgent1nMachine_VisitorInput, m_input, 0, "hkpCollisionInput*") // const struct hkpCollisionInput*
    HK_TRACKER_MEMBER(hkpAgent1nMachine_VisitorInput, m_contactMgr, 0, "hkpContactMgr*") // class hkpContactMgr*
    HK_TRACKER_MEMBER(hkpAgent1nMachine_VisitorInput, m_constraintOwner, 0, "hkpConstraintOwner*") // class hkpConstraintOwner*
    HK_TRACKER_MEMBER(hkpAgent1nMachine_VisitorInput, m_clientData, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpAgent1nMachine_VisitorInput, s_libraryName)


// hkpAgentNmMachineBodyTemp ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAgentNmMachineBodyTemp)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpAgentNmMachineBodyTemp)
    HK_TRACKER_MEMBER(hkpAgentNmMachineBodyTemp, m_bodyA, 0, "hkpCdBody") // class hkpCdBody
    HK_TRACKER_MEMBER(hkpAgentNmMachineBodyTemp, m_bodyB, 0, "hkpCdBody") // class hkpCdBody
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpAgentNmMachineBodyTemp, s_libraryName)


// hkpShapeKeyPairLocal1n ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeKeyPairLocal1n, s_libraryName)


// hkpShapeKeyPairLocalNm ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeKeyPairLocalNm, s_libraryName)

#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nTrack.h>


// hkpAgent1nSector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAgent1nSector)
HK_TRACKER_DECLARE_CLASS_END
HK_TRACKER_IMPLEMENT_SCAN_CLASS(hkpAgent1nSector, s_libraryName)


// hkpAgent1nTrack ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAgent1nTrack)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpAgent1nTrack)
    HK_TRACKER_MEMBER(hkpAgent1nTrack, m_sectors, 0, "hkArray<hkpAgent1nSector*, hkContainerHeapAllocator>") // hkArray< struct hkpAgent1nSector*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpAgent1nTrack, s_libraryName)

#include <Physics2012/Collide/Agent3/Machine/1n/hkpCpuDoubleContainerIterator.h>


// hkpCpuDoubleContainerIterator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCpuDoubleContainerIterator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCpuDoubleContainerIterator)
    HK_TRACKER_MEMBER(hkpCpuDoubleContainerIterator, m_containerA, 0, "hkpShapeContainer*") // const class hkpShapeContainer*
    HK_TRACKER_MEMBER(hkpCpuDoubleContainerIterator, m_containerB, 0, "hkpShapeContainer*") // const class hkpShapeContainer*
    HK_TRACKER_MEMBER(hkpCpuDoubleContainerIterator, m_shapeKeys, 0, "hkpShapeKeyPair*") // const class hkpShapeKeyPair*
    HK_TRACKER_MEMBER(hkpCpuDoubleContainerIterator, m_extractedShapeA, 0, "hkpShape*") // const class hkpShape*
    HK_TRACKER_MEMBER(hkpCpuDoubleContainerIterator, m_extractedShapeB, 0, "hkpShape*") // const class hkpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCpuDoubleContainerIterator, s_libraryName)

#include <Physics2012/Collide/Agent3/Machine/1n/hkpCpuSingleContainerIterator.h>


// hkpCpuSingleContainerIterator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCpuSingleContainerIterator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCpuSingleContainerIterator)
    HK_TRACKER_MEMBER(hkpCpuSingleContainerIterator, m_container, 0, "hkpShapeContainer*") // const class hkpShapeContainer*
    HK_TRACKER_MEMBER(hkpCpuSingleContainerIterator, m_shapeKeys, 0, "hkUint32*") // const hkUint32*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCpuSingleContainerIterator, s_libraryName)

#include <Physics2012/Collide/Agent3/Machine/Midphase/hkpMidphaseAgentData.h>


// hkpMidphaseAgentData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMidphaseAgentData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMidphaseAgentData)
    HK_TRACKER_MEMBER(hkpMidphaseAgentData, m_agent1nTrack, 0, "hkpAgent1nTrack") // struct hkpAgent1nTrack
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpMidphaseAgentData, s_libraryName)

#include <Physics2012/Collide/Agent3/Machine/Midphase/hkpShapeKeyTrack.h>


// hkpShapeKeyTrack ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeKeyTrack, s_libraryName)

// Skipping Class hkpShapeKeyTrackWriter as it is derived from a template

// Skipping Class hkpShapeKeyTrackConsumer as it is derived from a template

#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>


// hkpAgentNnMachinePaddedEntry ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpAgentNnMachinePaddedEntry, s_libraryName)


// hkpAgentNnMachineTimEntry ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpAgentNnMachineTimEntry, s_libraryName)

#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnTrack.h>


// hkpAgentNnEntry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAgentNnEntry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpAgentNnEntry)
    HK_TRACKER_MEMBER(hkpAgentNnEntry, m_contactMgr, 0, "hkpContactMgr*") // class hkpContactMgr*
    HK_TRACKER_MEMBER(hkpAgentNnEntry, m_collidable, 0, "hkpLinkedCollidable* [2]") // class hkpLinkedCollidable* [2]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpAgentNnEntry, s_libraryName, hkpAgentEntry)


// hkpAgentNnSector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAgentNnSector)
HK_TRACKER_DECLARE_CLASS_END
HK_TRACKER_IMPLEMENT_SCAN_CLASS(hkpAgentNnSector, s_libraryName)


// hkpAgentNnTrack ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAgentNnTrack)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpAgentNnTrack)
    HK_TRACKER_MEMBER(hkpAgentNnTrack, m_sectors, 0, "hkInplaceArray<hkpAgentNnSector*, 1, hkContainerHeapAllocator>") // class hkInplaceArray< struct hkpAgentNnSector*, 1, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpAgentNnTrack, s_libraryName)

// None hkpAgentNnTrackType
HK_TRACKER_IMPLEMENT_SIMPLE(hkpAgentNnTrackType, s_libraryName)
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpLinkedCollidable.h>


// hkpLinkedCollidable ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpLinkedCollidable)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CollisionEntry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpLinkedCollidable)
    HK_TRACKER_MEMBER(hkpLinkedCollidable, m_collisionEntries, 0, "hkArray<hkpLinkedCollidable::CollisionEntry, hkContainerHeapAllocator>") // hkArray< struct hkpLinkedCollidable::CollisionEntry, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpLinkedCollidable, s_libraryName, hkpCollidable)


// CollisionEntry hkpLinkedCollidable

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpLinkedCollidable::CollisionEntry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpLinkedCollidable::CollisionEntry)
    HK_TRACKER_MEMBER(hkpLinkedCollidable::CollisionEntry, m_agentEntry, 0, "hkpAgentNnEntry*") // struct hkpAgentNnEntry*
    HK_TRACKER_MEMBER(hkpLinkedCollidable::CollisionEntry, m_partner, 0, "hkpLinkedCollidable*") // class hkpLinkedCollidable*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpLinkedCollidable::CollisionEntry, s_libraryName)

#include <Physics2012/Collide/Agent3/hkpAgent3.h>


// hkpAgentEntry ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpAgentEntry, s_libraryName)


// hkpAgent3Input ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAgent3Input)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpAgent3Input)
    HK_TRACKER_MEMBER(hkpAgent3Input, m_bodyA, 0, "hkPadSpu<hkpCdBody*>") // class hkPadSpu< const class hkpCdBody* >
    HK_TRACKER_MEMBER(hkpAgent3Input, m_bodyB, 0, "hkPadSpu<hkpCdBody*>") // class hkPadSpu< const class hkpCdBody* >
    HK_TRACKER_MEMBER(hkpAgent3Input, m_overrideBodyA, 0, "hkPadSpu<hkpCdBody*>") // class hkPadSpu< const class hkpCdBody* >
    HK_TRACKER_MEMBER(hkpAgent3Input, m_input, 0, "hkPadSpu<hkpProcessCollisionInput*>") // class hkPadSpu< const struct hkpProcessCollisionInput* >
    HK_TRACKER_MEMBER(hkpAgent3Input, m_contactMgr, 0, "hkPadSpu<hkpContactMgr*>") // class hkPadSpu< class hkpContactMgr* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpAgent3Input, s_libraryName)


// hkpAgent3ProcessInput ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpAgent3ProcessInput, s_libraryName)

// hkAgent3 StreamCommand
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkAgent3::StreamCommand, s_libraryName, hkAgent3_StreamCommand)
// hkAgent3 Symmetric
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkAgent3::Symmetric, s_libraryName, hkAgent3_Symmetric)
#include <Physics2012/Collide/BoxBox/hkpBoxBoxContactPoint.h>


// hkpFeatureContactPoint ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpFeatureContactPoint, s_libraryName)

#include <Physics2012/Collide/BoxBox/hkpBoxBoxManifold.h>


// hkpBoxBoxManifold ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpBoxBoxManifold, s_libraryName)

#include <Physics2012/Collide/BroadPhase/3AxisSweep/hkp3AxisSweep.h>


// hkp3AxisSweep ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkp3AxisSweep)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpBpMarker)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpBpNode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpBpEndPoint)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpBpAxis)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MarkerHandling)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpBpMarkerUse)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpBpQuerySingleType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkp3AxisSweep)
    HK_TRACKER_MEMBER(hkp3AxisSweep, m_nodes, 0, "hkArray<hkp3AxisSweep::hkpBpNode, hkContainerHeapAllocator>") // hkArray< class hkp3AxisSweep::hkpBpNode, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkp3AxisSweep, m_axis, 0, "hkp3AxisSweep::hkpBpAxis [3]") // class hkp3AxisSweep::hkpBpAxis [3]
    HK_TRACKER_MEMBER(hkp3AxisSweep, m_markers, 0, "hkp3AxisSweep::hkpBpMarker*") // class hkp3AxisSweep::hkpBpMarker*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkp3AxisSweep, s_libraryName, hkpBroadPhase)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkp3AxisSweep, MarkerHandling, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkp3AxisSweep, hkpBpMarkerUse, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkp3AxisSweep, hkpBpQuerySingleType, s_libraryName)


// hkpBpMarker hkp3AxisSweep

HK_TRACKER_DECLARE_CLASS_BEGIN(hkp3AxisSweep::hkpBpMarker)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkp3AxisSweep::hkpBpMarker)
    HK_TRACKER_MEMBER(hkp3AxisSweep::hkpBpMarker, m_overlappingObjects, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkp3AxisSweep::hkpBpMarker, s_libraryName)


// hkpBpNode hkp3AxisSweep

HK_TRACKER_DECLARE_CLASS_BEGIN(hkp3AxisSweep::hkpBpNode)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkp3AxisSweep::hkpBpNode)
    HK_TRACKER_MEMBER(hkp3AxisSweep::hkpBpNode, m_handle, 0, "hkpBroadPhaseHandle*") // class hkpBroadPhaseHandle*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkp3AxisSweep::hkpBpNode, s_libraryName)


// hkpBpEndPoint hkp3AxisSweep
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkp3AxisSweep, hkpBpEndPoint, s_libraryName)


// hkpBpAxis hkp3AxisSweep

HK_TRACKER_DECLARE_CLASS_BEGIN(hkp3AxisSweep::hkpBpAxis)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkp3AxisSweep::hkpBpAxis)
    HK_TRACKER_MEMBER(hkp3AxisSweep::hkpBpAxis, m_endPoints, 0, "hkArray<hkp3AxisSweep::hkpBpEndPoint, hkContainerHeapAllocator>") // hkArray< class hkp3AxisSweep::hkpBpEndPoint, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkp3AxisSweep::hkpBpAxis, s_libraryName)

#include <Physics2012/Collide/BroadPhase/hkpBroadPhase.h>


// hkpBroadPhase ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBroadPhase)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpCastRayInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpCastAabbInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Enable32BitBroadPhase)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Enable32BitTreeBroadPhase)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Enable32BitHybridBroadPhase)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BroadPhaseType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Capabilities)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBroadPhase)
    HK_TRACKER_MEMBER(hkpBroadPhase, m_criticalSection, 0, "hkCriticalSection*") // class hkCriticalSection*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpBroadPhase, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBroadPhase, BroadPhaseType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBroadPhase, Capabilities, s_libraryName)


// hkpCastRayInput hkpBroadPhase

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBroadPhase::hkpCastRayInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBroadPhase::hkpCastRayInput)
    HK_TRACKER_MEMBER(hkpBroadPhase::hkpCastRayInput, m_toBase, 0, "hkVector4*") // const hkVector4*
    HK_TRACKER_MEMBER(hkpBroadPhase::hkpCastRayInput, m_aabbCacheInfo, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpBroadPhase::hkpCastRayInput, s_libraryName)


// hkpCastAabbInput hkpBroadPhase

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBroadPhase::hkpCastAabbInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBroadPhase::hkpCastAabbInput)
    HK_TRACKER_MEMBER(hkpBroadPhase::hkpCastAabbInput, m_aabbCacheInfo, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpBroadPhase::hkpCastAabbInput, s_libraryName)


// Enable32BitBroadPhase hkpBroadPhase
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBroadPhase, Enable32BitBroadPhase, s_libraryName)


// Enable32BitTreeBroadPhase hkpBroadPhase
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBroadPhase, Enable32BitTreeBroadPhase, s_libraryName)


// Enable32BitHybridBroadPhase hkpBroadPhase
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBroadPhase, Enable32BitHybridBroadPhase, s_libraryName)

#include <Physics2012/Collide/BroadPhase/hkpBroadPhaseCastCollector.h>

// hk.MemoryTracker ignore hkpBroadPhaseCastCollector
#include <Physics2012/Collide/BroadPhase/hkpBroadPhaseHandle.h>


// hkpBroadPhaseHandle ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpBroadPhaseHandle, s_libraryName)

#include <Physics2012/Collide/BroadPhase/hkpBroadPhaseHandlePair.h>


// hkpBroadPhaseHandlePair ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBroadPhaseHandlePair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBroadPhaseHandlePair)
    HK_TRACKER_MEMBER(hkpBroadPhaseHandlePair, m_a, 0, "hkpBroadPhaseHandle*") // class hkpBroadPhaseHandle*
    HK_TRACKER_MEMBER(hkpBroadPhaseHandlePair, m_b, 0, "hkpBroadPhaseHandle*") // class hkpBroadPhaseHandle*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpBroadPhaseHandlePair, s_libraryName)

#include <Physics2012/Collide/Dispatch/BroadPhase/hkpBroadPhaseListener.h>

// hk.MemoryTracker ignore hkpBroadPhaseListener
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpNullBroadPhaseListener.h>


// hkpNullBroadPhaseListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpNullBroadPhaseListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpNullBroadPhaseListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpNullBroadPhaseListener, s_libraryName, hkReferencedObject)

#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseDispatcher.h>


// hkpTypedBroadPhaseDispatcher ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpTypedBroadPhaseDispatcher)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpTypedBroadPhaseDispatcher)
    HK_TRACKER_MEMBER(hkpTypedBroadPhaseDispatcher, m_broadPhaseListeners, 0, "hkpBroadPhaseListener* [8] [8]") // class hkpBroadPhaseListener* [8] [8]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpTypedBroadPhaseDispatcher, s_libraryName)

#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseHandle.h>


// hkpTypedBroadPhaseHandle ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpTypedBroadPhaseHandle, s_libraryName)

#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseHandlePair.h>


// hkpTypedBroadPhaseHandlePair ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpTypedBroadPhaseHandlePair, s_libraryName)

#include <Physics2012/Collide/Dispatch/ContactMgr/hkpContactMgrFactory.h>


// hkpContactMgrFactory ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpContactMgrFactory)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpContactMgrFactory)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpContactMgrFactory, s_libraryName, hkReferencedObject)

#include <Physics2012/Collide/Dispatch/ContactMgr/hkpNullContactMgr.h>


// hkpNullContactMgr ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpNullContactMgr)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpNullContactMgr)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpNullContactMgr, s_libraryName, hkpContactMgr)

#include <Physics2012/Collide/Dispatch/ContactMgr/hkpNullContactMgrFactory.h>


// hkpNullContactMgrFactory ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpNullContactMgrFactory)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpNullContactMgrFactory)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpNullContactMgrFactory, s_libraryName, hkpContactMgrFactory)

#include <Physics2012/Collide/Dispatch/hkpAgentRegisterUtil.h>


// hkpAgentRegisterUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpAgentRegisterUtil, s_libraryName)

#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>


// hkpCollisionDispatcher ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCollisionDispatcher)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AgentFuncs)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Agent3Funcs)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(InitCollisionQualityInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ShapeInheritance)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AgentEntry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Agent3Entry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Agent3FuncsIntern)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DebugEntry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IsAgentPredictive)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CollisionQualityLevel)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCollisionDispatcher)
    HK_TRACKER_MEMBER(hkpCollisionDispatcher, m_contactMgrFactory, 0, "hkpContactMgrFactory* [8] [8]") // class hkpContactMgrFactory* [8] [8]
    HK_TRACKER_MEMBER(hkpCollisionDispatcher, m_shapeInheritance, 0, "hkArray<hkpCollisionDispatcher::ShapeInheritance, hkContainerHeapAllocator>") // hkArray< struct hkpCollisionDispatcher::ShapeInheritance, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCollisionDispatcher, m_debugAgent2Table, 0, "hkpCollisionDispatcher::DebugEntry [35] [35]*") // struct hkpCollisionDispatcher::DebugEntry [35] [35]*
    HK_TRACKER_MEMBER(hkpCollisionDispatcher, m_debugAgent2TablePred, 0, "hkpCollisionDispatcher::DebugEntry [35] [35]*") // struct hkpCollisionDispatcher::DebugEntry [35] [35]*
    HK_TRACKER_MEMBER(hkpCollisionDispatcher, m_debugAgent3Table, 0, "hkpCollisionDispatcher::DebugEntry [35] [35]*") // struct hkpCollisionDispatcher::DebugEntry [35] [35]*
    HK_TRACKER_MEMBER(hkpCollisionDispatcher, m_debugAgent3TablePred, 0, "hkpCollisionDispatcher::DebugEntry [35] [35]*") // struct hkpCollisionDispatcher::DebugEntry [35] [35]*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCollisionDispatcher, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollisionDispatcher, IsAgentPredictive, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollisionDispatcher, CollisionQualityLevel, s_libraryName)


// AgentFuncs hkpCollisionDispatcher
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollisionDispatcher, AgentFuncs, s_libraryName)


// Agent3Funcs hkpCollisionDispatcher
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollisionDispatcher, Agent3Funcs, s_libraryName)


// InitCollisionQualityInfo hkpCollisionDispatcher
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollisionDispatcher, InitCollisionQualityInfo, s_libraryName)


// ShapeInheritance hkpCollisionDispatcher
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollisionDispatcher, ShapeInheritance, s_libraryName)


// AgentEntry hkpCollisionDispatcher
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollisionDispatcher, AgentEntry, s_libraryName)


// Agent3Entry hkpCollisionDispatcher
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollisionDispatcher, Agent3Entry, s_libraryName)


// Agent3FuncsIntern hkpCollisionDispatcher
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollisionDispatcher, Agent3FuncsIntern, s_libraryName)


// DebugEntry hkpCollisionDispatcher
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollisionDispatcher, DebugEntry, s_libraryName)

// None hkpCollisionDispatcherAgentType
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCollisionDispatcherAgentType, s_libraryName)
#include <Physics2012/Collide/Filter/DefaultConvexList/hkpDefaultConvexListFilter.h>


// hkpDefaultConvexListFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDefaultConvexListFilter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDefaultConvexListFilter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDefaultConvexListFilter, s_libraryName, hkpConvexListFilter)

#include <Physics2012/Collide/Filter/Group/hkpGroupFilter.h>


// hkpGroupFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpGroupFilter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpGroupFilter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpGroupFilter, s_libraryName, hkpCollisionFilter)

#include <Physics2012/Collide/Filter/Group/hkpGroupFilterSetup.h>


// hkpGroupFilterSetup ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpGroupFilterSetup, s_libraryName)

#include <Physics2012/Collide/Filter/List/hkpCollisionFilterList.h>


// hkpCollisionFilterList ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCollisionFilterList)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCollisionFilterList)
    HK_TRACKER_MEMBER(hkpCollisionFilterList, m_collisionFilters, 0, "hkArray<hkpCollisionFilter*, hkContainerHeapAllocator>") // hkArray< class hkpCollisionFilter*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCollisionFilterList, s_libraryName, hkpCollisionFilter)

#include <Physics2012/Collide/Filter/Null/hkpNullCollisionFilter.h>


// hkpNullCollisionFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpNullCollisionFilter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpNullCollisionFilter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpNullCollisionFilter, s_libraryName, hkpCollisionFilter)

#include <Physics2012/Collide/Filter/hkpCollidableCollidableFilter.h>

// hk.MemoryTracker ignore hkpCollidableCollidableFilter
#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>


// hkpCollisionFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCollisionFilter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpFilterType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCollisionFilter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpCollisionFilter, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollisionFilter, hkpFilterType, s_libraryName)

#include <Physics2012/Collide/Filter/hkpConvexListFilter.h>


// hkpConvexListFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexListFilter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConvexListCollisionType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexListFilter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpConvexListFilter, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConvexListFilter, ConvexListCollisionType, s_libraryName)

#include <Physics2012/Collide/Filter/hkpRayCollidableFilter.h>

// hk.MemoryTracker ignore hkpRayCollidableFilter
#include <Physics2012/Collide/Filter/hkpShapeCollectionFilter.h>

// hk.MemoryTracker ignore hkpShapeCollectionFilter
#include <Physics2012/Collide/Query/CastUtil/hkpLinearCastInput.h>


// hkpLinearCastInput ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpLinearCastInput, s_libraryName)

#include <Physics2012/Collide/Query/CastUtil/hkpSimpleWorldRayCaster.h>


// hkpSimpleWorldRayCaster ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimpleWorldRayCaster)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSimpleWorldRayCaster)
    HK_TRACKER_MEMBER(hkpSimpleWorldRayCaster, m_input, 0, "hkpWorldRayCastInput*") // const struct hkpWorldRayCastInput*
    HK_TRACKER_MEMBER(hkpSimpleWorldRayCaster, m_filter, 0, "hkpRayCollidableFilter*") // const class hkpRayCollidableFilter*
    HK_TRACKER_MEMBER(hkpSimpleWorldRayCaster, m_result, 0, "hkpWorldRayCastOutput*") // struct hkpWorldRayCastOutput*
    HK_TRACKER_MEMBER(hkpSimpleWorldRayCaster, m_shapeInput, 0, "hkpShapeRayCastInput") // struct hkpShapeRayCastInput
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSimpleWorldRayCaster, s_libraryName, hkpBroadPhaseCastCollector)

#include <Physics2012/Collide/Query/CastUtil/hkpWorldLinearCaster.h>


// hkpWorldLinearCaster ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldLinearCaster)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldLinearCaster)
    HK_TRACKER_MEMBER(hkpWorldLinearCaster, m_input, 0, "hkpLinearCastInput*") // const struct hkpLinearCastInput*
    HK_TRACKER_MEMBER(hkpWorldLinearCaster, m_filter, 0, "hkpCollidableCollidableFilter*") // const class hkpCollidableCollidableFilter*
    HK_TRACKER_MEMBER(hkpWorldLinearCaster, m_castCollector, 0, "hkpCdPointCollector*") // class hkpCdPointCollector*
    HK_TRACKER_MEMBER(hkpWorldLinearCaster, m_startPointCollector, 0, "hkpCdPointCollector*") // class hkpCdPointCollector*
    HK_TRACKER_MEMBER(hkpWorldLinearCaster, m_collidableA, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpWorldLinearCaster, m_shapeInput, 0, "hkpLinearCastCollisionInput") // struct hkpLinearCastCollisionInput
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpWorldLinearCaster, s_libraryName, hkpBroadPhaseCastCollector)

#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastInput.h>


// hkpWorldRayCastInput ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpWorldRayCastInput, s_libraryName)

#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastOutput.h>


// hkpWorldRayCastOutput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldRayCastOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldRayCastOutput)
    HK_TRACKER_MEMBER(hkpWorldRayCastOutput, m_rootCollidable, 0, "hkpCollidable*") // const class hkpCollidable*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpWorldRayCastOutput, s_libraryName, hkpShapeRayCastOutput)

#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCaster.h>


// hkpWorldRayCaster ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldRayCaster)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldRayCaster)
    HK_TRACKER_MEMBER(hkpWorldRayCaster, m_input, 0, "hkpWorldRayCastInput*") // const struct hkpWorldRayCastInput*
    HK_TRACKER_MEMBER(hkpWorldRayCaster, m_filter, 0, "hkpRayCollidableFilter*") // const class hkpRayCollidableFilter*
    HK_TRACKER_MEMBER(hkpWorldRayCaster, m_collectorBase, 0, "hkpRayHitCollector*") // class hkpRayHitCollector*
    HK_TRACKER_MEMBER(hkpWorldRayCaster, m_shapeInput, 0, "hkpShapeRayCastInput") // struct hkpShapeRayCastInput
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpWorldRayCaster, s_libraryName, hkpBroadPhaseCastCollector)

#include <Physics2012/Collide/Query/Collector/BodyPairCollector/hkpAllCdBodyPairCollector.h>


// hkpAllCdBodyPairCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAllCdBodyPairCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpAllCdBodyPairCollector)
    HK_TRACKER_MEMBER(hkpAllCdBodyPairCollector, m_hits, 0, "hkInplaceArray<hkpRootCdBodyPair, 16, hkContainerHeapAllocator>") // class hkInplaceArray< struct hkpRootCdBodyPair, 16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpAllCdBodyPairCollector, s_libraryName, hkpCdBodyPairCollector)

#include <Physics2012/Collide/Query/Collector/BodyPairCollector/hkpFirstCdBodyPairCollector.h>


// hkpFirstCdBodyPairCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpFirstCdBodyPairCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpFirstCdBodyPairCollector)
    HK_TRACKER_MEMBER(hkpFirstCdBodyPairCollector, m_cdBodyPair, 0, "hkpRootCdBodyPair") // struct hkpRootCdBodyPair
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpFirstCdBodyPairCollector, s_libraryName, hkpCdBodyPairCollector)

#include <Physics2012/Collide/Query/Collector/BodyPairCollector/hkpFlagCdBodyPairCollector.h>


// hkpFlagCdBodyPairCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpFlagCdBodyPairCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpFlagCdBodyPairCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpFlagCdBodyPairCollector, s_libraryName, hkpCdBodyPairCollector)

#include <Physics2012/Collide/Query/Collector/BodyPairCollector/hkpRootCdBodyPair.h>


// hkpRootCdBodyPair ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRootCdBodyPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRootCdBodyPair)
    HK_TRACKER_MEMBER(hkpRootCdBodyPair, m_rootCollidableA, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpRootCdBodyPair, m_rootCollidableB, 0, "hkpCollidable*") // const class hkpCollidable*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpRootCdBodyPair, s_libraryName)

#include <Physics2012/Collide/Query/Collector/PointCollector/hkpAllCdPointCollector.h>


// hkpAllCdPointCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAllCdPointCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpAllCdPointCollector)
    HK_TRACKER_MEMBER(hkpAllCdPointCollector, m_hits, 0, "hkInplaceArray<hkpRootCdPoint, 8, hkContainerHeapAllocator>") // class hkInplaceArray< struct hkpRootCdPoint, 8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpAllCdPointCollector, s_libraryName, hkpCdPointCollector)

#include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>


// hkpClosestCdPointCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpClosestCdPointCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpClosestCdPointCollector)
    HK_TRACKER_MEMBER(hkpClosestCdPointCollector, m_hitPoint, 0, "hkpRootCdPoint") // struct hkpRootCdPoint
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpClosestCdPointCollector, s_libraryName, hkpCdPointCollector)

#include <Physics2012/Collide/Query/Collector/PointCollector/hkpFixedBufferCdPointCollector.h>


// hkpFixedBufferCdPointCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpFixedBufferCdPointCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpFixedBufferCdPointCollector)
    HK_TRACKER_MEMBER(hkpFixedBufferCdPointCollector, m_pointsArrayBase, 0, "hkPadSpu<hkpRootCdPoint*>") // class hkPadSpu< struct hkpRootCdPoint* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpFixedBufferCdPointCollector, s_libraryName, hkpCdPointCollector)

#include <Physics2012/Collide/Query/Collector/PointCollector/hkpRootCdPoint.h>


// hkpRootCdPoint ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRootCdPoint)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRootCdPoint)
    HK_TRACKER_MEMBER(hkpRootCdPoint, m_rootCollidableA, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpRootCdPoint, m_rootCollidableB, 0, "hkpCollidable*") // const class hkpCollidable*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpRootCdPoint, s_libraryName)

#include <Physics2012/Collide/Query/Collector/PointCollector/hkpSimpleClosestContactCollector.h>


// hkpSimpleClosestContactCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimpleClosestContactCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSimpleClosestContactCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSimpleClosestContactCollector, s_libraryName, hkpCdPointCollector)

#include <Physics2012/Collide/Query/Collector/RayCollector/hkpAllRayHitCollector.h>


// hkpAllRayHitCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAllRayHitCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpAllRayHitCollector)
    HK_TRACKER_MEMBER(hkpAllRayHitCollector, m_hits, 0, "hkInplaceArray<hkpWorldRayCastOutput, 8, hkContainerHeapAllocator>") // class hkInplaceArray< struct hkpWorldRayCastOutput, 8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpAllRayHitCollector, s_libraryName, hkpRayHitCollector)

#include <Physics2012/Collide/Query/Collector/RayCollector/hkpClosestRayHitCollector.h>


// hkpClosestRayHitCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpClosestRayHitCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpClosestRayHitCollector)
    HK_TRACKER_MEMBER(hkpClosestRayHitCollector, m_rayHit, 0, "hkpWorldRayCastOutput") // struct hkpWorldRayCastOutput
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpClosestRayHitCollector, s_libraryName, hkpRayHitCollector)

#include <Physics2012/Collide/Query/Collector/RayCollector/hkpFixedBufferRayHitCollector.h>


// hkpFixedBufferRayHitCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpFixedBufferRayHitCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpFixedBufferRayHitCollector)
    HK_TRACKER_MEMBER(hkpFixedBufferRayHitCollector, m_rayCastOutputBase, 0, "hkPadSpu<hkpWorldRayCastOutput*>") // class hkPadSpu< struct hkpWorldRayCastOutput* >
    HK_TRACKER_MEMBER(hkpFixedBufferRayHitCollector, m_nextFreeOutput, 0, "hkPadSpu<hkpWorldRayCastOutput*>") // class hkPadSpu< struct hkpWorldRayCastOutput* >
    HK_TRACKER_MEMBER(hkpFixedBufferRayHitCollector, m_collidableOnPpu, 0, "hkPadSpu<hkpCollidable*>") // class hkPadSpu< const class hkpCollidable* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpFixedBufferRayHitCollector, s_libraryName, hkpRayHitCollector)

#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuWorldGetClosestPointsJob.h>


// hkCpuWorldGetClosestPointsCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCpuWorldGetClosestPointsCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCpuWorldGetClosestPointsCollector)
    HK_TRACKER_MEMBER(hkCpuWorldGetClosestPointsCollector, m_filter, 0, "hkpGroupFilter*") // const class hkpGroupFilter*
    HK_TRACKER_MEMBER(hkCpuWorldGetClosestPointsCollector, m_castCollector, 0, "hkpFixedBufferCdPointCollector*") // class hkpFixedBufferCdPointCollector*
    HK_TRACKER_MEMBER(hkCpuWorldGetClosestPointsCollector, m_collidable, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkCpuWorldGetClosestPointsCollector, m_input, 0, "hkpLinearCastCollisionInput") // struct hkpLinearCastCollisionInput
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkCpuWorldGetClosestPointsCollector, s_libraryName, hkpBroadPhaseCastCollector)

#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobQueueUtils.h>


// hkpCollisionQueryJobQueueUtils ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCollisionQueryJobQueueUtils, s_libraryName)

#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobs.h>


// hkpCollisionQueryJobHeader ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCollisionQueryJobHeader, s_libraryName)


// hkpCollisionQueryJob ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCollisionQueryJob)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(JobSubType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCollisionQueryJob)
    HK_TRACKER_MEMBER(hkpCollisionQueryJob, m_semaphore, 0, "hkSemaphore*") // class hkSemaphore*
    HK_TRACKER_MEMBER(hkpCollisionQueryJob, m_sharedJobHeaderOnPpu, 0, "hkpCollisionQueryJobHeader*") // struct hkpCollisionQueryJobHeader*
    HK_TRACKER_MEMBER(hkpCollisionQueryJob, m_collisionInput, 0, "hkpProcessCollisionInput*") // const struct hkpProcessCollisionInput*
    HK_TRACKER_MEMBER(hkpCollisionQueryJob, m_jobDoneFlag, 0, "hkUint32*") // hkUint32*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCollisionQueryJob, s_libraryName, hkJob)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollisionQueryJob, JobSubType, s_libraryName)


// hkpPairLinearCastCommand ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPairLinearCastCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPairLinearCastCommand)
    HK_TRACKER_MEMBER(hkpPairLinearCastCommand, m_collidableA, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpPairLinearCastCommand, m_collidableB, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpPairLinearCastCommand, m_results, 0, "hkpRootCdPoint*") // struct hkpRootCdPoint*
    HK_TRACKER_MEMBER(hkpPairLinearCastCommand, m_startPointResults, 0, "hkpRootCdPoint*") // struct hkpRootCdPoint*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpPairLinearCastCommand, s_libraryName)


// hkpPairLinearCastJob ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPairLinearCastJob)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPairLinearCastJob)
    HK_TRACKER_MEMBER(hkpPairLinearCastJob, m_commandArray, 0, "hkpPairLinearCastCommand*") // const struct hkpPairLinearCastCommand*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPairLinearCastJob, s_libraryName, hkpCollisionQueryJob)


// hkpWorldLinearCastCommand ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldLinearCastCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldLinearCastCommand)
    HK_TRACKER_MEMBER(hkpWorldLinearCastCommand, m_collidable, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpWorldLinearCastCommand, m_results, 0, "hkpRootCdPoint*") // struct hkpRootCdPoint*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpWorldLinearCastCommand, s_libraryName)


// hkpWorldLinearCastJob ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldLinearCastJob)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldLinearCastJob)
    HK_TRACKER_MEMBER(hkpWorldLinearCastJob, m_commandArray, 0, "hkpWorldLinearCastCommand*") // const struct hkpWorldLinearCastCommand*
    HK_TRACKER_MEMBER(hkpWorldLinearCastJob, m_broadphase, 0, "hkpBroadPhase*") // const class hkpBroadPhase*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpWorldLinearCastJob, s_libraryName, hkpCollisionQueryJob)


// hkpMoppAabbCommand ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppAabbCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppAabbCommand)
    HK_TRACKER_MEMBER(hkpMoppAabbCommand, m_results, 0, "hkUint32 [2048]*") // hkUint32 [2048]*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpMoppAabbCommand, s_libraryName)


// hkpMoppAabbJob ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppAabbJob)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppAabbJob)
    HK_TRACKER_MEMBER(hkpMoppAabbJob, m_moppCodeData, 0, "hkUint8*") // const hkUint8*
    HK_TRACKER_MEMBER(hkpMoppAabbJob, m_commandArray, 0, "hkpMoppAabbCommand*") // const struct hkpMoppAabbCommand*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppAabbJob, s_libraryName, hkpCollisionQueryJob)


// hkpPairGetClosestPointsCommand ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPairGetClosestPointsCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPairGetClosestPointsCommand)
    HK_TRACKER_MEMBER(hkpPairGetClosestPointsCommand, m_collidableA, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpPairGetClosestPointsCommand, m_collidableB, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpPairGetClosestPointsCommand, m_results, 0, "hkpRootCdPoint*") // struct hkpRootCdPoint*
    HK_TRACKER_MEMBER(hkpPairGetClosestPointsCommand, m_indexIntoSharedResults, 0, "hkUint32*") // hkUint32*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpPairGetClosestPointsCommand, s_libraryName)


// hkpPairGetClosestPointsJob ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPairGetClosestPointsJob)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPairGetClosestPointsJob)
    HK_TRACKER_MEMBER(hkpPairGetClosestPointsJob, m_commandArray, 0, "hkpPairGetClosestPointsCommand*") // const struct hkpPairGetClosestPointsCommand*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPairGetClosestPointsJob, s_libraryName, hkpCollisionQueryJob)


// hkpWorldGetClosestPointsCommand ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldGetClosestPointsCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldGetClosestPointsCommand)
    HK_TRACKER_MEMBER(hkpWorldGetClosestPointsCommand, m_collidable, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpWorldGetClosestPointsCommand, m_results, 0, "hkpRootCdPoint*") // struct hkpRootCdPoint*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpWorldGetClosestPointsCommand, s_libraryName)


// hkpWorldGetClosestPointsJob ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldGetClosestPointsJob)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldGetClosestPointsJob)
    HK_TRACKER_MEMBER(hkpWorldGetClosestPointsJob, m_broadphase, 0, "hkpBroadPhase*") // const class hkpBroadPhase*
    HK_TRACKER_MEMBER(hkpWorldGetClosestPointsJob, m_commandArray, 0, "hkpWorldGetClosestPointsCommand*") // const struct hkpWorldGetClosestPointsCommand*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpWorldGetClosestPointsJob, s_libraryName, hkpCollisionQueryJob)

#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/Cpu/hkpCpuWorldRaycastJob.h>


// hkCpuWorldRayCastCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCpuWorldRayCastCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCpuWorldRayCastCollector)
    HK_TRACKER_MEMBER(hkCpuWorldRayCastCollector, m_filter, 0, "hkpCollisionFilter*") // const class hkpCollisionFilter*
    HK_TRACKER_MEMBER(hkCpuWorldRayCastCollector, m_command, 0, "hkpWorldRayCastCommand*") // struct hkpWorldRayCastCommand*
    HK_TRACKER_MEMBER(hkCpuWorldRayCastCollector, m_originalInput, 0, "hkpWorldRayCastInput*") // const struct hkpWorldRayCastInput*
    HK_TRACKER_MEMBER(hkCpuWorldRayCastCollector, m_result, 0, "hkpWorldRayCastOutput*") // struct hkpWorldRayCastOutput*
    HK_TRACKER_MEMBER(hkCpuWorldRayCastCollector, m_nextFreeResult, 0, "hkpWorldRayCastOutput*") // struct hkpWorldRayCastOutput*
    HK_TRACKER_MEMBER(hkCpuWorldRayCastCollector, m_workInput, 0, "hkpShapeRayCastInput") // struct hkpShapeRayCastInput
    HK_TRACKER_MEMBER(hkCpuWorldRayCastCollector, m_fixedBufferRayHitCollector, 0, "hkpFixedBufferRayHitCollector*") // class hkpFixedBufferRayHitCollector*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkCpuWorldRayCastCollector, s_libraryName, hkpBroadPhaseCastCollector)

#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/Util/hkpCollisionQueryUtil.h>


// hkpShapeRayCastJobUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeRayCastJobUtil, s_libraryName)


// hkpWorldRayCastJobUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpWorldRayCastJobUtil, s_libraryName)

#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/hkpRayCastQueryJobQueueUtils.h>


// hkpRayCastQueryJobQueueUtils ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpRayCastQueryJobQueueUtils, s_libraryName)

#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/hkpRayCastQueryJobs.h>


// hkpRayCastQueryJob ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRayCastQueryJob)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(JobSubType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRayCastQueryJob)
    HK_TRACKER_MEMBER(hkpRayCastQueryJob, m_semaphore, 0, "hkSemaphore*") // class hkSemaphore*
    HK_TRACKER_MEMBER(hkpRayCastQueryJob, m_sharedJobHeaderOnPpu, 0, "hkpCollisionQueryJobHeader*") // struct hkpCollisionQueryJobHeader*
    HK_TRACKER_MEMBER(hkpRayCastQueryJob, m_jobDoneFlag, 0, "hkUint32*") // hkUint32*
    HK_TRACKER_MEMBER(hkpRayCastQueryJob, m_collisionInput, 0, "hkpProcessCollisionInput*") // const struct hkpProcessCollisionInput*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRayCastQueryJob, s_libraryName, hkJob)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpRayCastQueryJob, JobSubType, s_libraryName)


// hkpShapeRayCastCommand ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeRayCastCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeRayCastCommand)
    HK_TRACKER_MEMBER(hkpShapeRayCastCommand, m_rayInput, 0, "hkpShapeRayCastInput") // struct hkpShapeRayCastInput
    HK_TRACKER_MEMBER(hkpShapeRayCastCommand, m_results, 0, "hkpWorldRayCastOutput*") // struct hkpWorldRayCastOutput*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpShapeRayCastCommand, s_libraryName)


// hkpShapeRayCastJob ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeRayCastJob)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeRayCastJob)
    HK_TRACKER_MEMBER(hkpShapeRayCastJob, m_commandArray, 0, "hkpShapeRayCastCommand*") // const struct hkpShapeRayCastCommand*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpShapeRayCastJob, s_libraryName, hkpRayCastQueryJob)


// hkpWorldRayCastCommand ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldRayCastCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldRayCastCommand)
    HK_TRACKER_MEMBER(hkpWorldRayCastCommand, m_results, 0, "hkpWorldRayCastOutput*") // struct hkpWorldRayCastOutput*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpWorldRayCastCommand, s_libraryName)


// hkpWorldRayCastBundleCommand ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldRayCastBundleCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldRayCastBundleCommand)
    HK_TRACKER_MEMBER(hkpWorldRayCastBundleCommand, m_results, 0, "hkpWorldRayCastOutput* [4]") // struct hkpWorldRayCastOutput* [4]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpWorldRayCastBundleCommand, s_libraryName)


// hkpWorldRayCastJob ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldRayCastJob)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldRayCastJob)
    HK_TRACKER_MEMBER(hkpWorldRayCastJob, m_commandArray, 0, "hkpWorldRayCastCommand*") // const struct hkpWorldRayCastCommand*
    HK_TRACKER_MEMBER(hkpWorldRayCastJob, m_bundleCommandArray, 0, "hkpWorldRayCastBundleCommand*") // const struct hkpWorldRayCastBundleCommand*
    HK_TRACKER_MEMBER(hkpWorldRayCastJob, m_broadphase, 0, "hkpBroadPhase*") // const class hkpBroadPhase*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpWorldRayCastJob, s_libraryName, hkpRayCastQueryJob)

#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>


// hkpListShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpListShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ChildInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ListShapeFlags)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpListShape)
    HK_TRACKER_MEMBER(hkpListShape, m_childInfo, 0, "hkArray<hkpListShape::ChildInfo, hkContainerHeapAllocator>") // hkArray< struct hkpListShape::ChildInfo, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpListShape, s_libraryName, hkpShapeCollection)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpListShape, ListShapeFlags, s_libraryName)


// ChildInfo hkpListShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpListShape::ChildInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpListShape::ChildInfo)
    HK_TRACKER_MEMBER(hkpListShape::ChildInfo, m_shape, 0, "hkpShape*") // const class hkpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpListShape::ChildInfo, s_libraryName)

#include <Physics2012/Collide/Shape/Compound/Collection/Mesh/hkpMeshMaterial.h>


// hkpMeshMaterial ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMeshMaterial, s_libraryName)

#include <Physics2012/Collide/Shape/Compound/Collection/Mesh/hkpNamedMeshMaterial.h>


// hkpNamedMeshMaterial ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpNamedMeshMaterial)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpNamedMeshMaterial)
    HK_TRACKER_MEMBER(hkpNamedMeshMaterial, m_name, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpNamedMeshMaterial, s_libraryName, hkpMeshMaterial)

#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>


// hkpShapeCollection ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeCollection)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CollectionType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeCollection)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpShapeCollection, s_libraryName, hkpShape)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpShapeCollection, CollectionType, s_libraryName)

#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/Modifiers/hkpRemoveTerminalsMoppModifier.h>


// hkpRemoveTerminalsMoppModifier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRemoveTerminalsMoppModifier)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRemoveTerminalsMoppModifier)
    HK_TRACKER_MEMBER(hkpRemoveTerminalsMoppModifier, m_removeInfo, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpRemoveTerminalsMoppModifier, m_tempShapesToRemove, 0, "hkArray<hkUint32, hkContainerHeapAllocator>*") // const hkArray< hkUint32, struct hkContainerHeapAllocator >*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRemoveTerminalsMoppModifier, s_libraryName, hkReferencedObject)


// hkpRemoveTerminalsMoppModifier2 ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRemoveTerminalsMoppModifier2)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRemoveTerminalsMoppModifier2)
    HK_TRACKER_MEMBER(hkpRemoveTerminalsMoppModifier2, m_enabledKeys, 0, "hkBitField*") // const class hkBitField*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRemoveTerminalsMoppModifier2, s_libraryName, hkpRemoveTerminalsMoppModifier)

#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>


// hkMoppBvTreeShapeBase ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMoppBvTreeShapeBase)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMoppBvTreeShapeBase)
    HK_TRACKER_MEMBER(hkMoppBvTreeShapeBase, m_code, 0, "hkpMoppCode*") // const class hkpMoppCode*
    HK_TRACKER_MEMBER(hkMoppBvTreeShapeBase, m_moppData, 0, "hkUint8*") // const hkUint8*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkMoppBvTreeShapeBase, s_libraryName, hkpBvTreeShape)


// hkpMoppBvTreeShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppBvTreeShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppBvTreeShape)
    HK_TRACKER_MEMBER(hkpMoppBvTreeShape, m_child, 0, "hkpSingleShapeContainer") // class hkpSingleShapeContainer
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppBvTreeShape, s_libraryName, hkMoppBvTreeShapeBase)

#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppCompilerInput.h>


// hkpMoppCompilerInput ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppCompilerInput, s_libraryName)

#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppUtility.h>


// hkpMoppUtility ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppUtility, s_libraryName)

#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>


// hkpBvTreeShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBvTreeShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BvTreeType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBvTreeShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpBvTreeShape, s_libraryName, hkpShape)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBvTreeShape, BvTreeType, s_libraryName)

#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>


// hkpBoxShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBoxShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBoxShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBoxShape, s_libraryName, hkpConvexShape)

#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>


// hkpCapsuleShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCapsuleShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(RayHitType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCapsuleShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCapsuleShape, s_libraryName, hkpConvexShape)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCapsuleShape, RayHitType, s_libraryName)

#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>


// hkpConvexTransformShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexTransformShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexTransformShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConvexTransformShape, s_libraryName, hkpConvexTransformShapeBase)

#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>


// hkpConvexTranslateShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexTranslateShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexTranslateShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConvexTranslateShape, s_libraryName, hkpConvexTransformShapeBase)

#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesConnectivity.h>


// hkpConvexVerticesConnectivity ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexVerticesConnectivity)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexVerticesConnectivity)
    HK_TRACKER_MEMBER(hkpConvexVerticesConnectivity, m_vertexIndices, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpConvexVerticesConnectivity, m_numVerticesPerFace, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConvexVerticesConnectivity, s_libraryName, hkReferencedObject)

#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesConnectivityUtil.h>


// hkpConvexVerticesConnectivityUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexVerticesConnectivityUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Edge)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FaceEdge)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VertexInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpConvexVerticesConnectivityUtil, s_libraryName)


// Edge hkpConvexVerticesConnectivityUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConvexVerticesConnectivityUtil, Edge, s_libraryName)


// FaceEdge hkpConvexVerticesConnectivityUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexVerticesConnectivityUtil::FaceEdge)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexVerticesConnectivityUtil::FaceEdge)
    HK_TRACKER_MEMBER(hkpConvexVerticesConnectivityUtil::FaceEdge, m_next, 0, "hkpConvexVerticesConnectivityUtil::FaceEdge*") // struct hkpConvexVerticesConnectivityUtil::FaceEdge*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpConvexVerticesConnectivityUtil::FaceEdge, s_libraryName)


// VertexInfo hkpConvexVerticesConnectivityUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConvexVerticesConnectivityUtil, VertexInfo, s_libraryName)

#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>


// hkpConvexVerticesShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexVerticesShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BuildConfig)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexVerticesShape)
    HK_TRACKER_MEMBER(hkpConvexVerticesShape, m_rotatedVertices, 0, "hkArray<hkFourTransposedPointsf, hkContainerHeapAllocator>") // hkArray< class hkFourTransposedPointsf, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpConvexVerticesShape, m_planeEquations, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpConvexVerticesShape, m_connectivity, 0, "hkpConvexVerticesConnectivity*") // const class hkpConvexVerticesConnectivity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConvexVerticesShape, s_libraryName, hkpConvexShape)


// BuildConfig hkpConvexVerticesShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConvexVerticesShape, BuildConfig, s_libraryName)

#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>


// hkpCylinderShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCylinderShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VertexIdEncoding)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCylinderShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCylinderShape, s_libraryName, hkpConvexShape)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCylinderShape, VertexIdEncoding, s_libraryName)

#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>


// hkpSphereShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSphereShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSphereShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSphereShape, s_libraryName, hkpConvexShape)

#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>


// hkpTriangleShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpTriangleShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpTriangleShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpTriangleShape, s_libraryName, hkpConvexShape)

#include <Physics2012/Collide/Shape/Convex/hkpConvexShape.h>


// hkpConvexShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WeldResult)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpConvexShape, s_libraryName, hkpSphereRepShape)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConvexShape, WeldResult, s_libraryName)


// hkpConvexTransformShapeBase ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexTransformShapeBase)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexTransformShapeBase)
    HK_TRACKER_MEMBER(hkpConvexTransformShapeBase, m_childShape, 0, "hkpSingleShapeContainer") // class hkpSingleShapeContainer
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpConvexTransformShapeBase, s_libraryName, hkpConvexShape)

#include <Physics2012/Collide/Shape/Deprecated/CompressedMesh/hkTjunctionDetector.h>


// hkTjunctionDetector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTjunctionDetector)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ProximityInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ProximityInfoEnum)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkTjunctionDetector, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTjunctionDetector, ProximityInfoEnum, s_libraryName)


// ProximityInfo hkTjunctionDetector
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTjunctionDetector, ProximityInfo, s_libraryName)

#include <Physics2012/Collide/Shape/Deprecated/CompressedMesh/hkpCompressedMeshShapeBuilder.h>


// hkpCompressedMeshShapeBuilder ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCompressedMeshShapeBuilder)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Statistics)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TriangleMapping)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MappingTree)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Subpart)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCompressedMeshShapeBuilder)
    HK_TRACKER_MEMBER(hkpCompressedMeshShapeBuilder, m_leftOver, 0, "hkGeometry") // struct hkGeometry
    HK_TRACKER_MEMBER(hkpCompressedMeshShapeBuilder, m_Tjunctions, 0, "hkArray<hkTjunctionDetector::ProximityInfo, hkContainerHeapAllocator>") // hkArray< struct hkTjunctionDetector::ProximityInfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShapeBuilder, m_weldedVertices, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShapeBuilder, m_bigMapping, 0, "hkArray<hkpCompressedMeshShapeBuilder::TriangleMapping, hkContainerHeapAllocator>") // hkArray< struct hkpCompressedMeshShapeBuilder::TriangleMapping, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShapeBuilder, subparts, 0, "hkArray<hkpCompressedMeshShapeBuilder::Subpart, hkContainerHeapAllocator>") // hkArray< struct hkpCompressedMeshShapeBuilder::Subpart, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShapeBuilder, m_shapeKeys, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShapeBuilder, m_geometry, 0, "hkGeometry") // struct hkGeometry
    HK_TRACKER_MEMBER(hkpCompressedMeshShapeBuilder, m_mesh, 0, "hkpCompressedMeshShape*") // class hkpCompressedMeshShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCompressedMeshShapeBuilder, s_libraryName)


// Statistics hkpCompressedMeshShapeBuilder
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCompressedMeshShapeBuilder, Statistics, s_libraryName)


// TriangleMapping hkpCompressedMeshShapeBuilder
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCompressedMeshShapeBuilder, TriangleMapping, s_libraryName)


// MappingTree hkpCompressedMeshShapeBuilder

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCompressedMeshShapeBuilder::MappingTree)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCompressedMeshShapeBuilder::MappingTree)
    HK_TRACKER_MEMBER(hkpCompressedMeshShapeBuilder::MappingTree, m_triangles, 0, "hkArray<hkpCompressedMeshShapeBuilder::TriangleMapping, hkContainerHeapAllocator>") // hkArray< struct hkpCompressedMeshShapeBuilder::TriangleMapping, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShapeBuilder::MappingTree, m_left, 0, "hkpCompressedMeshShapeBuilder::MappingTree*") // class hkpCompressedMeshShapeBuilder::MappingTree*
    HK_TRACKER_MEMBER(hkpCompressedMeshShapeBuilder::MappingTree, m_right, 0, "hkpCompressedMeshShapeBuilder::MappingTree*") // class hkpCompressedMeshShapeBuilder::MappingTree*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCompressedMeshShapeBuilder::MappingTree, s_libraryName)


// Subpart hkpCompressedMeshShapeBuilder
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCompressedMeshShapeBuilder, Subpart, s_libraryName)

#include <Physics2012/Collide/Shape/Deprecated/ConvexList/hkpConvexListShape.h>


// hkpConvexListShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexListShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexListShape)
    HK_TRACKER_MEMBER(hkpConvexListShape, m_childShapes, 0, "hkArray<hkpConvexShape*, hkContainerHeapAllocator>") // hkArray< const class hkpConvexShape*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConvexListShape, s_libraryName, hkpConvexShape)

#include <Physics2012/Collide/Shape/Deprecated/ConvexPieceMesh/hkpConvexPieceShape.h>


// hkpConvexPieceShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexPieceShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexPieceShape)
    HK_TRACKER_MEMBER(hkpConvexPieceShape, m_displayMesh, 0, "hkpShapeCollection*") // const class hkpShapeCollection*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConvexPieceShape, s_libraryName, hkpConvexShape)

#include <Physics2012/Collide/Shape/Deprecated/MultiSphere/hkpMultiSphereShape.h>


// hkpMultiSphereShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMultiSphereShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMultiSphereShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMultiSphereShape, s_libraryName, hkpSphereRepShape)

#include <Physics2012/Collide/Shape/HeightField/SampledHeightField/hkpSampledHeightFieldBaseCinfo.h>


// hkpSampledHeightFieldBaseCinfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSampledHeightFieldBaseCinfo, s_libraryName)

#include <Physics2012/Collide/Shape/HeightField/hkpSphereRepShape.h>


// hkpSphereRepShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSphereRepShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSphereRepShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSphereRepShape, s_libraryName, hkpShape)

#include <Physics2012/Collide/Shape/Misc/Bv/hkpBvShape.h>


// hkpBvShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBvShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBvShape)
    HK_TRACKER_MEMBER(hkpBvShape, m_boundingVolumeShape, 0, "hkpShape*") // const class hkpShape*
    HK_TRACKER_MEMBER(hkpBvShape, m_childShape, 0, "hkpSingleShapeContainer") // class hkpSingleShapeContainer
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBvShape, s_libraryName, hkpShape)

#include <Physics2012/Collide/Shape/Misc/MultiRay/hkpMultiRayShape.h>


// hkpMultiRayShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMultiRayShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Ray)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMultiRayShape)
    HK_TRACKER_MEMBER(hkpMultiRayShape, m_rays, 0, "hkArray<hkpMultiRayShape::Ray, hkContainerHeapAllocator>") // hkArray< struct hkpMultiRayShape::Ray, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMultiRayShape, s_libraryName, hkpShape)


// Ray hkpMultiRayShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMultiRayShape, Ray, s_libraryName)

#include <Physics2012/Collide/Shape/Misc/PhantomCallback/hkpPhantomCallbackShape.h>


// hkpPhantomCallbackShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPhantomCallbackShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPhantomCallbackShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpPhantomCallbackShape, s_libraryName, hkpShape)

#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>


// hkpTransformShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpTransformShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpTransformShape)
    HK_TRACKER_MEMBER(hkpTransformShape, m_childShape, 0, "hkpSingleShapeContainer") // class hkpSingleShapeContainer
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpTransformShape, s_libraryName, hkpShape)

#include <Physics2012/Collide/Shape/Query/hkpAabbCastCollector.h>


// hkpAabbCastCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAabbCastCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpAabbCastCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkpAabbCastCollector, s_libraryName)

#include <Physics2012/Collide/Shape/Query/hkpRayHitCollector.h>

// hk.MemoryTracker ignore hkpRayHitCollector
#include <Physics2012/Collide/Shape/Query/hkpRayShapeCollectionFilter.h>

// hk.MemoryTracker ignore hkpRayShapeCollectionFilter
#include <Physics2012/Collide/Shape/Query/hkpShapeRayBundleCastInput.h>


// hkpShapeRayBundleCastInput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeRayBundleCastInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeRayBundleCastInput)
    HK_TRACKER_MEMBER(hkpShapeRayBundleCastInput, m_rayShapeCollectionFilter, 0, "hkpRayShapeCollectionFilter*") // const class hkpRayShapeCollectionFilter*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpShapeRayBundleCastInput, s_libraryName)

#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastCollectorOutput.h>


// hkpShapeRayCastCollectorOutput ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeRayCastCollectorOutput, s_libraryName)

#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>


// hkpShapeRayCastInput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeRayCastInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeRayCastInput)
    HK_TRACKER_MEMBER(hkpShapeRayCastInput, m_rayShapeCollectionFilter, 0, "hkpRayShapeCollectionFilter*") // const class hkpRayShapeCollectionFilter*
    HK_TRACKER_MEMBER(hkpShapeRayCastInput, m_collidable, 0, "hkpCollidable*") // const class hkpCollidable*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpShapeRayCastInput, s_libraryName)

#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastOutput.h>


// hkpShapeRayCastOutput ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeRayCastOutput, s_libraryName)


// hkpShapeRayBundleCastOutput ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeRayBundleCastOutput, s_libraryName)

#include <Physics2012/Collide/Shape/hkpShape.h>


// hkpShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CalcSizeForSpuInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpShape, s_libraryName, hkpShapeBase)


// CalcSizeForSpuInput hkpShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpShape, CalcSizeForSpuInput, s_libraryName)


// hkpShapeKeyPair ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeKeyPair, s_libraryName)

// hk.MemoryTracker ignore hkpShapeModifier
#include <Physics2012/Collide/Shape/hkpShapeBase.h>


// hkpShapeBase ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeBase)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeBase)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpShapeBase, s_libraryName, hkcdShape)

#include <Physics2012/Collide/Shape/hkpShapeBuffer.h>

// None hkpBufferSize
HK_TRACKER_IMPLEMENT_SIMPLE(hkpBufferSize, s_libraryName)
#include <Physics2012/Collide/Shape/hkpShapeContainer.h>


// hkpShapeContainer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeContainer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ReferencePolicy)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeContainer)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkpShapeContainer, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpShapeContainer, ReferencePolicy, s_libraryName)


// hkpSingleShapeContainer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSingleShapeContainer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSingleShapeContainer)
    HK_TRACKER_MEMBER(hkpSingleShapeContainer, m_childShape, 0, "hkpShape*") // const class hkpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSingleShapeContainer, s_libraryName, hkpShapeContainer)

#include <Physics2012/Collide/Util/ShapeCutter/hkpShapeCutterUtil.h>


// hkpShapeConnectedCalculator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeConnectedCalculator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeConnectedCalculator)
    HK_TRACKER_MEMBER(hkpShapeConnectedCalculator, m_dispatcher, 0, "hkpCollisionDispatcher*") // class hkpCollisionDispatcher*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpShapeConnectedCalculator, s_libraryName, hkReferencedObject)


// hkpShapeCutterUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeCutterUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SubtractShapeInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IntersectShapeInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConvexShapeCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeCutterUtil, s_libraryName)


// SubtractShapeInput hkpShapeCutterUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeCutterUtil::SubtractShapeInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeCutterUtil::SubtractShapeInput)
    HK_TRACKER_MEMBER(hkpShapeCutterUtil::SubtractShapeInput, m_dispatcher, 0, "hkpCollisionDispatcher*") // class hkpCollisionDispatcher*
    HK_TRACKER_MEMBER(hkpShapeCutterUtil::SubtractShapeInput, m_subtractShape, 0, "hkpConvexVerticesShape*") // const class hkpConvexVerticesShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpShapeCutterUtil::SubtractShapeInput, s_libraryName)


// IntersectShapeInput hkpShapeCutterUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpShapeCutterUtil, IntersectShapeInput, s_libraryName)


// ConvexShapeCollector hkpShapeCutterUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeCutterUtil::ConvexShapeCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeCutterUtil::ConvexShapeCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkpShapeCutterUtil::ConvexShapeCollector, s_libraryName)

#include <Physics2012/Collide/Util/ShapeDepth/hkpShapeDepthUtil.h>


// hkpShapeDepthUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeDepthUtil, s_libraryName)

#include <Physics2012/Collide/Util/ShapeInfo/hkpShapeInfo.h>


// hkpShapeInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeInfo)
    HK_TRACKER_MEMBER(hkpShapeInfo, m_shape, 0, "hkpShape *") // const class hkpShape *
    HK_TRACKER_MEMBER(hkpShapeInfo, m_childShapeNames, 0, "hkArray<hkStringPtr, hkContainerHeapAllocator>") // hkArray< hkStringPtr, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpShapeInfo, m_childTransforms, 0, "hkArray<hkTransformf, hkContainerHeapAllocator>") // hkArray< hkTransformf, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpShapeInfo, s_libraryName, hkReferencedObject)

#include <Physics2012/Collide/Util/ShapeShrinker/hkpShapeShrinker.h>


// hkpShapeShrinker ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeShrinker)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ShapePair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeShrinker, s_libraryName)


// ShapePair hkpShapeShrinker

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeShrinker::ShapePair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeShrinker::ShapePair)
    HK_TRACKER_MEMBER(hkpShapeShrinker::ShapePair, originalShape, 0, "hkpShape*") // class hkpShape*
    HK_TRACKER_MEMBER(hkpShapeShrinker::ShapePair, newShape, 0, "hkpShape*") // class hkpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpShapeShrinker::ShapePair, s_libraryName)

#include <Physics2012/Collide/Util/ShapeSkinning/hkpShapeSkinningUtil.h>


// hkpShapeSkinningUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeSkinningUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Input)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeSkinningUtil, s_libraryName)


// Input hkpShapeSkinningUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeSkinningUtil::Input)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeSkinningUtil::Input)
    HK_TRACKER_MEMBER(hkpShapeSkinningUtil::Input, m_collisionDispatcher, 0, "hkpCollisionDispatcher*") // class hkpCollisionDispatcher*
    HK_TRACKER_MEMBER(hkpShapeSkinningUtil::Input, m_shapes, 0, "hkpShape**") // const class hkpShape**
    HK_TRACKER_MEMBER(hkpShapeSkinningUtil::Input, m_transforms, 0, "hkTransform*") // const hkTransform*
    HK_TRACKER_MEMBER(hkpShapeSkinningUtil::Input, m_vertexPositions, 0, "hkVector4*") // hkVector4*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpShapeSkinningUtil::Input, s_libraryName)

#include <Physics2012/Collide/Util/ShapeVirtualTable/hkpShapeVirtualTableUtil.h>


// hkpShapeVirtualTableUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeVirtualTableUtil, s_libraryName)

#include <Physics2012/Collide/Util/Welding/hkpMeshWeldingUtility.h>


// hkpMeshWeldingUtility ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMeshWeldingUtility)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ShapeInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WindingConsistency)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpMeshWeldingUtility, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMeshWeldingUtility, WindingConsistency, s_libraryName)


// ShapeInfo hkpMeshWeldingUtility

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMeshWeldingUtility::ShapeInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMeshWeldingUtility::ShapeInfo)
    HK_TRACKER_MEMBER(hkpMeshWeldingUtility::ShapeInfo, m_shape, 0, "hkpBvTreeShape*") // const class hkpBvTreeShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpMeshWeldingUtility::ShapeInfo, s_libraryName)

#include <Physics2012/Collide/Util/Welding/hkpWeldingUtility.h>


// hkpWeldingUtility ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWeldingUtility)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SinCosTableEntry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WeldingType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SectorType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(NumAngles)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpWeldingUtility, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWeldingUtility, WeldingType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWeldingUtility, SectorType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWeldingUtility, NumAngles, s_libraryName)


// SinCosTableEntry hkpWeldingUtility
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWeldingUtility, SinCosTableEntry, s_libraryName)

#include <Physics2012/Collide/Util/hkpCollideTriangleUtil.h>


// hkpFeatureOutput ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpFeatureOutput, s_libraryName)


// hkpCollideTriangleUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCollideTriangleUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClosestPointTriangleCache)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClosestPointTriangleResult)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PointTriangleDistanceCache)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClosestPointTriangleStatus)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpCollideTriangleUtil, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollideTriangleUtil, ClosestPointTriangleStatus, s_libraryName)


// ClosestPointTriangleCache hkpCollideTriangleUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollideTriangleUtil, ClosestPointTriangleCache, s_libraryName)


// ClosestPointTriangleResult hkpCollideTriangleUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollideTriangleUtil, ClosestPointTriangleResult, s_libraryName)


// PointTriangleDistanceCache hkpCollideTriangleUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollideTriangleUtil, PointTriangleDistanceCache, s_libraryName)

#include <Physics2012/Collide/Util/hkpTriangleUtil.h>


// hkpTriangleUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpTriangleUtil, s_libraryName)

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
