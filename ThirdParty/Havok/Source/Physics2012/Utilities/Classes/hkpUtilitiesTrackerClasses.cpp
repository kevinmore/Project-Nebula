/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Common/Base/hkBase.h>
static const char s_libraryName[] = "hkpUtilities";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpUtilitiesRegister() {}

#include <Physics2012/Utilities/Actions/AngularDashpot/hkpAngularDashpotAction.h>


// hkpAngularDashpotAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAngularDashpotAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpAngularDashpotAction)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpAngularDashpotAction, s_libraryName, hkpBinaryAction)

#include <Physics2012/Utilities/Actions/Dashpot/hkpDashpotAction.h>


// hkpDashpotAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDashpotAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDashpotAction)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDashpotAction, s_libraryName, hkpBinaryAction)

#include <Physics2012/Utilities/Actions/EaseConstraints/hkpEaseConstraintsAction.h>


// hkpEaseConstraintsAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpEaseConstraintsAction)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CollectSupportedConstraints)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpEaseConstraintsAction)
    HK_TRACKER_MEMBER(hkpEaseConstraintsAction, m_originalConstraints, 0, "hkArray<hkpConstraintInstance*, hkContainerHeapAllocator>") // hkArray< class hkpConstraintInstance*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpEaseConstraintsAction, m_originalLimits, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpEaseConstraintsAction, s_libraryName, hkpArrayAction)


// CollectSupportedConstraints hkpEaseConstraintsAction

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpEaseConstraintsAction::CollectSupportedConstraints)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpEaseConstraintsAction::CollectSupportedConstraints)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpEaseConstraintsAction::CollectSupportedConstraints, s_libraryName, hkpConstraintUtils::CollectConstraintsFilter)

#include <Physics2012/Utilities/Actions/EasePenetration/hkpEasePenetrationAction.h>


// hkpEasePenetrationAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpEasePenetrationAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpEasePenetrationAction)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpEasePenetrationAction, s_libraryName, hkpUnaryAction)

#include <Physics2012/Utilities/Actions/Motor/hkpMotorAction.h>


// hkpMotorAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMotorAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMotorAction)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMotorAction, s_libraryName, hkpUnaryAction)

#include <Physics2012/Utilities/Actions/MouseSpring/hkpMouseSpringAction.h>


// hkpMouseSpringAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMouseSpringAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMouseSpringAction)
    HK_TRACKER_MEMBER(hkpMouseSpringAction, m_applyCallbacks, 0, "hkArray<void*, hkContainerHeapAllocator>") // hkArray< void*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMouseSpringAction, s_libraryName, hkpUnaryAction)

#include <Physics2012/Utilities/Actions/Reorient/hkpReorientAction.h>


// hkpReorientAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpReorientAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpReorientAction)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpReorientAction, s_libraryName, hkpUnaryAction)

#include <Physics2012/Utilities/Actions/Spring/hkpSpringAction.h>


// hkpSpringAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSpringAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSpringAction)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSpringAction, s_libraryName, hkpBinaryAction)

#include <Physics2012/Utilities/Actions/Wind/hkpPrevailingWind.h>


// hkpPrevailingWind ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPrevailingWind)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Oscillator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Triple)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPrevailingWind)
    HK_TRACKER_MEMBER(hkpPrevailingWind, m_oscillators, 0, "hkArray<hkpPrevailingWind::Triple, hkContainerHeapAllocator>") // hkArray< struct hkpPrevailingWind::Triple, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPrevailingWind, s_libraryName, hkpWind)


// Oscillator hkpPrevailingWind

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPrevailingWind::Oscillator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPrevailingWind::Oscillator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpPrevailingWind::Oscillator, s_libraryName)


// Triple hkpPrevailingWind
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpPrevailingWind, Triple, s_libraryName)

#include <Physics2012/Utilities/Actions/Wind/hkpWind.h>


// hkpWind ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWind)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWind)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpWind, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/Actions/Wind/hkpWindAction.h>


// hkpWindAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWindAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWindAction)
    HK_TRACKER_MEMBER(hkpWindAction, m_wind, 0, "hkpWind*") // const class hkpWind*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpWindAction, s_libraryName, hkpUnaryAction)

#include <Physics2012/Utilities/Actions/Wind/hkpWindRegion.h>


// hkpWindRegion ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWindRegion)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWindRegion)
    HK_TRACKER_MEMBER(hkpWindRegion, m_phantom, 0, "hkpAabbPhantom*") // class hkpAabbPhantom*
    HK_TRACKER_MEMBER(hkpWindRegion, m_wind, 0, "hkpWind*") // const const class hkpWind*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpWindRegion, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/Cpu/hkpCpuCharacterProxyCollector.h>


// hkpCpuCharacterProxyCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCpuCharacterProxyCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCpuCharacterProxyCollector)
    HK_TRACKER_MEMBER(hkpCpuCharacterProxyCollector, m_charactersCollidable, 0, "hkpCollidable*") // const class hkpCollidable*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCpuCharacterProxyCollector, s_libraryName, hkpAllCdPointCollector)

#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/Cpu/hkpCpuCharacterProxyUtil.h>


// hkpCpuCharacterProxyUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCpuCharacterProxyUtil, s_libraryName)

#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/Util/hkpCharacterProxyJobUtil.h>


// hkpCharacterProxyJobUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterProxyJobUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(JobData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CharacterJobBatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpCharacterProxyJobUtil, s_libraryName)


// JobData hkpCharacterProxyJobUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterProxyJobUtil::JobData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterProxyJobUtil::JobData)
    HK_TRACKER_MEMBER(hkpCharacterProxyJobUtil::JobData, m_jobQueue, 0, "hkJobQueue*") // class hkJobQueue*
    HK_TRACKER_MEMBER(hkpCharacterProxyJobUtil::JobData, m_world, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpCharacterProxyJobUtil::JobData, m_collisionInput, 0, "hkpProcessCollisionInput*") // struct hkpProcessCollisionInput*
    HK_TRACKER_MEMBER(hkpCharacterProxyJobUtil::JobData, m_characters, 0, "hkArray<hkpCharacterProxy*, hkContainerHeapAllocator>*") // hkArray< class hkpCharacterProxy*, struct hkContainerHeapAllocator >*
    HK_TRACKER_MEMBER(hkpCharacterProxyJobUtil::JobData, m_broadphase, 0, "hkpBroadPhase*") // class hkpBroadPhase*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterProxyJobUtil::JobData, s_libraryName, hkReferencedObject)


// CharacterJobBatch hkpCharacterProxyJobUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterProxyJobUtil::CharacterJobBatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterProxyJobUtil::CharacterJobBatch)
    HK_TRACKER_MEMBER(hkpCharacterProxyJobUtil::CharacterJobBatch, m_commandStorage, 0, "hkpCharacterProxyIntegrateCommand*") // struct hkpCharacterProxyIntegrateCommand*
    HK_TRACKER_MEMBER(hkpCharacterProxyJobUtil::CharacterJobBatch, m_characterStorage, 0, "hkpCharacterProxy*") // class hkpCharacterProxy*
    HK_TRACKER_MEMBER(hkpCharacterProxyJobUtil::CharacterJobBatch, m_collidableStorage, 0, "hkpCollidable*") // class hkpCollidable*
    HK_TRACKER_MEMBER(hkpCharacterProxyJobUtil::CharacterJobBatch, m_objectInteractionStorage, 0, "hkpCharacterProxyInteractionResults*") // struct hkpCharacterProxyInteractionResults*
    HK_TRACKER_MEMBER(hkpCharacterProxyJobUtil::CharacterJobBatch, m_triggerVolumeStorage, 0, "hkpTriggerVolume**") // class hkpTriggerVolume**
    HK_TRACKER_MEMBER(hkpCharacterProxyJobUtil::CharacterJobBatch, m_jobHeaders, 0, "hkpCharacterProxyJobHeader*") // struct hkpCharacterProxyJobHeader*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCharacterProxyJobUtil::CharacterJobBatch, s_libraryName)

#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/hkpCharacterProxyJobQueueUtils.h>


// hkpCharacterProxyJobQueueUtils ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCharacterProxyJobQueueUtils, s_libraryName)

#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/hkpCharacterProxyJobs.h>


// hkpCharacterProxyJobHeader ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCharacterProxyJobHeader, s_libraryName)


// hkpCharacterProxyJob ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterProxyJob)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(JobSubType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterProxyJob)
    HK_TRACKER_MEMBER(hkpCharacterProxyJob, m_semaphore, 0, "hkSemaphore*") // class hkSemaphore*
    HK_TRACKER_MEMBER(hkpCharacterProxyJob, m_sharedJobHeaderOnPpu, 0, "hkpCharacterProxyJobHeader*") // struct hkpCharacterProxyJobHeader*
    HK_TRACKER_MEMBER(hkpCharacterProxyJob, m_jobDoneFlag, 0, "hkUint32*") // hkUint32*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterProxyJob, s_libraryName, hkJob)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCharacterProxyJob, JobSubType, s_libraryName)


// hkpCharacterProxyInteractionResults ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterProxyInteractionResults)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterProxyInteractionResults)
    HK_TRACKER_MEMBER(hkpCharacterProxyInteractionResults, m_collidingBody, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterProxyInteractionResults, s_libraryName, hkpCharacterObjectInteractionResult)


// hkpCharacterProxyIntegrateCommand ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterProxyIntegrateCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterProxyIntegrateCommand)
    HK_TRACKER_MEMBER(hkpCharacterProxyIntegrateCommand, m_character, 0, "hkpCharacterProxy*") // class hkpCharacterProxy*
    HK_TRACKER_MEMBER(hkpCharacterProxyIntegrateCommand, m_collidable, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpCharacterProxyIntegrateCommand, m_objectInteraction, 0, "hkpCharacterProxyInteractionResults*") // struct hkpCharacterProxyInteractionResults*
    HK_TRACKER_MEMBER(hkpCharacterProxyIntegrateCommand, m_triggerVolumeAndFlags, 0, "hkpTriggerVolume**") // class hkpTriggerVolume**
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCharacterProxyIntegrateCommand, s_libraryName)


// hkpCharacterProxyIntegrateJob ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterProxyIntegrateJob)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterProxyIntegrateJob)
    HK_TRACKER_MEMBER(hkpCharacterProxyIntegrateJob, m_commandArray, 0, "hkpCharacterProxyIntegrateCommand*") // const struct hkpCharacterProxyIntegrateCommand*
    HK_TRACKER_MEMBER(hkpCharacterProxyIntegrateJob, m_collisionInput, 0, "hkpProcessCollisionInput*") // const struct hkpProcessCollisionInput*
    HK_TRACKER_MEMBER(hkpCharacterProxyIntegrateJob, m_broadphase, 0, "hkpBroadPhase*") // const class hkpBroadPhase*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterProxyIntegrateJob, s_libraryName, hkpCharacterProxyJob)

#include <Physics2012/Utilities/CharacterControl/CharacterProxy/hkpCharacterProxy.h>


// hkpCharacterProxy ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterProxy)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterProxy)
    HK_TRACKER_MEMBER(hkpCharacterProxy, m_manifold, 0, "hkArray<hkpRootCdPoint, hkContainerHeapAllocator>") // hkArray< struct hkpRootCdPoint, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCharacterProxy, m_bodies, 0, "hkArray<hkpRigidBody*, hkContainerHeapAllocator>") // hkArray< class hkpRigidBody*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCharacterProxy, m_phantoms, 0, "hkArray<hkpPhantom*, hkContainerHeapAllocator>") // hkArray< class hkpPhantom*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCharacterProxy, m_overlappingTriggerVolumes, 0, "hkArray<hkpTriggerVolume*, hkContainerHeapAllocator>") // hkArray< class hkpTriggerVolume*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCharacterProxy, m_shapePhantom, 0, "hkpShapePhantom*") // class hkpShapePhantom*
    HK_TRACKER_MEMBER(hkpCharacterProxy, m_listeners, 0, "hkArray<hkpCharacterProxyListener*, hkContainerHeapAllocator>") // hkArray< class hkpCharacterProxyListener*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterProxy, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/CharacterControl/CharacterProxy/hkpCharacterProxyCinfo.h>


// hkpCharacterProxyCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterProxyCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterProxyCinfo)
    HK_TRACKER_MEMBER(hkpCharacterProxyCinfo, m_shapePhantom, 0, "hkpShapePhantom*") // class hkpShapePhantom*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterProxyCinfo, s_libraryName, hkpCharacterControllerCinfo)

#include <Physics2012/Utilities/CharacterControl/CharacterProxy/hkpCharacterProxyListener.h>


// hkpCharacterObjectInteractionEvent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterObjectInteractionEvent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterObjectInteractionEvent)
    HK_TRACKER_MEMBER(hkpCharacterObjectInteractionEvent, m_body, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCharacterObjectInteractionEvent, s_libraryName)


// hkpCharacterObjectInteractionResult ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCharacterObjectInteractionResult, s_libraryName)


// hkpCharacterProxyListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterProxyListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterProxyListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCharacterProxyListener, s_libraryName)

#include <Physics2012/Utilities/CharacterControl/CharacterRigidBody/hkpCharacterRigidBody.h>


// hkpCharacterRigidBody ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterRigidBody)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SupportInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VertPointInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterRigidBody)
    HK_TRACKER_MEMBER(hkpCharacterRigidBody, m_character, 0, "hkpRigidBody*") // class hkpRigidBody*
    HK_TRACKER_MEMBER(hkpCharacterRigidBody, m_listener, 0, "hkpCharacterRigidBodyListener*") // class hkpCharacterRigidBodyListener*
    HK_TRACKER_MEMBER(hkpCharacterRigidBody, m_verticalContactPoints, 0, "hkArray<hkpCharacterRigidBody::VertPointInfo, hkContainerHeapAllocator>") // hkArray< struct hkpCharacterRigidBody::VertPointInfo, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterRigidBody, s_libraryName, hkReferencedObject)


// SupportInfo hkpCharacterRigidBody

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterRigidBody::SupportInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterRigidBody::SupportInfo)
    HK_TRACKER_MEMBER(hkpCharacterRigidBody::SupportInfo, m_rigidBody, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCharacterRigidBody::SupportInfo, s_libraryName)


// VertPointInfo hkpCharacterRigidBody

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterRigidBody::VertPointInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterRigidBody::VertPointInfo)
    HK_TRACKER_MEMBER(hkpCharacterRigidBody::VertPointInfo, m_mgr, 0, "hkpSimpleConstraintContactMgr*") // class hkpSimpleConstraintContactMgr*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCharacterRigidBody::VertPointInfo, s_libraryName)

#include <Physics2012/Utilities/CharacterControl/CharacterRigidBody/hkpCharacterRigidBodyCinfo.h>


// hkpCharacterRigidBodyCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterRigidBodyCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterRigidBodyCinfo)
    HK_TRACKER_MEMBER(hkpCharacterRigidBodyCinfo, m_shape, 0, "hkpShape*") // class hkpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterRigidBodyCinfo, s_libraryName, hkpCharacterControllerCinfo)

#include <Physics2012/Utilities/CharacterControl/CharacterRigidBody/hkpCharacterRigidBodyListener.h>


// hkpCharacterRigidBodyListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterRigidBodyListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterRigidBodyListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterRigidBodyListener, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/CharacterControl/FirstPersonCharacter/hkpFirstPersonCharacter.h>


// hkpFirstPersonCharacter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpFirstPersonCharacter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CharacterControls)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ControlFlags)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpFirstPersonCharacter)
    HK_TRACKER_MEMBER(hkpFirstPersonCharacter, m_world, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpFirstPersonCharacter, m_characterRb, 0, "hkpCharacterRigidBody*") // class hkpCharacterRigidBody*
    HK_TRACKER_MEMBER(hkpFirstPersonCharacter, m_characterRbContext, 0, "hkpCharacterContext*") // class hkpCharacterContext*
    HK_TRACKER_MEMBER(hkpFirstPersonCharacter, m_currentGun, 0, "hkpFirstPersonGun*") // class hkpFirstPersonGun*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpFirstPersonCharacter, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpFirstPersonCharacter, ControlFlags, s_libraryName)


// CharacterControls hkpFirstPersonCharacter
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpFirstPersonCharacter, CharacterControls, s_libraryName)


// hkpFirstPersonCharacterCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpFirstPersonCharacterCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpFirstPersonCharacterCinfo)
    HK_TRACKER_MEMBER(hkpFirstPersonCharacterCinfo, m_characterRb, 0, "hkpCharacterRigidBody*") // class hkpCharacterRigidBody*
    HK_TRACKER_MEMBER(hkpFirstPersonCharacterCinfo, m_context, 0, "hkpCharacterContext*") // class hkpCharacterContext*
    HK_TRACKER_MEMBER(hkpFirstPersonCharacterCinfo, m_world, 0, "hkpWorld*") // class hkpWorld*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpFirstPersonCharacterCinfo, s_libraryName)

#include <Physics2012/Utilities/CharacterControl/StateMachine/Climbing/hkpCharacterStateClimbing.h>


// hkpCharacterStateClimbing ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterStateClimbing)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterStateClimbing)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterStateClimbing, s_libraryName, hkpCharacterState)

#include <Physics2012/Utilities/CharacterControl/StateMachine/Flying/hkpCharacterStateFlying.h>


// hkpCharacterStateFlying ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterStateFlying)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterStateFlying)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterStateFlying, s_libraryName, hkpCharacterState)

#include <Physics2012/Utilities/CharacterControl/StateMachine/InAir/hkpCharacterStateInAir.h>


// hkpCharacterStateInAir ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterStateInAir)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterStateInAir)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterStateInAir, s_libraryName, hkpCharacterState)

#include <Physics2012/Utilities/CharacterControl/StateMachine/Jumping/hkpCharacterStateJumping.h>


// hkpCharacterStateJumping ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterStateJumping)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterStateJumping)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterStateJumping, s_libraryName, hkpCharacterState)

#include <Physics2012/Utilities/CharacterControl/StateMachine/OnGround/hkpCharacterStateOnGround.h>


// hkpCharacterStateOnGround ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterStateOnGround)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterStateOnGround)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterStateOnGround, s_libraryName, hkpCharacterState)

#include <Physics2012/Utilities/CharacterControl/StateMachine/Util/hkpCharacterMovementUtil.h>


// hkpCharacterMovementUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterMovementUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpMovementUtilInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpCharacterMovementUtil, s_libraryName)


// hkpMovementUtilInput hkpCharacterMovementUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCharacterMovementUtil, hkpMovementUtilInput, s_libraryName)

#include <Physics2012/Utilities/CharacterControl/StateMachine/hkpCharacterContext.h>


// hkpCharacterInput ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCharacterInput, s_libraryName)


// hkpCharacterOutput ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCharacterOutput, s_libraryName)


// hkpCharacterContext ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterContext)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CharacterType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterContext)
    HK_TRACKER_MEMBER(hkpCharacterContext, m_stateManager, 0, "hkpCharacterStateManager*") // const class hkpCharacterStateManager*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterContext, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCharacterContext, CharacterType, s_libraryName)

#include <Physics2012/Utilities/CharacterControl/StateMachine/hkpCharacterState.h>


// hkpCharacterState ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterState)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterState)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpCharacterState, s_libraryName, hkReferencedObject)

// None hkpCharacterStateType
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCharacterStateType, s_libraryName)
#include <Physics2012/Utilities/CharacterControl/StateMachine/hkpCharacterStateManager.h>


// hkpCharacterStateManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterStateManager)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterStateManager)
    HK_TRACKER_MEMBER(hkpCharacterStateManager, m_registeredState, 0, "hkpCharacterState* [11]") // class hkpCharacterState* [11]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterStateManager, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/CharacterControl/hkpCharacterControl.h>


// hkpSurfaceInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSurfaceInfo, s_libraryName)


// hkpSurfaceInfoDeprecated ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSurfaceInfoDeprecated, s_libraryName)

#include <Physics2012/Utilities/CharacterControl/hkpCharacterControllerCinfo.h>


// hkpCharacterControllerCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterControllerCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterControllerCinfo)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterControllerCinfo, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/Collide/ContactModifiers/CenterOfMassChanger/hkpCenterOfMassChangerUtil.h>


// hkpCenterOfMassChangerUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCenterOfMassChangerUtil)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCenterOfMassChangerUtil)
    HK_TRACKER_MEMBER(hkpCenterOfMassChangerUtil, m_bodyA, 0, "hkpRigidBody*") // class hkpRigidBody*
    HK_TRACKER_MEMBER(hkpCenterOfMassChangerUtil, m_bodyB, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCenterOfMassChangerUtil, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/Collide/ContactModifiers/MassChanger/hkpCollisionMassChangerUtil.h>


// hkpCollisionMassChangerUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCollisionMassChangerUtil)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCollisionMassChangerUtil)
    HK_TRACKER_MEMBER(hkpCollisionMassChangerUtil, m_bodyA, 0, "hkpRigidBody*") // class hkpRigidBody*
    HK_TRACKER_MEMBER(hkpCollisionMassChangerUtil, m_bodyB, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCollisionMassChangerUtil, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/Collide/ContactModifiers/SoftContact/hkpSoftContactUtil.h>


// hkpSoftContactUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSoftContactUtil)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSoftContactUtil)
    HK_TRACKER_MEMBER(hkpSoftContactUtil, m_bodyA, 0, "hkpRigidBody*") // class hkpRigidBody*
    HK_TRACKER_MEMBER(hkpSoftContactUtil, m_bodyB, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSoftContactUtil, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/Collide/ContactModifiers/SurfaceVelocity/Filtered/hkpFilteredSurfaceVelocityUtil.h>


// hkpFilteredSurfaceVelocityUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpFilteredSurfaceVelocityUtil)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpFilteredSurfaceVelocityUtil)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpFilteredSurfaceVelocityUtil, s_libraryName, hkpSurfaceVelocityUtil)

#include <Physics2012/Utilities/Collide/ContactModifiers/SurfaceVelocity/hkpSurfaceVelocityUtil.h>


// hkpSurfaceVelocityUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSurfaceVelocityUtil)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSurfaceVelocityUtil)
    HK_TRACKER_MEMBER(hkpSurfaceVelocityUtil, m_rigidBody, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSurfaceVelocityUtil, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/Collide/ContactModifiers/ViscoseSurface/hkpViscoseSurfaceUtil.h>


// hkpViscoseSurfaceUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpViscoseSurfaceUtil)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpViscoseSurfaceUtil)
    HK_TRACKER_MEMBER(hkpViscoseSurfaceUtil, m_entity, 0, "hkpEntity*") // class hkpEntity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpViscoseSurfaceUtil, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/Collide/Filter/GroupFilter/hkpGroupFilterUtil.h>


// hkpGroupFilterUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpGroupFilterUtil, s_libraryName)

#include <Physics2012/Utilities/Collide/RemoveContact/hkpRemoveContactUtil.h>


// hkpRemoveContactUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpRemoveContactUtil, s_libraryName)

#include <Physics2012/Utilities/Collide/ShapeUtils/CollapseTransform/hkpTransformCollapseUtil.h>


// hkpTransformCollapseUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpTransformCollapseUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Options)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Results)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SharedShapeBehaviour)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpTransformCollapseUtil, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpTransformCollapseUtil, SharedShapeBehaviour, s_libraryName)


// Options hkpTransformCollapseUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpTransformCollapseUtil, Options, s_libraryName)


// Results hkpTransformCollapseUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpTransformCollapseUtil, Results, s_libraryName)

#include <Physics2012/Utilities/Collide/ShapeUtils/CreateShape/hkpCreateShapeUtility.h>


// hkpCreateShapeUtility ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCreateShapeUtility)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CreateShapeInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ShapeInfoOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCreateShapeUtility)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCreateShapeUtility, s_libraryName, hkReferencedObject)


// CreateShapeInput hkpCreateShapeUtility

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCreateShapeUtility::CreateShapeInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCreateShapeUtility::CreateShapeInput)
    HK_TRACKER_MEMBER(hkpCreateShapeUtility::CreateShapeInput, m_vertices, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCreateShapeUtility::CreateShapeInput, m_szMeshName, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCreateShapeUtility::CreateShapeInput, s_libraryName)


// ShapeInfoOutput hkpCreateShapeUtility

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCreateShapeUtility::ShapeInfoOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCreateShapeUtility::ShapeInfoOutput)
    HK_TRACKER_MEMBER(hkpCreateShapeUtility::ShapeInfoOutput, m_shape, 0, "hkpShape*") // class hkpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCreateShapeUtility::ShapeInfoOutput, s_libraryName)

#include <Physics2012/Utilities/Collide/ShapeUtils/MoppCodeStreamer/hkpMoppCodeStreamer.h>


// hkpMoppCodeStreamer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppCodeStreamer, s_libraryName)

#include <Physics2012/Utilities/Collide/ShapeUtils/ShapeConverter/hkpShapeConverter.h>


// hkpShapeConverter ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeConverter, s_libraryName)

#include <Physics2012/Utilities/Collide/ShapeUtils/ShapeKeyPath/hkpShapeKeyPath.h>


// hkpShapeKeyPath ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeKeyPath)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Iterator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeKeyPath)
    HK_TRACKER_MEMBER(hkpShapeKeyPath, m_rootShape, 0, "hkpShape*") // const class hkpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpShapeKeyPath, s_libraryName)


// Iterator hkpShapeKeyPath

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeKeyPath::Iterator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeKeyPath::Iterator)
    HK_TRACKER_MEMBER(hkpShapeKeyPath::Iterator, m_path, 0, "hkpShapeKeyPath*") // const class hkpShapeKeyPath*
    HK_TRACKER_MEMBER(hkpShapeKeyPath::Iterator, m_currentShape, 0, "hkpShape*") // const class hkpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpShapeKeyPath::Iterator, s_libraryName)

#include <Physics2012/Utilities/Collide/ShapeUtils/ShapeMerger/hkpShapeMergeUtility.h>


// hkpShapeMergeUtility ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeMergeUtility, s_libraryName)

#include <Physics2012/Utilities/Collide/ShapeUtils/ShapeScaling/hkpShapeScalingUtility.h>


// hkpShapeScalingUtility ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeScalingUtility)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ShapePair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeScalingUtility, s_libraryName)


// ShapePair hkpShapeScalingUtility

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeScalingUtility::ShapePair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeScalingUtility::ShapePair)
    HK_TRACKER_MEMBER(hkpShapeScalingUtility::ShapePair, originalShape, 0, "hkpShape*") // class hkpShape*
    HK_TRACKER_MEMBER(hkpShapeScalingUtility::ShapePair, newShape, 0, "hkpShape*") // class hkpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpShapeScalingUtility::ShapePair, s_libraryName)

#include <Physics2012/Utilities/Collide/ShapeUtils/ShapeSharing/hkpShapeSharingUtil.h>


// hkpShapeSharingUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeSharingUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Options)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Results)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeSharingUtil, s_libraryName)


// Options hkpShapeSharingUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpShapeSharingUtil, Options, s_libraryName)


// Results hkpShapeSharingUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpShapeSharingUtil, Results, s_libraryName)

#include <Physics2012/Utilities/Collide/ShapeUtils/ShapeToMeshConverter/hkpShapeToMeshConverter.h>


// hkpShapeToMeshConverter ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeToMeshConverter, s_libraryName)

#include <Physics2012/Utilities/Collide/ShapeUtils/SimpleMeshTklStreamer/hkpSimpleMeshTklStreamer.h>


// hkpTklStreamer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpTklStreamer, s_libraryName)

#include <Physics2012/Utilities/Collide/TriggerVolume/hkpTriggerVolume.h>


// hkpTriggerVolume ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpTriggerVolume)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(EventInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(EventType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Operation)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpTriggerVolume)
    HK_TRACKER_MEMBER(hkpTriggerVolume, m_overlappingBodies, 0, "hkArray<hkpRigidBody*, hkContainerHeapAllocator>") // hkArray< class hkpRigidBody*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpTriggerVolume, m_eventQueue, 0, "hkArray<hkpTriggerVolume::EventInfo, hkContainerHeapAllocator>") // hkArray< struct hkpTriggerVolume::EventInfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpTriggerVolume, m_triggerBody, 0, "hkpRigidBody*") // class hkpRigidBody*
    HK_TRACKER_MEMBER(hkpTriggerVolume, m_newOverlappingBodies, 0, "hkArray<hkpRigidBody*, hkContainerHeapAllocator>") // hkArray< class hkpRigidBody*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpTriggerVolume, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpTriggerVolume, EventType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpTriggerVolume, Operation, s_libraryName)


// EventInfo hkpTriggerVolume

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpTriggerVolume::EventInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpTriggerVolume::EventInfo)
    HK_TRACKER_MEMBER(hkpTriggerVolume::EventInfo, m_body, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpTriggerVolume::EventInfo, s_libraryName)

#include <Physics2012/Utilities/Collide/hkpShapeGenerator.h>


// hkpShapeGenerator ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeGenerator, s_libraryName)

#include <Physics2012/Utilities/Constraint/Bilateral/hkpConstraintUtils.h>

// hk.MemoryTracker ignore hkpConstraintUtils
#include <Physics2012/Utilities/Constraint/Chain/hkpConstraintChainUtil.h>


// hkpConstraintChainUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConstraintChainUtil, s_libraryName)

#include <Physics2012/Utilities/Constraint/Chain/hkpPoweredChainMapper.h>


// hkpPoweredChainMapper ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPoweredChainMapper)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Config)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ChainEndpoints)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Target)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LinkInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPoweredChainMapper)
    HK_TRACKER_MEMBER(hkpPoweredChainMapper, m_links, 0, "hkArray<hkpPoweredChainMapper::LinkInfo, hkContainerHeapAllocator>") // hkArray< struct hkpPoweredChainMapper::LinkInfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpPoweredChainMapper, m_targets, 0, "hkArray<hkpPoweredChainMapper::Target, hkContainerHeapAllocator>") // hkArray< struct hkpPoweredChainMapper::Target, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpPoweredChainMapper, m_chains, 0, "hkArray<hkpConstraintChainInstance*, hkContainerHeapAllocator>") // hkArray< class hkpConstraintChainInstance*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPoweredChainMapper, s_libraryName, hkReferencedObject)


// Config hkpPoweredChainMapper
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpPoweredChainMapper, Config, s_libraryName)


// ChainEndpoints hkpPoweredChainMapper

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPoweredChainMapper::ChainEndpoints)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPoweredChainMapper::ChainEndpoints)
    HK_TRACKER_MEMBER(hkpPoweredChainMapper::ChainEndpoints, m_start, 0, "hkpEntity*") // class hkpEntity*
    HK_TRACKER_MEMBER(hkpPoweredChainMapper::ChainEndpoints, m_end, 0, "hkpEntity*") // class hkpEntity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpPoweredChainMapper::ChainEndpoints, s_libraryName)


// Target hkpPoweredChainMapper

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPoweredChainMapper::Target)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPoweredChainMapper::Target)
    HK_TRACKER_MEMBER(hkpPoweredChainMapper::Target, m_chain, 0, "hkpPoweredChainData*") // class hkpPoweredChainData*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpPoweredChainMapper::Target, s_libraryName)


// LinkInfo hkpPoweredChainMapper

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPoweredChainMapper::LinkInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPoweredChainMapper::LinkInfo)
    HK_TRACKER_MEMBER(hkpPoweredChainMapper::LinkInfo, m_limitConstraint, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpPoweredChainMapper::LinkInfo, s_libraryName)

#include <Physics2012/Utilities/Constraint/Chain/hkpPoweredChainMapperUtil.h>


// hkpPoweredChainMapperUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpPoweredChainMapperUtil, s_libraryName)

#include <Physics2012/Utilities/Deprecated/ConstrainedSystem/hkpConstrainedSystemFilter.h>


// hkpConstrainedSystemFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstrainedSystemFilter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstrainedSystemFilter)
    HK_TRACKER_MEMBER(hkpConstrainedSystemFilter, m_otherFilter, 0, "hkpCollisionFilter*") // const class hkpCollisionFilter*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConstrainedSystemFilter, s_libraryName, hkpCollisionFilter)

#include <Physics2012/Utilities/Deprecated/DisableEntity/hkpDisableEntityCollisionFilter.h>


// hkpDisableEntityCollisionFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDisableEntityCollisionFilter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDisableEntityCollisionFilter)
    HK_TRACKER_MEMBER(hkpDisableEntityCollisionFilter, m_disabledEntities, 0, "hkArray<hkpEntity*, hkContainerHeapAllocator>") // hkArray< class hkpEntity*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDisableEntityCollisionFilter, s_libraryName, hkpCollisionFilter)

#include <Physics2012/Utilities/Deprecated/H1Group/hkpGroupCollisionFilter.h>


// hkpGroupCollisionFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpGroupCollisionFilter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpGroupCollisionFilter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpGroupCollisionFilter, s_libraryName, hkpCollisionFilter)

#include <Physics2012/Utilities/Deprecated/hkpCollapseTransformsDeprecated.h>


// hkpCollapseTransformsDeprecated ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCollapseTransformsDeprecated, s_libraryName)

#include <Physics2012/Utilities/Destruction/BreakOffParts/hkpBreakOffPartsUtil.h>


// hkpBreakOffPartsListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakOffPartsListener)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ContactImpulseLimitBreachedEvent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakOffPartsListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkpBreakOffPartsListener, s_libraryName)


// ContactImpulseLimitBreachedEvent hkpBreakOffPartsListener

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PointInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent)
    HK_TRACKER_MEMBER(hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent, m_breakingBody, 0, "hkpRigidBody*") // class hkpRigidBody*
    HK_TRACKER_MEMBER(hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent, m_points, 0, "hkInplaceArray<hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent::PointInfo, 4, hkContainerHeapAllocator>") // class hkInplaceArray< struct hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent::PointInfo, 4, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent, s_libraryName)


// PointInfo hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent::PointInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent::PointInfo)
    HK_TRACKER_MEMBER(hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent::PointInfo, m_collidingBody, 0, "hkpRigidBody*") // class hkpRigidBody*
    HK_TRACKER_MEMBER(hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent::PointInfo, m_contactPoint, 0, "hkContactPoint*") // const class hkContactPoint*
    HK_TRACKER_MEMBER(hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent::PointInfo, m_properties, 0, "hkpContactPointProperties*") // const class hkpContactPointProperties*
    HK_TRACKER_MEMBER(hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent::PointInfo, m_internalContactMgr, 0, "hkpSimpleConstraintContactMgr*") // const class hkpSimpleConstraintContactMgr*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent::PointInfo, s_libraryName)


// hkpBreakOffPartsUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakOffPartsUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LimitContactImpulseUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GameControlFunctor)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LimitContactImpulseUtilDefault)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LimitContactImpulseUtilCpuOnly)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BreakOffGameControlResult)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakOffPartsUtil)
    HK_TRACKER_MEMBER(hkpBreakOffPartsUtil, m_criticalSection, 0, "hkCriticalSection*") // class hkCriticalSection*
    HK_TRACKER_MEMBER(hkpBreakOffPartsUtil, m_breakOffPartsListener, 0, "hkpBreakOffPartsListener*") // class hkpBreakOffPartsListener*
    HK_TRACKER_MEMBER(hkpBreakOffPartsUtil, m_breakOffGameControlFunctor, 0, "hkpBreakOffPartsUtil::GameControlFunctor *") // class hkpBreakOffPartsUtil::GameControlFunctor *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBreakOffPartsUtil, s_libraryName, hkpWorldExtension)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBreakOffPartsUtil, BreakOffGameControlResult, s_libraryName)


// LimitContactImpulseUtil hkpBreakOffPartsUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakOffPartsUtil::LimitContactImpulseUtil)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakOffPartsUtil::LimitContactImpulseUtil)
    HK_TRACKER_MEMBER(hkpBreakOffPartsUtil::LimitContactImpulseUtil, m_entity, 0, "hkpEntity*") // class hkpEntity*
    HK_TRACKER_MEMBER(hkpBreakOffPartsUtil::LimitContactImpulseUtil, m_shapeKeyToMaxImpulse, 0, "hkPointerMap<hkUint32, hkUint8, hkContainerHeapAllocator>") // class hkPointerMap< hkUint32, hkUint8, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpBreakOffPartsUtil::LimitContactImpulseUtil, m_breakOffUtil, 0, "hkpBreakOffPartsUtil*") // class hkpBreakOffPartsUtil*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBreakOffPartsUtil::LimitContactImpulseUtil, s_libraryName, hkReferencedObject)


// GameControlFunctor hkpBreakOffPartsUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakOffPartsUtil::GameControlFunctor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakOffPartsUtil::GameControlFunctor)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBreakOffPartsUtil::GameControlFunctor, s_libraryName, hkReferencedObject)


// LimitContactImpulseUtilDefault hkpBreakOffPartsUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakOffPartsUtil::LimitContactImpulseUtilDefault)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakOffPartsUtil::LimitContactImpulseUtilDefault)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBreakOffPartsUtil::LimitContactImpulseUtilDefault, s_libraryName, hkpBreakOffPartsUtil::LimitContactImpulseUtil)


// LimitContactImpulseUtilCpuOnly hkpBreakOffPartsUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakOffPartsUtil::LimitContactImpulseUtilCpuOnly)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakOffPartsUtil::LimitContactImpulseUtilCpuOnly)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBreakOffPartsUtil::LimitContactImpulseUtilCpuOnly, s_libraryName, hkpBreakOffPartsUtil::LimitContactImpulseUtil)

#include <Physics2012/Utilities/Dynamics/EntityContactCollector/hkpEntityContactCollector.h>


// hkpEntityContactCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpEntityContactCollector)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ContactPoint)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpEntityContactCollector)
    HK_TRACKER_MEMBER(hkpEntityContactCollector, m_contactPoints, 0, "hkArray<hkpEntityContactCollector::ContactPoint, hkContainerHeapAllocator>") // hkArray< struct hkpEntityContactCollector::ContactPoint, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpEntityContactCollector, m_entities, 0, "hkInplaceArray<hkpEntity*, 1, hkContainerHeapAllocator>") // class hkInplaceArray< class hkpEntity*, 1, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpEntityContactCollector, s_libraryName, hkpContactListener)


// ContactPoint hkpEntityContactCollector

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpEntityContactCollector::ContactPoint)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpEntityContactCollector::ContactPoint)
    HK_TRACKER_MEMBER(hkpEntityContactCollector::ContactPoint, m_bodyA, 0, "hkpRigidBody*") // class hkpRigidBody*
    HK_TRACKER_MEMBER(hkpEntityContactCollector::ContactPoint, m_bodyB, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpEntityContactCollector::ContactPoint, s_libraryName)

#include <Physics2012/Utilities/Dynamics/ImpulseAccumulator/hkpImpulseAccumulator.h>


// hkpImpulseAccumulator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpImpulseAccumulator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpImpulseAccumulator)
    HK_TRACKER_MEMBER(hkpImpulseAccumulator, m_body, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpImpulseAccumulator, s_libraryName)

#include <Physics2012/Utilities/Dynamics/Inertia/hkpAccurateInertiaTensorComputer.h>


// hkpAccurateInertiaTensorComputer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpAccurateInertiaTensorComputer, s_libraryName)

#include <Physics2012/Utilities/Dynamics/Inertia/hkpInertiaTensorComputer.h>


// hkpInertiaTensorComputer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpInertiaTensorComputer, s_libraryName)

#include <Physics2012/Utilities/Dynamics/KeyFrame/hkpKeyFrameUtility.h>


// hkpKeyFrameUtility ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpKeyFrameUtility)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(KeyFrameInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AccelerationInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpKeyFrameUtility, s_libraryName)


// KeyFrameInfo hkpKeyFrameUtility
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpKeyFrameUtility, KeyFrameInfo, s_libraryName)


// AccelerationInfo hkpKeyFrameUtility
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpKeyFrameUtility, AccelerationInfo, s_libraryName)

#include <Physics2012/Utilities/Dynamics/Lazyadd/hkpLazyAddToWorld.h>


// hkpLazyAddToWorld ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpLazyAddToWorld)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpLazyAddToWorld)
    HK_TRACKER_MEMBER(hkpLazyAddToWorld, m_world, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpLazyAddToWorld, m_entities, 0, "hkArray<hkpEntity*, hkContainerHeapAllocator>") // hkArray< class hkpEntity*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpLazyAddToWorld, m_actions, 0, "hkArray<hkpAction*, hkContainerHeapAllocator>") // hkArray< class hkpAction*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpLazyAddToWorld, m_constraints, 0, "hkArray<hkpConstraintInstance*, hkContainerHeapAllocator>") // hkArray< class hkpConstraintInstance*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpLazyAddToWorld, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/Dynamics/PhantomBatchMove/hkpPhantomBatchMoveUtil.h>


// hkpPhantomBatchMoveUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpPhantomBatchMoveUtil, s_libraryName)

#include <Physics2012/Utilities/Dynamics/RigidBodyReset/hkpRigidBodyResetUtil.h>


// hkpRigidBodyResetUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRigidBodyResetUtil)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRigidBodyResetUtil)
    HK_TRACKER_MEMBER(hkpRigidBodyResetUtil, m_mainRB, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRigidBodyResetUtil, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/Dynamics/SaveContactPoints/hkpPhysicsSystemWithContacts.h>


// hkpPhysicsSystemWithContacts ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPhysicsSystemWithContacts)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPhysicsSystemWithContacts)
    HK_TRACKER_MEMBER(hkpPhysicsSystemWithContacts, m_contacts, 0, "hkArray<hkpSerializedAgentNnEntry*, hkContainerHeapAllocator>") // hkArray< struct hkpSerializedAgentNnEntry*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPhysicsSystemWithContacts, s_libraryName, hkpPhysicsSystem)

#include <Physics2012/Utilities/Dynamics/SaveContactPoints/hkpSaveContactPointsEndianUtil.h>


// hkpSaveContactPointsEndianUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSaveContactPointsEndianUtil, s_libraryName)

#include <Physics2012/Utilities/Dynamics/SaveContactPoints/hkpSaveContactPointsUtil.h>


// hkpSaveContactPointsUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSaveContactPointsUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SavePointsInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LoadPointsInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpSaveContactPointsUtil, s_libraryName)


// SavePointsInput hkpSaveContactPointsUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSaveContactPointsUtil, SavePointsInput, s_libraryName)


// LoadPointsInput hkpSaveContactPointsUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSaveContactPointsUtil, LoadPointsInput, s_libraryName)

// hk.MemoryTracker ignore EntitySelector
#include <Physics2012/Utilities/Dynamics/SaveContactPoints/hkpSerializedAgentNnEntry.h>


// hkpSerializedTrack1nInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSerializedTrack1nInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSerializedTrack1nInfo)
    HK_TRACKER_MEMBER(hkpSerializedTrack1nInfo, m_sectors, 0, "hkArray<hkpAgent1nSector*, hkContainerHeapAllocator>") // hkArray< struct hkpAgent1nSector*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpSerializedTrack1nInfo, m_subTracks, 0, "hkArray<hkpSerializedSubTrack1nInfo*, hkContainerHeapAllocator>") // hkArray< struct hkpSerializedSubTrack1nInfo*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpSerializedTrack1nInfo, s_libraryName)


// hkpSerializedSubTrack1nInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSerializedSubTrack1nInfo, s_libraryName)


// hkpSerializedAgentNnEntry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSerializedAgentNnEntry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SerializedAgentType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSerializedAgentNnEntry)
    HK_TRACKER_MEMBER(hkpSerializedAgentNnEntry, m_bodyA, 0, "hkpEntity*") // class hkpEntity*
    HK_TRACKER_MEMBER(hkpSerializedAgentNnEntry, m_bodyB, 0, "hkpEntity*") // class hkpEntity*
    HK_TRACKER_MEMBER(hkpSerializedAgentNnEntry, m_propertiesStream, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpSerializedAgentNnEntry, m_contactPoints, 0, "hkArray<hkContactPoint, hkContainerHeapAllocator>") // hkArray< class hkContactPoint, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpSerializedAgentNnEntry, m_cpIdMgr, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpSerializedAgentNnEntry, m_trackInfo, 0, "hkpSerializedTrack1nInfo") // struct hkpSerializedTrack1nInfo
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSerializedAgentNnEntry, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSerializedAgentNnEntry, SerializedAgentType, s_libraryName)

#include <Physics2012/Utilities/Dynamics/ScaleSystem/hkpSystemScalingUtility.h>


// hkpSystemScalingUtility ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSystemScalingUtility, s_libraryName)

#include <Physics2012/Utilities/Dynamics/SuspendInactiveAgents/hkpSuspendInactiveAgentsUtil.h>


// hkpSuspendInactiveAgentsUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSuspendInactiveAgentsUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(OperationMode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(InitContactsMode)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSuspendInactiveAgentsUtil)
    HK_TRACKER_MEMBER(hkpSuspendInactiveAgentsUtil, m_world, 0, "hkpWorld*") // class hkpWorld*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSuspendInactiveAgentsUtil, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSuspendInactiveAgentsUtil, OperationMode, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSuspendInactiveAgentsUtil, InitContactsMode, s_libraryName)

#include <Physics2012/Utilities/Dynamics/TimeSteppers/hkpVariableTimestepper.h>


// hkpVariableTimestepper ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpVariableTimestepper, s_libraryName)

#include <Physics2012/Utilities/Geometry/hkpGeometryConverter.h>


// hkpGeometryConverter ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpGeometryConverter, s_libraryName)

#include <Physics2012/Utilities/Serialize/Display/hkpSerializedDisplayMarker.h>


// hkpSerializedDisplayMarker ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSerializedDisplayMarker)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSerializedDisplayMarker)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSerializedDisplayMarker, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/Serialize/Display/hkpSerializedDisplayMarkerList.h>


// hkpSerializedDisplayMarkerList ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSerializedDisplayMarkerList)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSerializedDisplayMarkerList)
    HK_TRACKER_MEMBER(hkpSerializedDisplayMarkerList, m_markers, 0, "hkArray<hkpSerializedDisplayMarker*, hkContainerHeapAllocator>") // hkArray< class hkpSerializedDisplayMarker*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSerializedDisplayMarkerList, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/Serialize/Display/hkpSerializedDisplayRbTransforms.h>


// hkpSerializedDisplayRbTransforms ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSerializedDisplayRbTransforms)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DisplayTransformPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSerializedDisplayRbTransforms)
    HK_TRACKER_MEMBER(hkpSerializedDisplayRbTransforms, m_transforms, 0, "hkArray<hkpSerializedDisplayRbTransforms::DisplayTransformPair, hkContainerHeapAllocator>") // hkArray< struct hkpSerializedDisplayRbTransforms::DisplayTransformPair, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSerializedDisplayRbTransforms, s_libraryName, hkReferencedObject)


// DisplayTransformPair hkpSerializedDisplayRbTransforms

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSerializedDisplayRbTransforms::DisplayTransformPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSerializedDisplayRbTransforms::DisplayTransformPair)
    HK_TRACKER_MEMBER(hkpSerializedDisplayRbTransforms::DisplayTransformPair, m_rb, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpSerializedDisplayRbTransforms::DisplayTransformPair, s_libraryName)

#include <Physics2012/Utilities/Serialize/hkpDisplayBindingData.h>


// hkpDisplayBindingData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDisplayBindingData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(RigidBody)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PhysicsSystem)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDisplayBindingData)
    HK_TRACKER_MEMBER(hkpDisplayBindingData, m_rigidBodyBindings, 0, "hkArray<hkpDisplayBindingData::RigidBody *, hkContainerHeapAllocator>") // hkArray< struct hkpDisplayBindingData::RigidBody *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpDisplayBindingData, m_physicsSystemBindings, 0, "hkArray<hkpDisplayBindingData::PhysicsSystem *, hkContainerHeapAllocator>") // hkArray< struct hkpDisplayBindingData::PhysicsSystem *, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDisplayBindingData, s_libraryName, hkReferencedObject)


// RigidBody hkpDisplayBindingData

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDisplayBindingData::RigidBody)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDisplayBindingData::RigidBody)
    HK_TRACKER_MEMBER(hkpDisplayBindingData::RigidBody, m_rigidBody, 0, "hkpRigidBody *") // class hkpRigidBody *
    HK_TRACKER_MEMBER(hkpDisplayBindingData::RigidBody, m_displayObjectPtr, 0, "hkReferencedObject *") // class hkReferencedObject *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDisplayBindingData::RigidBody, s_libraryName, hkReferencedObject)


// PhysicsSystem hkpDisplayBindingData

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDisplayBindingData::PhysicsSystem)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDisplayBindingData::PhysicsSystem)
    HK_TRACKER_MEMBER(hkpDisplayBindingData::PhysicsSystem, m_bindings, 0, "hkArray<hkpDisplayBindingData::RigidBody *, hkContainerHeapAllocator>") // hkArray< struct hkpDisplayBindingData::RigidBody *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpDisplayBindingData::PhysicsSystem, m_system, 0, "hkpPhysicsSystem *") // class hkpPhysicsSystem *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDisplayBindingData::PhysicsSystem, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/Serialize/hkpHavokSnapshot.h>


// hkpHavokSnapshot ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpHavokSnapshot)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConvertListener)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Options)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SnapshotOptionsBits)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpHavokSnapshot, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpHavokSnapshot, SnapshotOptionsBits, s_libraryName)


// ConvertListener hkpHavokSnapshot

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpHavokSnapshot::ConvertListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpHavokSnapshot::ConvertListener)
    HK_TRACKER_MEMBER(hkpHavokSnapshot::ConvertListener, m_objects, 0, "hkArray<hkReferencedObject*, hkContainerHeapAllocator>") // hkArray< class hkReferencedObject*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpHavokSnapshot::ConvertListener, s_libraryName, hkPackfileWriter::AddObjectListener)


// Options hkpHavokSnapshot
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpHavokSnapshot, Options, s_libraryName)

#include <Physics2012/Utilities/Serialize/hkpPhysicsData.h>


// hkpPhysicsData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPhysicsData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SplitPhysicsSystemsOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPhysicsData)
    HK_TRACKER_MEMBER(hkpPhysicsData, m_worldCinfo, 0, "hkpWorldCinfo*") // class hkpWorldCinfo*
    HK_TRACKER_MEMBER(hkpPhysicsData, m_systems, 0, "hkArray<hkpPhysicsSystem*, hkContainerHeapAllocator>") // hkArray< class hkpPhysicsSystem*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPhysicsData, s_libraryName, hkReferencedObject)


// SplitPhysicsSystemsOutput hkpPhysicsData

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPhysicsData::SplitPhysicsSystemsOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPhysicsData::SplitPhysicsSystemsOutput)
    HK_TRACKER_MEMBER(hkpPhysicsData::SplitPhysicsSystemsOutput, m_unconstrainedFixedBodies, 0, "hkpPhysicsSystem*") // class hkpPhysicsSystem*
    HK_TRACKER_MEMBER(hkpPhysicsData::SplitPhysicsSystemsOutput, m_unconstrainedKeyframedBodies, 0, "hkpPhysicsSystem*") // class hkpPhysicsSystem*
    HK_TRACKER_MEMBER(hkpPhysicsData::SplitPhysicsSystemsOutput, m_unconstrainedMovingBodies, 0, "hkpPhysicsSystem*") // class hkpPhysicsSystem*
    HK_TRACKER_MEMBER(hkpPhysicsData::SplitPhysicsSystemsOutput, m_phantoms, 0, "hkpPhysicsSystem*") // class hkpPhysicsSystem*
    HK_TRACKER_MEMBER(hkpPhysicsData::SplitPhysicsSystemsOutput, m_constrainedSystems, 0, "hkArray<hkpPhysicsSystem*, hkContainerHeapAllocator>") // hkArray< class hkpPhysicsSystem*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpPhysicsData::SplitPhysicsSystemsOutput, s_libraryName)

#include <Physics2012/Utilities/Serialize/hkpPhysicsToSceneDataBridge.h>


// hkpPhysicsToSceneDataBridge ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPhysicsToSceneDataBridge)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(RootLevelContainer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPhysicsToSceneDataBridge)
    HK_TRACKER_MEMBER(hkpPhysicsToSceneDataBridge, m_sceneDataContext, 0, "hkxSceneDataContext*") // class hkxSceneDataContext*
    HK_TRACKER_MEMBER(hkpPhysicsToSceneDataBridge, m_physicsWorld, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpPhysicsToSceneDataBridge, m_loadedContainers, 0, "hkArray<hkpPhysicsToSceneDataBridge::RootLevelContainer, hkContainerHeapAllocator>") // hkArray< struct hkpPhysicsToSceneDataBridge::RootLevelContainer, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpPhysicsToSceneDataBridge, m_rigidBodyToMeshIdMap, 0, "hkPointerMap<hkpRigidBody*, hkUlong, hkContainerHeapAllocator>") // class hkPointerMap< class hkpRigidBody*, hkUlong, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpPhysicsToSceneDataBridge, m_rigidBodyToTransformMap, 0, "hkPointerMap<hkpRigidBody*, hkTransformf*, hkContainerHeapAllocator>") // class hkPointerMap< class hkpRigidBody*, hkTransformf*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpPhysicsToSceneDataBridge, m_meshIdToScaleAndSkewTransformMap, 0, "hkPointerMap<hkUlong, hkMatrix4f*, hkContainerHeapAllocator>") // class hkPointerMap< hkUlong, hkMatrix4f*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPhysicsToSceneDataBridge, s_libraryName, hkReferencedObject)


// RootLevelContainer hkpPhysicsToSceneDataBridge

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPhysicsToSceneDataBridge::RootLevelContainer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPhysicsToSceneDataBridge::RootLevelContainer)
    HK_TRACKER_MEMBER(hkpPhysicsToSceneDataBridge::RootLevelContainer, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkpPhysicsToSceneDataBridge::RootLevelContainer, m_container, 0, "hkRootLevelContainer") // class hkRootLevelContainer
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpPhysicsToSceneDataBridge::RootLevelContainer, s_libraryName)

#include <Physics2012/Utilities/VisualDebugger/ShapeHash/hkpShapeHashUtil.h>


// hkpShapeHashUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpShapeHashUtil, s_libraryName)


// hkpUserShapeHashUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpUserShapeHashUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(UserShapeHashFunctions)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpUserShapeHashUtil)
    HK_TRACKER_MEMBER(hkpUserShapeHashUtil, m_userShapeHashFunctions, 0, "hkArray<hkpUserShapeHashUtil::UserShapeHashFunctions, hkContainerHeapAllocator>") // hkArray< struct hkpUserShapeHashUtil::UserShapeHashFunctions, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpUserShapeHashUtil, s_libraryName, hkReferencedObject)


// UserShapeHashFunctions hkpUserShapeHashUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpUserShapeHashUtil, UserShapeHashFunctions, s_libraryName)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpActiveContactPointViewer.h>


// hkpActiveContactPointViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpActiveContactPointViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpActiveContactPointViewer)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpActiveContactPointViewer, s_libraryName, hkpContactPointViewer)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpBroadphaseViewer.h>


// hkpBroadphaseViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBroadphaseViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBroadphaseViewer)
    HK_TRACKER_MEMBER(hkpBroadphaseViewer, m_broadPhaseDisplayGeometries, 0, "hkArray<hkDisplayAABB, hkContainerHeapAllocator>") // hkArray< class hkDisplayAABB, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBroadphaseViewer, s_libraryName, hkpWorldViewerBase)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpCollideDebugUtil.h>


// hkpCollideDebugUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCollideDebugUtil, s_libraryName)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpContactPointViewer.h>


// hkpContactPointViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpContactPointViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpContactPointViewer)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpContactPointViewer, s_libraryName, hkpWorldViewerBase)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpConvexRadiusBuilder.h>


// hkpConvexRadiusBuilder ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexRadiusBuilder)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpConvexRadiusBuilderEnvironment)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexRadiusBuilder)
    HK_TRACKER_MEMBER(hkpConvexRadiusBuilder, m_currentGeometry, 0, "hkDisplayGeometry*") // class hkDisplayGeometry*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConvexRadiusBuilder, s_libraryName, hkDisplayGeometryBuilder)


// hkpConvexRadiusBuilderEnvironment hkpConvexRadiusBuilder
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConvexRadiusBuilder, hkpConvexRadiusBuilderEnvironment, s_libraryName)


// hkpUserConvexRadiusBuilder ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpUserConvexRadiusBuilder)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(UserShapeBuilder)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpUserConvexRadiusBuilder)
    HK_TRACKER_MEMBER(hkpUserConvexRadiusBuilder, m_userConvexRadiusBuilders, 0, "hkArray<hkpUserConvexRadiusBuilder::UserShapeBuilder, hkContainerHeapAllocator>") // hkArray< struct hkpUserConvexRadiusBuilder::UserShapeBuilder, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpUserConvexRadiusBuilder, s_libraryName, hkReferencedObject)


// UserShapeBuilder hkpUserConvexRadiusBuilder
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpUserConvexRadiusBuilder, UserShapeBuilder, s_libraryName)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpConvexRadiusViewer.h>


// hkpConvexRadiusViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexRadiusViewer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WorldToEntityData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexRadiusViewer)
    HK_TRACKER_MEMBER(hkpConvexRadiusViewer, m_worldEntities, 0, "hkArray<hkpConvexRadiusViewer::WorldToEntityData*, hkContainerHeapAllocator>") // hkArray< struct hkpConvexRadiusViewer::WorldToEntityData*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpConvexRadiusViewer, m_builder, 0, "hkpConvexRadiusBuilder*") // class hkpConvexRadiusBuilder*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConvexRadiusViewer, s_libraryName, hkpWorldViewerBase)


// WorldToEntityData hkpConvexRadiusViewer

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexRadiusViewer::WorldToEntityData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexRadiusViewer::WorldToEntityData)
    HK_TRACKER_MEMBER(hkpConvexRadiusViewer::WorldToEntityData, world, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpConvexRadiusViewer::WorldToEntityData, entitiesCreated, 0, "hkArray<hkUlong, hkContainerHeapAllocator>") // hkArray< hkUlong, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpConvexRadiusViewer::WorldToEntityData, s_libraryName)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpInactiveContactPointViewer.h>


// hkpInactiveContactPointViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpInactiveContactPointViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpInactiveContactPointViewer)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpInactiveContactPointViewer, s_libraryName, hkpContactPointViewer)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpInconsistentWindingViewer.h>


// hkpInconsistentWindingViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpInconsistentWindingViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpInconsistentWindingViewer)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpInconsistentWindingViewer, s_libraryName, hkpWorldViewerBase)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpMidphaseViewer.h>


// hkpMidphaseViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMidphaseViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMidphaseViewer)
    HK_TRACKER_MEMBER(hkpMidphaseViewer, m_broadPhaseDisplayGeometries, 0, "hkArray<hkDisplayAABB, hkContainerHeapAllocator>") // hkArray< class hkDisplayAABB, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMidphaseViewer, s_libraryName, hkpWorldViewerBase)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpShapeDisplayViewer.h>


// hkpShapeDisplayViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeDisplayViewer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ShapeDisplayViewerOptions)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WorldToEntityData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeDisplayViewer)
    HK_TRACKER_MEMBER(hkpShapeDisplayViewer, m_worldEntities, 0, "hkArray<hkpShapeDisplayViewer::WorldToEntityData*, hkContainerHeapAllocator>") // hkArray< struct hkpShapeDisplayViewer::WorldToEntityData*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpShapeDisplayViewer, m_instancedShapeToGeomID, 0, "hkPointerMap<hkpShape*, hkUlong, hkContainerHeapAllocator>") // class hkPointerMap< const class hkpShape*, hkUlong, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpShapeDisplayViewer, m_instancedShapeToUsageCount, 0, "hkPointerMap<hkpShape*, hkUlong, hkContainerHeapAllocator>") // class hkPointerMap< const class hkpShape*, hkUlong, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpShapeDisplayViewer, m_cachedShapes, 0, "hkPointerMap<hkpShape*, hkUlong, hkContainerHeapAllocator>") // class hkPointerMap< const class hkpShape*, hkUlong, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpShapeDisplayViewer, m_builder, 0, "hkpShapeDisplayBuilder*") // class hkpShapeDisplayBuilder*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpShapeDisplayViewer, s_libraryName, hkpWorldViewerBase)


// ShapeDisplayViewerOptions hkpShapeDisplayViewer

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeDisplayViewer::ShapeDisplayViewerOptions)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeDisplayViewer::ShapeDisplayViewerOptions)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpShapeDisplayViewer::ShapeDisplayViewerOptions, s_libraryName, hkReferencedObject)


// WorldToEntityData hkpShapeDisplayViewer

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeDisplayViewer::WorldToEntityData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeDisplayViewer::WorldToEntityData)
    HK_TRACKER_MEMBER(hkpShapeDisplayViewer::WorldToEntityData, world, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpShapeDisplayViewer::WorldToEntityData, entitiesCreated, 0, "hkArray<hkUlong, hkContainerHeapAllocator>") // hkArray< hkUlong, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpShapeDisplayViewer::WorldToEntityData, s_libraryName)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpToiContactPointViewer.h>


// hkpToiContactPointViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpToiContactPointViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpToiContactPointViewer)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpToiContactPointViewer, s_libraryName, hkpWorldViewerBase)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpToiCountViewer.h>


// hkpToiCountViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpToiCountViewer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DisplayPosition)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpToiCountViewer)
    HK_TRACKER_MEMBER(hkpToiCountViewer, m_toiCounts, 0, "hkPointerMap<hkpRigidBody*, hkUint32, hkContainerHeapAllocator>") // class hkPointerMap< class hkpRigidBody*, hkUint32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpToiCountViewer, s_libraryName, hkpWorldViewerBase)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpToiCountViewer, DisplayPosition, s_libraryName)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpWeldingViewer.h>


// hkpWeldingViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWeldingViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWeldingViewer)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpWeldingViewer, s_libraryName, hkpWorldViewerBase)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/Constraint/Drawer/hkpConstraintChainDrawer.h>


// hkpConstraintChainDrawer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConstraintChainDrawer, s_libraryName)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpConstraintViewer.h>


// hkpConstraintViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintViewer)
    HK_TRACKER_MEMBER(hkpConstraintViewer, m_constraints, 0, "hkArray<hkpConstraintInstance*, hkContainerHeapAllocator>") // hkArray< class hkpConstraintInstance*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConstraintViewer, s_libraryName, hkpWorldViewerBase)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpPhantomDisplayViewer.h>


// hkpPhantomDisplayViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPhantomDisplayViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPhantomDisplayViewer)
    HK_TRACKER_MEMBER(hkpPhantomDisplayViewer, m_phantomShapesCreated, 0, "hkArray<hkpWorldObject*, hkContainerHeapAllocator>") // hkArray< class hkpWorldObject*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPhantomDisplayViewer, s_libraryName, hkpWorldViewerBase)


// hkpUserShapePhantomTypeIdentifier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpUserShapePhantomTypeIdentifier)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpUserShapePhantomTypeIdentifier)
    HK_TRACKER_MEMBER(hkpUserShapePhantomTypeIdentifier, m_shapePhantomTypes, 0, "hkArray<hkpPhantomType, hkContainerHeapAllocator>") // hkArray< enum hkpPhantomType, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpUserShapePhantomTypeIdentifier, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpRigidBodyCentreOfMassViewer.h>


// hkpRigidBodyCentreOfMassViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRigidBodyCentreOfMassViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRigidBodyCentreOfMassViewer)
    HK_TRACKER_MEMBER(hkpRigidBodyCentreOfMassViewer, m_entitiesCreated, 0, "hkArray<hkpRigidBody*, hkContainerHeapAllocator>") // hkArray< class hkpRigidBody*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRigidBodyCentreOfMassViewer, s_libraryName, hkpWorldViewerBase)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpRigidBodyInertiaViewer.h>


// hkpRigidBodyInertiaViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRigidBodyInertiaViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRigidBodyInertiaViewer)
    HK_TRACKER_MEMBER(hkpRigidBodyInertiaViewer, m_entitiesCreated, 0, "hkArray<hkpRigidBody*, hkContainerHeapAllocator>") // hkArray< class hkpRigidBody*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpRigidBodyInertiaViewer, m_displayBoxes, 0, "hkArray<hkDisplayBox, hkContainerHeapAllocator>") // hkArray< class hkDisplayBox, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRigidBodyInertiaViewer, s_libraryName, hkpWorldViewerBase)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpRigidBodyLocalFrameViewer.h>


// hkpRigidBodyLocalFrameViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRigidBodyLocalFrameViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRigidBodyLocalFrameViewer)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRigidBodyLocalFrameViewer, s_libraryName, hkpWorldViewerBase)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpSimulationIslandViewer.h>


// hkpSimulationIslandViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimulationIslandViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSimulationIslandViewer)
    HK_TRACKER_MEMBER(hkpSimulationIslandViewer, m_inactiveIslandDisplayGeometries, 0, "hkArray<hkDisplayAABB, hkContainerHeapAllocator>") // hkArray< class hkDisplayAABB, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpSimulationIslandViewer, m_activeIslandDisplayGeometries, 0, "hkArray<hkDisplayAABB, hkContainerHeapAllocator>") // hkArray< class hkDisplayAABB, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSimulationIslandViewer, s_libraryName, hkpWorldViewerBase)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpSingleBodyConstraintViewer.h>


// hkpSingleBodyConstraintViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSingleBodyConstraintViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSingleBodyConstraintViewer)
    HK_TRACKER_MEMBER(hkpSingleBodyConstraintViewer, m_currentWorld, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpSingleBodyConstraintViewer, m_pickedBody, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSingleBodyConstraintViewer, s_libraryName, hkpWorldViewerBase)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpSweptTransformDisplayViewer.h>


// hkpSweptTransformDisplayViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSweptTransformDisplayViewer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WorldToEntityData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSweptTransformDisplayViewer)
    HK_TRACKER_MEMBER(hkpSweptTransformDisplayViewer, m_worldEntities, 0, "hkArray<hkpSweptTransformDisplayViewer::WorldToEntityData*, hkContainerHeapAllocator>") // hkArray< struct hkpSweptTransformDisplayViewer::WorldToEntityData*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpSweptTransformDisplayViewer, m_builder, 0, "hkpShapeDisplayBuilder*") // class hkpShapeDisplayBuilder*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSweptTransformDisplayViewer, s_libraryName, hkpWorldViewerBase)


// WorldToEntityData hkpSweptTransformDisplayViewer

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSweptTransformDisplayViewer::WorldToEntityData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSweptTransformDisplayViewer::WorldToEntityData)
    HK_TRACKER_MEMBER(hkpSweptTransformDisplayViewer::WorldToEntityData, world, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpSweptTransformDisplayViewer::WorldToEntityData, entitiesCreated, 0, "hkArray<hkUlong, hkContainerHeapAllocator>") // hkArray< hkUlong, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpSweptTransformDisplayViewer::WorldToEntityData, s_libraryName)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpWorldSnapshotViewer.h>


// hkpWorldSnapshotViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldSnapshotViewer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Type)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldSnapshotViewer)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpWorldSnapshotViewer, s_libraryName, hkpWorldViewerBase)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWorldSnapshotViewer, Type, s_libraryName)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpWorldViewerBase.h>


// hkpWorldViewerBase ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldViewerBase)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldViewerBase)
    HK_TRACKER_MEMBER(hkpWorldViewerBase, m_context, 0, "hkpPhysicsContext*") // class hkpPhysicsContext*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpWorldViewerBase, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Utilities/hkpMousePickingViewer.h>


// hkpMousePickingViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMousePickingViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMousePickingViewer)
    HK_TRACKER_MEMBER(hkpMousePickingViewer, m_currentWorld, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpMousePickingViewer, m_mouseSpring, 0, "hkpMouseSpringAction*") // class hkpMouseSpringAction*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMousePickingViewer, s_libraryName, hkpWorldViewerBase)

#include <Physics2012/Utilities/VisualDebugger/Viewer/Vehicle/hkpVehicleViewer.h>


// hkpVehicleViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleViewer)
    HK_TRACKER_MEMBER(hkpVehicleViewer, m_vehicles, 0, "hkArray<hkpVehicleInstance*, hkContainerHeapAllocator>") // hkArray< class hkpVehicleInstance*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleViewer, s_libraryName, hkpWorldViewerBase)

#include <Physics2012/Utilities/VisualDebugger/Viewer/hkpShapeContinueData.h>


// hkpShapeContinueData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeContinueData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeContinueData)
    HK_TRACKER_MEMBER(hkpShapeContinueData, m_shapeKeys, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpShapeContinueData, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/VisualDebugger/Viewer/hkpShapeDisplayBuilder.h>


// hkpShapeDisplayBuilder ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeDisplayBuilder)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpShapeDisplayBuilderEnvironment)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeDisplayBuilder)
    HK_TRACKER_MEMBER(hkpShapeDisplayBuilder, m_currentGeometry, 0, "hkDisplayGeometry*") // class hkDisplayGeometry*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpShapeDisplayBuilder, s_libraryName, hkDisplayGeometryBuilder)


// hkpShapeDisplayBuilderEnvironment hkpShapeDisplayBuilder
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpShapeDisplayBuilder, hkpShapeDisplayBuilderEnvironment, s_libraryName)


// hkpUserShapeDisplayBuilder ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpUserShapeDisplayBuilder)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(UserShapeBuilder)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpUserShapeDisplayBuilder)
    HK_TRACKER_MEMBER(hkpUserShapeDisplayBuilder, m_userShapeBuilders, 0, "hkArray<hkpUserShapeDisplayBuilder::UserShapeBuilder, hkContainerHeapAllocator>") // hkArray< struct hkpUserShapeDisplayBuilder::UserShapeBuilder, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpUserShapeDisplayBuilder, s_libraryName, hkReferencedObject)


// UserShapeBuilder hkpUserShapeDisplayBuilder
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpUserShapeDisplayBuilder, UserShapeBuilder, s_libraryName)

#include <Physics2012/Utilities/VisualDebugger/hkpPhysicsContext.h>


// hkpPhysicsContextWorldListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPhysicsContextWorldListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPhysicsContextWorldListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkpPhysicsContextWorldListener, s_libraryName)


// hkpPhysicsContext ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPhysicsContext)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPhysicsContext)
    HK_TRACKER_MEMBER(hkpPhysicsContext, m_worlds, 0, "hkArray<hkpWorld*, hkContainerHeapAllocator>") // hkArray< class hkpWorld*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpPhysicsContext, m_worldCinfos, 0, "hkArray<hkpWorldCinfo, hkContainerHeapAllocator>") // hkArray< class hkpWorldCinfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpPhysicsContext, m_addListeners, 0, "hkArray<hkpPhysicsContextWorldListener*, hkContainerHeapAllocator>") // hkArray< class hkpPhysicsContextWorldListener*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPhysicsContext, s_libraryName, hkReferencedObject)

#include <Physics2012/Utilities/Weapons/hkpBallGun.h>


// hkpBallGun ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBallGun)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBallGun)
    HK_TRACKER_MEMBER(hkpBallGun, m_addedBodies, 0, "hkQueue<hkpRigidBody*>*") // class hkQueue< class hkpRigidBody* >*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBallGun, s_libraryName, hkpFirstPersonGun)

#include <Physics2012/Utilities/Weapons/hkpFirstPersonGun.h>


// hkpFirstPersonGunBulletListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpFirstPersonGunBulletListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpFirstPersonGunBulletListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpFirstPersonGunBulletListener, s_libraryName, hkReferencedObject)


// hkpFirstPersonGun ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpFirstPersonGun)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SweepSphereOut)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Type)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(KeyboardKey)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpFirstPersonGun)
    HK_TRACKER_MEMBER(hkpFirstPersonGun, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkpFirstPersonGun, m_listeners, 0, "hkArray<hkpFirstPersonGunBulletListener*, hkContainerHeapAllocator>") // hkArray< class hkpFirstPersonGunBulletListener*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpFirstPersonGun, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpFirstPersonGun, Type, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpFirstPersonGun, KeyboardKey, s_libraryName)


// SweepSphereOut hkpFirstPersonGun

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpFirstPersonGun::SweepSphereOut)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpFirstPersonGun::SweepSphereOut)
    HK_TRACKER_MEMBER(hkpFirstPersonGun::SweepSphereOut, m_body, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpFirstPersonGun::SweepSphereOut, s_libraryName)

#include <Physics2012/Utilities/Weapons/hkpGravityGun.h>


// hkpGravityGun ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpGravityGun)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpGravityGun)
    HK_TRACKER_MEMBER(hkpGravityGun, m_grabbedBodies, 0, "hkArray<hkpRigidBody*, hkContainerHeapAllocator>") // hkArray< class hkpRigidBody*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpGravityGun, s_libraryName, hkpFirstPersonGun)

#include <Physics2012/Utilities/Weapons/hkpMountedBallGun.h>


// hkpMountedBallGun ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMountedBallGun)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMountedBallGun)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMountedBallGun, s_libraryName, hkpBallGun)

#include <Physics2012/Utilities/Weapons/hkpProjectileGun.h>


// hkpGunProjectile ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpGunProjectile)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Flags)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpGunProjectile)
    HK_TRACKER_MEMBER(hkpGunProjectile, m_body, 0, "hkpRigidBody*") // class hkpRigidBody*
    HK_TRACKER_MEMBER(hkpGunProjectile, m_gun, 0, "hkpProjectileGun*") // class hkpProjectileGun*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpGunProjectile, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpGunProjectile, Flags, s_libraryName)


// hkpProjectileGun ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpProjectileGun)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpProjectileGun)
    HK_TRACKER_MEMBER(hkpProjectileGun, m_projectiles, 0, "hkArray<hkpGunProjectile*, hkContainerHeapAllocator>") // hkArray< class hkpGunProjectile*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpProjectileGun, m_world, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpProjectileGun, m_destructionWorld, 0, "hkdWorld*") // class hkdWorld*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpProjectileGun, s_libraryName, hkpFirstPersonGun)

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
