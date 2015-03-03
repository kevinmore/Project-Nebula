/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Physics2012/Dynamics/hkpDynamics.h>
static const char s_libraryName[] = "hkpDynamics";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpDynamicsRegister() {}

#include <Physics2012/Dynamics/Action/hkpAction.h>


// hkpAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpAction)
    HK_TRACKER_MEMBER(hkpAction, m_world, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpAction, m_island, 0, "hkpSimulationIsland*") // class hkpSimulationIsland*
    HK_TRACKER_MEMBER(hkpAction, m_name, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpAction, s_libraryName, hkReferencedObject)

#include <Physics2012/Dynamics/Action/hkpActionListener.h>

// hk.MemoryTracker ignore hkpActionListener
#include <Physics2012/Dynamics/Action/hkpArrayAction.h>


// hkpArrayAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpArrayAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpArrayAction)
    HK_TRACKER_MEMBER(hkpArrayAction, m_entities, 0, "hkArray<hkpEntity*, hkContainerHeapAllocator>") // hkArray< class hkpEntity*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpArrayAction, s_libraryName, hkpAction)

#include <Physics2012/Dynamics/Action/hkpBinaryAction.h>


// hkpBinaryAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBinaryAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBinaryAction)
    HK_TRACKER_MEMBER(hkpBinaryAction, m_entityA, 0, "hkpEntity*") // class hkpEntity*
    HK_TRACKER_MEMBER(hkpBinaryAction, m_entityB, 0, "hkpEntity*") // class hkpEntity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpBinaryAction, s_libraryName, hkpAction)

#include <Physics2012/Dynamics/Action/hkpUnaryAction.h>


// hkpUnaryAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpUnaryAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpUnaryAction)
    HK_TRACKER_MEMBER(hkpUnaryAction, m_entity, 0, "hkpEntity*") // class hkpEntity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpUnaryAction, s_libraryName, hkpAction)

#include <Physics2012/Dynamics/Collide/ContactListener/Util/hkpAddModifierUtil.h>


// hkpAddModifierUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpAddModifierUtil, s_libraryName)

#include <Physics2012/Dynamics/Collide/ContactListener/Util/hkpCollisionCallbackUtil.h>


// hkpCollisionCallbackUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCollisionCallbackUtil)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCollisionCallbackUtil)
    HK_TRACKER_MEMBER(hkpCollisionCallbackUtil, m_endOfStepCallbackUtil, 0, "hkpEndOfStepCallbackUtil") // class hkpEndOfStepCallbackUtil
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCollisionCallbackUtil, s_libraryName, hkpWorldExtension)

#include <Physics2012/Dynamics/Collide/ContactListener/Util/hkpEndOfStepCallbackUtil.h>


// hkpEndOfStepCallbackUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpEndOfStepCallbackUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Collision)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(NewCollision)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpEndOfStepCallbackUtil)
    HK_TRACKER_MEMBER(hkpEndOfStepCallbackUtil, m_collisions, 0, "hkArray<hkpEndOfStepCallbackUtil::Collision, hkContainerHeapAllocator>") // hkArray< struct hkpEndOfStepCallbackUtil::Collision, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpEndOfStepCallbackUtil, m_newCollisions, 0, "hkArray<hkpEndOfStepCallbackUtil::NewCollision, hkContainerHeapAllocator>") // hkArray< struct hkpEndOfStepCallbackUtil::NewCollision, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpEndOfStepCallbackUtil, m_removedCollisions, 0, "hkArray<hkpEndOfStepCallbackUtil::Collision, hkContainerHeapAllocator>") // hkArray< struct hkpEndOfStepCallbackUtil::Collision, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpEndOfStepCallbackUtil, s_libraryName, hkReferencedObject)


// Collision hkpEndOfStepCallbackUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpEndOfStepCallbackUtil::Collision)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpEndOfStepCallbackUtil::Collision)
    HK_TRACKER_MEMBER(hkpEndOfStepCallbackUtil::Collision, m_mgr, 0, "hkpSimpleConstraintContactMgr*") // class hkpSimpleConstraintContactMgr*
    HK_TRACKER_MEMBER(hkpEndOfStepCallbackUtil::Collision, m_listener, 0, "hkpContactListener*") // class hkpContactListener*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpEndOfStepCallbackUtil::Collision, s_libraryName)


// NewCollision hkpEndOfStepCallbackUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpEndOfStepCallbackUtil, NewCollision, s_libraryName)

#include <Physics2012/Dynamics/Collide/ContactListener/hkpCollisionEvent.h>


// hkpCollisionEvent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCollisionEvent)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CallbackSource)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCollisionEvent)
    HK_TRACKER_MEMBER(hkpCollisionEvent, m_bodies, 0, "hkpRigidBody* [2]") // class hkpRigidBody* [2]
    HK_TRACKER_MEMBER(hkpCollisionEvent, m_contactMgr, 0, "hkpSimpleConstraintContactMgr*") // class hkpSimpleConstraintContactMgr*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCollisionEvent, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCollisionEvent, CallbackSource, s_libraryName)

#include <Physics2012/Dynamics/Collide/ContactListener/hkpContactListener.h>

// hk.MemoryTracker ignore hkpContactListener
#include <Physics2012/Dynamics/Collide/ContactListener/hkpContactPointEvent.h>


// hkpContactPointEvent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpContactPointEvent)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Type)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpContactPointEvent)
    HK_TRACKER_MEMBER(hkpContactPointEvent, m_contactPoint, 0, "hkContactPoint*") // class hkContactPoint*
    HK_TRACKER_MEMBER(hkpContactPointEvent, m_contactPointProperties, 0, "hkpContactPointProperties*") // class hkpContactPointProperties*
    HK_TRACKER_MEMBER(hkpContactPointEvent, m_separatingVelocity, 0, "float*") // float*
    HK_TRACKER_MEMBER(hkpContactPointEvent, m_rotateNormal, 0, "float*") // float*
    HK_TRACKER_MEMBER(hkpContactPointEvent, m_shapeKeyStorage, 0, "hkUint32*") // hkUint32*
    HK_TRACKER_MEMBER(hkpContactPointEvent, m_accumulators, 0, "hkpVelocityAccumulator* [2]") // class hkpVelocityAccumulator* [2]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpContactPointEvent, s_libraryName, hkpCollisionEvent)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpContactPointEvent, Type, s_libraryName)

#include <Physics2012/Dynamics/Collide/Deprecated/hkpCollisionEvents.h>


// hkpContactPointAddedEvent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpContactPointAddedEvent)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Type)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpContactPointAddedEvent)
    HK_TRACKER_MEMBER(hkpContactPointAddedEvent, m_bodyA, 0, "hkpCdBody*") // const class hkpCdBody*
    HK_TRACKER_MEMBER(hkpContactPointAddedEvent, m_bodyB, 0, "hkpCdBody*") // const class hkpCdBody*
    HK_TRACKER_MEMBER(hkpContactPointAddedEvent, m_callbackFiredFrom, 0, "hkpEntity*") // class hkpEntity*
    HK_TRACKER_MEMBER(hkpContactPointAddedEvent, m_contactPoint, 0, "hkContactPoint*") // const class hkContactPoint*
    HK_TRACKER_MEMBER(hkpContactPointAddedEvent, m_gskCache, 0, "hkpGskCache*") // const class hkpGskCache*
    HK_TRACKER_MEMBER(hkpContactPointAddedEvent, m_contactPointProperties, 0, "hkpContactPointProperties*") // class hkpContactPointProperties*
    HK_TRACKER_MEMBER(hkpContactPointAddedEvent, m_internalContactMgr, 0, "hkpDynamicsContactMgr*") // class hkpDynamicsContactMgr*
    HK_TRACKER_MEMBER(hkpContactPointAddedEvent, m_collisionInput, 0, "hkpProcessCollisionInput*") // const struct hkpProcessCollisionInput*
    HK_TRACKER_MEMBER(hkpContactPointAddedEvent, m_collisionOutput, 0, "hkpProcessCollisionOutput*") // struct hkpProcessCollisionOutput*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpContactPointAddedEvent, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpContactPointAddedEvent, Type, s_libraryName)


// hkpToiPointAddedEvent ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpToiPointAddedEvent, s_libraryName)


// hkpManifoldPointAddedEvent ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpManifoldPointAddedEvent, s_libraryName)


// hkpContactPointConfirmedEvent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpContactPointConfirmedEvent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpContactPointConfirmedEvent)
    HK_TRACKER_MEMBER(hkpContactPointConfirmedEvent, m_collidableA, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpContactPointConfirmedEvent, m_collidableB, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpContactPointConfirmedEvent, m_callbackFiredFrom, 0, "hkpEntity*") // class hkpEntity*
    HK_TRACKER_MEMBER(hkpContactPointConfirmedEvent, m_contactPoint, 0, "hkContactPoint*") // class hkContactPoint*
    HK_TRACKER_MEMBER(hkpContactPointConfirmedEvent, m_contactPointProperties, 0, "hkpContactPointProperties*") // class hkpContactPointProperties*
    HK_TRACKER_MEMBER(hkpContactPointConfirmedEvent, m_contactData, 0, "hkpSimpleContactConstraintData*") // const class hkpSimpleContactConstraintData*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpContactPointConfirmedEvent, s_libraryName)


// hkpContactPointRemovedEvent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpContactPointRemovedEvent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpContactPointRemovedEvent)
    HK_TRACKER_MEMBER(hkpContactPointRemovedEvent, m_contactPointProperties, 0, "hkpContactPointProperties*") // class hkpContactPointProperties*
    HK_TRACKER_MEMBER(hkpContactPointRemovedEvent, m_entityA, 0, "hkpEntity*") // class hkpEntity*
    HK_TRACKER_MEMBER(hkpContactPointRemovedEvent, m_entityB, 0, "hkpEntity*") // class hkpEntity*
    HK_TRACKER_MEMBER(hkpContactPointRemovedEvent, m_callbackFiredFrom, 0, "hkpEntity*") // class hkpEntity*
    HK_TRACKER_MEMBER(hkpContactPointRemovedEvent, m_internalContactMgr, 0, "hkpDynamicsContactMgr*") // class hkpDynamicsContactMgr*
    HK_TRACKER_MEMBER(hkpContactPointRemovedEvent, m_constraintOwner, 0, "hkpConstraintOwner*") // class hkpConstraintOwner*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpContactPointRemovedEvent, s_libraryName)


// hkpContactProcessEvent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpContactProcessEvent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpContactProcessEvent)
    HK_TRACKER_MEMBER(hkpContactProcessEvent, m_collidableA, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpContactProcessEvent, m_collidableB, 0, "hkpCollidable*") // const class hkpCollidable*
    HK_TRACKER_MEMBER(hkpContactProcessEvent, m_callbackFiredFrom, 0, "hkpEntity*") // class hkpEntity*
    HK_TRACKER_MEMBER(hkpContactProcessEvent, m_collisionData, 0, "hkpProcessCollisionData*") // struct hkpProcessCollisionData*
    HK_TRACKER_MEMBER(hkpContactProcessEvent, m_contactPointProperties, 0, "hkpContactPointProperties* [256]") // class hkpContactPointProperties* [256]
    HK_TRACKER_MEMBER(hkpContactProcessEvent, m_internalContactMgr, 0, "hkpDynamicsContactMgr*") // class hkpDynamicsContactMgr*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpContactProcessEvent, s_libraryName)

// None hkpContactPointAccept
HK_TRACKER_IMPLEMENT_SIMPLE(hkpContactPointAccept, s_libraryName)
#include <Physics2012/Dynamics/Collide/Deprecated/hkpCollisionListener.h>


// hkpCollisionListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCollisionListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCollisionListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCollisionListener, s_libraryName, hkpContactListener)

#include <Physics2012/Dynamics/Collide/Deprecated/hkpContactUpdater.h>


// hkpContactUpdateEvent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpContactUpdateEvent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpContactUpdateEvent)
    HK_TRACKER_MEMBER(hkpContactUpdateEvent, m_contactPointIds, 0, "hkInplaceArray<hkUint16, 256, hkContainerHeapAllocator>") // class hkInplaceArray< hkUint16, 256, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpContactUpdateEvent, m_callbackFiredFrom, 0, "hkpEntity*") // class hkpEntity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpContactUpdateEvent, s_libraryName)


// hkpContactUpdater ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpContactUpdater, s_libraryName)

#include <Physics2012/Dynamics/Collide/Filter/Constraint/hkpConstraintCollisionFilter.h>


// hkpConstraintCollisionFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintCollisionFilter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintCollisionFilter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConstraintCollisionFilter, s_libraryName, hkpPairCollisionFilter)

#include <Physics2012/Dynamics/Collide/Filter/Pair/hkpPairCollisionFilter.h>


// hkpPairCollisionFilter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPairCollisionFilter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PairFilterKey)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PairFilterPointerMapOperations)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MapPairFilterKeyOverrideType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPairCollisionFilter)
    HK_TRACKER_MEMBER(hkpPairCollisionFilter, m_disabledPairs, 0, "hkMap<hkpPairCollisionFilter::PairFilterKey, hkUint64, hkpPairCollisionFilter::PairFilterPointerMapOperations, hkContainerHeapAllocator>") // class hkMap< struct hkpPairCollisionFilter::PairFilterKey, hkUint64, struct hkpPairCollisionFilter::PairFilterPointerMapOperations, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpPairCollisionFilter, m_childFilter, 0, "hkpCollisionFilter*") // const class hkpCollisionFilter*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPairCollisionFilter, s_libraryName, hkpCollisionFilter)


// PairFilterKey hkpPairCollisionFilter

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPairCollisionFilter::PairFilterKey)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPairCollisionFilter::PairFilterKey)
    HK_TRACKER_MEMBER(hkpPairCollisionFilter::PairFilterKey, m_a, 0, "hkpEntity*") // class hkpEntity*
    HK_TRACKER_MEMBER(hkpPairCollisionFilter::PairFilterKey, m_b, 0, "hkpEntity*") // class hkpEntity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpPairCollisionFilter::PairFilterKey, s_libraryName)


// PairFilterPointerMapOperations hkpPairCollisionFilter
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpPairCollisionFilter, PairFilterPointerMapOperations, s_libraryName)


// MapPairFilterKeyOverrideType hkpPairCollisionFilter

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPairCollisionFilter::MapPairFilterKeyOverrideType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPairCollisionFilter::MapPairFilterKeyOverrideType)
    HK_TRACKER_MEMBER(hkpPairCollisionFilter::MapPairFilterKeyOverrideType, m_elem, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpPairCollisionFilter::MapPairFilterKeyOverrideType, s_libraryName)

#include <Physics2012/Dynamics/Collide/hkpDynamicsContactMgr.h>


// hkpDynamicsContactMgr ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDynamicsContactMgr)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDynamicsContactMgr)
    HK_TRACKER_MEMBER(hkpDynamicsContactMgr, m_world, 0, "hkpWorld*") // class hkpWorld*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpDynamicsContactMgr, s_libraryName, hkpContactMgr)

#include <Physics2012/Dynamics/Collide/hkpResponseModifier.h>


// hkpResponseModifier ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpResponseModifier, s_libraryName)

#include <Physics2012/Dynamics/Collide/hkpSimpleConstraintContactMgr.h>


// hkpSimpleConstraintContactMgr ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimpleConstraintContactMgr)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Factory)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSimpleConstraintContactMgr)
    HK_TRACKER_MEMBER(hkpSimpleConstraintContactMgr, m_contactConstraintData, 0, "hkpSimpleContactConstraintData") // class hkpSimpleContactConstraintData
    HK_TRACKER_MEMBER(hkpSimpleConstraintContactMgr, m_constraint, 0, "hkpConstraintInstance") // class hkpConstraintInstance
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSimpleConstraintContactMgr, s_libraryName, hkpDynamicsContactMgr)


// Factory hkpSimpleConstraintContactMgr

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimpleConstraintContactMgr::Factory)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSimpleConstraintContactMgr::Factory)
    HK_TRACKER_MEMBER(hkpSimpleConstraintContactMgr::Factory, m_world, 0, "hkpWorld*") // class hkpWorld*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSimpleConstraintContactMgr::Factory, s_libraryName, hkpContactMgrFactory)

#include <Physics2012/Dynamics/Common/hkpMaterial.h>


// hkpMaterial ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMaterial, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Atom/hkpModifierConstraintAtom.h>


// hkpModifierConstraintAtom ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpModifierConstraintAtom)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpModifierConstraintAtom)
    HK_TRACKER_MEMBER(hkpModifierConstraintAtom, m_child, 0, "hkpConstraintAtom*") // struct hkpConstraintAtom*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpModifierConstraintAtom, s_libraryName, hkpConstraintAtom)


// hkpMassChangerModifierConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMassChangerModifierConstraintAtom, s_libraryName)


// hkpCenterOfMassChangerModifierConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCenterOfMassChangerModifierConstraintAtom, s_libraryName)


// hkpSoftContactModifierConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSoftContactModifierConstraintAtom, s_libraryName)


// hkpViscousSurfaceModifierConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpViscousSurfaceModifierConstraintAtom, s_libraryName)


// hkpMovingSurfaceModifierConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMovingSurfaceModifierConstraintAtom, s_libraryName)


// hkpIgnoreModifierConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpIgnoreModifierConstraintAtom, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Atom/hkpSimpleContactConstraintAtom.h>


// hkpSimpleContactConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSimpleContactConstraintAtom, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Atom/hkpSimpleContactConstraintAtomUtil.h>


// hkpSimpleContactConstraintAtomUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSimpleContactConstraintAtomUtil, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Breakable/hkpBreakableConstraintData.h>


// hkpBreakableConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakableConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakableConstraintData)
    HK_TRACKER_MEMBER(hkpBreakableConstraintData, m_atoms, 0, "hkpBridgeAtoms") // struct hkpBridgeAtoms
    HK_TRACKER_MEMBER(hkpBreakableConstraintData, m_constraintData, 0, "hkpConstraintData*") // class hkpConstraintData*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBreakableConstraintData, s_libraryName, hkpConstraintData)


// Runtime hkpBreakableConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBreakableConstraintData, Runtime, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Breakable/hkpBreakableListener.h>


// hkpBreakableConstraintEvent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakableConstraintEvent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakableConstraintEvent)
    HK_TRACKER_MEMBER(hkpBreakableConstraintEvent, m_constraintInstance, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
    HK_TRACKER_MEMBER(hkpBreakableConstraintEvent, m_breakableConstraintData, 0, "hkpBreakableConstraintData*") // class hkpBreakableConstraintData*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpBreakableConstraintEvent, s_libraryName)


// hkpBreakableListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakableListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakableListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkpBreakableListener, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Chain/BallSocket/hkpBallSocketChainData.h>


// hkpBallSocketChainData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBallSocketChainData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConstraintInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBallSocketChainData)
    HK_TRACKER_MEMBER(hkpBallSocketChainData, m_atoms, 0, "hkpBridgeAtoms") // struct hkpBridgeAtoms
    HK_TRACKER_MEMBER(hkpBallSocketChainData, m_infos, 0, "hkArray<hkpBallSocketChainData::ConstraintInfo, hkContainerHeapAllocator>") // hkArray< struct hkpBallSocketChainData::ConstraintInfo, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBallSocketChainData, s_libraryName, hkpConstraintChainData)


// Runtime hkpBallSocketChainData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBallSocketChainData, Runtime, s_libraryName)


// ConstraintInfo hkpBallSocketChainData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBallSocketChainData, ConstraintInfo, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Chain/StiffSpring/hkpStiffSpringChainData.h>


// hkpStiffSpringChainData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpStiffSpringChainData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConstraintInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpStiffSpringChainData)
    HK_TRACKER_MEMBER(hkpStiffSpringChainData, m_atoms, 0, "hkpBridgeAtoms") // struct hkpBridgeAtoms
    HK_TRACKER_MEMBER(hkpStiffSpringChainData, m_infos, 0, "hkArray<hkpStiffSpringChainData::ConstraintInfo, hkContainerHeapAllocator>") // hkArray< struct hkpStiffSpringChainData::ConstraintInfo, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpStiffSpringChainData, s_libraryName, hkpConstraintChainData)


// Runtime hkpStiffSpringChainData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpStiffSpringChainData, Runtime, s_libraryName)


// ConstraintInfo hkpStiffSpringChainData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpStiffSpringChainData, ConstraintInfo, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Chain/hkpConstraintChainData.h>


// hkpConstraintChainData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintChainData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintChainData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpConstraintChainData, s_libraryName, hkpConstraintData)

#include <Physics2012/Dynamics/Constraint/Chain/hkpConstraintChainInstance.h>


// hkpConstraintChainInstance ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintChainInstance)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintChainInstance)
    HK_TRACKER_MEMBER(hkpConstraintChainInstance, m_chainedEntities, 0, "hkArray<hkpEntity*, hkContainerHeapAllocator>") // hkArray< class hkpEntity*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpConstraintChainInstance, m_action, 0, "hkpConstraintChainInstanceAction*") // class hkpConstraintChainInstanceAction*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConstraintChainInstance, s_libraryName, hkpConstraintInstance)

#include <Physics2012/Dynamics/Constraint/Chain/hkpConstraintChainInstanceAction.h>


// hkpConstraintChainInstanceAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintChainInstanceAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintChainInstanceAction)
    HK_TRACKER_MEMBER(hkpConstraintChainInstanceAction, m_constraintInstance, 0, "hkpConstraintChainInstance*") // class hkpConstraintChainInstance*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConstraintChainInstanceAction, s_libraryName, hkpAction)

#include <Physics2012/Dynamics/Constraint/ConstraintKit/hkpConstraintConstructionKit.h>


// hkpConstraintConstructionKit ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintConstructionKit)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintConstructionKit)
    HK_TRACKER_MEMBER(hkpConstraintConstructionKit, m_constraint, 0, "hkpGenericConstraintData*") // class hkpGenericConstraintData*
    HK_TRACKER_MEMBER(hkpConstraintConstructionKit, m_scheme, 0, "hkpGenericConstraintDataScheme*") // class hkpGenericConstraintDataScheme*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpConstraintConstructionKit, s_libraryName)

#include <Physics2012/Dynamics/Constraint/ConstraintKit/hkpConstraintModifier.h>

// hk.MemoryTracker ignore hkpConstraintModifier
#include <Physics2012/Dynamics/Constraint/ConstraintKit/hkpGenericConstraintData.h>


// hkpGenericConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpGenericConstraintData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpGenericConstraintData)
    HK_TRACKER_MEMBER(hkpGenericConstraintData, m_atoms, 0, "hkpBridgeAtoms") // struct hkpBridgeAtoms
    HK_TRACKER_MEMBER(hkpGenericConstraintData, m_scheme, 0, "hkpGenericConstraintDataScheme") // class hkpGenericConstraintDataScheme
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpGenericConstraintData, s_libraryName, hkpConstraintData)

#include <Physics2012/Dynamics/Constraint/ConstraintKit/hkpGenericConstraintParameters.h>


// hkpGenericConstraintDataParameters ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpGenericConstraintDataParameters)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpGenericConstraintDataParameters)
    HK_TRACKER_MEMBER(hkpGenericConstraintDataParameters, m_rbA, 0, "hkTransform*") // const hkTransform*
    HK_TRACKER_MEMBER(hkpGenericConstraintDataParameters, m_rbB, 0, "hkTransform*") // const hkTransform*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpGenericConstraintDataParameters, s_libraryName)

#include <Physics2012/Dynamics/Constraint/ConstraintKit/hkpGenericConstraintScheme.h>


// hkpGenericConstraintDataScheme ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpGenericConstraintDataScheme)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConstraintInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpGenericConstraintDataScheme)
    HK_TRACKER_MEMBER(hkpGenericConstraintDataScheme, m_data, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpGenericConstraintDataScheme, m_commands, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpGenericConstraintDataScheme, m_modifiers, 0, "hkArray<hkpConstraintModifier*, hkContainerHeapAllocator>") // hkArray< class hkpConstraintModifier*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpGenericConstraintDataScheme, m_motors, 0, "hkArray<hkpConstraintMotor*, hkContainerHeapAllocator>") // hkArray< class hkpConstraintMotor*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpGenericConstraintDataScheme, s_libraryName)


// ConstraintInfo hkpGenericConstraintDataScheme
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpGenericConstraintDataScheme, ConstraintInfo, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Contact/hkpContactImpulseLimitBreachedListener.h>


// hkpContactImpulseLimitBreachedListenerInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpContactImpulseLimitBreachedListenerInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SingleImpulseElem)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpContactImpulseLimitBreachedListenerInfo)
    HK_TRACKER_MEMBER(hkpContactImpulseLimitBreachedListenerInfo, m_data, 0, "hkpContactImpulseLimitBreachedListenerInfo::ListenerData") // struct hkpContactImpulseLimitBreachedListenerInfo::ListenerData
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpContactImpulseLimitBreachedListenerInfo, s_libraryName)


// SingleImpulseElem hkpContactImpulseLimitBreachedListenerInfo

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpContactImpulseLimitBreachedListenerInfo::SingleImpulseElem)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpContactImpulseLimitBreachedListenerInfo::SingleImpulseElem)
    HK_TRACKER_MEMBER(hkpContactImpulseLimitBreachedListenerInfo::SingleImpulseElem, m_constraintInstance, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
    HK_TRACKER_MEMBER(hkpContactImpulseLimitBreachedListenerInfo::SingleImpulseElem, m_properties, 0, "hkpContactPointProperties*") // class hkpContactPointProperties*
    HK_TRACKER_MEMBER(hkpContactImpulseLimitBreachedListenerInfo::SingleImpulseElem, m_contactPoint, 0, "hkContactPoint*") // class hkContactPoint*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpContactImpulseLimitBreachedListenerInfo::SingleImpulseElem, s_libraryName)


// hkpContactImpulseLimitBreachedListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpContactImpulseLimitBreachedListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpContactImpulseLimitBreachedListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkpContactImpulseLimitBreachedListener, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Contact/hkpContactPointProperties.h>


// hkpContactPointProperties ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpContactPointProperties, s_libraryName)


// hkContactPointPropertiesWithExtendedUserData16 ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkContactPointPropertiesWithExtendedUserData16, s_libraryName)


// hkpContactPointPropertiesStream ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpContactPointPropertiesStream, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Contact/hkpDynamicsCpIdMgr.h>


// hkpDynamicsCpIdMgr ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDynamicsCpIdMgr)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDynamicsCpIdMgr)
    HK_TRACKER_MEMBER(hkpDynamicsCpIdMgr, m_values, 0, "hkInplaceArray<hkUint8, 8, hkContainerHeapAllocator>") // class hkInplaceArray< hkUint8, 8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpDynamicsCpIdMgr, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Contact/hkpSimpleContactConstraintData.h>


// hkpSimpleContactConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimpleContactConstraintData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSimpleContactConstraintData)
    HK_TRACKER_MEMBER(hkpSimpleContactConstraintData, m_idMgrA, 0, "hkpDynamicsCpIdMgr") // class hkpDynamicsCpIdMgr
    HK_TRACKER_MEMBER(hkpSimpleContactConstraintData, m_clientData, 0, "void*") // void*
    HK_TRACKER_MEMBER(hkpSimpleContactConstraintData, m_constraint, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
    HK_TRACKER_MEMBER(hkpSimpleContactConstraintData, m_atom, 0, "hkpSimpleContactConstraintAtom*") // struct hkpSimpleContactConstraintAtom*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSimpleContactConstraintData, s_libraryName, hkpConstraintData)

#include <Physics2012/Dynamics/Constraint/Contact/hkpSimpleContactConstraintUtil.h>


// hkpSimpleContactConstraintUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSimpleContactConstraintUtil, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Malleable/hkpMalleableConstraintData.h>


// hkpMalleableConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMalleableConstraintData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMalleableConstraintData)
    HK_TRACKER_MEMBER(hkpMalleableConstraintData, m_constraintData, 0, "hkpConstraintData*") // class hkpConstraintData*
    HK_TRACKER_MEMBER(hkpMalleableConstraintData, m_atoms, 0, "hkpBridgeAtoms") // struct hkpBridgeAtoms
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMalleableConstraintData, s_libraryName, hkpConstraintData)

#include <Physics2012/Dynamics/Constraint/Response/hkpSimpleCollisionResponse.h>


// hkpSimpleCollisionResponse ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimpleCollisionResponse)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SolveSingleOutput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SolveSingleOutput2)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpSimpleCollisionResponse, s_libraryName)


// SolveSingleOutput hkpSimpleCollisionResponse
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSimpleCollisionResponse, SolveSingleOutput, s_libraryName)


// SolveSingleOutput2 hkpSimpleCollisionResponse
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSimpleCollisionResponse, SolveSingleOutput2, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Setup/hkpConstraintSolverSetup.h>


// hkpConstraintSchemaInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintSchemaInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintSchemaInfo)
    HK_TRACKER_MEMBER(hkpConstraintSchemaInfo, m_constraint, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
    HK_TRACKER_MEMBER(hkpConstraintSchemaInfo, m_schema, 0, "hkpJacobianSchema*") // class hkpJacobianSchema*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpConstraintSchemaInfo, s_libraryName)


// hkpBuildJacobianTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBuildJacobianTask)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AtomInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBuildJacobianTask)
    HK_TRACKER_MEMBER(hkpBuildJacobianTask, m_next, 0, "hkpBuildJacobianTask*") // struct hkpBuildJacobianTask*
    HK_TRACKER_MEMBER(hkpBuildJacobianTask, m_taskHeader, 0, "hkpBuildJacobianTaskHeader*") // struct hkpBuildJacobianTaskHeader*
    HK_TRACKER_MEMBER(hkpBuildJacobianTask, m_accumulators, 0, "hkpVelocityAccumulator*") // const class hkpVelocityAccumulator*
    HK_TRACKER_MEMBER(hkpBuildJacobianTask, m_schemas, 0, "hkpJacobianSchema*") // class hkpJacobianSchema*
    HK_TRACKER_MEMBER(hkpBuildJacobianTask, m_schemasOfNextTask, 0, "hkpJacobianSchema*") // class hkpJacobianSchema*
    HK_TRACKER_MEMBER(hkpBuildJacobianTask, m_atomInfos, 0, "hkpBuildJacobianTask::AtomInfo [144]") // struct hkpBuildJacobianTask::AtomInfo [144]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpBuildJacobianTask, s_libraryName)


// AtomInfo hkpBuildJacobianTask

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBuildJacobianTask::AtomInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBuildJacobianTask::AtomInfo)
    HK_TRACKER_MEMBER(hkpBuildJacobianTask::AtomInfo, m_atoms, 0, "hkpConstraintAtom*") // struct hkpConstraintAtom*
    HK_TRACKER_MEMBER(hkpBuildJacobianTask::AtomInfo, m_instance, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
    HK_TRACKER_MEMBER(hkpBuildJacobianTask::AtomInfo, m_runtime, 0, "hkpConstraintRuntime*") // struct hkpConstraintRuntime*
    HK_TRACKER_MEMBER(hkpBuildJacobianTask::AtomInfo, m_transformA, 0, "hkTransform*") // const hkTransform*
    HK_TRACKER_MEMBER(hkpBuildJacobianTask::AtomInfo, m_transformB, 0, "hkTransform*") // const hkTransform*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpBuildJacobianTask::AtomInfo, s_libraryName)


// hkpBuildJacobianTaskCollection ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBuildJacobianTaskCollection)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CallbackPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpBuildJacobianTaskCollection, s_libraryName)


// CallbackPair hkpBuildJacobianTaskCollection

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBuildJacobianTaskCollection::CallbackPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBuildJacobianTaskCollection::CallbackPair)
    HK_TRACKER_MEMBER(hkpBuildJacobianTaskCollection::CallbackPair, m_callbackConstraints, 0, "hkConstraintInternal*") // const struct hkConstraintInternal*
    HK_TRACKER_MEMBER(hkpBuildJacobianTaskCollection::CallbackPair, m_atomInfo, 0, "hkpBuildJacobianTask::AtomInfo*") // struct hkpBuildJacobianTask::AtomInfo*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpBuildJacobianTaskCollection::CallbackPair, s_libraryName)


// hkpSolveJacobiansTaskCollection ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSolveJacobiansTaskCollection)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSolveJacobiansTaskCollection)
    HK_TRACKER_MEMBER(hkpSolveJacobiansTaskCollection, m_firstSolveJacobiansTask, 0, "hkpSolveConstraintBatchTask*") // struct hkpSolveConstraintBatchTask*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpSolveJacobiansTaskCollection, s_libraryName)


// hkpBuildJacobianTaskHeader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBuildJacobianTaskHeader)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBuildJacobianTaskHeader)
    HK_TRACKER_MEMBER(hkpBuildJacobianTaskHeader, m_buffer, 0, "void*") // void*
    HK_TRACKER_MEMBER(hkpBuildJacobianTaskHeader, m_impulseLimitsBreached, 0, "hkpImpulseLimitBreachedHeader*") // struct hkpImpulseLimitBreachedHeader*
    HK_TRACKER_MEMBER(hkpBuildJacobianTaskHeader, m_accumulatorsBase, 0, "hkpVelocityAccumulator*") // class hkpVelocityAccumulator*
    HK_TRACKER_MEMBER(hkpBuildJacobianTaskHeader, m_accumulatorsEnd, 0, "hkpVelocityAccumulator*") // class hkpVelocityAccumulator*
    HK_TRACKER_MEMBER(hkpBuildJacobianTaskHeader, m_schemasBase, 0, "hkpJacobianSchema*") // class hkpJacobianSchema*
    HK_TRACKER_MEMBER(hkpBuildJacobianTaskHeader, m_solverTempBase, 0, "hkpSolverElemTemp*") // struct hkpSolverElemTemp*
    HK_TRACKER_MEMBER(hkpBuildJacobianTaskHeader, m_constraintQueryIn, 0, "hkpConstraintQueryIn*") // const class hkpConstraintQueryIn*
    HK_TRACKER_MEMBER(hkpBuildJacobianTaskHeader, m_newSplitIslands, 0, "hkArray<hkpSimulationIsland*, hkContainerHeapAllocator>") // hkArray< class hkpSimulationIsland*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpBuildJacobianTaskHeader, m_tasks, 0, "hkpBuildJacobianTaskCollection") // struct hkpBuildJacobianTaskCollection
    HK_TRACKER_MEMBER(hkpBuildJacobianTaskHeader, m_solveTasks, 0, "hkpSolveJacobiansTaskCollection") // struct hkpSolveJacobiansTaskCollection
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpBuildJacobianTaskHeader, s_libraryName)


// hkpSolveConstraintBatchTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSolveConstraintBatchTask)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSolveConstraintBatchTask)
    HK_TRACKER_MEMBER(hkpSolveConstraintBatchTask, m_next, 0, "hkpSolveConstraintBatchTask*") // struct hkpSolveConstraintBatchTask*
    HK_TRACKER_MEMBER(hkpSolveConstraintBatchTask, m_taskHeader, 0, "hkpBuildJacobianTaskHeader*") // struct hkpBuildJacobianTaskHeader*
    HK_TRACKER_MEMBER(hkpSolveConstraintBatchTask, m_accumulators, 0, "hkpVelocityAccumulator*") // class hkpVelocityAccumulator*
    HK_TRACKER_MEMBER(hkpSolveConstraintBatchTask, m_schemas, 0, "hkpJacobianSchema*") // const class hkpJacobianSchema*
    HK_TRACKER_MEMBER(hkpSolveConstraintBatchTask, m_solverElemTemp, 0, "hkpSolverElemTemp*") // struct hkpSolverElemTemp*
    HK_TRACKER_MEMBER(hkpSolveConstraintBatchTask, m_firstTaskInNextBatch, 0, "hkpSolveConstraintBatchTask*") // struct hkpSolveConstraintBatchTask*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpSolveConstraintBatchTask, s_libraryName)


// hkpConstraintSolverResources ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintSolverResources)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VelocityAccumTransformBackup)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintSolverResources)
    HK_TRACKER_MEMBER(hkpConstraintSolverResources, m_stepInfo, 0, "hkStepInfo*") // class hkStepInfo*
    HK_TRACKER_MEMBER(hkpConstraintSolverResources, m_solverInfo, 0, "hkpSolverInfo*") // struct hkpSolverInfo*
    HK_TRACKER_MEMBER(hkpConstraintSolverResources, m_constraintQueryInput, 0, "hkpConstraintQueryIn*") // class hkpConstraintQueryIn*
    HK_TRACKER_MEMBER(hkpConstraintSolverResources, m_accumulators, 0, "hkpVelocityAccumulator*") // class hkpVelocityAccumulator*
    HK_TRACKER_MEMBER(hkpConstraintSolverResources, m_accumulatorsEnd, 0, "hkpVelocityAccumulator*") // class hkpVelocityAccumulator*
    HK_TRACKER_MEMBER(hkpConstraintSolverResources, m_accumulatorsCurrent, 0, "hkpVelocityAccumulator*") // class hkpVelocityAccumulator*
    HK_TRACKER_MEMBER(hkpConstraintSolverResources, m_accumulatorsBackup, 0, "hkpConstraintSolverResources::VelocityAccumTransformBackup*") // struct hkpConstraintSolverResources::VelocityAccumTransformBackup*
    HK_TRACKER_MEMBER(hkpConstraintSolverResources, m_accumulatorsBackupEnd, 0, "hkpConstraintSolverResources::VelocityAccumTransformBackup*") // struct hkpConstraintSolverResources::VelocityAccumTransformBackup*
    HK_TRACKER_MEMBER(hkpConstraintSolverResources, m_schemas, 0, "hkpConstraintSolverResources::BufferState<hkpJacobianSchema> [3]") // struct hkpConstraintSolverResources::BufferState< class hkpJacobianSchema > [3]
    HK_TRACKER_MEMBER(hkpConstraintSolverResources, m_elemTemp, 0, "hkpSolverElemTemp*") // struct hkpSolverElemTemp*
    HK_TRACKER_MEMBER(hkpConstraintSolverResources, m_elemTempEnd, 0, "hkpSolverElemTemp*") // struct hkpSolverElemTemp*
    HK_TRACKER_MEMBER(hkpConstraintSolverResources, m_elemTempCurrent, 0, "hkpSolverElemTemp*") // struct hkpSolverElemTemp*
    HK_TRACKER_MEMBER(hkpConstraintSolverResources, m_elemTempLastProcessed, 0, "hkpSolverElemTemp*") // struct hkpSolverElemTemp*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpConstraintSolverResources, s_libraryName)


// VelocityAccumTransformBackup hkpConstraintSolverResources
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConstraintSolverResources, VelocityAccumTransformBackup, s_libraryName)


// hkpConstraintSolverSetup ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConstraintSolverSetup, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintCallbackUtil.h>


// hkpConstraintCallbackUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConstraintCallbackUtil, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintChainLengthUtil.h>


// hkpConstraintChainLengthUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintChainLengthUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(RopeInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintChainLengthUtil)
    HK_TRACKER_MEMBER(hkpConstraintChainLengthUtil, m_segmentCinfo, 0, "hkpRigidBodyCinfo") // class hkpRigidBodyCinfo
    HK_TRACKER_MEMBER(hkpConstraintChainLengthUtil, m_instance, 0, "hkpConstraintChainInstance*") // class hkpConstraintChainInstance*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpConstraintChainLengthUtil, s_libraryName)


// RopeInfo hkpConstraintChainLengthUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConstraintChainLengthUtil, RopeInfo, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintDataCloningUtil.h>


// hkpConstraintDataCloningUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConstraintDataCloningUtil, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintPivotsUtil.h>


// hkpConstraintPivotsUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConstraintPivotsUtil, s_libraryName)

#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintStabilizationUtil.h>


// hkpConstraintStabilizationUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConstraintStabilizationUtil, s_libraryName)

#include <Physics2012/Dynamics/Constraint/hkpConstraintInstance.h>


// hkpConstraintInstance ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintInstance)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SmallArraySerializeOverrideType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConstraintPriority)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(InstanceType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AddReferences)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CloningMode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(OnDestructionRemapInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintInstance)
    HK_TRACKER_MEMBER(hkpConstraintInstance, m_owner, 0, "hkpConstraintOwner*") // class hkpConstraintOwner*
    HK_TRACKER_MEMBER(hkpConstraintInstance, m_data, 0, "hkpConstraintData*") // class hkpConstraintData*
    HK_TRACKER_MEMBER(hkpConstraintInstance, m_constraintModifiers, 0, "hkpModifierConstraintAtom*") // struct hkpModifierConstraintAtom*
    HK_TRACKER_MEMBER(hkpConstraintInstance, m_entities, 0, "hkpEntity* [2]") // class hkpEntity* [2]
    HK_TRACKER_MEMBER(hkpConstraintInstance, m_listeners, 0, "hkSmallArray<hkpConstraintListener*>") // class hkSmallArray< class hkpConstraintListener* >
    HK_TRACKER_MEMBER(hkpConstraintInstance, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkpConstraintInstance, m_internal, 0, "hkConstraintInternal*") // struct hkConstraintInternal*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConstraintInstance, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConstraintInstance, ConstraintPriority, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConstraintInstance, InstanceType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConstraintInstance, AddReferences, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConstraintInstance, CloningMode, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConstraintInstance, OnDestructionRemapInfo, s_libraryName)


// SmallArraySerializeOverrideType hkpConstraintInstance

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintInstance::SmallArraySerializeOverrideType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintInstance::SmallArraySerializeOverrideType)
    HK_TRACKER_MEMBER(hkpConstraintInstance::SmallArraySerializeOverrideType, m_data, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpConstraintInstance::SmallArraySerializeOverrideType, s_libraryName)


// hkConstraintInternal ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkConstraintInternal)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkConstraintInternal)
    HK_TRACKER_MEMBER(hkConstraintInternal, m_constraint, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
    HK_TRACKER_MEMBER(hkConstraintInternal, m_entities, 0, "hkpEntity* [2]") // class hkpEntity* [2]
    HK_TRACKER_MEMBER(hkConstraintInternal, m_atoms, 0, "hkpConstraintAtom*") // struct hkpConstraintAtom*
    HK_TRACKER_MEMBER(hkConstraintInternal, m_runtime, 0, "hkpConstraintRuntime*") // struct hkpConstraintRuntime*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkConstraintInternal, s_libraryName)

#include <Physics2012/Dynamics/Constraint/hkpConstraintListener.h>


// hkpConstraintTrackerData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintTrackerData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintTrackerData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConstraintTrackerData, s_libraryName, hkReferencedObject)


// hkpConstraintBrokenEvent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintBrokenEvent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintBrokenEvent)
    HK_TRACKER_MEMBER(hkpConstraintBrokenEvent, m_world, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpConstraintBrokenEvent, m_constraintInstance, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
    HK_TRACKER_MEMBER(hkpConstraintBrokenEvent, m_eventSource, 0, "hkClass*") // const hkClass*
    HK_TRACKER_MEMBER(hkpConstraintBrokenEvent, m_eventSourceDetails, 0, "hkpConstraintTrackerData*") // class hkpConstraintTrackerData*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpConstraintBrokenEvent, s_libraryName)


// hkpConstraintRepairedEvent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintRepairedEvent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintRepairedEvent)
    HK_TRACKER_MEMBER(hkpConstraintRepairedEvent, m_world, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpConstraintRepairedEvent, m_constraintInstance, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
    HK_TRACKER_MEMBER(hkpConstraintRepairedEvent, m_eventSource, 0, "hkClass*") // const hkClass*
    HK_TRACKER_MEMBER(hkpConstraintRepairedEvent, m_eventSourceDetails, 0, "hkpConstraintTrackerData*") // class hkpConstraintTrackerData*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpConstraintRepairedEvent, s_libraryName)


// hkpConstraintListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpConstraintListener, s_libraryName)

#include <Physics2012/Dynamics/Constraint/hkpConstraintOwner.h>


// hkpConstraintOwner ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintOwner)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintOwner)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConstraintOwner, s_libraryName, hkReferencedObject)

#include <Physics2012/Dynamics/Destruction/BreakableBody/hkpBreakableBody.h>


// hkpBreakableBody ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakableBody)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Controller)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakableBody)
    HK_TRACKER_MEMBER(hkpBreakableBody, m_controller, 0, "hkpBreakableBody::Controller *") // class hkpBreakableBody::Controller *
    HK_TRACKER_MEMBER(hkpBreakableBody, m_breakableShape, 0, "hkpBreakableShape *") // const class hkpBreakableShape *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpBreakableBody, s_libraryName, hkReferencedObject)


// Controller hkpBreakableBody

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakableBody::Controller)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakableBody::Controller)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBreakableBody::Controller, s_libraryName, hkReferencedObject)

#include <Physics2012/Dynamics/Destruction/BreakableMaterial/MultiMaterial/CompoundShape/hkpScsBreakableMaterial.h>


// hkpStaticCompoundShapeBreakableMaterial ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpStaticCompoundShapeBreakableMaterial)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpStaticCompoundShapeBreakableMaterial)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpStaticCompoundShapeBreakableMaterial, s_libraryName, hkpBreakableMultiMaterial)

#include <Physics2012/Dynamics/Destruction/BreakableMaterial/MultiMaterial/ExtendedMeshShape/hkpEmsBreakableMaterial.h>


// hkpExtendedMeshShapeBreakableMaterial ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpExtendedMeshShapeBreakableMaterial)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpExtendedMeshShapeBreakableMaterial)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpExtendedMeshShapeBreakableMaterial, s_libraryName, hkpBreakableMultiMaterial)

#include <Physics2012/Dynamics/Destruction/BreakableMaterial/MultiMaterial/ListShape/hkpListShapeBreakableMaterial.h>


// hkpListShapeBreakableMaterial ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpListShapeBreakableMaterial)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpListShapeBreakableMaterial)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpListShapeBreakableMaterial, s_libraryName, hkpBreakableMultiMaterial)

#include <Physics2012/Dynamics/Destruction/BreakableMaterial/MultiMaterial/hkpBreakableMultiMaterial.h>


// hkpBreakableMultiMaterial ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakableMultiMaterial)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(InverseMappingDescriptor)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(InverseMapping)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakableMultiMaterial)
    HK_TRACKER_MEMBER(hkpBreakableMultiMaterial, m_subMaterials, 0, "hkArray<hkpBreakableMaterial *, hkContainerHeapAllocator>") // hkArray< class hkpBreakableMaterial *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpBreakableMultiMaterial, m_inverseMapping, 0, "hkpBreakableMultiMaterial::InverseMapping *") // struct hkpBreakableMultiMaterial::InverseMapping *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpBreakableMultiMaterial, s_libraryName, hkpBreakableMaterial)


// InverseMappingDescriptor hkpBreakableMultiMaterial
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBreakableMultiMaterial, InverseMappingDescriptor, s_libraryName)


// InverseMapping hkpBreakableMultiMaterial

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakableMultiMaterial::InverseMapping)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakableMultiMaterial::InverseMapping)
    HK_TRACKER_MEMBER(hkpBreakableMultiMaterial::InverseMapping, m_descriptors, 0, "hkArray<hkpBreakableMultiMaterial::InverseMappingDescriptor, hkContainerHeapAllocator>") // hkArray< struct hkpBreakableMultiMaterial::InverseMappingDescriptor, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpBreakableMultiMaterial::InverseMapping, m_subShapeIds, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBreakableMultiMaterial::InverseMapping, s_libraryName, hkReferencedObject)

#include <Physics2012/Dynamics/Destruction/BreakableMaterial/Simple/hkpSimpleBreakableMaterial.h>


// hkpSimpleBreakableMaterial ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimpleBreakableMaterial)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSimpleBreakableMaterial)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSimpleBreakableMaterial, s_libraryName, hkpBreakableMaterial)

#include <Physics2012/Dynamics/Destruction/BreakableMaterial/hkpBreakableMaterial.h>


// hkpBreakableMaterial ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakableMaterial)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ShapeKeyCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakableMaterial)
    HK_TRACKER_MEMBER(hkpBreakableMaterial, m_properties, 0, "hkRefCountedProperties*") // class hkRefCountedProperties*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpBreakableMaterial, s_libraryName, hkReferencedObject)


// ShapeKeyCollector hkpBreakableMaterial

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakableMaterial::ShapeKeyCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakableMaterial::ShapeKeyCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkpBreakableMaterial::ShapeKeyCollector, s_libraryName)

#include <Physics2012/Dynamics/Destruction/BreakableMaterial/hkpBreakableMaterialUtil.h>


// hkpBreakableMaterialUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpBreakableMaterialUtil, s_libraryName)

#include <Physics2012/Dynamics/Destruction/BreakableShape/hkpBreakableShape.h>


// hkpBreakableShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBreakableShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBreakableShape)
    HK_TRACKER_MEMBER(hkpBreakableShape, m_physicsShape, 0, "hkcdShape *") // const class hkcdShape *
    HK_TRACKER_MEMBER(hkpBreakableShape, m_material, 0, "hkpBreakableMaterial *") // class hkpBreakableMaterial *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBreakableShape, s_libraryName, hkReferencedObject)

#include <Physics2012/Dynamics/Destruction/Utilities/hkpDestructionBreakOffUtil.h>


// hkpDestructionBreakOffUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDestructionBreakOffUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GameControlFunctor)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ContactListener)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ContactListenerSpu)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ContactListenerPpu)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BreakOffGameControlResult)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDestructionBreakOffUtil)
    HK_TRACKER_MEMBER(hkpDestructionBreakOffUtil, m_entityContactsListener, 0, "hkpDestructionBreakOffUtil::ContactListener*") // class hkpDestructionBreakOffUtil::ContactListener*
    HK_TRACKER_MEMBER(hkpDestructionBreakOffUtil, m_criticalSection, 0, "hkCriticalSection*") // class hkCriticalSection*
    HK_TRACKER_MEMBER(hkpDestructionBreakOffUtil, m_breakOffPartsListener, 0, "hkpBreakOffPartsListener*") // class hkpBreakOffPartsListener*
    HK_TRACKER_MEMBER(hkpDestructionBreakOffUtil, m_breakOffControlFunctor, 0, "hkpDestructionBreakOffUtil::GameControlFunctor *") // class hkpDestructionBreakOffUtil::GameControlFunctor *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDestructionBreakOffUtil, s_libraryName, hkpWorldExtension)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpDestructionBreakOffUtil, BreakOffGameControlResult, s_libraryName)


// GameControlFunctor hkpDestructionBreakOffUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDestructionBreakOffUtil::GameControlFunctor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDestructionBreakOffUtil::GameControlFunctor)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDestructionBreakOffUtil::GameControlFunctor, s_libraryName, hkReferencedObject)


// ContactListener hkpDestructionBreakOffUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDestructionBreakOffUtil::ContactListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDestructionBreakOffUtil::ContactListener)
    HK_TRACKER_MEMBER(hkpDestructionBreakOffUtil::ContactListener, m_breakOffUtil, 0, "hkpDestructionBreakOffUtil*") // class hkpDestructionBreakOffUtil*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDestructionBreakOffUtil::ContactListener, s_libraryName, hkReferencedObject)


// ContactListenerSpu hkpDestructionBreakOffUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDestructionBreakOffUtil::ContactListenerSpu)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDestructionBreakOffUtil::ContactListenerSpu)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDestructionBreakOffUtil::ContactListenerSpu, s_libraryName, hkpDestructionBreakOffUtil::ContactListener)


// ContactListenerPpu hkpDestructionBreakOffUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDestructionBreakOffUtil::ContactListenerPpu)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDestructionBreakOffUtil::ContactListenerPpu)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDestructionBreakOffUtil::ContactListenerPpu, s_libraryName, hkpDestructionBreakOffUtil::ContactListener)

#include <Physics2012/Dynamics/Entity/Util/hkpEntityAabbUtil.h>


// hkValueIndexPair ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkValueIndexPair, s_libraryName)


// hkpEntityAabbUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpEntityAabbUtil, s_libraryName)

#include <Physics2012/Dynamics/Entity/Util/hkpEntityCallbackUtil.h>


// hkpEntityCallbackUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpEntityCallbackUtil, s_libraryName)

#include <Physics2012/Dynamics/Entity/hkpEntity.h>


// hkpEntity ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpEntity)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SmallArraySerializeOverrideType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SpuCollisionCallback)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ExtendedListeners)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SpuCollisionCallbackEventFilter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpEntity)
    HK_TRACKER_MEMBER(hkpEntity, m_limitContactImpulseUtilAndFlag, 0, "void*") // void*
    HK_TRACKER_MEMBER(hkpEntity, m_breakableBody, 0, "hkpBreakableBody*") // class hkpBreakableBody*
    HK_TRACKER_MEMBER(hkpEntity, m_constraintsMaster, 0, "hkSmallArray<hkConstraintInternal>") // class hkSmallArray< struct hkConstraintInternal >
    HK_TRACKER_MEMBER(hkpEntity, m_constraintsSlave, 0, "hkArray<hkpConstraintInstance*, hkContainerHeapAllocator>") // hkArray< class hkpConstraintInstance*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpEntity, m_constraintRuntime, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpEntity, m_simulationIsland, 0, "hkpSimulationIsland*") // class hkpSimulationIsland*
    HK_TRACKER_MEMBER(hkpEntity, m_spuCollisionCallback, 0, "hkpEntity::SpuCollisionCallback") // struct hkpEntity::SpuCollisionCallback
    HK_TRACKER_MEMBER(hkpEntity, m_motion, 0, "hkpMaxSizeMotion") // class hkpMaxSizeMotion
    HK_TRACKER_MEMBER(hkpEntity, m_contactListeners, 0, "hkSmallArray<hkpContactListener*>") // class hkSmallArray< class hkpContactListener* >
    HK_TRACKER_MEMBER(hkpEntity, m_actions, 0, "hkSmallArray<hkpAction*>") // class hkSmallArray< class hkpAction* >
    HK_TRACKER_MEMBER(hkpEntity, m_localFrame, 0, "hkLocalFrame *") // class hkLocalFrame *
    HK_TRACKER_MEMBER(hkpEntity, m_extendedListeners, 0, "hkpEntity::ExtendedListeners*") // struct hkpEntity::ExtendedListeners*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpEntity, s_libraryName, hkpWorldObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpEntity, SpuCollisionCallbackEventFilter, s_libraryName)


// SmallArraySerializeOverrideType hkpEntity

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpEntity::SmallArraySerializeOverrideType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpEntity::SmallArraySerializeOverrideType)
    HK_TRACKER_MEMBER(hkpEntity::SmallArraySerializeOverrideType, m_data, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpEntity::SmallArraySerializeOverrideType, s_libraryName)


// SpuCollisionCallback hkpEntity

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpEntity::SpuCollisionCallback)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpEntity::SpuCollisionCallback)
    HK_TRACKER_MEMBER(hkpEntity::SpuCollisionCallback, m_util, 0, "hkSpuCollisionCallbackUtil*") // class hkSpuCollisionCallbackUtil*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpEntity::SpuCollisionCallback, s_libraryName)


// ExtendedListeners hkpEntity

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpEntity::ExtendedListeners)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpEntity::ExtendedListeners)
    HK_TRACKER_MEMBER(hkpEntity::ExtendedListeners, m_activationListeners, 0, "hkSmallArray<hkpEntityActivationListener*>") // class hkSmallArray< class hkpEntityActivationListener* >
    HK_TRACKER_MEMBER(hkpEntity::ExtendedListeners, m_entityListeners, 0, "hkSmallArray<hkpEntityListener*>") // class hkSmallArray< class hkpEntityListener* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpEntity::ExtendedListeners, s_libraryName)

#include <Physics2012/Dynamics/Entity/hkpEntityActivationListener.h>

// hk.MemoryTracker ignore hkpEntityActivationListener
#include <Physics2012/Dynamics/Entity/hkpEntityListener.h>

// hk.MemoryTracker ignore hkpEntityListener
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>


// hkpRigidBody ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRigidBody)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRigidBody)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRigidBody, s_libraryName, hkpEntity)

#include <Physics2012/Dynamics/Entity/hkpRigidBodyCinfo.h>


// hkpRigidBodyCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRigidBodyCinfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SolverDeactivation)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRigidBodyCinfo)
    HK_TRACKER_MEMBER(hkpRigidBodyCinfo, m_shape, 0, "hkpShape*") // const class hkpShape*
    HK_TRACKER_MEMBER(hkpRigidBodyCinfo, m_localFrame, 0, "hkLocalFrame*") // class hkLocalFrame*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpRigidBodyCinfo, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpRigidBodyCinfo, SolverDeactivation, s_libraryName)

#include <Physics2012/Dynamics/Motion/Rigid/ThinBoxMotion/hkpThinBoxMotion.h>


// hkpThinBoxMotion ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpThinBoxMotion)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpThinBoxMotion)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpThinBoxMotion, s_libraryName, hkpBoxMotion)

#include <Physics2012/Dynamics/Motion/Rigid/hkpBoxMotion.h>


// hkpBoxMotion ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBoxMotion)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBoxMotion)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBoxMotion, s_libraryName, hkpMotion)

#include <Physics2012/Dynamics/Motion/Rigid/hkpCharacterMotion.h>


// hkpCharacterMotion ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCharacterMotion)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCharacterMotion)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCharacterMotion, s_libraryName, hkpMotion)

#include <Physics2012/Dynamics/Motion/Rigid/hkpFixedRigidMotion.h>


// hkpFixedRigidMotion ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpFixedRigidMotion)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpFixedRigidMotion)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpFixedRigidMotion, s_libraryName, hkpKeyframedRigidMotion)

#include <Physics2012/Dynamics/Motion/Rigid/hkpKeyframedRigidMotion.h>


// hkpKeyframedRigidMotion ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpKeyframedRigidMotion)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpKeyframedRigidMotion)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpKeyframedRigidMotion, s_libraryName, hkpMotion)


// hkpMaxSizeMotion ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMaxSizeMotion)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMaxSizeMotion)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMaxSizeMotion, s_libraryName, hkpKeyframedRigidMotion)

#include <Physics2012/Dynamics/Motion/Rigid/hkpSphereMotion.h>


// hkpSphereMotion ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSphereMotion)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSphereMotion)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSphereMotion, s_libraryName, hkpMotion)

#include <Physics2012/Dynamics/Motion/hkpMotion.h>


// hkpMotion ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMotion)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MotionType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMotion)
    HK_TRACKER_MEMBER(hkpMotion, m_savedMotion, 0, "hkpMaxSizeMotion*") // class hkpMaxSizeMotion*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpMotion, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMotion, MotionType, s_libraryName)


// hkpRigidMotion ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRigidMotion)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRigidMotion)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpRigidMotion, s_libraryName, hkpMotion)

#include <Physics2012/Dynamics/Phantom/hkpAabbPhantom.h>


// hkpAabbPhantom ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAabbPhantom)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpAabbPhantom)
    HK_TRACKER_MEMBER(hkpAabbPhantom, m_overlappingCollidables, 0, "hkArray<hkpCollidable*, hkContainerHeapAllocator>") // hkArray< class hkpCollidable*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpAabbPhantom, s_libraryName, hkpPhantom)

#include <Physics2012/Dynamics/Phantom/hkpCachingShapePhantom.h>


// hkpCachingShapePhantom ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCachingShapePhantom)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CollisionDetail)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCachingShapePhantom)
    HK_TRACKER_MEMBER(hkpCachingShapePhantom, m_collisionDetails, 0, "hkArray<hkpCachingShapePhantom::CollisionDetail, hkContainerHeapAllocator>") // hkArray< struct hkpCachingShapePhantom::CollisionDetail, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCachingShapePhantom, s_libraryName, hkpShapePhantom)


// CollisionDetail hkpCachingShapePhantom

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCachingShapePhantom::CollisionDetail)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCachingShapePhantom::CollisionDetail)
    HK_TRACKER_MEMBER(hkpCachingShapePhantom::CollisionDetail, m_agent, 0, "hkpCollisionAgent*") // class hkpCollisionAgent*
    HK_TRACKER_MEMBER(hkpCachingShapePhantom::CollisionDetail, m_collidable, 0, "hkpCollidable*") // class hkpCollidable*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCachingShapePhantom::CollisionDetail, s_libraryName)

#include <Physics2012/Dynamics/Phantom/hkpPhantom.h>


// hkpPhantom ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPhantom)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPhantom)
    HK_TRACKER_MEMBER(hkpPhantom, m_overlapListeners, 0, "hkArray<hkpPhantomOverlapListener*, hkContainerHeapAllocator>") // hkArray< class hkpPhantomOverlapListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpPhantom, m_phantomListeners, 0, "hkArray<hkpPhantomListener*, hkContainerHeapAllocator>") // hkArray< class hkpPhantomListener*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpPhantom, s_libraryName, hkpWorldObject)

#include <Physics2012/Dynamics/Phantom/hkpPhantomBroadPhaseListener.h>


// hkpPhantomBroadPhaseListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPhantomBroadPhaseListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPhantomBroadPhaseListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPhantomBroadPhaseListener, s_libraryName, hkReferencedObject)

#include <Physics2012/Dynamics/Phantom/hkpPhantomListener.h>

// hk.MemoryTracker ignore hkpPhantomListener
#include <Physics2012/Dynamics/Phantom/hkpPhantomOverlapListener.h>


// hkpCollidableAddedEvent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCollidableAddedEvent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCollidableAddedEvent)
    HK_TRACKER_MEMBER(hkpCollidableAddedEvent, m_phantom, 0, "hkpPhantom*") // const class hkpPhantom*
    HK_TRACKER_MEMBER(hkpCollidableAddedEvent, m_collidable, 0, "hkpCollidable*") // const class hkpCollidable*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCollidableAddedEvent, s_libraryName)


// hkpCollidableRemovedEvent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCollidableRemovedEvent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCollidableRemovedEvent)
    HK_TRACKER_MEMBER(hkpCollidableRemovedEvent, m_phantom, 0, "hkpPhantom*") // const class hkpPhantom*
    HK_TRACKER_MEMBER(hkpCollidableRemovedEvent, m_collidable, 0, "hkpCollidable*") // const class hkpCollidable*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCollidableRemovedEvent, s_libraryName)


// hkpPhantomOverlapListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPhantomOverlapListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPhantomOverlapListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkpPhantomOverlapListener, s_libraryName)

// None hkpCollidableAccept
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCollidableAccept, s_libraryName)
#include <Physics2012/Dynamics/Phantom/hkpPhantomType.h>

// None hkpPhantomType
HK_TRACKER_IMPLEMENT_SIMPLE(hkpPhantomType, s_libraryName)
#include <Physics2012/Dynamics/Phantom/hkpShapePhantom.h>


// hkpShapePhantom ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapePhantom)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapePhantom)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpShapePhantom, s_libraryName, hkpPhantom)

#include <Physics2012/Dynamics/Phantom/hkpSimpleShapePhantom.h>


// hkpSimpleShapePhantom ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimpleShapePhantom)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CollisionDetail)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSimpleShapePhantom)
    HK_TRACKER_MEMBER(hkpSimpleShapePhantom, m_collisionDetails, 0, "hkArray<hkpSimpleShapePhantom::CollisionDetail, hkContainerHeapAllocator>") // hkArray< struct hkpSimpleShapePhantom::CollisionDetail, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSimpleShapePhantom, s_libraryName, hkpShapePhantom)


// CollisionDetail hkpSimpleShapePhantom

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimpleShapePhantom::CollisionDetail)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSimpleShapePhantom::CollisionDetail)
    HK_TRACKER_MEMBER(hkpSimpleShapePhantom::CollisionDetail, m_collidable, 0, "hkpCollidable*") // class hkpCollidable*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpSimpleShapePhantom::CollisionDetail, s_libraryName)

#include <Physics2012/Dynamics/World/BroadPhaseBorder/hkpBroadPhaseBorder.h>


// hkpBroadPhaseBorder ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBroadPhaseBorder)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBroadPhaseBorder)
    HK_TRACKER_MEMBER(hkpBroadPhaseBorder, m_world, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpBroadPhaseBorder, m_phantoms, 0, "hkpPhantom* [6]") // class hkpPhantom* [6]
    HK_TRACKER_MEMBER(hkpBroadPhaseBorder, m_entitiesExitingBroadPhase, 0, "hkArray<hkpEntity*, hkContainerHeapAllocator>") // hkArray< class hkpEntity*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBroadPhaseBorder, s_libraryName, hkReferencedObject)

#include <Physics2012/Dynamics/World/CommandQueue/hkpPhysicsCommand.h>


// hkpPhysicsCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpPhysicsCommand, s_libraryName)


// hkpConstraintInfoExtended ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintInfoExtended)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintInfoExtended)
    HK_TRACKER_MEMBER(hkpConstraintInfoExtended, m_constraint, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConstraintInfoExtended, s_libraryName, hkpConstraintInfo)

#include <Physics2012/Dynamics/World/CommandQueue/hkpPhysicsCommandQueue.h>


// hkpPhysicsCommandQueue ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpPhysicsCommandQueue, s_libraryName)

#include <Physics2012/Dynamics/World/Extensions/hkpWorldExtension.h>


// hkpWorldExtension ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldExtension)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldExtension)
    HK_TRACKER_MEMBER(hkpWorldExtension, m_world, 0, "hkpWorld*") // class hkpWorld*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpWorldExtension, s_libraryName, hkReferencedObject)

// None hkpKnownWorldExtensionIds
HK_TRACKER_IMPLEMENT_SIMPLE(hkpKnownWorldExtensionIds, s_libraryName)
#include <Physics2012/Dynamics/World/Listener/hkpIslandActivationListener.h>

// hk.MemoryTracker ignore hkpIslandActivationListener
#include <Physics2012/Dynamics/World/Listener/hkpIslandPostCollideListener.h>

// hk.MemoryTracker ignore hkpIslandPostCollideListener
#include <Physics2012/Dynamics/World/Listener/hkpIslandPostIntegrateListener.h>

// hk.MemoryTracker ignore hkpIslandPostIntegrateListener
#include <Physics2012/Dynamics/World/Listener/hkpWorldDeletionListener.h>

// hk.MemoryTracker ignore hkpWorldDeletionListener
#include <Physics2012/Dynamics/World/Listener/hkpWorldPostCollideListener.h>

// hk.MemoryTracker ignore hkpWorldPostCollideListener
#include <Physics2012/Dynamics/World/Listener/hkpWorldPostIntegrateListener.h>

// hk.MemoryTracker ignore hkpWorldPostIntegrateListener
#include <Physics2012/Dynamics/World/Listener/hkpWorldPostSimulationListener.h>

// hk.MemoryTracker ignore hkpWorldPostSimulationListener
#include <Physics2012/Dynamics/World/Memory/Default/hkpDefaultWorldMemoryWatchDog.h>


// hkpDefaultWorldMemoryWatchDog ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDefaultWorldMemoryWatchDog)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDefaultWorldMemoryWatchDog)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDefaultWorldMemoryWatchDog, s_libraryName, hkWorldMemoryAvailableWatchDog)

#include <Physics2012/Dynamics/World/Memory/hkpWorldMemoryAvailableWatchDog.h>


// hkWorldMemoryAvailableWatchDog ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldMemoryAvailableWatchDog)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MemUsageInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldMemoryAvailableWatchDog)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkWorldMemoryAvailableWatchDog, s_libraryName, hkReferencedObject)


// MemUsageInfo hkWorldMemoryAvailableWatchDog

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldMemoryAvailableWatchDog::MemUsageInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldMemoryAvailableWatchDog::MemUsageInfo)
    HK_TRACKER_MEMBER(hkWorldMemoryAvailableWatchDog::MemUsageInfo, m_largestSimulationIsland, 0, "hkpSimulationIsland*") // class hkpSimulationIsland*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkWorldMemoryAvailableWatchDog::MemUsageInfo, s_libraryName)

#include <Physics2012/Dynamics/World/Simulation/hkpSimulation.h>


// hkpSimulation ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimulation)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FindContacts)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ResetCollisionInformation)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LastProcessingStep)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSimulation)
    HK_TRACKER_MEMBER(hkpSimulation, m_world, 0, "hkpWorld*") // class hkpWorld*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSimulation, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSimulation, FindContacts, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSimulation, ResetCollisionInformation, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSimulation, LastProcessingStep, s_libraryName)

#include <Physics2012/Dynamics/World/Util/BroadPhase/hkpBroadPhaseBorderListener.h>


// hkpBroadPhaseBorderListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBroadPhaseBorderListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBroadPhaseBorderListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBroadPhaseBorderListener, s_libraryName, hkReferencedObject)

#include <Physics2012/Dynamics/World/Util/BroadPhase/hkpEntityEntityBroadPhaseListener.h>


// hkpEntityEntityBroadPhaseListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpEntityEntityBroadPhaseListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpEntityEntityBroadPhaseListener)
    HK_TRACKER_MEMBER(hkpEntityEntityBroadPhaseListener, m_world, 0, "hkpWorld*") // class hkpWorld*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpEntityEntityBroadPhaseListener, s_libraryName, hkReferencedObject)

#include <Physics2012/Dynamics/World/Util/hkpBodyOperation.h>


// hkpBodyOperation ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBodyOperation)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(UpdateInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ExecutionState)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBodyOperation)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpBodyOperation, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBodyOperation, ExecutionState, s_libraryName)


// UpdateInfo hkpBodyOperation
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBodyOperation, UpdateInfo, s_libraryName)

#include <Physics2012/Dynamics/World/Util/hkpNullAction.h>


// hkpNullAction ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpNullAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpNullAction)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpNullAction, s_libraryName, hkpAction)

#include <Physics2012/Dynamics/World/Util/hkpWorldAgentUtil.h>


// hkpWorldAgentUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpWorldAgentUtil, s_libraryName)

#include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>


// hkpWorldCallbackUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpWorldCallbackUtil, s_libraryName)

#include <Physics2012/Dynamics/World/Util/hkpWorldConstraintUtil.h>


// hkpWorldConstraintUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpWorldConstraintUtil, s_libraryName)

#include <Physics2012/Dynamics/World/Util/hkpWorldMemoryUtil.h>


// hkpWorldMemoryUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpWorldMemoryUtil, s_libraryName)

#include <Physics2012/Dynamics/World/Util/hkpWorldOperationQueue.h>


// UserCallback hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::UserCallback)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::UserCallback)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkWorldOperation::UserCallback, s_libraryName, hkReferencedObject)


// BaseOperation hkWorldOperation
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkWorldOperation::BaseOperation, s_libraryName, hkWorldOperation_BaseOperation)


// AddEntity hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::AddEntity)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::AddEntity)
    HK_TRACKER_MEMBER(hkWorldOperation::AddEntity, m_entity, 0, "hkpEntity*") // class hkpEntity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::AddEntity, s_libraryName, hkWorldOperation::BaseOperation)


// RemoveEntity hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::RemoveEntity)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::RemoveEntity)
    HK_TRACKER_MEMBER(hkWorldOperation::RemoveEntity, m_entity, 0, "hkpEntity*") // class hkpEntity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::RemoveEntity, s_libraryName, hkWorldOperation::BaseOperation)


// SetRigidBodyMotionType hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::SetRigidBodyMotionType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::SetRigidBodyMotionType)
    HK_TRACKER_MEMBER(hkWorldOperation::SetRigidBodyMotionType, m_rigidBody, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::SetRigidBodyMotionType, s_libraryName, hkWorldOperation::BaseOperation)


// SetWorldObjectShape hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::SetWorldObjectShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::SetWorldObjectShape)
    HK_TRACKER_MEMBER(hkWorldOperation::SetWorldObjectShape, m_worldObject, 0, "hkpWorldObject*") // class hkpWorldObject*
    HK_TRACKER_MEMBER(hkWorldOperation::SetWorldObjectShape, m_shape, 0, "hkpShape*") // const class hkpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::SetWorldObjectShape, s_libraryName, hkWorldOperation::BaseOperation)


// UpdateWorldObjectShape hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::UpdateWorldObjectShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::UpdateWorldObjectShape)
    HK_TRACKER_MEMBER(hkWorldOperation::UpdateWorldObjectShape, m_worldObject, 0, "hkpWorldObject*") // class hkpWorldObject*
    HK_TRACKER_MEMBER(hkWorldOperation::UpdateWorldObjectShape, m_shapeModifier, 0, "hkpShapeModifier*") // class hkpShapeModifier*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::UpdateWorldObjectShape, s_libraryName, hkWorldOperation::BaseOperation)


// AddEntityBatch hkWorldOperation
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkWorldOperation::AddEntityBatch, s_libraryName, hkWorldOperation_AddEntityBatch)


// RemoveEntityBatch hkWorldOperation
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkWorldOperation::RemoveEntityBatch, s_libraryName, hkWorldOperation_RemoveEntityBatch)


// AddConstraint hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::AddConstraint)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::AddConstraint)
    HK_TRACKER_MEMBER(hkWorldOperation::AddConstraint, m_constraint, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::AddConstraint, s_libraryName, hkWorldOperation::BaseOperation)


// RemoveConstraint hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::RemoveConstraint)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::RemoveConstraint)
    HK_TRACKER_MEMBER(hkWorldOperation::RemoveConstraint, m_constraint, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::RemoveConstraint, s_libraryName, hkWorldOperation::BaseOperation)


// AddAction hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::AddAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::AddAction)
    HK_TRACKER_MEMBER(hkWorldOperation::AddAction, m_action, 0, "hkpAction*") // class hkpAction*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::AddAction, s_libraryName, hkWorldOperation::BaseOperation)


// RemoveAction hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::RemoveAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::RemoveAction)
    HK_TRACKER_MEMBER(hkWorldOperation::RemoveAction, m_action, 0, "hkpAction*") // class hkpAction*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::RemoveAction, s_libraryName, hkWorldOperation::BaseOperation)


// MergeIslands hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::MergeIslands)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::MergeIslands)
    HK_TRACKER_MEMBER(hkWorldOperation::MergeIslands, m_entities, 0, "hkpEntity* [2]") // class hkpEntity* [2]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::MergeIslands, s_libraryName, hkWorldOperation::BaseOperation)


// AddPhantom hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::AddPhantom)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::AddPhantom)
    HK_TRACKER_MEMBER(hkWorldOperation::AddPhantom, m_phantom, 0, "hkpPhantom*") // class hkpPhantom*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::AddPhantom, s_libraryName, hkWorldOperation::BaseOperation)


// RemovePhantom hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::RemovePhantom)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::RemovePhantom)
    HK_TRACKER_MEMBER(hkWorldOperation::RemovePhantom, m_phantom, 0, "hkpPhantom*") // class hkpPhantom*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::RemovePhantom, s_libraryName, hkWorldOperation::BaseOperation)


// AddPhantomBatch hkWorldOperation
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkWorldOperation::AddPhantomBatch, s_libraryName, hkWorldOperation_AddPhantomBatch)


// RemovePhantomBatch hkWorldOperation
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkWorldOperation::RemovePhantomBatch, s_libraryName, hkWorldOperation_RemovePhantomBatch)


// UpdateEntityBP hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::UpdateEntityBP)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::UpdateEntityBP)
    HK_TRACKER_MEMBER(hkWorldOperation::UpdateEntityBP, m_entity, 0, "hkpEntity*") // class hkpEntity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::UpdateEntityBP, s_libraryName, hkWorldOperation::BaseOperation)


// UpdatePhantomBP hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::UpdatePhantomBP)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::UpdatePhantomBP)
    HK_TRACKER_MEMBER(hkWorldOperation::UpdatePhantomBP, m_phantom, 0, "hkpPhantom*") // class hkpPhantom*
    HK_TRACKER_MEMBER(hkWorldOperation::UpdatePhantomBP, m_aabb, 0, "hkAabb*") // class hkAabb*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::UpdatePhantomBP, s_libraryName, hkWorldOperation::BaseOperation)


// UpdateFilterOnEntity hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::UpdateFilterOnEntity)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::UpdateFilterOnEntity)
    HK_TRACKER_MEMBER(hkWorldOperation::UpdateFilterOnEntity, m_entity, 0, "hkpEntity*") // class hkpEntity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::UpdateFilterOnEntity, s_libraryName, hkWorldOperation::BaseOperation)


// UpdateFilterOnEntityPair hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::UpdateFilterOnEntityPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::UpdateFilterOnEntityPair)
    HK_TRACKER_MEMBER(hkWorldOperation::UpdateFilterOnEntityPair, m_entityA, 0, "hkpEntity*") // class hkpEntity*
    HK_TRACKER_MEMBER(hkWorldOperation::UpdateFilterOnEntityPair, m_entityB, 0, "hkpEntity*") // class hkpEntity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::UpdateFilterOnEntityPair, s_libraryName, hkWorldOperation::BaseOperation)


// UpdateFilterOnPhantom hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::UpdateFilterOnPhantom)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::UpdateFilterOnPhantom)
    HK_TRACKER_MEMBER(hkWorldOperation::UpdateFilterOnPhantom, m_phantom, 0, "hkpPhantom*") // class hkpPhantom*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::UpdateFilterOnPhantom, s_libraryName, hkWorldOperation::BaseOperation)


// UpdateFilterOnWorld hkWorldOperation
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkWorldOperation::UpdateFilterOnWorld, s_libraryName, hkWorldOperation_UpdateFilterOnWorld)


// ReintegrateAndRecollideEntityBatch hkWorldOperation
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkWorldOperation::ReintegrateAndRecollideEntityBatch, s_libraryName, hkWorldOperation_ReintegrateAndRecollideEntityBatch)


// UpdateMovedBodyInfo hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::UpdateMovedBodyInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::UpdateMovedBodyInfo)
    HK_TRACKER_MEMBER(hkWorldOperation::UpdateMovedBodyInfo, m_entity, 0, "hkpEntity*") // class hkpEntity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::UpdateMovedBodyInfo, s_libraryName, hkWorldOperation::BaseOperation)


// SetRigidBodyPositionAndRotation hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::SetRigidBodyPositionAndRotation)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::SetRigidBodyPositionAndRotation)
    HK_TRACKER_MEMBER(hkWorldOperation::SetRigidBodyPositionAndRotation, m_rigidBody, 0, "hkpRigidBody*") // class hkpRigidBody*
    HK_TRACKER_MEMBER(hkWorldOperation::SetRigidBodyPositionAndRotation, m_positionAndRotation, 0, "hkVector4*") // hkVector4*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::SetRigidBodyPositionAndRotation, s_libraryName, hkWorldOperation::BaseOperation)


// SetRigidBodyLinearVelocity hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::SetRigidBodyLinearVelocity)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::SetRigidBodyLinearVelocity)
    HK_TRACKER_MEMBER(hkWorldOperation::SetRigidBodyLinearVelocity, m_rigidBody, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::SetRigidBodyLinearVelocity, s_libraryName, hkWorldOperation::BaseOperation)


// SetRigidBodyAngularVelocity hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::SetRigidBodyAngularVelocity)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::SetRigidBodyAngularVelocity)
    HK_TRACKER_MEMBER(hkWorldOperation::SetRigidBodyAngularVelocity, m_rigidBody, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::SetRigidBodyAngularVelocity, s_libraryName, hkWorldOperation::BaseOperation)


// ApplyRigidBodyLinearImpulse hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::ApplyRigidBodyLinearImpulse)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::ApplyRigidBodyLinearImpulse)
    HK_TRACKER_MEMBER(hkWorldOperation::ApplyRigidBodyLinearImpulse, m_rigidBody, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::ApplyRigidBodyLinearImpulse, s_libraryName, hkWorldOperation::BaseOperation)


// ApplyRigidBodyPointImpulse hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::ApplyRigidBodyPointImpulse)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::ApplyRigidBodyPointImpulse)
    HK_TRACKER_MEMBER(hkWorldOperation::ApplyRigidBodyPointImpulse, m_rigidBody, 0, "hkpRigidBody*") // class hkpRigidBody*
    HK_TRACKER_MEMBER(hkWorldOperation::ApplyRigidBodyPointImpulse, m_pointAndImpulse, 0, "hkVector4*") // hkVector4*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::ApplyRigidBodyPointImpulse, s_libraryName, hkWorldOperation::BaseOperation)


// ApplyRigidBodyAngularImpulse hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::ApplyRigidBodyAngularImpulse)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::ApplyRigidBodyAngularImpulse)
    HK_TRACKER_MEMBER(hkWorldOperation::ApplyRigidBodyAngularImpulse, m_rigidBody, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::ApplyRigidBodyAngularImpulse, s_libraryName, hkWorldOperation::BaseOperation)


// AddReference hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::AddReference)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::AddReference)
    HK_TRACKER_MEMBER(hkWorldOperation::AddReference, m_worldObject, 0, "hkpWorldObject*") // class hkpWorldObject*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::AddReference, s_libraryName, hkWorldOperation::BaseOperation)


// RemoveReference hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::RemoveReference)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::RemoveReference)
    HK_TRACKER_MEMBER(hkWorldOperation::RemoveReference, m_worldObject, 0, "hkpWorldObject*") // class hkpWorldObject*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::RemoveReference, s_libraryName, hkWorldOperation::BaseOperation)


// ActivateRegion hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::ActivateRegion)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::ActivateRegion)
    HK_TRACKER_MEMBER(hkWorldOperation::ActivateRegion, m_aabb, 0, "hkAabb*") // class hkAabb*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::ActivateRegion, s_libraryName, hkWorldOperation::BaseOperation)


// ActivateEntity hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::ActivateEntity)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::ActivateEntity)
    HK_TRACKER_MEMBER(hkWorldOperation::ActivateEntity, m_entity, 0, "hkpEntity*") // class hkpEntity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::ActivateEntity, s_libraryName, hkWorldOperation::BaseOperation)


// RequestDeactivateEntity hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::RequestDeactivateEntity)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::RequestDeactivateEntity)
    HK_TRACKER_MEMBER(hkWorldOperation::RequestDeactivateEntity, m_entity, 0, "hkpEntity*") // class hkpEntity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::RequestDeactivateEntity, s_libraryName, hkWorldOperation::BaseOperation)


// DeactivateEntity hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::DeactivateEntity)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::DeactivateEntity)
    HK_TRACKER_MEMBER(hkWorldOperation::DeactivateEntity, m_entity, 0, "hkpEntity*") // class hkpEntity*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::DeactivateEntity, s_libraryName, hkWorldOperation::BaseOperation)


// ConstraintCollisionFilterConstraintBroken hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::ConstraintCollisionFilterConstraintBroken)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::ConstraintCollisionFilterConstraintBroken)
    HK_TRACKER_MEMBER(hkWorldOperation::ConstraintCollisionFilterConstraintBroken, m_filter, 0, "hkpConstraintCollisionFilter*") // class hkpConstraintCollisionFilter*
    HK_TRACKER_MEMBER(hkWorldOperation::ConstraintCollisionFilterConstraintBroken, m_constraintInstance, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::ConstraintCollisionFilterConstraintBroken, s_libraryName, hkWorldOperation::BaseOperation)


// UserCallbackOperation hkWorldOperation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorldOperation::UserCallbackOperation)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorldOperation::UserCallbackOperation)
    HK_TRACKER_MEMBER(hkWorldOperation::UserCallbackOperation, m_userCallback, 0, "hkWorldOperation::UserCallback*") // class hkWorldOperation::UserCallback*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkWorldOperation::UserCallbackOperation, s_libraryName, hkWorldOperation::BaseOperation)


// BiggestOperation hkWorldOperation
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkWorldOperation::BiggestOperation, s_libraryName, hkWorldOperation_BiggestOperation)


// hkpBodyOperationEntry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBodyOperationEntry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBodyOperationEntry)
    HK_TRACKER_MEMBER(hkpBodyOperationEntry, m_entity, 0, "hkpEntity*") // class hkpEntity*
    HK_TRACKER_MEMBER(hkpBodyOperationEntry, m_operation, 0, "hkpBodyOperation*") // class hkpBodyOperation*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpBodyOperationEntry, s_libraryName)


// hkpWorldOperationQueue ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldOperationQueue)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldOperationQueue)
    HK_TRACKER_MEMBER(hkpWorldOperationQueue, m_pending, 0, "hkArray<hkWorldOperation::BiggestOperation, hkContainerHeapAllocator>") // hkArray< struct hkWorldOperation::BiggestOperation, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorldOperationQueue, m_world, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpWorldOperationQueue, m_islandMerges, 0, "hkArray<hkWorldOperation::BiggestOperation, hkContainerHeapAllocator>") // hkArray< struct hkWorldOperation::BiggestOperation, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorldOperationQueue, m_pendingBodyOperations, 0, "hkArray<hkpBodyOperationEntry, hkContainerHeapAllocator>") // hkArray< struct hkpBodyOperationEntry, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpWorldOperationQueue, s_libraryName)


// hkpDebugInfoOnPendingOperationQueues ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDebugInfoOnPendingOperationQueues)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDebugInfoOnPendingOperationQueues)
    HK_TRACKER_MEMBER(hkpDebugInfoOnPendingOperationQueues, m_pending, 0, "hkArray<hkWorldOperation::BiggestOperation, hkContainerHeapAllocator>*") // hkArray< struct hkWorldOperation::BiggestOperation, struct hkContainerHeapAllocator >*
    HK_TRACKER_MEMBER(hkpDebugInfoOnPendingOperationQueues, m_nextQueue, 0, "hkpDebugInfoOnPendingOperationQueues*") // struct hkpDebugInfoOnPendingOperationQueues*
    HK_TRACKER_MEMBER(hkpDebugInfoOnPendingOperationQueues, m_prevQueue, 0, "hkpDebugInfoOnPendingOperationQueues*") // struct hkpDebugInfoOnPendingOperationQueues*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpDebugInfoOnPendingOperationQueues, s_libraryName)

// hkWorldOperation Type
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkWorldOperation::Type, s_libraryName, hkWorldOperation_Type)
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>


// hkpWorldOperationUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpWorldOperationUtil, s_libraryName)

#include <Physics2012/Dynamics/World/hkpPhysicsSystem.h>


// hkpPhysicsSystem ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPhysicsSystem)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CloneConstraintMode)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPhysicsSystem)
    HK_TRACKER_MEMBER(hkpPhysicsSystem, m_rigidBodies, 0, "hkArray<hkpRigidBody*, hkContainerHeapAllocator>") // hkArray< class hkpRigidBody*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpPhysicsSystem, m_constraints, 0, "hkArray<hkpConstraintInstance*, hkContainerHeapAllocator>") // hkArray< class hkpConstraintInstance*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpPhysicsSystem, m_actions, 0, "hkArray<hkpAction*, hkContainerHeapAllocator>") // hkArray< class hkpAction*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpPhysicsSystem, m_phantoms, 0, "hkArray<hkpPhantom*, hkContainerHeapAllocator>") // hkArray< class hkpPhantom*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpPhysicsSystem, m_name, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPhysicsSystem, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpPhysicsSystem, CloneConstraintMode, s_libraryName)

#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>


// hkpSimulationIsland ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimulationIsland)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSimulationIsland)
    HK_TRACKER_MEMBER(hkpSimulationIsland, m_world, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpSimulationIsland, m_actions, 0, "hkArray<hkpAction*, hkContainerHeapAllocator>") // hkArray< class hkpAction*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpSimulationIsland, m_entities, 0, "hkInplaceArray<hkpEntity*, 1, hkContainerHeapAllocator>") // class hkInplaceArray< class hkpEntity*, 1, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpSimulationIsland, m_midphaseAgentTrack, 0, "hkpAgentNnTrack") // struct hkpAgentNnTrack
    HK_TRACKER_MEMBER(hkpSimulationIsland, m_narrowphaseAgentTrack, 0, "hkpAgentNnTrack") // struct hkpAgentNnTrack
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSimulationIsland, s_libraryName, hkpConstraintOwner)

#include <Physics2012/Dynamics/World/hkpWorld.h>


// hkpWorldDynamicsStepInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpWorldDynamicsStepInfo, s_libraryName)


// hkpMultithreadConfig ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMultithreadConfig, s_libraryName)


// hkpWorld ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorld)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IgnoreForceMultithreadedSimulation)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ReintegrationRecollideMode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MtAccessChecking)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CachedAabbUpdate)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorld)
    HK_TRACKER_MEMBER(hkpWorld, m_simulation, 0, "hkpSimulation*") // class hkpSimulation*
    HK_TRACKER_MEMBER(hkpWorld, m_fixedIsland, 0, "hkpSimulationIsland*") // class hkpSimulationIsland*
    HK_TRACKER_MEMBER(hkpWorld, m_fixedRigidBody, 0, "hkpRigidBody*") // class hkpRigidBody*
    HK_TRACKER_MEMBER(hkpWorld, m_activeSimulationIslands, 0, "hkArray<hkpSimulationIsland*, hkContainerHeapAllocator>") // hkArray< class hkpSimulationIsland*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_inactiveSimulationIslands, 0, "hkArray<hkpSimulationIsland*, hkContainerHeapAllocator>") // hkArray< class hkpSimulationIsland*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_dirtySimulationIslands, 0, "hkArray<hkpSimulationIsland*, hkContainerHeapAllocator>") // hkArray< class hkpSimulationIsland*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_maintenanceMgr, 0, "hkpWorldMaintenanceMgr*") // class hkpWorldMaintenanceMgr*
    HK_TRACKER_MEMBER(hkpWorld, m_memoryWatchDog, 0, "hkWorldMemoryAvailableWatchDog *") // class hkWorldMemoryAvailableWatchDog *
    HK_TRACKER_MEMBER(hkpWorld, m_broadPhase, 0, "hkpBroadPhase*") // class hkpBroadPhase*
    HK_TRACKER_MEMBER(hkpWorld, m_broadPhaseDispatcher, 0, "hkpTypedBroadPhaseDispatcher*") // class hkpTypedBroadPhaseDispatcher*
    HK_TRACKER_MEMBER(hkpWorld, m_phantomBroadPhaseListener, 0, "hkpPhantomBroadPhaseListener*") // class hkpPhantomBroadPhaseListener*
    HK_TRACKER_MEMBER(hkpWorld, m_entityEntityBroadPhaseListener, 0, "hkpEntityEntityBroadPhaseListener*") // class hkpEntityEntityBroadPhaseListener*
    HK_TRACKER_MEMBER(hkpWorld, m_broadPhaseBorderListener, 0, "hkpBroadPhaseBorderListener*") // class hkpBroadPhaseBorderListener*
    HK_TRACKER_MEMBER(hkpWorld, m_multithreadedSimulationJobData, 0, "hkpMtThreadStructure*") // struct hkpMtThreadStructure*
    HK_TRACKER_MEMBER(hkpWorld, m_collisionInput, 0, "hkpProcessCollisionInput*") // struct hkpProcessCollisionInput*
    HK_TRACKER_MEMBER(hkpWorld, m_collisionFilter, 0, "hkpCollisionFilter*") // class hkpCollisionFilter*
    HK_TRACKER_MEMBER(hkpWorld, m_collisionDispatcher, 0, "hkpCollisionDispatcher*") // class hkpCollisionDispatcher*
    HK_TRACKER_MEMBER(hkpWorld, m_convexListFilter, 0, "hkpConvexListFilter*") // class hkpConvexListFilter*
    HK_TRACKER_MEMBER(hkpWorld, m_pendingOperations, 0, "hkpWorldOperationQueue*") // class hkpWorldOperationQueue*
    HK_TRACKER_MEMBER(hkpWorld, m_pendingOperationQueues, 0, "hkpDebugInfoOnPendingOperationQueues*") // struct hkpDebugInfoOnPendingOperationQueues*
    HK_TRACKER_MEMBER(hkpWorld, m_modifyConstraintCriticalSection, 0, "hkCriticalSection*") // class hkCriticalSection*
    HK_TRACKER_MEMBER(hkpWorld, m_islandDirtyListCriticalSection, 0, "hkCriticalSection*") // class hkCriticalSection*
    HK_TRACKER_MEMBER(hkpWorld, m_propertyMasterLock, 0, "hkCriticalSection*") // class hkCriticalSection*
    HK_TRACKER_MEMBER(hkpWorld, m_phantoms, 0, "hkArray<hkpPhantom*, hkContainerHeapAllocator>") // hkArray< class hkpPhantom*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_actionListeners, 0, "hkArray<hkpActionListener*, hkContainerHeapAllocator>") // hkArray< class hkpActionListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_entityListeners, 0, "hkArray<hkpEntityListener*, hkContainerHeapAllocator>") // hkArray< class hkpEntityListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_phantomListeners, 0, "hkArray<hkpPhantomListener*, hkContainerHeapAllocator>") // hkArray< class hkpPhantomListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_constraintListeners, 0, "hkArray<hkpConstraintListener*, hkContainerHeapAllocator>") // hkArray< class hkpConstraintListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_worldDeletionListeners, 0, "hkArray<hkpWorldDeletionListener*, hkContainerHeapAllocator>") // hkArray< class hkpWorldDeletionListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_islandActivationListeners, 0, "hkArray<hkpIslandActivationListener*, hkContainerHeapAllocator>") // hkArray< class hkpIslandActivationListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_worldPostSimulationListeners, 0, "hkArray<hkpWorldPostSimulationListener*, hkContainerHeapAllocator>") // hkArray< class hkpWorldPostSimulationListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_worldPostIntegrateListeners, 0, "hkArray<hkpWorldPostIntegrateListener*, hkContainerHeapAllocator>") // hkArray< class hkpWorldPostIntegrateListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_worldPostCollideListeners, 0, "hkArray<hkpWorldPostCollideListener*, hkContainerHeapAllocator>") // hkArray< class hkpWorldPostCollideListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_islandPostIntegrateListeners, 0, "hkArray<hkpIslandPostIntegrateListener*, hkContainerHeapAllocator>") // hkArray< class hkpIslandPostIntegrateListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_islandPostCollideListeners, 0, "hkArray<hkpIslandPostCollideListener*, hkContainerHeapAllocator>") // hkArray< class hkpIslandPostCollideListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_contactListeners, 0, "hkArray<hkpContactListener*, hkContainerHeapAllocator>") // hkArray< class hkpContactListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_contactImpulseLimitBreachedListeners, 0, "hkArray<hkpContactImpulseLimitBreachedListener*, hkContainerHeapAllocator>") // hkArray< class hkpContactImpulseLimitBreachedListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_worldExtensions, 0, "hkArray<hkpWorldExtension*, hkContainerHeapAllocator>") // hkArray< class hkpWorldExtension*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpWorld, m_violatedConstraintArray, 0, "hkpViolatedConstraintArray*") // struct hkpViolatedConstraintArray*
    HK_TRACKER_MEMBER(hkpWorld, m_broadPhaseBorder, 0, "hkpBroadPhaseBorder*") // class hkpBroadPhaseBorder*
    HK_TRACKER_MEMBER(hkpWorld, m_destructionWorld, 0, "hkdWorld*") // class hkdWorld*
    HK_TRACKER_MEMBER(hkpWorld, m_npWorld, 0, "hknpWorld*") // class hknpWorld*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpWorld, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWorld, ReintegrationRecollideMode, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWorld, MtAccessChecking, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWorld, CachedAabbUpdate, s_libraryName)


// IgnoreForceMultithreadedSimulation hkpWorld
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWorld, IgnoreForceMultithreadedSimulation, s_libraryName)

// None hkpUpdateCollisionFilterOnWorldMode
HK_TRACKER_IMPLEMENT_SIMPLE(hkpUpdateCollisionFilterOnWorldMode, s_libraryName)
// None hkpUpdateCollisionFilterOnEntityMode
HK_TRACKER_IMPLEMENT_SIMPLE(hkpUpdateCollisionFilterOnEntityMode, s_libraryName)
// None hkpEntityActivation
HK_TRACKER_IMPLEMENT_SIMPLE(hkpEntityActivation, s_libraryName)
// None hkpUpdateCollectionFilterMode
HK_TRACKER_IMPLEMENT_SIMPLE(hkpUpdateCollectionFilterMode, s_libraryName)
// None hkpStepResult
HK_TRACKER_IMPLEMENT_SIMPLE(hkpStepResult, s_libraryName)
#include <Physics2012/Dynamics/World/hkpWorldCinfo.h>


// hkpWorldCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldCinfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SolverType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SimulationType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ContactPointGeneration)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BroadPhaseType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BroadPhaseBorderBehaviour)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldCinfo)
    HK_TRACKER_MEMBER(hkpWorldCinfo, m_collisionFilter, 0, "hkpCollisionFilter *") // class hkpCollisionFilter *
    HK_TRACKER_MEMBER(hkpWorldCinfo, m_convexListFilter, 0, "hkpConvexListFilter *") // class hkpConvexListFilter *
    HK_TRACKER_MEMBER(hkpWorldCinfo, m_memoryWatchDog, 0, "hkWorldMemoryAvailableWatchDog *") // class hkWorldMemoryAvailableWatchDog *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpWorldCinfo, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWorldCinfo, SolverType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWorldCinfo, SimulationType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWorldCinfo, ContactPointGeneration, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWorldCinfo, BroadPhaseType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWorldCinfo, BroadPhaseBorderBehaviour, s_libraryName)

#include <Physics2012/Dynamics/World/hkpWorldObject.h>


// hkpWorldObject ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWorldObject)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MtChecks)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BroadPhaseType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWorldObject)
    HK_TRACKER_MEMBER(hkpWorldObject, m_world, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpWorldObject, m_collidable, 0, "hkpLinkedCollidable") // class hkpLinkedCollidable
    HK_TRACKER_MEMBER(hkpWorldObject, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkpWorldObject, m_properties, 0, "hkArray<hkSimpleProperty, hkContainerHeapAllocator>") // hkArray< class hkSimpleProperty, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpWorldObject, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWorldObject, MtChecks, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWorldObject, BroadPhaseType, s_libraryName)

// hkWorldOperation Result
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkWorldOperation::Result, s_libraryName, hkWorldOperation_Result)

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
