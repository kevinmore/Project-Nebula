/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>

#include <Physics2012/Dynamics/World/hkpPhysicsSystem.h>
#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintPivotsUtil.h>
#include <Physics/Constraint/Data/hkpConstraintDataUtils.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Dynamics/Destruction/Utilities/hkpDestructionBreakOffUtil.h>

//
//	Constructor

hkpDestructionBreakOffUtil::hkpDestructionBreakOffUtil(hkpBreakOffPartsListener* breakOffPartsListener)
:	hkpWorldExtension(HK_WORLD_EXTENSION_ANONYMOUS)
,	m_breakOffPartsListener(breakOffPartsListener)
{
	// Create contact listener
	m_entityContactsListener = new DefaultContactListener(this);

	// Create critical section
	m_criticalSection = new hkCriticalSection(1);

	// Create break-off control functor
	GameControlFunctor* func = new GameControlFunctor;
	m_breakOffControlFunctor = func;
	func->removeReference();
}

//
//	Destructor

hkpDestructionBreakOffUtil::~hkpDestructionBreakOffUtil()
{
	if ( m_criticalSection )
	{
		delete m_criticalSection;
		m_criticalSection = HK_NULL;
	}

	if ( m_entityContactsListener )
	{
		m_entityContactsListener->removeReference();
		m_entityContactsListener = HK_NULL;
	}
}

//
//	Entity contacts listener constructor

hkpDestructionBreakOffUtil::ContactListener::ContactListener(hkpDestructionBreakOffUtil* breakOffUtil)
:	m_breakOffUtil(breakOffUtil)
{}

//
//	Entity contacts listener destructor

hkpDestructionBreakOffUtil::ContactListener::~ContactListener()
{}

//
//	Given a hkpCdBody, it returns the owning entity and shape key

static HK_FORCE_INLINE void getRootEntityAndKey(const hkpCdBody* bodyIn, hkpEntity*& entityOut, hkpShapeKey& keyOut)
{
	const hkpCdBody* rootCollidable = bodyIn;
	const hkpCdBody* topChild = rootCollidable;
	while ( rootCollidable->m_parent )
	{
		topChild = rootCollidable;
		rootCollidable = rootCollidable->m_parent;
	}
	keyOut		= topChild->getShapeKey();
	entityOut	= hkpGetRigidBody(reinterpret_cast<const hkpCollidable*>(rootCollidable));
}

//
//	The decision function that says whether the \a limitedBody should be broken by impact of \a otherBody.
//	The \a key is the shape key in contact in the \a otherBody and the currently set \a maxImpulse for breaking the \a limitedBody.
//	This implementation simply returns USE_LIMIT.


hkpDestructionBreakOffUtil::BreakOffGameControlResult hkpDestructionBreakOffUtil::GameControlFunctor::test(	const hkpRigidBody* limitedBody, 
																											const hkpRigidBody* otherBody, 
																											const hkpShapeKey key, 
																											const hkUFloat8& maxImpulse)
{
	return USE_LIMIT;
}


//
//	Called after a contact point is created. This contact point is still a potential one.

void hkpDestructionBreakOffUtil::ContactListenerSpu::contactPointAddedCallback(hkpContactPointAddedEvent& event)
{
	// Early out if the breakOffPartsUtil has been removed from the world.
	if ( !m_breakOffUtil->getWorld() )
	{
		return;
	}

	// Get the body that fired the callback
	hkpEntity* currentEntity = event.m_callbackFiredFrom;
	hkpEntity* otherEntity;

	// Find the shape key that was hit
	hkpShapeKey contactShapeKey = HK_INVALID_SHAPE_KEY;
	{
		// Resolve entities and keys
		hkpShapeKey keyA;	hkpEntity* entityA;		getRootEntityAndKey(event.m_bodyA, entityA, keyA);
		hkpShapeKey keyB;	hkpEntity* entityB;		getRootEntityAndKey(event.m_bodyB, entityB, keyB);

		HK_ASSERT2(0x260810e1, entityA && entityB && ((entityA == currentEntity) || (entityB == currentEntity)), "Failed to resolve entities and keys!!!");
		if ( entityA == currentEntity )	
		{
			currentEntity = entityA;	
			otherEntity = entityB;	
			contactShapeKey = keyA;	
		}
		else
		{
			currentEntity = entityB;
			otherEntity = entityA;	
			contactShapeKey = keyB;	
		}		
	}

	const bool isEntityIdSmaller	= (currentEntity->getUid() < otherEntity->getUid());
	
	// Get the maximum impulse that can be applied on this contact
	hkUFloat8 maxImpulse = getMaxImpulseForContactPoint(contactShapeKey, currentEntity, otherEntity);
	{
		BreakOffGameControlResult gameControl = m_breakOffUtil->m_breakOffControlFunctor->test(	reinterpret_cast<hkpRigidBody*>(otherEntity), 
																								reinterpret_cast<hkpRigidBody*>(currentEntity),
																								contactShapeKey, maxImpulse);

		switch (gameControl)
		{
		case BREAK_OFF:			// Zero limit = force breaking
			maxImpulse.m_value = 1;
		case USE_LIMIT:
		break;

		case DO_NOT_BREAK_OFF:	// No limit = no breaking
		return;

		default:
			HK_ASSERT2(0x23750120, 0, "Break off game control error");
		break;
		}
	}

	// Enforce impulse limit
	limitContactImpulse(maxImpulse, isEntityIdSmaller, event.m_contactPointProperties);
}

//
// Search for the last valid key, or HK_INVALID_SHAPE_KEY if there wasn't one.

static HK_FORCE_INLINE hkpShapeKey HK_CALL hkpDestructionBreakOffUtil_findTopShapeKey(const hkpShapeKey* leafKeyPtr, int sizeOfKeyStorage)
{
	hkpShapeKey key = HK_INVALID_SHAPE_KEY;
	const hkpShapeKey *const end = leafKeyPtr + sizeOfKeyStorage;
	const hkpShapeKey* kp = leafKeyPtr;
	while ( ( *kp != HK_INVALID_SHAPE_KEY ) && ( kp < end ) )
	{
		key = *kp;
		kp++;
	}

	return key;
}

//
//	Called after a contact point is created.

void hkpDestructionBreakOffUtil::ContactListenerPpu::contactPointCallback(const hkpContactPointEvent& event)
{
	// Early out if the breakOffPartsUtil has been removed from the world or the contact is not new
	if ( !m_breakOffUtil->getWorld() || !(event.m_contactPointProperties->m_flags & hkContactPointMaterial::CONTACT_IS_NEW))
	{
		return;
	}

	// Get the body that fired the callback
	hkpRigidBody* currentEntity	= event.getBody(event.m_source);
	hkpRigidBody* otherEntity	= event.getBody(1 - event.m_source);
	const bool isEntityIdSmaller = (currentEntity->getUid() < otherEntity->getUid());

	HK_ASSERT2(0x4f79a1be, event.m_source != hkpContactPointEvent::SOURCE_WORLD, "This listener should not be attached to the world.");
	const hkpShapeKey contactShapeKey = hkpDestructionBreakOffUtil_findTopShapeKey(event.getShapeKeys(event.m_source), currentEntity->m_numShapeKeysInContactPointProperties);	

	// Get the maximum impulse that can be applied on this contact
	hkUFloat8 maxImpulse = getMaxImpulseForContactPoint(contactShapeKey, currentEntity, otherEntity);
	{
		BreakOffGameControlResult gameControl = m_breakOffUtil->m_breakOffControlFunctor->test(currentEntity, otherEntity, contactShapeKey, maxImpulse);

		switch (gameControl)
		{
		case BREAK_OFF:			// Zero limit = force breaking
			maxImpulse.m_value = 1;
		case USE_LIMIT:
			break;

		case DO_NOT_BREAK_OFF:	// No limit = no breaking
			return;

		default:
			HK_ASSERT2(0x23750120, 0, "Break off game control error");
			break;
		}
	}
	
	// Enforce impulse limit
	limitContactImpulse(maxImpulse, isEntityIdSmaller, event.m_contactPointProperties);
}

//
//	Returns the constraint strength set on the given entity

inline hkReal HK_CALL hkpDestructionBreakOffUtil::getConstraintStrength(const hkpEntity* entity)
{
	hkReal constraintStrength = HK_REAL_MAX;

	if ( entity )
	{
		// Get breakable body
		const hkpBreakableBody* breakableBody = entity->m_breakableBody;

		if ( breakableBody && (breakableBody->getConstraintStrength() > 0.0f) )
		{
			constraintStrength = breakableBody->getConstraintStrength();
		}
	}

	// No constraint strength set. Constraint is unbreakable!
	return constraintStrength;
}

//
//	Called when a constraint is added to the world.

void hkpDestructionBreakOffUtil::constraintAddedCallback(hkpConstraintInstance* constraint)
{
	// Ignore contact constraints
	if ( constraint->getData()->getType() == hkpConstraintData::CONSTRAINT_TYPE_CONTACT )
	{
		return;
	}

	// Get constraint strengths set on the constrained bodies
	const hkReal maxImpulseA	= getConstraintStrength(constraint->getEntityA());
	const hkReal maxImpulseB	= getConstraintStrength(constraint->getEntityB());
	const hkReal maxImpulseAB	= hkMath::min2(maxImpulseA, maxImpulseB);

	// Write impulse to constraint
	hkpConstraintData* constraintData = const_cast<hkpConstraintData*>(constraint->getData());
	if ( maxImpulseAB < HK_REAL_MAX )
	{
		const int notifiedBodyIdx = (maxImpulseA == maxImpulseAB) ? 0 : 1;
		constraintData->setBodyToNotify(notifiedBodyIdx);
	}

	// We have to call this always as when a previously-breakable body is broken into pieces, and pieces are not
	// breakable anymore. We don't want to limit the solver's impulses on such bodies then.
	constraintData->setBreachImpulse(maxImpulseAB);
}

//
//	Calculate a maxImpulse value given that current entity is colliding with other at key.

inline hkUFloat8 HK_CALL hkpDestructionBreakOffUtil::ContactListener::getMaxImpulseForContactPoint(hkpShapeKey key, hkpEntity* entity, hkpEntity* other)
{
	hkUFloat8 maxImpulse;
	maxImpulse.m_value = 0;
	if ( !getMaxImpulseForKey(key, entity, maxImpulse) )
	{
		return maxImpulse;	// Not breakable
	}

	if ( other->m_damageMultiplier != 1.0f )
	{
		maxImpulse = maxImpulse / hkFloat32(other->m_damageMultiplier);	
	}
	
	maxImpulse.m_value = hkMath::max2(hkUchar(1), maxImpulse.m_value);
	return maxImpulse;
}

//
//	Used by the CPU callbacks (the contactPointCallback and the contactPointAddedCallback), but not used by the SPU version.

inline void HK_CALL hkpDestructionBreakOffUtil::ContactListener::limitContactImpulse(const hkUFloat8& maxImpulse, bool isEntityIdSmaller, hkpContactPointProperties* properties)
{
	if ( !maxImpulse.m_value ||											// Not breakable, or
		( properties->m_maxImpulse.m_value &&							// Breakable, but
		(maxImpulse.m_value >= properties->m_maxImpulse.m_value) ) )	// existing limit is lower than the one we're trying to set
	{
		return;
	}

	// Set impulse limit
	properties->setMaxImpulsePerStep(maxImpulse);
	if ( isEntityIdSmaller )
	{
		properties->m_flags |= hkContactPointMaterial::CONTACT_BREAKOFF_OBJECT_ID_SMALLER;
	}
	else
	{
		properties->m_flags &= ~hkContactPointMaterial::CONTACT_BREAKOFF_OBJECT_ID_SMALLER;
	}
}

//
//	Enables / disables all contact callbacks on PPU. By default, callbacks are processed on Spu.

void hkpDestructionBreakOffUtil::enableSpuCallbacks(bool doEnable)
{
	// Delete current listener
	if ( m_entityContactsListener )
	{
		m_entityContactsListener->removeReference();
		m_entityContactsListener = HK_NULL;
	}

	// Create the new listener
	if ( doEnable )
	{
		m_entityContactsListener = new ContactListenerSpu(this);
	}
	else
	{
		m_entityContactsListener = new ContactListenerPpu(this);
	}
}

//
//	Marks the given body as breakable

void hkpDestructionBreakOffUtil::ContactListener::markBreakableBody(hkpRigidBody* body)
{
	// Add the contact listener on the body
	body->addContactListener(this);
}

//
//	Marks the given body as breakable

void hkpDestructionBreakOffUtil::ContactListenerSpu::markBreakableBody(hkpRigidBody* body)
{
	// Call base class
	hkpDestructionBreakOffUtil::ContactListener::markBreakableBody(body);

	// Flag it to run on SPU
	body->m_limitContactImpulseUtilAndFlag = (void*)(hkUlong)(1);
}

//
//	Returns true if the body can be broken-off

hkBool hkpDestructionBreakOffUtil::ContactListener::canBreak(const hkpRigidBody* body) const
{
	hkpContactListener* cl = const_cast<hkpDestructionBreakOffUtil::ContactListener*>(this);
	return body && (body->getContactListeners().indexOf(cl) >= 0);
}

//
//	Returns true if the body can be broken-off

hkBool hkpDestructionBreakOffUtil::ContactListenerSpu::canBreak(const hkpRigidBody* body) const
{
	return body && body->m_limitContactImpulseUtilAndFlag;
}

//
//	Flag a rigid body to be breakable

void hkpDestructionBreakOffUtil::markBreakableBody(hkpRigidBody* body)
{
	// If the body is already marked, do nothing!
	const int listenerIdx = body->getContactListeners().indexOf(m_entityContactsListener);
	if ( listenerIdx < 0 )
	{
		// Mark the entity as breakable
		m_entityContactsListener->markBreakableBody(body);
	}
}

//
//	Flags a rigid body as unbreakable

void hkpDestructionBreakOffUtil::markUnbreakableBody(hkpRigidBody* body)
{
	// If the body was not previously marked, do nothing
	if ( body->getContactListeners().indexOf(m_entityContactsListener) >= 0 )
	{
		// Un-mark it as breakable
		_markUnbreakableBody(body);
	}
}

//
//	Called when this extension is attached to the physics world.

void hkpDestructionBreakOffUtil::performAttachments(hkpWorld* world)
{
	// Register this as a contact / constraint impulse listener
	world->addContactImpulseLimitBreachedListener(this);
	world->addConstraintListener(this);
}

//
//	Called when this extension is detached from the physics world.

void hkpDestructionBreakOffUtil::performDetachments(hkpWorld* world)
{
	// Un-register this as a contact / constraint impulse listener
	world->removeConstraintListener(this);
	world->removeContactImpulseLimitBreachedListener(this);
}

//
//	Maximum impulse breached event info struct

struct MaxImpulseBreachedEventInfo
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DESTRUCTION_2012, MaxImpulseBreachedEventInfo);

	typedef hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent Event;

	// The event
	Event* m_event;

	// True if the object should be marked as unbreakable once hit
	hkBool32 m_disableFurtherBreakOff;
};

//
//	Queries the contact manager for the breaking body, shape key and whether the body should be marked as unbreakable after this hit.
//	Returns true if the body is breakable, false otherwise. The returned info is only valid if the function returned true.

static HK_FORCE_INLINE bool HK_CALL queryImpulseLimitBreachedInfo(	const hkpDestructionBreakOffUtil* utilPtr, const hkpContactImpulseLimitBreachedListenerInfo& bi, hkpRigidBody*& breakingBodyOut,
																	hkpShapeKey& shapeKeyOut, hkBool32& disableFurtherBreakOff)
{
	// Get the rigid bodies
	hkpRigidBody* entities[2] = { bi.getBodyA(), bi.getBodyB() };

	// Get the current body
	hkpContactPointProperties* props = bi.getContactPointProperties();
	int currentBodyId = -1;
	
	if(props->m_flags & hkContactPointMaterial::CONTACT_BREAKOFF_OBJECT_ID_SMALLER)
	{  // if the breaking body uid is smaller than the other
		
		currentBodyId = (entities[0]->getUid() < entities[1]->getUid() ?  0 : 1 );
	}
	else
	{
		currentBodyId = (entities[0]->getUid() < entities[1]->getUid() ?  1 : 0 );
	}
	breakingBodyOut = entities[currentBodyId];
	// Check if body is still breakable
	if ( !utilPtr->canBreak(breakingBodyOut) )
	{
		return false;	// Body not breakable!
	}

	// Get shape key
	const hkpSimpleContactConstraintAtom* atom = bi.getContactMgr()->getAtom();
	shapeKeyOut = props->getExtendedUserData(atom, currentBodyId, 0);

	// If the shape key is valid, i.e. this is a shape collection (compound), then the 
	// breakable shape would have a maximum impulse value associated with each shape key
	// (because the mapping is done with materials, which cover the entire range of shape keys).
	// In this case, compound shapes will never be disabled on impact, but simple shapes will always be
	// disabled.
	disableFurtherBreakOff = (shapeKeyOut == HK_INVALID_SHAPE_KEY);

	// Body is breakable
	return true;
}

//
//	Queries whether the given body is breakable and whether it should be marked as unbreakable after this hit.
//	Returns true if the body is breakable, false otherwise. The returned info is only valid if the function returned true.

static HK_FORCE_INLINE bool HK_CALL queryImpulseLimitBreachedInfo(	const hkpDestructionBreakOffUtil* utilPtr, const hkpRigidBody* rigidBody, hkpShapeKey closestShapeKey,
																	hkBool32& disableFurtherBreakOff)
{
	// If the shape key is valid, i.e. this is a shape collection (compound), then the 
	// breakable shape would have a maximum impulse value associated with each shape key
	// (because the mapping is done with materials, which cover the entire range of shape keys).
	// In this case, compound shapes will never be disabled on impact, but simple shapes will always be
	// disabled.
	disableFurtherBreakOff = (closestShapeKey == HK_INVALID_SHAPE_KEY);

	return utilPtr->canBreak(rigidBody);
}

#ifndef HK_PLATFORM_SPU

//
//	Called by the constraint solver when it needs to apply more than the maximum allowed impulse in order to maintain the given contacts.

void hkpDestructionBreakOffUtil::contactImpulseLimitBreachedCallback(const hkpContactImpulseLimitBreachedListenerInfo* breachedContacts, int numBreachedContacts)
{
	if ( breachedContacts[0].isToi() )
	{
		HK_ASSERT2(0xad875533, numBreachedContacts == 1, "TOI contact points are always processed individually.");
		contactImpulseLimitBreachedCallbackToi(&breachedContacts[0]);
		return;
	}

	hkInplaceArray<hkpEntity*, 128> bodiesToCollide;
	hkpPhysicsSystem newSystem;

	// Single Threaded Section
	{
		hkLocalArray<MaxImpulseBreachedEventInfo> eventInfos(numBreachedContacts); // Allocate worst-case size
		hkLocalArray<Event*> constraintEvents(numBreachedContacts);
		hkLocalArray<const hkpConstraintInstance*> constraints(numBreachedContacts);
		hkLocalArray<hkContactPoint> constraintPoints(numBreachedContacts);

		hkCriticalSectionLock lock(m_criticalSection);
		{
			hkPointerMap<hkpRigidBody*, int> bodyToEventIndexMap;

			for (int i = 0; i < numBreachedContacts; i++)
			{
				const hkpContactImpulseLimitBreachedListenerInfo& bi = breachedContacts[i];
				HK_ASSERT2(0xad8533a3, !bi.isToi(), "This function is for non-TOI contact points only.");

				// Check whether this is a contact or a constraint
				if ( bi.isContact() )
				{
					// Get the breaking body (if any)
					hkpRigidBody* breakingBody		= HK_NULL;
					hkpShapeKey key					= HK_INVALID_SHAPE_KEY;
					hkBool32 disableFurtherBreakOff = false;
					if ( !queryImpulseLimitBreachedInfo(this, bi, breakingBody, key, disableFurtherBreakOff) )
					{
						continue;	// Body is not breakable!
					}

					// Find or create an event
					Event* event = HK_NULL;
					hkPointerMap<hkpRigidBody*, int>::Iterator it = bodyToEventIndexMap.findKey(breakingBody);
					if ( bodyToEventIndexMap.isValid(it) )
					{
						// We already have an event for this body
						const int index = bodyToEventIndexMap.getValue(it);
						MaxImpulseBreachedEventInfo& info = eventInfos[index];
						event = info.m_event;
						HK_ASSERT2(0xad7644aa, event->m_breakingBody == breakingBody, "Entity pointers don't match.");
						info.m_disableFurtherBreakOff = info.m_disableFurtherBreakOff | disableFurtherBreakOff;
					}
					else
					{
						// The body is new, must create a new event!
						bodyToEventIndexMap.insert(breakingBody, eventInfos.getSize());
						MaxImpulseBreachedEventInfo& info = *eventInfos.expandByUnchecked(1);
						info.m_event = new Event();
						event = info.m_event;
						event->m_breakingBody = breakingBody;
						info.m_disableFurtherBreakOff = disableFurtherBreakOff;
					}

					// Add the new contact point to the event!
					PointInfo& pointInfo = event->m_points.expandOne();
					pointInfo.m_collidingBody			= hkSelectOther(event->m_breakingBody, bi.getBodyA(), bi.getBodyB());
					pointInfo.m_contactPointDirection	= ( event->m_breakingBody ==  bi.getBodyA() ) ? 1.0f : -1.0f;
					pointInfo.m_brokenShapeKey			= key;
					pointInfo.m_isContact				= true;
					pointInfo.m_contactPoint			= bi.getContactPoint();
					hkpContactPointProperties* cpp		= bi.getContactPointProperties();
					pointInfo.m_properties				= cpp;
					pointInfo.m_breakingImpulse			= cpp->m_maxImpulse;
					pointInfo.m_internalContactMgr		= bi.getContactMgr();
				}
				else // Process constraints
				{
					// Create a new event.
					const hkpConstraintInstance* instance = bi.getConstraintInstance();
					const hkpConstraintData* data = static_cast<const hkpConstraintData*>(instance->getData());

					hkUint8 notifiedBodyIdx = data->getNotifiedBodyIndex();
					Event* constraintEvent = new Event();
					constraintEvents.pushBackUnchecked(constraintEvent);
					PointInfo& pointInfo = *constraintEvent->m_points.expandByUnchecked(1);
					constraints.pushBackUnchecked(instance);

					hkpRigidBody* breakingBody		= static_cast<hkpRigidBody*>(instance->getEntity(notifiedBodyIdx));
					const hkpWorld* physicsWorld	= breakingBody->getWorld();
					const hkVector4 vPivot			= hkpConstraintDataUtils::getPivot(data, notifiedBodyIdx);
					constraintEvent->m_breakingBody		= breakingBody;
					pointInfo.m_collidingBody			= static_cast<hkpRigidBody*>(instance->getEntity(1 - notifiedBodyIdx));
					pointInfo.m_brokenShapeKey			= hkpConstraintPivotsUtil::findClosestShapeKey(physicsWorld, breakingBody->getCollidable()->getShape(), vPivot);
					pointInfo.m_isContact				= false;
					pointInfo.m_properties				= HK_NULL;
					pointInfo.m_breakingImpulse			= HK_REAL_MAX;
					pointInfo.m_internalContactMgr		= HK_NULL;
					pointInfo.m_contactPointDirection	= 1.0f;
					hkContactPoint* cp					= constraintPoints.expandByUnchecked(1);
					pointInfo.m_contactPoint			= cp;

					// Compute contact point.
					hkVector4 pivotA, pivotB;
					instance->getPivotsInWorld(pivotA, pivotB);

					hkVector4 displacement;
					if ( notifiedBodyIdx )
					{
						cp->setPosition(pivotB);
						displacement.setSub(pivotA, pivotB);
					}
					else
					{
						cp->setPosition(pivotA);
						displacement.setSub(pivotB, pivotA); 
					}

					const hkSimdReal distance = displacement.length<3>();
					if ( distance.isLess(hkSimdReal_Eps) )
					{
						displacement = hkVector4::getConstant<HK_QUADREAL_1000>();
					}
					else
					{
						displacement.normalize<3>();
					}
					displacement.setComponent<3>(-distance);

					// Set separating normal and penetration depth
					cp->setSeparatingNormal(displacement);
				}
			}
		}

		// Trigger break-off callbacks for contact points and remove keys from corresponding utils.
		for (int ui = 0; ui < eventInfos.getSize(); ui++)
		{
			MaxImpulseBreachedEventInfo& info = eventInfos[ui];
			Event& event = *info.m_event;

			hkResult result = m_breakOffPartsListener->breakOffSubPart(event, newSystem);

			if ( result == HK_SUCCESS)
			{
				// The body was broken
				bodiesToCollide.pushBack(event.m_breakingBody);

				// Mark it as unbreakable, it has already been broken
// 				if ( info.m_disableFurtherBreakOff )
// 				{
// 					_markUnbreakableBody(event.m_breakingBody);
// 				}

				for (int p = 0; p < event.m_points.getSize(); p++)
				{
					PointInfo& pointInfo = event.m_points[p];

					// Disable contact. This is only necessary if this function is called 
					// from a simple collision response, which happens before solving. In this case
					// the contact point will still get into the solver, so we simple disable it by setting
					// the distance to infinity

					// The const cast is safe: we got this point from hkpContactImpulseLimitBreachedListenerInfo above..
					const_cast<hkContactPoint*>(pointInfo.m_contactPoint)->setDistance(HK_REAL_HIGH);
				}
			}

			delete info.m_event;
		}

		// Trigger break-off callbacks for constraints.
		for (int i = 0; i < constraintEvents.getSize(); i++)
		{
			Event& event = *constraintEvents[i];

			// Get the breaking body (if any)
			hkBool32 disableFurtherBreakOff = false;
			const bool bodyIsBreakable = queryImpulseLimitBreachedInfo(this, event.m_breakingBody, event.m_points[0].m_brokenShapeKey, disableFurtherBreakOff);

			// Trigger destruction
			hkResult result = m_breakOffPartsListener->breakOffSubPart(event, newSystem);
			if ( (result == HK_SUCCESS) && bodyIsBreakable )
			{
				// The body was broken
				bodiesToCollide.pushBack(event.m_breakingBody);

				// Mark it as unbreakable, it has already been broken
// 				if ( disableFurtherBreakOff )
// 				{
// 					_markUnbreakableBody(event.m_breakingBody);
// 				}
			}

			delete constraintEvents[i];
		}
	}

	// Multi Threaded Section
	{
		// Add newly created bodies
		m_world->addPhysicsSystem(&newSystem);

		// Collide newly created and modified bodies
		bodiesToCollide.insertAt(bodiesToCollide.getSize(), (hkpEntity**)newSystem.getRigidBodies().begin(), newSystem.getRigidBodies().getSize());
		if ( bodiesToCollide.getSize() )
		{
			m_world->reintegrateAndRecollideEntities(bodiesToCollide.begin(), bodiesToCollide.getSize(), hkpWorld::RR_MODE_RECOLLIDE_NARROWPHASE);
			hkReferencedObject::lockAll();
			newSystem.removeAll();
			hkReferencedObject::unlockAll();
		}
	}
}

//
//	Called by the constraint solver when it needs to apply more than the maximum allowed impulse in order to maintain the given contacts.

void hkpDestructionBreakOffUtil::contactImpulseLimitBreachedCallbackToi(const hkpContactImpulseLimitBreachedListenerInfo* breachedContact)
{
	hkpPhysicsSystem newSystem;
	hkpRigidBody* breakingBody	= HK_NULL;
	hkpRigidBody* collidingBody	= HK_NULL;

	// Single Threaded Section
	bool recollideBody = false;
	{
		hkCriticalSectionLock lock(m_criticalSection);

		const hkpContactImpulseLimitBreachedListenerInfo& bi = *breachedContact;

		// Get the breaking body (if any)
		hkpShapeKey key					= HK_INVALID_SHAPE_KEY;
		hkBool32 disableFurtherBreakOff	= false;
		if ( !queryImpulseLimitBreachedInfo(this, bi, breakingBody, key, disableFurtherBreakOff) )
		{
			return;	// Body is not breakable!
		}

		// Create the event
		Event event;
		event.m_breakingBody = breakingBody;

		// Create the one and only contact point
		PointInfo& pointInfo				= event.m_points.expandOne();
		pointInfo.m_collidingBody			= hkSelectOther(breakingBody, bi.getBodyA(), bi.getBodyB());
		pointInfo.m_contactPointDirection	= (breakingBody ==  bi.getBodyA()) ? 1.0f : -1.0f;
		pointInfo.m_brokenShapeKey			= key;
		pointInfo.m_isContact				= true;
		pointInfo.m_contactPoint			= bi.getContactPoint();
		pointInfo.m_properties				= bi.getContactPointProperties();
		pointInfo.m_breakingImpulse			= pointInfo.m_properties->m_maxImpulse;
		pointInfo.m_internalContactMgr		= bi.getContactMgr();
		collidingBody = pointInfo.m_collidingBody;

		hkResult result = m_breakOffPartsListener->breakOffSubPart(event, newSystem);
		if ( result == HK_FAILURE)
		{
			return;	// Don't break it off
		}

		// The body was broken
		recollideBody = true;

		// Mark it as unbreakable, it has already been broken
// 		if ( disableFurtherBreakOff )
// 		{
// 			_markUnbreakableBody(breakingBody);
// 		}
	}

	// Multi Threaded Section
	{
		// Update broken body
		hkpRigidBody* bodyToCollide = HK_NULL;
		{
			// If we have a TOI, we have to re-collide our current pair
			if ( !breakingBody->isFixed() )
			{
				// Redo all collisions of the breaking moving body
				bodyToCollide = breakingBody;
			}
			else
			{
				// Just update the pair (as soon as havok supports it) : m_world->recollideTwoEntities( breakingBody, collidingBody ); // implement this
				// For the time being we simply update the moving body (the fixed body might be huge and would give us 
				// an incredible CPU spike).
				// As a caveat there
				// might be still TOIs scheduled for the fixed breakingBody and another object.
				// In this case we just let this false TOI happen.
				bodyToCollide = collidingBody;
			}
		}

		// Add the new bodies
		m_world->addPhysicsSystem(&newSystem);

		// Collide the new bodies and the modified body
		if ( recollideBody )
		{
			hkReferencedObject::lockAll();
			newSystem.addRigidBody(bodyToCollide);
			hkReferencedObject::unlockAll();
		}

		if ( newSystem.getRigidBodies().getSize() )
		{
			m_world->reintegrateAndRecollideEntities((hkpEntity**)newSystem.getRigidBodies().begin(), newSystem.getRigidBodies().getSize(), hkpWorld::RR_MODE_RECOLLIDE_NARROWPHASE);
			hkReferencedObject::lockAll();
			newSystem.removeAll();
			hkReferencedObject::unlockAll();
		}
	}
}

#endif	//	!HK_PLATFORM_SPU

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
