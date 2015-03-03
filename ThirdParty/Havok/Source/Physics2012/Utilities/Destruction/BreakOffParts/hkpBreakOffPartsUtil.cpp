/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/BitField/hkBitField.h>

#include <Physics2012/Utilities/Destruction/BreakOffParts/hkpBreakOffPartsUtil.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/Modifiers/hkpRemoveTerminalsMoppModifier.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/hkpPhysicsSystem.h>

#include <Physics/Constraint/Data/BallAndSocket/hkpBallAndSocketConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintPivotsUtil.h>
#include <Physics/Constraint/Data/hkpConstraintDataUtils.h>



hkpBreakOffPartsUtil::BreakOffGameControlResult 
hkpBreakOffPartsUtil::GameControlFunctor::test(const hkpRigidBody* limitedBody, 
											   const hkpRigidBody* otherBody, 
											   const hkpShapeKey key, 
											   const hkUFloat8& maxImpulse)
{
	return USE_LIMIT;
}


hkpBreakOffPartsUtil::hkpBreakOffPartsUtil( hkpBreakOffPartsListener* listenerInterface )
	: hkpWorldExtension( HK_WORLD_EXTENSION_BREAK_OFF_PARTS )
	, m_criticalSection(new hkCriticalSection(1))
	, m_breakOffPartsListener(listenerInterface)
{
	GameControlFunctor* func = new GameControlFunctor;
	m_breakOffGameControlFunctor = func;
	func->removeReference();
}

hkpBreakOffPartsUtil::~hkpBreakOffPartsUtil()
{
	delete m_criticalSection;
}

void hkpBreakOffPartsUtil::removeKeysFromListShape( hkpEntity* entity, hkpShapeKey* keysToRemove, int numKeys)
{
	hkAabbUint32* childAabbs = entity->getCollidable()->m_boundingVolumeData.m_childShapeAabbs;
	const hkpShape* shape = entity->getCollidable()->getShape();

	switch( shape->getType() )
	{
		case hkcdShapeType::LIST:
doListShape:
			{
				hkpListShape* list = const_cast<hkpListShape*>(static_cast<const hkpListShape*>(shape));
				HK_ASSERT2( 0x6b24907c, !childAabbs || entity->getCollidable()->m_boundingVolumeData.m_capacityChildShapeAabbs >= list->getNumChildShapes(), "The cached number of child shapes is invalid" );
				for (int i = 0; i < numKeys; i++ )
				{
					list->disableChild( keysToRemove[i] );
				}
			}
			break;

		case hkcdShapeType::MOPP:
			{
				hkpMoppBvTreeShape* moppShape = const_cast<hkpMoppBvTreeShape*>(static_cast<const hkpMoppBvTreeShape*>( shape ));
				const hkpShapeCollection* collection = moppShape->getShapeCollection();
				hkArray<hkpShapeKey> keys(keysToRemove,numKeys,numKeys);

				hkpRemoveTerminalsMoppModifier modifier(moppShape->getMoppCode(), collection, keys);
				modifier.applyRemoveTerminals( const_cast<hkpMoppCode*>(moppShape->getMoppCode()) );

				if ( collection->getType() == hkcdShapeType::LIST )
				{
					shape = collection;
					goto doListShape;
				}
			}
			break;

		default:
			HK_ASSERT2( 0xf02efe56, 0, "Unsupported shape type" );
			return;
	}

	// disable cached AABBs in the entity
	if ( childAabbs )
	{
		
		hkPointerMap<hkpShapeKey, int> keysRemovedMap;
		for(int ki = 0; ki < numKeys; ki++)
		{
			keysRemovedMap.insert(keysToRemove[ki], 1);
		}

		// Find the first removed element.
		int numAabbs = entity->getCollidable()->m_boundingVolumeData.m_numChildShapeAabbs;
		hkpShapeKey* childKeys = entity->getCollidable()->m_boundingVolumeData.m_childShapeKeys;

		for( int ai = 0; ai < numAabbs ; ai++) 
		{
			if (keysRemovedMap.getWithDefault(childKeys[ai], 0))
			{
				childAabbs[ai].setInvalidY();
			}
		}
	}
}

void hkpBreakOffPartsUtil::removeKeysFromListShapeByEnabledList( hkpEntity* entity, const hkBitField& enabledKeys )
{
	hkAabbUint32* childAabbs = entity->getCollidable()->m_boundingVolumeData.m_childShapeAabbs;
	const hkpShape* shape = entity->getCollidable()->getShape();

	switch( shape->getType() )
	{
	case hkcdShapeType::LIST:
doListShape:
		{
			hkpListShape* list = const_cast<hkpListShape*>(static_cast<const hkpListShape*>(shape));
			HK_ASSERT2( 0xf0345465, !childAabbs || entity->getCollidable()->m_boundingVolumeData.m_capacityChildShapeAabbs >= list->getNumChildShapes(), "The cached number of child shapes is invalid" );
			HK_ASSERT2( 0xf0436ede, enabledKeys.getSize() == list->m_childInfo.getSize(), "The bitfield does not match the list shape" );
			list->setEnabledChildren( enabledKeys );
			break;	
		}

	case hkcdShapeType::MOPP:
		{
			hkpMoppBvTreeShape* moppShape = const_cast<hkpMoppBvTreeShape*>(static_cast<const hkpMoppBvTreeShape*>( shape ));
			const hkpShapeCollection* collection = moppShape->getShapeCollection();

			hkpRemoveTerminalsMoppModifier2 modifier(moppShape->getMoppCode(), enabledKeys);
			modifier.applyRemoveTerminals( const_cast<hkpMoppCode*>(moppShape->getMoppCode()) );

			if ( collection->getType() == hkcdShapeType::LIST )
			{
				shape = collection;
				goto doListShape;
			}
			HK_WARN( 0xf0235456, "Cannot remove keys from a non list shape" );
			break;
		}

	default:
		HK_ASSERT2( 0xf02efe56, 0, "Unsupported shape type" );
		return;
	}

	// disable cached AABBs in the entity
	if ( childAabbs )
	{
		int numAabbs = entity->getCollidable()->m_boundingVolumeData.m_numChildShapeAabbs;

		for (int i = 0; i < numAabbs ; i++) 
		{
			if ( enabledKeys.get(i) == 0 )
			{
				childAabbs[i].setInvalidY();
			}
		}
	}
}

void hkpBreakOffPartsUtil::worldDeletedCallback( hkpWorld* world)
{
	m_world->removeContactImpulseLimitBreachedListener(this);
	m_world = HK_NULL;
	removeReference();
}

void hkpBreakOffPartsUtil::entityRemovedCallback( hkpEntity* entity )
{
}

hkpBreakOffPartsUtil::LimitContactImpulseUtil::LimitContactImpulseUtil( hkpBreakOffPartsUtil* breakUtil, hkpEntity* entity )
	: m_entity(entity)
	, m_maxConstraintImpulse(HK_REAL_MAX)
	, m_breakOffUtil(breakUtil)
{
	HK_ASSERT2( 0xf0323454, entity->m_limitContactImpulseUtilAndFlag == HK_NULL, "The hkpEntity::m_limitContactImpulseUtilAndFlag member is already in use" );
	m_maxImpulse.m_value = hkUFloat8::MAX_VALUE-1; // aka unbreakable
	entity->m_limitContactImpulseUtilAndFlag = this;
	entity->addContactListener( this ); 
	m_breakOffUtil->addReference();
}

hkpBreakOffPartsUtil::LimitContactImpulseUtil::~LimitContactImpulseUtil()
{
	m_breakOffUtil->removeReference();
}

hkpBreakOffPartsUtil::LimitContactImpulseUtil* hkpBreakOffPartsUtil::createLimitContactImpulseUtil( hkpEntity* entity )
{
	return new LimitContactImpulseUtilDefault( this, entity );
}

hkpBreakOffPartsUtil::LimitContactImpulseUtil* hkpBreakOffPartsUtil::getOrCreateUtil( hkpEntity* entity )
{
	// search entry for entity
	LimitContactImpulseUtil* limitUtil = getLimitContactImpulseUtilPtr( entity );
	return limitUtil ? limitUtil : createLimitContactImpulseUtil( entity );
}

void hkpBreakOffPartsUtil::markPieceBreakable( hkpEntity* entity, hkpShapeKey key, hkReal maxImpulse )
{
	HK_ASSERT2 ( 0xf0ffee34, entity->m_numShapeKeysInContactPointProperties >= 1, "You have to set hkpEntity::m_numShapeKeysInContactPointProperties to a minimum of 1 " );
	LimitContactImpulseUtil* limitUtil = getOrCreateUtil( entity ); 
	hkUFloat8 mf = hkFloat32(maxImpulse);
	mf.m_value = hkUint8(hkMath::max2( 1, int(mf.m_value) ));	// 0 is a special flag, so use a minimum of 1
	limitUtil->setMaxImpulseForShapeKey(key, mf.m_value);
}

void hkpBreakOffPartsUtil::markEntityBreakable( hkpEntity* entity, hkReal maxImpulse )
{
	LimitContactImpulseUtil* limitUtil = getOrCreateUtil( entity );
	hkUFloat8 mf = hkFloat32(maxImpulse);
	limitUtil->m_maxImpulse.m_value = hkUint8(hkMath::max2( 1, int(mf.m_value) ));	// 0 is a special flag, so use a minimum of 1
}

void hkpBreakOffPartsUtil::unmarkPieceBreakable( hkpEntity* entity, hkpShapeKey key )
{
	HK_ASSERT2 ( 0xf0ffae34, entity->m_numShapeKeysInContactPointProperties >= 1, "You have to set hkpEntity::m_numShapeKeysInContactPointProperties to a minimum of 1 " );
	LimitContactImpulseUtil* limitUtil = getLimitContactImpulseUtilPtr( entity );
	if (limitUtil)
	{
		limitUtil->removeKey(key);

		if (! limitUtil->hasShapeKeySpecificMaxImpulses())
		{
			unmarkEntityBreakable(entity);
		}
	}
}

void hkpBreakOffPartsUtil::unmarkEntityBreakable( hkpEntity* entity )
{
	LimitContactImpulseUtil* limitUtil = getLimitContactImpulseUtilPtr( entity );
	if (limitUtil)
	{
		entity->removeContactListener(limitUtil);
		delete limitUtil;
		entity->m_limitContactImpulseUtilAndFlag = HK_NULL;
	}
}

void hkpBreakOffPartsUtil::setMaxConstraintImpulse( hkpEntity* entity, hkReal maxConstraintImpulse )
{
	LimitContactImpulseUtil* limitUtil = getOrCreateUtil( entity );
	limitUtil->m_maxConstraintImpulse = maxConstraintImpulse;
}

void hkpBreakOffPartsUtil::LimitContactImpulseUtil::limitContactImpulse( const hkUFloat8& maxImpulse, bool isEntityIdSmaller, hkpContactPointProperties* properties )
{
	if ( !maxImpulse.m_value )
	{
		return;
	}

	// if the piece is flagged to be breakable, than take the one with lower force
	if ( properties->m_maxImpulse.m_value )
	{
		// integer compare is faster
		if ( maxImpulse.m_value >= properties->m_maxImpulse.m_value )
		{
			return;	// existing limits is already lower than my one
		}
	}

	properties->setMaxImpulsePerStep( maxImpulse );
	if ( isEntityIdSmaller )
	{
		properties->m_flags |= hkContactPointMaterial::CONTACT_BREAKOFF_OBJECT_ID_SMALLER;
	}
	else
	{
		properties->m_flags &= ~hkContactPointMaterial::CONTACT_BREAKOFF_OBJECT_ID_SMALLER;
	}
}

void hkpBreakOffPartsUtil::LimitContactImpulseUtilDefault::contactPointAddedCallback( hkpContactPointAddedEvent& event )
{
	// Early out if the breakOffPartsUtil has been removed from the world.
	if ( !m_breakOffUtil->getWorld() )
	{
		return;
	}

	// get the body just one below the root, the one which should hold the shapekey
	const hkpCdBody* rootA = event.m_bodyA;
	const hkpCdBody* topChildA = rootA;
	while ( rootA->m_parent ){ topChildA = rootA; rootA = rootA->m_parent; }

	hkpShapeKey key = topChildA->m_shapeKey;

	// see which body we work on (this callback is called twice on CPU/PPU)
	if ( m_entity->getCollidableMtUnchecked() != rootA )
	{
		const hkpCdBody* rootB = event.m_bodyB;
		const hkpCdBody* topChildB = rootB;
		while ( rootB->m_parent ){ topChildB = rootB; rootB = rootB->m_parent; }

		key = topChildB->m_shapeKey;
	}
	
	if (event.m_internalContactMgr->getType() == hkpContactMgr::TYPE_SIMPLE_CONSTRAINT_CONTACT_MGR)
	{
		// find out whether we are object A or B in the contact
		hkpSimpleConstraintContactMgr* mgr = reinterpret_cast<hkpSimpleConstraintContactMgr*>(event.m_internalContactMgr);
		hkpEntity* e = mgr->m_constraint.getEntityA();
		if ( e == m_entity )
		{
			hkpRigidBody* rbA = reinterpret_cast<hkpRigidBody*>( e );
			hkpRigidBody* rbB = reinterpret_cast<hkpRigidBody*>( mgr->m_constraint.getEntityB() );

			hkUFloat8 maxImpulse = getMaxImpulseForContactPoint(key, rbB);
			BreakOffGameControlResult gameControl = m_breakOffUtil->m_breakOffGameControlFunctor->test(rbA, rbB, key, maxImpulse);

			switch (gameControl)
			{
				case BREAK_OFF: // zero limit = force breaking
					maxImpulse.m_value = 1;
					break;

				case DO_NOT_BREAK_OFF: // no limit = no breaking
					return;

				case USE_LIMIT:
					break;

				default:
					HK_ASSERT2(0x23750120, 0, "Break off game control error");
					break;
			}
			const bool isEntityIdSmaller = (rbA->getUid() < rbB->getUid());
			limitContactImpulse( maxImpulse, isEntityIdSmaller, event.m_contactPointProperties );
		}
		else
		{
			hkpRigidBody* rbA = reinterpret_cast<hkpRigidBody*>( e );
			hkpRigidBody* rbB = reinterpret_cast<hkpRigidBody*>( m_entity );

			hkUFloat8 maxImpulse = getMaxImpulseForContactPoint(key, rbA);
			BreakOffGameControlResult gameControl = m_breakOffUtil->m_breakOffGameControlFunctor->test(rbB, rbA, key, maxImpulse);

			switch (gameControl)
			{
				case BREAK_OFF: // zero limit = force breaking
					maxImpulse.m_value = 1; 
					break;

				case DO_NOT_BREAK_OFF: // no limit = no breaking
					return;

				case USE_LIMIT:
					break;

				default:
					HK_ASSERT2(0x23750121, 0, "Break off game control error");
					break;
			}
			const bool isEntityIdSmaller = (rbB->getUid() < rbA->getUid());
			limitContactImpulse( maxImpulse, isEntityIdSmaller, event.m_contactPointProperties );
		}
	}
	else
	{
		HK_WARN( 0xf023ef45, "This utility only works with the default contact mgr. A different manager type was found and this event is ignored.");
	}
}

// Search for the last valid key, or HK_INVALID_SHAPE_KEY if there wasn't one.
static inline hkpShapeKey findTopShapeKey( const hkpShapeKey* leafKeyPtr, int sizeOfKeyStorage )
{
	hkpShapeKey key = HK_INVALID_SHAPE_KEY;
	const hkpShapeKey *const end = leafKeyPtr + sizeOfKeyStorage;
	const hkpShapeKey* kp = leafKeyPtr;
	while ( ( *kp != HK_INVALID_SHAPE_KEY ) && ( kp < end ) )
	{
		key = *kp;
		++kp;
	}
	return key;
}

void hkpBreakOffPartsUtil::LimitContactImpulseUtilCpuOnly::contactPointCallback( const hkpContactPointEvent& event )
{
	// Early out if the breakOffPartsUtil has been removed from the world.
	if ( !m_breakOffUtil->getWorld() )
	{
		return;
	}

	if ( event.m_contactPointProperties->m_flags & hkContactPointMaterial::CONTACT_IS_NEW )
	{
		HK_ASSERT2( 0x4f79a1be, event.m_source != hkpContactPointEvent::SOURCE_WORLD, "This listener should not be attached to the world." );
		HK_ASSERT2( 0xfe7ab8a1, event.getBody( event.m_source ) == m_entity, "This listener should be attached to the breakable entity." );
		const hkpShapeKey key = findTopShapeKey( event.getShapeKeys( event.m_source ), m_entity->m_numShapeKeysInContactPointProperties );	

		if ( event.m_source == hkpContactPointEvent::SOURCE_A )
		{
			hkpRigidBody* rbA = event.getBody(event.m_source);
			hkpRigidBody* rbB = event.getBody(1 - event.m_source);

			hkUFloat8 maxImpulse = getMaxImpulseForContactPoint(key, rbB);
			BreakOffGameControlResult gameControl = m_breakOffUtil->m_breakOffGameControlFunctor->test(rbA, rbB, key, maxImpulse);

			switch (gameControl)
			{
			case BREAK_OFF: // zero limit = force breaking
				maxImpulse.m_value = 1;
				break;

			case DO_NOT_BREAK_OFF: // no limit = no breaking
				return;

			case USE_LIMIT:
				break;

			default:
				HK_ASSERT2(0x23750122, 0, "Break off game control error");
				break;
			}
			const bool isEntityIdSmaller = (rbA->getUid() < rbB->getUid());
			limitContactImpulse( maxImpulse, isEntityIdSmaller, event.m_contactPointProperties );
		}
		else
		{
			hkpRigidBody* rbA = event.getBody(1 - event.m_source);
			hkpRigidBody* rbB = event.getBody(event.m_source);

			hkUFloat8 maxImpulse = getMaxImpulseForContactPoint(key, rbA);
			BreakOffGameControlResult gameControl = m_breakOffUtil->m_breakOffGameControlFunctor->test(rbB, rbA, key, maxImpulse);

			switch (gameControl)
			{
			case BREAK_OFF: // zero limit = force breaking
				maxImpulse.m_value = 1;
				break;

			case DO_NOT_BREAK_OFF: // no limit = no breaking
				return;

			case USE_LIMIT:
				break;

			default:
				HK_ASSERT2(0x23750123, 0, "Break off game control error");
				break;
			}
			const bool isEntityIdSmaller = (rbB->getUid() < rbA->getUid());
			limitContactImpulse( maxImpulse, isEntityIdSmaller, event.m_contactPointProperties );
		}
	}
}

void hkpBreakOffPartsUtil::constraintAddedCallback( hkpConstraintInstance* constraint )
{
	if (constraint->getData()->getType() == hkpConstraintData::CONSTRAINT_TYPE_CONTACT)
	{
		return;
	}

	// Check if either of the bodies has a breakOffUtil attached.
	hkpBreakOffPartsUtil::LimitContactImpulseUtil* utilA = getLimitContactImpulseUtilPtr( constraint->getEntityA() );
	hkpBreakOffPartsUtil::LimitContactImpulseUtil* utilB = getLimitContactImpulseUtilPtr( constraint->getEntityB() );

	hkpConstraintData* constraintData = const_cast<hkpConstraintData*>(constraint->getData());

	hkReal maxImpulse = HK_REAL_MAX;
	if (utilA || utilB)
	{
		// Find max impulse
		hkReal maxImpulseA = HK_REAL_MAX;
		hkReal maxImpulseB = HK_REAL_MAX;

		if (utilA)
		{
			maxImpulseA = utilA->m_maxConstraintImpulse;
		}

		if (utilB)
		{
			maxImpulseB = utilB->m_maxConstraintImpulse;
		}
		maxImpulse = hkMath::min2(maxImpulseA, maxImpulseB);

		// Write data to the constraint
		int notifiedBodyIdx = maxImpulseA < maxImpulseB ? 0 : 1;
		constraintData->setBodyToNotify(notifiedBodyIdx);
	}

	// set the constraint's force limits 
	// we have to call this always as when a previously-breakable body is broken into pieces, and pieces are not
	// breakable anymore and do not have an util connected. --- we don't want to limit the solver's
	// impulses on such bodies then.
	constraintData->setBreachImpulse(maxImpulse);

}

static inline hkpBreakOffPartsUtil::LimitContactImpulseUtil* findUtil( hkpSimpleConstraintContactMgr* mgr, hkpContactPointProperties* props, hkBool& defaultValueHitOut, hkpShapeKey& key, hkUFloat8& maxImpulseOut )
{
	int id = -1;

	if (props->m_flags & hkContactPointMaterial::CONTACT_BREAKOFF_OBJECT_ID_SMALLER )
	{
		id = (mgr->getConstraintInstance()->getEntityA()->getUid() < mgr->getConstraintInstance()->getEntityB()->getUid() ? 0 :	1 );
	}
	else
	{
		id = (mgr->getConstraintInstance()->getEntityA()->getUid() < mgr->getConstraintInstance()->getEntityB()->getUid() ? 1 :	0 );
	}
	
	hkpBreakOffPartsUtil::LimitContactImpulseUtil* util = hkpBreakOffPartsUtil::getLimitContactImpulseUtilPtr( mgr->m_constraint.getEntity(id) );

	int maxImpulse = props->m_maxImpulse.m_value;
	maxImpulseOut.m_value = hkInt8(maxImpulse);

	if ( !util )
	{
		return HK_NULL;
	}

	const hkpSimpleContactConstraintAtom* atom = mgr->getAtom();
	hkpShapeKey* startOfKeys = props->getStartOfExtendedUserData( atom ) + ( id ? props->getNumExtendedUserDatas( atom, 0 ) : 0 );
	key = findTopShapeKey( startOfKeys, props->getNumExtendedUserDatas( atom, id ) );

	key = props->getExtendedUserData(atom, id, 0);
	{
		// lets find the object
		if ( key != HK_INVALID_SHAPE_KEY )
		{
			int maxA = util->getMaxImpulseForKey(key);
			if ( maxA != 0 )
			{
				return util;
			}
		}
	}
	if ( util->m_maxImpulse.m_value != 0 )
	{
		defaultValueHitOut = true;
		return util;
	}
	return HK_NULL;
}

static inline hkpBreakOffPartsUtil::LimitContactImpulseUtil* findUtil( const hkpRigidBody* body, const hkpConstraintInstance* constraint, hkBool& defaultValueHitOut, hkpShapeKey& key, hkUFloat8& maxImpulseOut )
{
	hkpBreakOffPartsUtil::LimitContactImpulseUtil* util = hkpBreakOffPartsUtil::getLimitContactImpulseUtilPtr( body );

	hkFloat32 maxImpulse = hkFloat32(constraint->getData()->getBreachImpulse());
	maxImpulseOut.m_value = hkUint8(hkUFloat8(maxImpulse));

	if ( !util )
	{
		return HK_NULL;
	}

	// Find attachment shapeKey
	//
	hkVector4 pivotInBodySpace = body == static_cast<hkpRigidBody*>(constraint->getEntityA()) ? hkpConstraintDataUtils::getPivotA(constraint->getData()) : hkpConstraintDataUtils::getPivotB(constraint->getData());
	key = hkpConstraintPivotsUtil::findClosestShapeKey(body->getWorld(), body->getCollidable()->getShape(), pivotInBodySpace);

	{
		// lets find the object
		if ( key != HK_INVALID_SHAPE_KEY )
		{
			int maxA = util->getMaxImpulseForKey(key);
			if ( maxA != 0 )
			{
				return util;
			}
		}
	}
	if ( util->m_maxImpulse.m_value != 0 )
	{
		defaultValueHitOut = true;
		return util;
	}
	return HK_NULL;
}

namespace 
{
	struct hkpLimitBreachedEventInfo
	{
		hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent* m_event;
		hkBool32 m_defaultValueHit;
		hkpBreakOffPartsUtil::LimitContactImpulseUtil* m_util;
		//hkBool32 m_hasOneOrMoreToi;
	};

}

void hkpBreakOffPartsUtil::contactImpulseLimitBreachedCallback( const hkpContactImpulseLimitBreachedListenerInfo* breachedContacts, int numBreachedContacts )
{
  	if (breachedContacts[0].isToi())
	{
		HK_ASSERT2(0xad875533, numBreachedContacts == 1, "TOI contact points are always processed individually.");
		contactImpulseLimitBreachedCallback_forToi(&breachedContacts[0]);
		return;
	}

	hkInplaceArray<hkpEntity*, 128> bodiesToCollide;
	hkpPhysicsSystem newSystem;

	//
	//	Single Threaded Section
	//
	{
		hkPointerMap<hkpBreakOffPartsUtil::LimitContactImpulseUtil*, int> utilPtrToEventIndexMap;
		hkLocalArray<hkpLimitBreachedEventInfo> eventInfos(numBreachedContacts); // allocate worst-case size
		hkLocalArray<hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent*> constraintEvents(numBreachedContacts);
		hkLocalArray<const hkpConstraintInstance*> constraints(numBreachedContacts);
		hkLocalArray<hkContactPoint> constraintPoints(numBreachedContacts);

		hkCriticalSectionLock lock(m_criticalSection);

		for (int i =0; i < numBreachedContacts; i++)
		{
			const hkpContactImpulseLimitBreachedListenerInfo& bi = breachedContacts[i];
			HK_ASSERT2(0xad8533a3, !bi.isToi(), "This function is for non-TOI contact points only.");

			if (bi.isContact())
			{
				// get the shape key
				hkpShapeKey key = HK_INVALID_SHAPE_KEY;

				// get the breakable object. This can be tricky if both objects are breakable.
				hkBool defaultValueHit = false;
				hkUFloat8 unusedMaxImpulse;
				hkpBreakOffPartsUtil::LimitContactImpulseUtil* util = findUtil( bi.getContactMgr(), bi.getContactPointProperties(), defaultValueHit, key, unusedMaxImpulse ); 

				if ( !util )
				{
					continue;	// already removed
				}

				// get event
				hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent* event = HK_NULL;
				hkPointerMap<hkpBreakOffPartsUtil::LimitContactImpulseUtil*, int>::Iterator it = utilPtrToEventIndexMap.findKey(util);
				if (utilPtrToEventIndexMap.isValid(it))
				{
					const int index = utilPtrToEventIndexMap.getValue(it);
					hkpLimitBreachedEventInfo& info = eventInfos[index];
					event = info.m_event;
					HK_ASSERT2(0xad7644aa, event->m_breakingBody == (hkpRigidBody*)util->m_entity, "Entity pointers don't match.");
					info.m_defaultValueHit = info.m_defaultValueHit | hkBool32(defaultValueHit);
				}
				else
				{
					utilPtrToEventIndexMap.insert(util, eventInfos.getSize());
					hkpLimitBreachedEventInfo& info = *eventInfos.expandByUnchecked(1);
					info.m_event = new hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent();
					info.m_util  = util;
					event = info.m_event;
					event->m_breakingBody = (hkpRigidBody*)util->m_entity;
					info.m_defaultValueHit = defaultValueHit;
				}

				hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent::PointInfo& pointInfo = event->m_points.expandOne();

				pointInfo.m_collidingBody  = hkSelectOther( event->m_breakingBody, bi.getBodyA(), bi.getBodyB() );
				pointInfo.m_contactPointDirection = (event->m_breakingBody ==  bi.getBodyA() ) ? 1.0f : -1.0f;
				pointInfo.m_brokenShapeKey = key;
				pointInfo.m_isContact = true;
				pointInfo.m_contactPoint   = bi.getContactPoint();
				hkpContactPointProperties* cpp = bi.getContactPointProperties();
				pointInfo.m_properties     = cpp;
				pointInfo.m_breakingImpulse= cpp->m_maxImpulse;
				pointInfo.m_internalContactMgr = bi.getContactMgr();
			}
			else // process constraints
			{
				//
				// Create a new event. Note that we don't search for the corresponding util -- so this cannot be used for breakables just yet.

				const hkpConstraintInstance* instance = bi.getConstraintInstance();
				const hkpConstraintData* data = static_cast<const hkpConstraintData*>(instance->getData());

				hkUint8 notifiedBodyIdx = data->getNotifiedBodyIndex();
				hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent* constraintEvent = new hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent();
				constraintEvents.pushBackUnchecked( constraintEvent );
				hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent::PointInfo& pointInfo = *constraintEvent->m_points.expandByUnchecked(1);
				constraints.pushBackUnchecked(instance);

				constraintEvent->m_breakingBody = static_cast<hkpRigidBody*>( notifiedBodyIdx ? instance->getEntityB() : instance->getEntityA() );
				pointInfo.m_collidingBody = static_cast<hkpRigidBody*>(instance->getOtherEntity(constraintEvent->m_breakingBody));
				pointInfo.m_brokenShapeKey = HK_INVALID_SHAPE_KEY;
				pointInfo.m_isContact = false;
				pointInfo.m_properties = HK_NULL;
				pointInfo.m_breakingImpulse = HK_REAL_MAX;
				pointInfo.m_internalContactMgr = HK_NULL;
				pointInfo.m_contactPointDirection = 1.0f;
				hkContactPoint* cp = constraintPoints.expandByUnchecked(1);
				pointInfo.m_contactPoint = cp;

				// 
				// Compute contact point.

				hkVector4 pivotA, pivotB;
				instance->getPivotsInWorld(pivotA, pivotB);
				cp->setPosition( instance->getEntityA() == constraintEvent->m_breakingBody ? pivotA : pivotB );

				hkVector4 displacement = cp->getSeparatingNormal(); 
				if (instance->getEntityA() == constraintEvent->m_breakingBody)
				{
					displacement.setSub(pivotB, pivotA); 
				}
				else
				{
					displacement.setSub(pivotA, pivotB);
				}

				const hkSimdReal distance = displacement.length<3>();
				if (distance < hkSimdReal_Eps)
				{
					displacement = hkVector4::getConstant<HK_QUADREAL_1000>();
				}
				else
				{
					displacement.normalize<3>();
				}

				// distance: 
				cp->setSeparatingNormal(displacement,-distance);


			}
		}

		// Trigger break-off callbacks for contact points and remove keys from corresponding utils.
		//
		for (int ui=0; ui < eventInfos.getSize(); ui++)
		{
			hkpLimitBreachedEventInfo& info = eventInfos[ui];
			hkpBreakOffPartsUtil::LimitContactImpulseUtil* util = info.m_util;
			hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent& event = *info.m_event;

			hkInplaceArray<hkpShapeKey,256> keysToRemove;
			hkResult result = m_breakOffPartsListener->breakOffSubPart( event, keysToRemove, newSystem );

			if ( result == HK_SUCCESS)
			{
				if ( keysToRemove.getSize() )
				{
					bodiesToCollide.pushBack(event.m_breakingBody);
				}
				// remove keys 
				for (int k = 0; k < keysToRemove.getSize(); k++)	{ util->removeKey( keysToRemove[k] );	}	

				// mark the entire object as unbreakable if it has been hit
				if ( info.m_defaultValueHit  )                      { util->m_maxImpulse.m_value = hkUFloat8::MAX_VALUE-1; }

				for (int p = 0; p < event.m_points.getSize(); p++)
				{
					hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent::PointInfo& pointInfo = event.m_points[p];

					// disable contact. This is only necessary if we this function is called 
					// from a simple collision response, which happens before solving. In this case
					// the contact point will still get into the solver, so we simple disable it by setting
					// the distance to infinity

					// The const cast is safe: we got this point from hkpContactImpulseLimitBreachedListenerInfo above..
					const_cast<hkContactPoint*>(pointInfo.m_contactPoint)->setDistance( HK_REAL_HIGH );
				}
			}

			delete info.m_event;
		}

		// Trigger break-off callbacks for constraints.
		//
		for (int i = 0; i < constraintEvents.getSize(); i++)
		{
			hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent& event = *constraintEvents[i];

			// an util may be connected to the body. but there may be none as well
			hkUFloat8 unusedMaxImpulse;
			hkBool defaultValueHit = false;
			hkpBreakOffPartsUtil::LimitContactImpulseUtil* util = findUtil(event.m_breakingBody, constraints[i], defaultValueHit, event.m_points[0].m_brokenShapeKey, unusedMaxImpulse );

			hkInplaceArray<hkpShapeKey,256> keysToRemove;
			hkResult result = m_breakOffPartsListener->breakOffSubPart(event, keysToRemove, newSystem);

			if ( result == HK_SUCCESS && util)
			{
				if ( keysToRemove.getSize() )
				{
					bodiesToCollide.pushBack(event.m_breakingBody);
				}
				// remove keys 
				for (int k = 0; k < keysToRemove.getSize(); k++)	{ util->removeKey( keysToRemove[k] );	}	

				// mark the entire object as unbreakable if it has been hit
				if ( defaultValueHit  )                      { util->m_maxImpulse.m_value = hkUFloat8::MAX_VALUE-1; }

			}

			delete constraintEvents[i];
		}
	}

	//
	//	Multi Threaded Section
	//

		// Add newly created bodies
	m_world->addPhysicsSystem( &newSystem );

		// Collide newly created and modified bodies
	bodiesToCollide.insertAt(bodiesToCollide.getSize(), (hkpEntity**)newSystem.getRigidBodies().begin(), newSystem.getRigidBodies().getSize());

	if (bodiesToCollide.getSize())
	{
		m_world->reintegrateAndRecollideEntities(bodiesToCollide.begin(), bodiesToCollide.getSize(), hkpWorld::RR_MODE_RECOLLIDE_NARROWPHASE );
		hkReferencedObject::lockAll();
		newSystem.removeAll();
		hkReferencedObject::unlockAll();
	}

}


void hkpBreakOffPartsUtil::contactImpulseLimitBreachedCallback_forToi( const hkpContactImpulseLimitBreachedListenerInfo* breachedContact )
{
	const hkpContactImpulseLimitBreachedListenerInfo& bi = *breachedContact;
	hkpRigidBody* breakingBody;
	hkpRigidBody* collidingBody;
	hkpPhysicsSystem newSystem;
	hkUFloat8 maxImpulse;

	//
	//	Single Threaded Section
	//
	bool recollideBody = false;
	{
		hkCriticalSectionLock lock(m_criticalSection);
		// get the shape key
		hkpShapeKey key = HK_INVALID_SHAPE_KEY; // SNC complains if passed to findUtil without initializing

		// get the breakable object. This can be tricky if both objects are breakable.
		hkBool defaultValueHit = false;
		hkpContactPointProperties* props = bi.getContactPointProperties();
		hkpBreakOffPartsUtil::LimitContactImpulseUtil* util = findUtil( bi.getContactMgr(), props, defaultValueHit, key, maxImpulse );

		if ( !util )
		{
			return;	// already removed
		}


		hkInplaceArray<hkpShapeKey,256> keysToRemove;

		breakingBody  = (hkpRigidBody*)util->m_entity;
		collidingBody = hkSelectOther( breakingBody, bi.getBodyA(), bi.getBodyB() );

		hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent event;
		event.m_breakingBody = breakingBody;

		hkpBreakOffPartsListener::ContactImpulseLimitBreachedEvent::PointInfo& pointInfo = event.m_points.expandOne();
		pointInfo.m_collidingBody  = collidingBody;
		pointInfo.m_brokenShapeKey = key;
		pointInfo.m_contactPoint   = bi.getContactPoint();
		pointInfo.m_properties     = bi.getContactPointProperties();
		pointInfo.m_breakingImpulse = pointInfo.m_properties->m_maxImpulse;
		pointInfo.m_contactPointDirection = (breakingBody ==  bi.getBodyA() ) ? 1.0f : -1.0f;
		pointInfo.m_isContact = true;

		hkResult result = m_breakOffPartsListener->breakOffSubPart( event,	keysToRemove, newSystem  );

		if ( result == HK_FAILURE)
		{
			return;	// don't break it off
		}

		// remove keys 
		if ( keysToRemove.getSize() )
		{
			recollideBody = true;
		}
		for (int k = 0; k < keysToRemove.getSize(); k++)	{ util->removeKey( keysToRemove[k] );	}	
		if ( defaultValueHit  )                             { util->m_maxImpulse.m_value = hkUFloat8::MAX_VALUE-1; }
	}

	//
	//	Multi Threaded Section
	//

	// update broken body
	hkpRigidBody* bodyToCollide = HK_NULL;
	{
		// if we have a TOI, we have to recollide our current pair
		if ( !breakingBody->isFixed() )
		{
			// redo all collisions of the breaking moving body
			bodyToCollide = breakingBody;
		}
		else
		{
			// just update the pair (as soon as havok supports it) : m_world->recollideTwoEntities( breakingBody, collidingBody ); // implement this
			// For the time being we simply update the moving body (the fixed body might be huge and would give us 
			// an incredible CPU spike).
			// As a caveat there
			// might be still TOIs scheduled for the fixed breakingBody and another object.
			// In this case we just let this false TOI happen.
			bodyToCollide = collidingBody;
		}
	}


	// add the new bodies
	m_world->addPhysicsSystem( &newSystem );

	// collide the new bodies and the modified body
	if ( recollideBody )
	{
		hkReferencedObject::lockAll();
		newSystem.addRigidBody(bodyToCollide);
		hkReferencedObject::unlockAll();
	}

	if ( newSystem.getRigidBodies().getSize() )
	{
		m_world->reintegrateAndRecollideEntities( (hkpEntity**)newSystem.getRigidBodies().begin(), newSystem.getRigidBodies().getSize(), hkpWorld::RR_MODE_RECOLLIDE_NARROWPHASE );
		hkReferencedObject::lockAll();
		newSystem.removeAll();
		hkReferencedObject::unlockAll();
	}

}

void hkpBreakOffPartsUtil::LimitContactImpulseUtil::findWeakestPoint( hkpShapeKey& keyOut, hkReal& weakestImpulseOut )
{
	if ( !hasShapeKeySpecificMaxImpulses() )
	{
		keyOut = HK_INVALID_SHAPE_KEY;
		weakestImpulseOut = m_maxImpulse;
		return;
	}

	hkUFloat8 minImpulse; minImpulse.m_value = hkUFloat8::MAX_VALUE-1;
	hkPointerMap<hkpShapeKey, hkUint8>::Iterator i;
	for ( i = m_shapeKeyToMaxImpulse.getIterator(); m_shapeKeyToMaxImpulse.isValid(i); i = m_shapeKeyToMaxImpulse.getNext(i))
	{
		hkUchar val = m_shapeKeyToMaxImpulse.getValue(i);
		if ( val < minImpulse.m_value)
		{
			minImpulse.m_value = val;
			keyOut = m_shapeKeyToMaxImpulse.getKey(i);
		}
	}
	weakestImpulseOut = minImpulse;
}

void hkpBreakOffPartsUtil::performAttachments( hkpWorld* world )
{
	world->addContactImpulseLimitBreachedListener( this ); 
	world->addConstraintListener( this );
}

void hkpBreakOffPartsUtil::performDetachments( hkpWorld* world )
{
	world->removeConstraintListener( this );
	world->removeContactImpulseLimitBreachedListener( this );
}

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
