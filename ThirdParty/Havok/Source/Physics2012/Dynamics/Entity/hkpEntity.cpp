/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Physics2012/Dynamics/Entity/hkpEntity.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>
#include <Common/Base/Types/Geometry/LocalFrame/hkLocalFrame.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>
#include <Physics2012/Collide/Shape/hkpShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>
#include <Physics2012/Dynamics/Entity/Util/hkpEntityAabbUtil.h>
#include <Physics2012/Dynamics/Entity/Util/hkpEntityCallbackUtil.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationQueue.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpFixedRigidMotion.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpKeyframedRigidMotion.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpSphereMotion.h>
#include <Physics2012/Dynamics/Motion/Rigid/ThinBoxMotion/hkpThinBoxMotion.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpCharacterMotion.h>

#include <Physics2012/Utilities/Destruction/BreakOffParts/hkpBreakOffPartsUtil.h>

#if defined(HK_PLATFORM_HAS_SPU)
#	include <Physics2012/Dynamics/Collide/Deprecated/Dispatch/hkpSpuCollisionCallbackUtil.h>
#endif

#if HK_POINTER_SIZE == 4 && !defined(HK_REAL_IS_DOUBLE)
	// make sure that the size of the entity is below 640. Else we are wasting lots of memory
	HK_COMPILE_TIME_ASSERT( sizeof (hkpEntity) <= 0x280 );
#endif

extern const hkClass hkpEntityExtendedListenersClass;

hkpEntity::hkpEntity( const hkpShape* shape )
:	hkpWorldObject( shape, BROAD_PHASE_ENTITY )
{
	// This is used by the simulation islands, and is not initialized correctly until the entity has been added to the world.
	m_storageIndex = hkObjectIndex(-1);
	m_simulationIsland = HK_NULL;
	m_extendedListeners = HK_NULL;
	m_limitContactImpulseUtilAndFlag = HK_NULL;
	m_uid = hkUint32(-1);
	m_solverData = 0;
	m_numShapeKeysInContactPointProperties = 0;
	m_breakableBody = HK_NULL;
	m_damageMultiplier = 1.0f;
	m_npData = 0xffffffff;
}

hkpEntity::hkpEntity( class hkFinishLoadedObjectFlag flag ) :
	hkpWorldObject( flag ),
	m_material( flag ),
	m_spuCollisionCallback( flag ),
	m_motion( flag ),
	m_localFrame( flag )
{
	if( flag.m_finishing )
	{
		void* motion = &m_motion;
		switch( m_motion.getType() )
		{
			case hkpMotion::MOTION_SPHERE_INERTIA:
				new (motion) hkpSphereMotion( flag );
				break;
			case hkpMotion::MOTION_BOX_INERTIA:
				new (motion) hkpBoxMotion( flag );
				break;
			case hkpMotion::MOTION_KEYFRAMED:
				new (motion) hkpKeyframedRigidMotion( flag );
				break;
			case hkpMotion::MOTION_FIXED:
				new (motion) hkpFixedRigidMotion( flag );
				break;
			case hkpMotion::MOTION_THIN_BOX_INERTIA:
				new (motion) hkpThinBoxMotion( flag );
				break;
			case hkpMotion::MOTION_CHARACTER:
				new (motion) hkpCharacterMotion( flag );
				break;
			default:
				HK_ASSERT3(0x5ba44035,false,"Invalid motion type " << m_motion.getType() );
		}
	}
}


void hkpEntity::setCachedShapeData(const hkpWorld* world, const hkpShape* shape)
{
	// Prepare data for child AABB caching
	{
		// Free old cache (if available).
		if ( m_collidable.m_boundingVolumeData.hasAllocations() )
		{
			m_collidable.m_boundingVolumeData.deallocate();
		}

		while (1)
		{
			if ( !world ) { break; }
			if ( !shape ) { break; }

			// hkpStaticCompoundShape and hkpBvCompressedMeshShape do not need midphase AABB caching, 
			// since they already store the AABBs internally
			if( shape->getType() == hkcdShapeType::STATIC_COMPOUND ||
				shape->getType() == hkcdShapeType::BV_COMPRESSED_MESH )
			{
				break;
			}

			bool hasBvTree = false;
			const hkpShapeContainer* container;
			if      ( world->m_collisionDispatcher->hasAlternateType(shape->getType(), hkcdShapeType::COLLECTION) ) { container = reinterpret_cast<const hkpShapeCollection*>(shape)->getContainer(); }
			else if ( world->m_collisionDispatcher->hasAlternateType(shape->getType(), hkcdShapeType::BV_TREE   ) ) { container = reinterpret_cast<const hkpBvTreeShape*>    (shape)->getContainer(); hasBvTree = true; }
			else { break; }

			// Optimization: we don't cache AABBs for static trees as these trees might be quite large, causing huge memory allocations.
			if ( isFixedOrKeyframed() && hasBvTree )
			{
				break;
			}

			// Get total number of child shapes, including any currently disabled ones which may get re-enabled later
			
			int numChildShapes;
			if ( shape->getType() == hkcdShapeType::LIST )
			{
				const hkpListShape* listShape = reinterpret_cast<const hkpListShape*> (shape);
				numChildShapes = listShape->m_childInfo.getSize();
			}
			else
			{
				numChildShapes = container->getNumChildShapes();
			}

			// This spu-code-block is duplicating the functionality in hkpListShape(or other collections shape)::calcSizeForSpu
#if defined(HK_PLATFORM_HAS_SPU)
			if ( numChildShapes >= hkpListShape::MAX_CHILDREN_FOR_SPU_MIDPHASE )
			{
				m_collidable.m_forceCollideOntoPpu |= hkpCollidable::FORCE_PPU_SHAPE_REQUEST;
			}
#endif

			m_collidable.m_boundingVolumeData.allocate(numChildShapes);
			m_collidable.m_boundingVolumeData.invalidate();

			if ( isFixed() )
			{
				updateCachedAabb();
			}

			break;
		}
	}

#if defined(HK_PLATFORM_HAS_SPU)
	if( shape && world )
	{
		// Check if the shape is compatible with chosen collision detection ELF
		if( world->m_useCompoundSpuElf )
		{
			if( shape->getType() == hkcdShapeType::EXTENDED_MESH || shape->getType() == hkcdShapeType::COMPRESSED_MESH )
			{
				HK_WARN(0x62518f78, "hkpWorld::m_useCompoundSpuElf must be disabled to use hkpExtendedMeshShape or hkpCompressedMeshShape on SPU. Forcing shape to PPU instead." );
				m_collidable.m_forceCollideOntoPpu |= hkpCollidable::FORCE_PPU_SHAPE_REQUEST;
			}
		}
		else
		{
			if( shape->getType() == hkcdShapeType::STATIC_COMPOUND || shape->getType() == hkcdShapeType::BV_COMPRESSED_MESH )
			{
				HK_WARN(0xf7a7f19, "hkpWorld::m_useCompoundSpuElf must be enabled to use hkpBvCompressedMeshShape or hkpStaticCompoundShape on SPU. Forcing shape to PPU instead." );
				m_collidable.m_forceCollideOntoPpu |= hkpCollidable::FORCE_PPU_SHAPE_REQUEST;
			}
		}
	}
#endif
}


void hkpEntity::updateCachedAabb()
{
	if ( m_world && m_collidable.m_shape )
	{
		HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );

		hkpEntity* thisObj = this;
		hkpEntityAabbUtil::entityBatchRecalcAabb(m_world->m_collisionInput, &thisObj, 1);
	}
}


#if defined(HK_PLATFORM_HAS_SPU)

void hkpEntity::setSpuCollisionCallbackUtil(hkSpuCollisionCallbackUtil* util, SpuCollisionCallbackEventFilter eventFilter, hkUint8 userFilter)
{
	util->addReference();

	// Important Note: we are only allowed to remove the reference after adding a new reference as the passed-in
	// utility might be the same as the previously set utility!
	if ( m_spuCollisionCallback.m_util )
	{
		m_spuCollisionCallback.m_util->removeReference();
	}

	m_spuCollisionCallback.m_util			= util;
	m_spuCollisionCallback.m_capacity		= hkUint16(util->m_capacity);
	m_spuCollisionCallback.m_eventFilter	= hkUint8(eventFilter);
	m_spuCollisionCallback.m_userFilter		= userFilter;
}

#endif

hkpEntity::~hkpEntity()
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	{
		hkpBreakOffPartsUtil::LimitContactImpulseUtil* util = hkpBreakOffPartsUtil::getLimitContactImpulseUtilPtr( this );
		if (util)
		{
			removeContactListener( util );
			delete util;
			m_limitContactImpulseUtilAndFlag = HK_NULL;
		}
	}

	HK_ASSERT(0xad000212, m_actions.isEmpty());
	HK_ASSERT(0xad000213, m_constraintsMaster.isEmpty());
	HK_ASSERT(0xad000214, m_constraintsSlave.isEmpty());

	hkpEntityCallbackUtil::fireEntityDeleted( this );

	// getCollidable()->getShape()->removeReference(); // CK: now done in the hkpWorldObject dtor

#if defined(HK_PLATFORM_HAS_SPU)
	if ( m_spuCollisionCallback.m_util )
	{
		m_spuCollisionCallback.m_util->removeReference();
	}
#endif
	delete m_extendedListeners;
	HK_ASSERT2(0x140a19b8,  getWorld() == HK_NULL, "removeReference() or destructor called while hkpEntity is still in simulation" );
}



void hkpEntity::addEntityListener( hkpEntityListener* el )
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	if ( !m_extendedListeners )
	{
		m_extendedListeners = new ExtendedListeners;
	}
	hkSmallArray<hkpEntityListener*>& arr = m_extendedListeners->m_entityListeners;

	HK_ASSERT2(0x63a5ad5d, arr.indexOf( el ) < 0, "You tried to add an entity listener twice" );
	int emptyIndex = arr.indexOf( HK_NULL );
	if ( emptyIndex >= 0)
	{
		arr[emptyIndex] = el;
	}
	else
	{
		arr.pushBack( el );
	}
}

void hkpEntity::removeEntityListener( hkpEntityListener* el)
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	if ( !m_extendedListeners )
	{
		m_extendedListeners = new ExtendedListeners;
	}
	hkSmallArray<hkpEntityListener*>& arr = m_extendedListeners->m_entityListeners;
	int i = arr.indexOf( el );
	HK_ASSERT2(0x79e1d7d7, i >= 0, "You tried to remove an entity listener, which was never added" );
	arr[i] = HK_NULL;
}

void hkpEntity::addEntityActivationListener( hkpEntityActivationListener* el)
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	if ( !m_extendedListeners )
	{
		m_extendedListeners = new ExtendedListeners;
	}
	hkSmallArray<hkpEntityActivationListener*>& arr = m_extendedListeners->m_activationListeners;

	HK_ASSERT2(0x6515b54d, arr.indexOf( el ) < 0, "You tried to add an entity activation listener twice" );

	int emptyIndex = arr.indexOf( HK_NULL );
	if ( emptyIndex >= 0)
	{
		arr[emptyIndex] = el;
	}
	else
	{
		arr.pushBack( el );
	}
}

void hkpEntity::removeEntityActivationListener( hkpEntityActivationListener* el)
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	HK_ASSERT2(0x271bc75a, m_extendedListeners, "You tried to remove an entity activation listener, which was never added" );

	hkSmallArray<hkpEntityActivationListener*>& arr = m_extendedListeners->m_activationListeners;
	int i = arr.indexOf( el );
	HK_ASSERT2(0x271bc759, i >= 0, "You tried to remove an entity activation listener, which was never added" );
	arr[i] = HK_NULL;
}


void hkpEntity::addContactListener( hkpContactListener* cl )
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	HK_ASSERT2(0x6515b54e, m_contactListeners.indexOf( cl ) < 0, "You tried to add a contact listener twice" );

	int emptyIndex = m_contactListeners.indexOf( HK_NULL );
	if ( emptyIndex >= 0)
	{
		m_contactListeners[emptyIndex] = cl;
	}
	else
	{
		m_contactListeners.pushBack(cl);
	}
}



void hkpEntity::removeContactListener( hkpContactListener* cl)
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	int i = m_contactListeners.indexOf( cl );
	HK_ASSERT2(0x403757e4, i >= 0, "You tried to remove a contact listener, which was never added" );

	m_contactListeners[i] = HK_NULL;
}



hkpDynamicsContactMgr* hkpEntity::findContactMgrTo(const hkpEntity* entity) const
{
	const hkArray<struct hkpLinkedCollidable::CollisionEntry>& entries = this->getLinkedCollidable()->getCollisionEntriesNonDeterministic();
	int numCollisionEntries = entries.getSize();
	{
		for (int i = 0; i < numCollisionEntries; i++)
		{
			const hkpLinkedCollidable::CollisionEntry& collisionEntry = entries[i];
			const hkpCollidable* collisionPartner = collisionEntry.m_partner;
			if ( collisionPartner == entity->getCollidable() )
			{
				return (hkpDynamicsContactMgr*)collisionEntry.m_agentEntry->m_contactMgr;
			}
		}
	}

	return HK_NULL;
}


static HK_FORCE_INLINE hkBool hkpEntity_isActive(const hkpEntity* entity)
{
	if ( entity->getSimulationIsland() == HK_NULL )
	{
		return false;
	}
	else
	{
		return entity->getSimulationIsland()->m_activeMark;
	}
}

hkBool hkpEntity::isActive() const
{
	return hkpEntity_isActive( this );
}

void hkpEntity::activate()
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );

	hkCheckDeterminismUtil::checkMt(0xf0000010, m_motion.getNumInactiveFrames(0));
	hkCheckDeterminismUtil::checkMt(0xf0000011, m_motion.getNumInactiveFrames(1));

	hkCheckDeterminismUtil::checkMt(0xf0000012, m_motion.m_deactivationNumInactiveFrames[0]);
	hkCheckDeterminismUtil::checkMt(0xf0000013, m_motion.m_deactivationNumInactiveFrames[1]);

	if (!hkpEntity_isActive(this) && !isFixed() && m_world)
	{
		// reset counters
		m_motion.m_deactivationNumInactiveFrames[0] = 0;
		m_motion.m_deactivationNumInactiveFrames[1] = 0;
		hkpWorldOperationUtil::markIslandActive(m_world, m_simulationIsland);

		// synchronize flags
		hkUint8* deactFlags = m_world->m_dynamicsStepInfo.m_solverInfo.m_deactivationNumInactiveFramesSelectFlag;
		m_motion.setWorldSelectFlagsNeg(deactFlags[0], deactFlags[1], m_world->m_dynamicsStepInfo.m_solverInfo.m_deactivationIntegrateCounter);
	}
}

void hkpEntity::requestDeactivation()
{
	HK_ASSERT2(0xf0ff0091, m_world, "hkpEntity::requestDeactivation() called on hkpEntity which has not been added to an hkpWorld");
	HK_ASSERT2(0xf0ff0092, !isFixed() || !hkpEntity_isActive(this), "hkpEntity::requestDeactivation() called on a fixed hkpEntity");
	HK_ASSERT2(0xf0ff0093, m_motion.isDeactivationEnabled(), "hkpEntity::requestDeactivation() called on an entity with deactivation disabled");

	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_RW, this, HK_ACCESS_RW );

	if ( hkpEntity_isActive(this) && m_motion.isDeactivationEnabled() )
	{
		// Make this entity deactivate the next time it is checked
		m_motion.requestDeactivation();
	}
}

void hkpEntity::deactivate()
{
	HK_WARN_ONCE(0xf0ff0099, "hkpEntity::deactivate() is deprecated. Please use hkpEntity::requestDeactivation() instead.");
	HK_ASSERT2(0xf0ff0091, m_world, "hkpEntity::deactivate() called for hkpEntity which has not been added to an hkpWorld");
	HK_ASSERT2(0xf0ff0092, !isFixed() || !hkpEntity_isActive(this), "hkpEntity::deactivate() about to be executed for a fixed body");

	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_RW, this, HK_ACCESS_RW );

	if (hkpEntity_isActive(this))
	{
		if ( m_simulationIsland->m_isSparse || m_simulationIsland->m_splitCheckRequested )
		{
			HK_ASSERT2( 0xf02343de, !m_world->areCriticalOperationsLocked(), "You cannot call this function from a callback. Call deactivateAsCriticalOperation instead");

			int old = m_world->m_minDesiredIslandSize;

			// force splitting this island: see details in hkpWorldOperationUtil::splitSimulationIsland
			m_world->m_minDesiredIslandSize = 0;
			m_simulationIsland->m_isSparse = false;
			m_simulationIsland->m_splitCheckRequested = true;
			hkpWorldOperationUtil::splitSimulationIsland( m_world, m_simulationIsland );
			m_world->m_minDesiredIslandSize = old;
		}
			// warning: m_simulationIsland might be changed here, we have to use the member variable

			// make sure we will get the deactivation. Check hkRigidMotionUtilCanDeactivateFinal for details
		for (int i = 0; i < m_simulationIsland->m_entities.getSize(); i++)
		{
			hkpEntity* other = m_simulationIsland->m_entities[i];
			other->m_motion.m_deactivationRefPosition[0].setW(hkSimdReal_Max); 
			other->m_motion.m_deactivationRefPosition[1].setW(hkSimdReal_Max); 
		}

		hkpWorldOperationUtil::markIslandInactive(m_world, m_simulationIsland);
	}
}

void hkpEntity::activateAsCriticalOperation()
{
	if (m_world && m_world->areCriticalOperationsLocked())
	{
		hkWorldOperation::ActivateEntity op;
		op.m_entity = this;
		m_world->queueOperation( op );
		return;
	}
	activate();
}

void hkpEntity::requestDeactivationAsCriticalOperation()
{
	if (m_world && m_world->areCriticalOperationsLocked())
	{
		hkWorldOperation::RequestDeactivateEntity op;
		op.m_entity = this;
		m_world->queueOperation( op );
		return;
	}

	requestDeactivation();
}

void hkpEntity::deactivateAsCriticalOperation()
{
	if (m_world && m_world->areCriticalOperationsLocked())
	{
		hkWorldOperation::DeactivateEntity op;
		op.m_entity = this;
		m_world->queueOperation( op );
		return;
	}

	deactivate();
}

void hkpEntity::deallocateInternalArrays()
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );

	// Need to deallocate any arrays in the entity that are 0 size
	// else warn user that they should call the in place destructor.


	// get rid of breakoff parts util
	{
		hkpBreakOffPartsUtil::LimitContactImpulseUtil* util = hkpBreakOffPartsUtil::getLimitContactImpulseUtilPtr( this );
		if (util)
		{
			removeContactListener( util );
			delete util;
			m_limitContactImpulseUtilAndFlag = HK_NULL;
		}
	}


	// If this entity was loaded by a pack file
	// and the motion pointer has been allocated separately by the user
	// then warn the user
	if (m_motion.m_memSizeAndFlags != 0)
	{
		HK_WARN(0x234f224a, "Entity at address " << this << " was loaded from a packfile but has a user allocated hkMotionState.\nPlease call in-place destructor to deallocate.\n");
	}


	// Linked Collidables collision entries
	hkArray<hkpLinkedCollidable::CollisionEntry>& collisionEntries = getLinkedCollidable()->getCollisionEntriesNonDeterministic();
	if (collisionEntries.getSize() == 0)
	{
		collisionEntries.clearAndDeallocate();
	}
	else
	{
		HK_ASSERT3(0x234f223a, 0, "Entity at address " << this << " has linked collidables with non-zero collision entries array.\nPlease call in-place destructor to deallocate.\n");
	}

	// Constraints Master
	if (m_constraintsMaster.getSize() == 0)
	{
		m_constraintsMaster.clearAndDeallocate();
	}
	else
	{
		HK_WARN(0x234f2249, "Entity at address " << this << " has non-zero m_constraintsMaster array.\nPlease call in-place destructor to deallocate.\n");
	}

	// Constraints Slave
	if (m_constraintsSlave.getSize() == 0)
	{
		m_constraintsSlave.clearAndDeallocate();
	}
	else
	{
		HK_WARN(0x234f2248, "Entity at address " << this << " has non-zero m_constraintsSlave array.\nPlease call in-place destructor to deallocate.\n");
	}

	// Constraints Runtime
	if (m_constraintRuntime.getSize() == 0)
	{
		m_constraintRuntime.clearAndDeallocate();
	}
	else
	{
		HK_WARN(0x234f2247, "Entity at address " << this << " has non-zero m_constraintRuntime array.\nPlease call in-place destructor to deallocate.\n");
	}



	// Collision Listeners
	{
		hkBool hasCollisionListeners = false;
		for (int i = 0; i < m_contactListeners.getSize(); i++)
		{
			if (m_contactListeners[i] != HK_NULL)
			{
				hasCollisionListeners = true;
				break;
			}
		}

		if (!hasCollisionListeners)
		{
			m_contactListeners.clearAndDeallocate();
		}
		else
		{
			HK_WARN(0x234f2246, "Entity at address " << this << " has non-zero m_collisionListeners array.\nPlease call in-place destructor to deallocate.\n");
		}
	}

	// Activation Listeners
	hkBool hasActivationListeners = false;
	{
		if ( m_extendedListeners )
		{
			hkSmallArray<hkpEntityActivationListener*>& arr = m_extendedListeners->m_activationListeners;
			for (int i = 0; i < arr.getSize(); i++)
			{
				if (arr[i] != HK_NULL)
				{
					hasActivationListeners = true;
					break;
				}
			}

			if (!hasActivationListeners)
			{
				arr.clearAndDeallocate();
			}
			else
			{
				HK_WARN(0x234f224b, "Entity at address " << this << " has non-zero m_activationListeners array.\nPlease call in-place destructor to deallocate.\n");
			}
		}
	}

	// Entity Listeners
	hkBool hasEntityListeners = false;
	{
		if ( m_extendedListeners )
		{
			hkSmallArray<hkpEntityListener*>& arr = m_extendedListeners->m_entityListeners;
			for (int i = 0; i < arr.getSize(); i++)
			{
				if (arr[i] != HK_NULL)
				{
					hasEntityListeners = true;
					break;
				}
			}

			if (!hasEntityListeners)
			{
				arr.clearAndDeallocate();
			}
			else
			{
				HK_WARN(0x234f224c, "Entity at address " << this << " has non-zero m_entityListeners array.\nPlease call in-place destructor to deallocate.\n");
			}
		}
	}
	if ( !hasActivationListeners && !hasEntityListeners)
	{
		delete m_extendedListeners;
		m_extendedListeners = HK_NULL;
	}

	// Actions
	// Rather than resize this array on removal of an action hkpWorld
	// just replaces the action with a null action. If all the actions
	// are null we can go ahead and safely deallocate
	if (m_actions.isEmpty())
	{
		m_actions.clearAndDeallocate();
	}
	else
	{
		HK_WARN(0x234f224d, "Entity at address " << this << " has non-zero m_actions array.\nPlease call in-place destructor to deallocate.\n");
	}
}

int hkpEntity::getNumConstraints() const
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, m_simulationIsland, HK_ACCESS_RO);
	return m_constraintsMaster.getSize() + m_constraintsSlave.getSize();
}

hkpConstraintInstance* hkpEntity::getConstraint( int i )
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, m_simulationIsland, HK_ACCESS_RW);
	HK_ASSERT2(0x52bba259, i >=0 && i < getNumConstraints(), "Constraint index out of range.");
	return i < m_constraintsMaster.getSize() ? m_constraintsMaster[i].m_constraint : m_constraintsSlave[i - m_constraintsMaster.getSize()];
}

const hkpConstraintInstance* hkpEntity::getConstraint( int i ) const
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, m_simulationIsland, HK_ACCESS_RO);
	HK_ASSERT2(0x2314c376, i >=0 && i < getNumConstraints(), "Constraint index out of range.");
	return i < m_constraintsMaster.getSize() ? m_constraintsMaster[i].m_constraint : m_constraintsSlave[i - m_constraintsMaster.getSize()];
}

void hkpEntity::getAllConstraints(hkArray<hkpConstraintInstance*>& constraints)
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, m_simulationIsland, HK_ACCESS_RO);

	int numConstraints = getNumConstraints();
	constraints.setSize(numConstraints);

	// Collect all constraints
	int ci = 0; // constraint index
	for (int i = 0; i < m_constraintsMaster.getSize(); i++)
	{
		constraints[ci++] = m_constraintsMaster[i].m_constraint;
	}
	for (int i = 0; i < m_constraintsSlave.getSize(); i++)
	{
		constraints[ci++] = m_constraintsSlave[i];
	}
}


/// Returns read only access to the internal constraint master list
const hkSmallArray<struct hkConstraintInternal>&  hkpEntity::getConstraintMastersImpl() const
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, m_simulationIsland, HK_ACCESS_RO);
	return m_constraintsMaster;
}

/// Returns read write access to the internal constraint master list
hkSmallArray<struct hkConstraintInternal>&  hkpEntity::getConstraintMastersRwImpl()
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, m_simulationIsland, HK_ACCESS_RW);
	return m_constraintsMaster;
}

const hkArray<class hkpConstraintInstance*>&  hkpEntity::getConstraintSlavesImpl() const
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, m_simulationIsland, HK_ACCESS_RO);
	if (isFixed())
	{
		HK_WARN_ONCE(0xad901260, "Calling hkpEntity::getConstraintSlavesImpl() for a fixed objects. Constraints' order is nondeterministic.");
	}
	return m_constraintsSlave;
}

#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
namespace {

class cmpLessConstraint
{
public:
	cmpLessConstraint(const hkpEntity* thisEntity) { m_thisEntity = thisEntity; }

	const hkpEntity* m_thisEntity;
	HK_FORCE_INLINE bool operator() (hkpConstraintInstance*  constraint0,  hkpConstraintInstance*  constraint1)
	{
		if (constraint0->m_uid < constraint1->m_uid) { return true; } 	 
		if (constraint0->m_uid > constraint1->m_uid) { return false; } 	 
		const hkpEntity* otherEntity0 = constraint0->getOtherEntity(m_thisEntity);
		const hkpEntity* otherEntity1 = constraint1->getOtherEntity(m_thisEntity);
		return otherEntity0->getUid() < otherEntity1->getUid(); 	 
	}
};

}
#endif

void hkpEntity::sortConstraintsSlavesDeterministically()
{
#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_RW, m_simulationIsland, HK_ACCESS_RW);
	if ( isFixed() )
	{
		// Disable determinism warning within this function
		HK_ON_DEBUG( const hkBool isWarningEnabled = hkError::getInstance().isEnabled(0xad901260);  )
		HK_ON_DEBUG( hkError::getInstance().setEnabled(0xad901260, false);  )

		// Sort constraint slaves
		hkpEntity* body = this;
		hkAlgorithm::quickSort((hkpConstraintInstance**)body->m_constraintsSlave.begin(), body->m_constraintsSlave.getSize(), cmpLessConstraint(this));

		// Now update the slave index in the hkpInternals,
		for (int si = 0; si < body->getConstraintSlaves().getSize(); si++)
		{
			hkpConstraintInstance* c = body->getConstraintSlaves()[si];
			hkpEntity* otherBody = c->getOtherEntity(body);
			for (int ii =0; ii < otherBody->getConstraintMasters().getSize(); ii++)
			{
				if (otherBody->getConstraintMasters()[ii].m_constraint == c)
				{
					otherBody->getConstraintMastersRw()[ii].m_slaveIndex = hkObjectIndex(si);
					break;
				}
			}
		}

		HK_ON_DEBUG( hkError::getInstance().setEnabled(0xad901260, isWarningEnabled ); )
	}
#endif
}

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
