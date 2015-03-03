/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Physics2012/Dynamics/Entity/Util/hkpEntityCallbackUtil.h>
#include <Physics2012/Dynamics/Entity/hkpEntityListener.h>
#include <Physics2012/Dynamics/Entity/hkpEntityActivationListener.h>
#include <Physics2012/Dynamics/Collide/Deprecated/hkpCollisionListener.h>
#include <Physics2012/Dynamics/Action/hkpAction.h>

static inline void HK_CALL hkpEntityCallbackUtil_cleanupNullPointers( hkSmallArray<void*>& cleanupArray )
{
	for (int i = cleanupArray.getSize() - 1; i >= 0; i-- )
	{
		if ( cleanupArray[i] == HK_NULL )
		{
			cleanupArray.removeAtAndCopy(i);
		}
	}
}

void HK_CALL hkpEntityCallbackUtil::fireEntityAdded( hkpEntity* entity ) 
{
	if ( entity->m_extendedListeners )
	{
		hkSmallArray<hkpEntityListener*> &listen = entity->m_extendedListeners->m_entityListeners;
		for ( int i = listen.getSize()-1; i >= 0; i-- )
		{
			if (listen[i] != HK_NULL)
			{
				HK_TIME_CODE_BLOCK("entAddCb", entity);
				listen[i]->entityAddedCallback( entity );
			}
		}
	}
	{
		HK_ASSERT2( 0xf0356434, entity->getNumConstraints()==0, "Constraints are alreade attached to the entity before the entity is added to the world" );
		HK_ASSERT2( 0xad000240, entity->getNumActions()    ==0, "Actions are already attached to the entity before the entity is added to the world" );
	}
}

void HK_CALL hkpEntityCallbackUtil::fireEntityRemoved( hkpEntity* entity ) 
{
	if ( entity->m_extendedListeners )
	{
		hkSmallArray<hkpEntityListener*> &listen = entity->m_extendedListeners->m_entityListeners;
		for ( int i = listen.getSize()-1; i >= 0; i-- )
		{
			if (listen[i] != HK_NULL)
			{
				HK_TIME_CODE_BLOCK("entRemCb", entity);
				listen[i]->entityRemovedCallback( entity );
			}
		}
	}

	// master constraints
	{
		hkSmallArray<hkConstraintInternal>& constraints = entity->m_constraintsMaster;
		while( constraints.getSize() )
		{
			HK_TIME_CODE_BLOCK("entRemCb", entity);
			HK_ON_DEBUG( int oldsize = constraints.getSize() );
			constraints[0].m_constraint->entityRemovedCallback( entity );
			HK_ASSERT2( 0xf0403423, constraints.getSize() < oldsize, "You have to remove the constraint in the entityRemovedCallback" );
		}
	}

	// slave constraints
	{
		hkArray<hkpConstraintInstance*>& constraints = entity->m_constraintsSlave;

#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
		entity->sortConstraintsSlavesDeterministically();
#endif
		while( constraints.getSize() )
		{
			HK_TIME_CODE_BLOCK("entRemCb", entity);
			HK_ON_DEBUG( int oldsize = constraints.getSize() );
			constraints[0]->entityRemovedCallback( entity );
			HK_ASSERT2( 0xf0403423, constraints.getSize() < oldsize, "You have to remove the constraint in the entityRemovedCallback" );
		}
	}

	// actions
	{
		hkSmallArray<hkpAction*>& actions = entity->m_actions;
		while( actions.getSize() )
		{
			HK_TIME_CODE_BLOCK("entRemCb", entity);
			HK_ON_DEBUG( int oldsize = actions.getSize() );
			actions[0]->entityRemovedCallback( entity );
			HK_ASSERT2( 0xad78dd56, actions.getSize() < oldsize, "You have to remove the action in the entityRemovedCallback." );
		}
	}
}


void HK_CALL hkpEntityCallbackUtil::fireEntityShapeSet( hkpEntity* entity )
{
	if ( entity->m_extendedListeners )
	{
		hkSmallArray<hkpEntityListener*> &listen = entity->m_extendedListeners->m_entityListeners;
		for ( int i = listen.getSize()-1; i >= 0; i-- )
		{
			if (listen[i] != HK_NULL)
			{
				HK_TIME_CODE_BLOCK("setShapeCb", entity);
				listen[i]->entityShapeSetCallback( entity );
			}
		}
	}
}

void HK_CALL hkpEntityCallbackUtil::fireEntitySetMotionType( hkpEntity* entity )
{
	if ( entity->m_extendedListeners )
	{
		hkSmallArray<hkpEntityListener*> &listen = entity->m_extendedListeners->m_entityListeners;
		for ( int i = listen.getSize()-1; i >= 0; i-- )
		{
			if (listen[i] != HK_NULL)
			{
				HK_TIME_CODE_BLOCK("setMotionTypeCb", entity);
				listen[i]->entitySetMotionTypeCallback( entity );
			}
		}
	}
}

void HK_CALL hkpEntityCallbackUtil::fireEntityDeleted( hkpEntity* entity )
{
	if ( entity->m_extendedListeners )
	{
		hkSmallArray<hkpEntityListener*> &listen = entity->m_extendedListeners->m_entityListeners;
		for ( int i = listen.getSize()-1; i >= 0; i-- )
		{
			if (listen[i] != HK_NULL)
			{
				HK_TIME_CODE_BLOCK("entDelCb", entity);
				listen[i]->entityDeletedCallback( entity );
			}
		}
	}
}


void HK_CALL hkpEntityCallbackUtil::fireContactPointCallbackInternal( hkpEntity* entity, hkpContactPointEvent& event )
{
	hkSmallArray<hkpContactListener*>& listen = entity->m_contactListeners;
	for ( int i = listen.getSize()-1; i >= 0; i-- )
	{
		if (listen[i] != HK_NULL)
		{
			HK_TIME_CODE_BLOCK("cpCb", entity);
			listen[i]->contactPointCallback( event );
		}
	}
	hkSmallArray<void*>& cleanupArray = reinterpret_cast<hkSmallArray<void*>&>(listen);	
	hkpEntityCallbackUtil_cleanupNullPointers( cleanupArray );
}


void HK_CALL hkpEntityCallbackUtil::fireContactConstraintAddedCallback( hkpEntity* entity, hkpCollisionEvent& event )
{
	hkSmallArray<hkpContactListener*>& listen = entity->m_contactListeners;
	for ( int i = listen.getSize()-1; i >= 0; i-- )
	{
		if (listen[i] != HK_NULL)
		{
			HK_TIME_CODE_BLOCK("NewCollCb", entity);
			listen[i]->collisionAddedCallback( event );
		}
	}
	hkSmallArray<void*>& cleanupArray = reinterpret_cast<hkSmallArray<void*>&>(listen);	
	hkpEntityCallbackUtil_cleanupNullPointers( cleanupArray );
}

void HK_CALL hkpEntityCallbackUtil::fireContactConstraintRemovedCallback( hkpEntity* entity, hkpCollisionEvent& event )
{
	hkSmallArray<hkpContactListener*>& listen = entity->m_contactListeners;
	for ( int i = listen.getSize()-1; i >= 0; i-- )
	{
		if (listen[i] != HK_NULL)
		{
			HK_TIME_CODE_BLOCK("FinCollCb", entity);
			listen[i]->collisionRemovedCallback( event );
		}
	}
	hkSmallArray<void*>& cleanupArray = reinterpret_cast<hkSmallArray<void*>&>(listen);	
	hkpEntityCallbackUtil_cleanupNullPointers( cleanupArray );
}


// Deprecated.
void HK_CALL hkpEntityCallbackUtil::fireContactPointAddedInternal( hkpEntity* entity, hkpContactPointAddedEvent& event)
{
	hkSmallArray<hkpContactListener*>& listen = entity->m_contactListeners;
	event.m_callbackFiredFrom = entity;
	for ( int i = listen.getSize()-1; i >= 0; i-- )
	{
		if (listen[i] != HK_NULL)
		{
			HK_TIME_CODE_BLOCK("cpAddCb", entity);
			listen[i]->contactPointAddedCallback( event ); 
		}
	}
	hkSmallArray<void*>& cleanupArray = reinterpret_cast<hkSmallArray<void*>&>(listen);	
	hkpEntityCallbackUtil_cleanupNullPointers( cleanupArray );
}


// Deprecated.
void HK_CALL hkpEntityCallbackUtil::fireContactPointRemovedInternal( hkpEntity* entity, hkpContactPointRemovedEvent& event )
{
	hkSmallArray<hkpContactListener*>& listen = entity->m_contactListeners;
    event.m_callbackFiredFrom = entity;
	for ( int i = listen.getSize()-1; i >= 0; i-- )
	{
		if (listen[i] != HK_NULL)
		{
			HK_TIME_CODE_BLOCK("cpRemCb", entity);
			listen[i]->contactPointRemovedCallback( event );
		}
	}
	hkSmallArray<void*>& cleanupArray = reinterpret_cast<hkSmallArray<void*>&>(listen);
	hkpEntityCallbackUtil_cleanupNullPointers( cleanupArray );
}

// Deprecated.
void HK_CALL hkpEntityCallbackUtil::fireContactProcessInternal( hkpEntity* entity, hkpContactProcessEvent& event )
{
	hkSmallArray<hkpContactListener*>& listen = entity->m_contactListeners;
    event.m_callbackFiredFrom = entity;
	for ( int i = listen.getSize()-1; i >= 0; i-- )
	{
		if (listen[i] != HK_NULL)
		{
			HK_TIME_CODE_BLOCK("cpProCb", entity);
			listen[i]->contactProcessCallback( event );
		}
	}
	
	hkSmallArray<void*>& cleanupArray = reinterpret_cast<hkSmallArray<void*>&>(listen);
	hkpEntityCallbackUtil_cleanupNullPointers( cleanupArray );
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
