/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>

#include <Physics2012/Dynamics/World/Util/hkpWorldOperationQueue.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>



#if defined (HK_PLATFORM_HAS_SPU)
HK_COMPILE_TIME_ASSERT( (HK_OFFSET_OF( hkpWorldObject, m_collidable )& 0xf) == 0 );
#endif

hkpWorldObject::hkpWorldObject( class hkFinishLoadedObjectFlag flag )
	:	hkReferencedObject(flag),
		m_collidable(flag),
		m_name(flag),
		m_properties(flag)
{
	if( flag.m_finishing )
	{
		m_collidable.setOwner(this);
	}
}


hkpWorldObject::hkpWorldObject( const hkpShape* shape, BroadPhaseType type )
:	m_world(HK_NULL),
	m_userData(HK_NULL),
	m_collidable( shape, (hkMotionState*)HK_NULL, type )
{
	m_collidable.setOwner( this );

	if (shape)
	{
		shape->addReference();
	}
}

hkWorldOperation::Result hkpWorldObject::setShape(const hkpShape* shape)
{
	HK_ASSERT2(0xad45fe22, false, "This function must be overridden in derived classes, if it's to be used.");

//	if (m_world && m_world->areCriticalOperationsLocked())
//	{
//		hkWorldOperation::SetWorldObjectShape op;
//		op.m_worldObject = this;
//		op.m_shape = shape;
//
//		m_world->queueOperation(op);
//		return hkWorldOperation::POSTPONED;
//	}
//
//	// Handle reference counting here.
//	if (getCollidable()->getShape())
//	{
//		getCollidable()->getShape()->removeReference();
//	}
//	getCollidable()->setShape(shape);
//	shape->addReference();

	return hkWorldOperation::DONE;
}

//
//	Updates the shape of an hkpEntity or an hkpPhantom.

hkWorldOperation::Result hkpWorldObject::updateShape(hkpShapeModifier* shapeModifier)
{
	HK_ASSERT2(0xad45fe22, false, "This function must be overridden in derived classes, if it's to be used.");

	return hkWorldOperation::DONE;
}

void hkpWorldObject::addProperty( hkUint32 key, hkpPropertyValue value)
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );

	for (int i = 0; i < m_properties.getSize(); ++i)
	{
		if (m_properties[i].m_key == key)
		{
			HK_ASSERT2(0x26ca3b52, 0, "You are trying to add a property to a world object, where a property of that type already exists");
			return;
		}
	}
	hkpProperty& p = m_properties.expandOne();
	p.m_value = value;
	p.m_key = key;
}


void hkpWorldObject::setProperty( hkUint32 key, hkpPropertyValue value)
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );

	for (int i = 0; i < m_properties.getSize(); ++i)
	{
		if (m_properties[i].m_key == key)
		{
			m_properties[i].m_value = value;
			return;
		}
	}
	hkpProperty& p = m_properties.expandOne();
	p.m_value = value;
	p.m_key = key;
}


hkpPropertyValue hkpWorldObject::removeProperty(hkUint32 key)
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );

	for (int i = 0; i < m_properties.getSize(); ++i)
	{
		if (m_properties[i].m_key == key)
		{
			hkpProperty found = m_properties[i];
			m_properties.removeAtAndCopy(i);
			return found.m_value;
		}
	}

	HK_ASSERT2(0x62ee448b, 0, "You are trying to remove a property from a world object, where a property of that type does not exist");

	hkpPropertyValue returnValue;
	returnValue.m_data = 0;

	return returnValue;
}

void hkpWorldObject::editProperty( hkUint32 key, hkpPropertyValue value, MtChecks mtCheck )
{
#ifdef HK_DEBUG_MULTI_THREADING
	if ( mtCheck == MULTI_THREADING_CHECKS_ENABLE && m_world && m_world->m_propertyMasterLock )
	{
		HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RO );
	}
#endif

	for (int i = 0; i < m_properties.getSize(); ++i)
	{
		if (m_properties[i].m_key == key)
		{
			m_properties[i].m_value = value;
			return;
		}
	}

	HK_ASSERT2(0x6c6f226b, 0, "You are trying to update a property of a world object, where a property of that type does not exist");
}

void hkpWorldObject::lockProperty( hkUint32 key )
{
	if ( !m_world || !m_world->m_propertyMasterLock )
	{
		return;
	}
	m_world->m_propertyMasterLock->enter();
}

/// unlocks a given locked property
void hkpWorldObject::unlockProperty( hkUint32 key )
{
	if ( !m_world || !m_world->m_propertyMasterLock )
	{
		return;
	}
	m_world->m_propertyMasterLock->leave();
}

void hkpWorldObject::markForWriteImpl( )
{
#ifdef HK_DEBUG_MULTI_THREADING
	if ( m_world )
	{
		HK_ASSERT2( 0xf0213de, !m_world->m_multiThreadCheck.isMarkedForReadRecursive(), "You cannot mark an entity read write, if it is already marked as read only by the hkpWorld::markForRead(RECURSIVE)" );
	}
	getMultiThreadCheck().markForWrite();
#endif
}

void hkpWorldObject::checkReadWrite()
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW);
}

void hkpWorldObject::checkReadOnly() const
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RO);
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
