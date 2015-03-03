/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Filter/List/hkpCollisionFilterList.h>

hkpCollisionFilterList::hkpCollisionFilterList( const hkArray<hkpCollisionFilter*>& collisionFilters )
{
	m_type = HK_FILTER_LIST;
	m_collisionFilters = collisionFilters;
	for (int i  = 0; i < m_collisionFilters.getSize(); ++i)
	{
		m_collisionFilters[i]->addReference();
	}
}

hkpCollisionFilterList::~hkpCollisionFilterList()
{
	for ( int i = 0; i < m_collisionFilters.getSize(); ++i )
	{
		m_collisionFilters[i]->removeReference();
	}
}



hkBool hkpCollisionFilterList::isCollisionEnabled( const hkpCollidable& a, const hkpCollidable& b ) const
{
	 // check filters
	for ( int i = m_collisionFilters.getSize()-1; i>=0; i-- )
	{
		if(! m_collisionFilters[i]->isCollisionEnabled( a, b ))
		{
			return false;
		}
	}
	return true;
}


hkBool hkpCollisionFilterList::isCollisionEnabled( const hkpCollisionInput& input, const hkpCdBody& a, const hkpCdBody& b, const hkpShapeContainer& bContainer, hkpShapeKey bKey  ) const
{
	 // check filters
	for ( int i = m_collisionFilters.getSize()-1; i>=0; i-- )
	{
		if(! m_collisionFilters[i]->isCollisionEnabled( input, a, b, bContainer, bKey ))
		{
			return false;
		}
	}
	return true;
}

hkBool hkpCollisionFilterList::isCollisionEnabled( const hkpShapeRayCastInput& aInput, const hkpShapeContainer& bContainer, hkpShapeKey bKey ) const
{
		 // check filters
	for ( int i = m_collisionFilters.getSize()-1; i>=0; i-- )
	{
		if(! m_collisionFilters[i]->isCollisionEnabled( aInput, bContainer, bKey ))
		{
			return false;
		}
	}
	return true;
}

hkBool hkpCollisionFilterList::isCollisionEnabled( const hkpCollisionInput& input, const hkpCdBody& collectionBodyA, const hkpCdBody& collectionBodyB, const HK_SHAPE_CONTAINER& containerShapeA, const HK_SHAPE_CONTAINER& containerShapeB, hkpShapeKey keyA, hkpShapeKey keyB ) const
{
	// check filters
	for ( int i = m_collisionFilters.getSize()-1; i>=0; i-- )
	{
		if(! m_collisionFilters[i]->isCollisionEnabled( input, collectionBodyA, collectionBodyB, containerShapeA, containerShapeB, keyA, keyB ))
		{
			return false;
		}
	}
	return true;
}


hkBool hkpCollisionFilterList::isCollisionEnabled( const hkpWorldRayCastInput& a, const hkpCollidable& collidableB ) const
{
		 // check filters
	for ( int i = m_collisionFilters.getSize()-1; i>=0; i-- )
	{
		if(! m_collisionFilters[i]->isCollisionEnabled( a, collidableB ))
		{
			return false;
		}
	}
	return true;
}


hkpCollisionFilterList::hkpCollisionFilterList()
{
}

void hkpCollisionFilterList::addCollisionFilter( hkpCollisionFilter* filter )
{
	filter->addReference();
	m_collisionFilters.pushBack( filter );
}

void hkpCollisionFilterList::removeCollisionFilter( hkpCollisionFilter* filter )
{
	int index = m_collisionFilters.indexOf(filter);
	HK_ASSERT2(0x509f9d0d,  index >= 0, "Collision filter was not found under existing filters" );
	m_collisionFilters.removeAt(  index );
	filter->removeReference();
}


const hkArray<hkpCollisionFilter*>& hkpCollisionFilterList::getCollisionFilters() const
{	
	return m_collisionFilters;	
}

void hkpCollisionFilterList::init( hkpWorld* world )
{
	for(int i = 0; i < m_collisionFilters.getSize(); i++)
	{
		m_collisionFilters[i]->init(world);
	}
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
