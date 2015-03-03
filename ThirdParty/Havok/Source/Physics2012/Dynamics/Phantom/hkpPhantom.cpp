/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>


#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

#include <Physics2012/Collide/BroadPhase/hkpBroadPhase.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseDispatcher.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseHandlePair.h>

#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>

#include <Physics2012/Dynamics/Phantom/hkpPhantom.h>
#include <Physics2012/Dynamics/Phantom/hkpPhantomListener.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/Simulation/hkpSimulation.h>

// TODO . this include is temporary: until hkPhantoms;:updateBroadPhase is moved to hkpWorldOperationUtil
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationQueue.h>

#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>

void hkpPhantom::removeNullOverlapListeners()
{
	for (int i = m_overlapListeners.getSize() - 1; i >= 0; i-- )
	{
		if ( m_overlapListeners[i] == HK_NULL )
		{
			m_overlapListeners.removeAtAndCopy(i);
		}
	}
}

void hkpPhantom::removeNullPhantomListeners()
{
	for (int i = m_phantomListeners.getSize() - 1; i >= 0; i-- )
	{
		if ( m_phantomListeners[i] == HK_NULL )
		{
			m_phantomListeners.removeAtAndCopy(i);
		}
	}
}

void hkpPhantom::firePhantomDeleted( )
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_RO, this, HK_ACCESS_RO );

	for ( int i = m_phantomListeners.getSize()-1; i >= 0; i-- )
	{
		if (m_phantomListeners[i] != HK_NULL)
		{
			m_phantomListeners[i]->phantomDeletedCallback( this );
		}
	}
	//cleanupNullPointers<hkpPhantomListener>( m_phantomListeners ); // not necessary, as object is deleted
}


void hkpPhantom::firePhantomRemoved( )
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_RO, this, HK_ACCESS_RO );

	for ( int i = m_phantomListeners.getSize()-1; i >= 0; i-- )
	{
		if (m_phantomListeners[i] != HK_NULL)
		{
			m_phantomListeners[i]->phantomRemovedCallback( this );
		}
	}
	removeNullPhantomListeners();
}


void hkpPhantom::firePhantomAdded( )
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_RO, this, HK_ACCESS_RO );

	for ( int i = m_phantomListeners.getSize()-1; i >= 0; i-- )
	{
		if (m_phantomListeners[i] != HK_NULL)
		{
			m_phantomListeners[i]->phantomAddedCallback( this );
		}
	}
	removeNullPhantomListeners();
}

void hkpPhantom::firePhantomShapeSet( )
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_RO, this, HK_ACCESS_RO );

	for ( int i = m_phantomListeners.getSize()-1; i >= 0; i-- )
	{
		if (m_phantomListeners[i] != HK_NULL)
		{
			m_phantomListeners[i]->phantomShapeSetCallback( this );
		}
	}
	removeNullPhantomListeners();
}

void hkpPhantom::updateBroadPhase( const hkAabb& aabb )
{
	if ( m_world != HK_NULL )
	{
		// Check if the world is locked, if so postpone the operation
		if (m_world->areCriticalOperationsLockedForPhantoms())
		{
			hkWorldOperation::UpdatePhantomBP op;
			op.m_phantom = this;
			op.m_aabb = const_cast<hkAabb*>(&aabb);
			m_world->queueOperation(op);
			return;
		}

		// Perform the actual operation
		HK_ACCESS_CHECK_OBJECT( m_world, HK_ACCESS_RW );

		m_world->lockCriticalOperations();

		hkLocalArray<hkpBroadPhaseHandlePair> newPairs( m_world->m_broadPhaseUpdateSize );
		hkLocalArray<hkpBroadPhaseHandlePair> delPairs( m_world->m_broadPhaseUpdateSize );

		hkpBroadPhaseHandle* thisObj = m_collidable.getBroadPhaseHandle();

		m_world->getBroadPhase()->lock();

		m_world->getBroadPhase()->updateAabbs( &thisObj, &aabb, 1, newPairs, delPairs );

		// check for changes
		if ( newPairs.getSize() != 0 || delPairs.getSize() != 0)
		{
			hkpTypedBroadPhaseDispatcher::removeDuplicates( newPairs, delPairs );

			m_world->m_broadPhaseDispatcher->removePairs(static_cast<hkpTypedBroadPhaseHandlePair*>(delPairs.begin()), delPairs.getSize());
			m_world->m_broadPhaseDispatcher->addPairs( static_cast<hkpTypedBroadPhaseHandlePair*>(newPairs.begin()), newPairs.getSize(),  m_world->getCollisionFilter() );

			removeNullOverlapListeners();
		}

		m_world->getBroadPhase()->unlock();

		setBoundingVolumeData(aabb);

		m_world->unlockAndAttemptToExecutePendingOperations();
	}
	else
	{
		//HK_WARN_ONCE(0x3a15c993,  "Updating the AABB of a phantom that has not been added to a hkpWorld");
	}
}

void hkpPhantom::setBoundingVolumeData(const hkAabb& aabb)
{
#ifdef HK_ARCH_ARM
	HK_ASSERT2(0x7f7a2b8a, (((hkUlong)&getCollidableRw()->m_boundingVolumeData) & 0x3) == 0, "Unaligned bounding volume data!");
#else
	HK_ASSERT2(0x7f7a2b8a, (((hkUlong)&getCollidableRw()->m_boundingVolumeData) & 0xF) == 0, "Unaligned bounding volume data!");
#endif
	hkAabbUint32& aabbUint32 = reinterpret_cast<hkAabbUint32&>(getCollidableRw()->m_boundingVolumeData);

	const hkpCollisionInput* collisionInput = m_world->getCollisionInput();

	hkAabbUtil::convertAabbToUint32(aabb, collisionInput->m_aabb32Info.m_bitOffsetLow, collisionInput->m_aabb32Info.m_bitOffsetHigh, collisionInput->m_aabb32Info.m_bitScale, aabbUint32 );
	hkAabbUtil::compressExpandedAabbUint32(aabbUint32, aabbUint32);
}

void hkpPhantom::addPhantomListener( hkpPhantomListener* el)
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	HK_ASSERT2(0x782c270a,  m_phantomListeners.indexOf( el ) < 0, "You cannot add a listener twice to a phantom" );
	m_phantomListeners.pushBack( el );
}

void hkpPhantom::removePhantomListener( hkpPhantomListener* el)
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	int i = m_phantomListeners.indexOf( el );
	HK_ASSERT2(0x5b2f9aa5,  i>=0, "Tried to remove a listener which was never added");
	m_phantomListeners.removeAtAndCopy( i );
}


void hkpPhantom::addPhantomOverlapListener( hkpPhantomOverlapListener* el)
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	HK_ASSERT2(0x5d027d23,  m_overlapListeners.indexOf( el ) < 0, "You cannot add a listener twice to a phantom" );
	m_overlapListeners.pushBack( el );
}

void hkpPhantom::removePhantomOverlapListener( hkpPhantomOverlapListener* el)
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	int i = m_overlapListeners.indexOf( el );
	HK_ASSERT2(0x5478016a,  i>=0, "Tried to remove a listener which was never added");
	m_overlapListeners.removeAtAndCopy( i );
}


hkpPhantom::~hkpPhantom()
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	firePhantomDeleted();
}

void hkpPhantom::deallocateInternalArrays()
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RW );

	// Need to deallocate any arrays in the phantom that are 0 size
	// else warn user that they should call the in place destructor

	// Overlap Listeners
	if (m_overlapListeners.getSize() == 0)
	{
		m_overlapListeners.clearAndDeallocate();
	}
	else
	{
		HK_WARN(0x234f224e, "Phantom at address " << this << " has non-zero m_overlapListeners array.\nPlease call in-place destructor to deallocate.\n");
	}

	// Phantom Listeners
	if (m_phantomListeners.getSize() == 0)
	{
		m_phantomListeners.clearAndDeallocate();
	}
	else
	{
		HK_WARN(0x234f224f, "Phantom at address " << this << " has non-zero m_phantomListeners array.\nPlease call in-place destructor to deallocate.\n");
	}
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
