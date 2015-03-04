/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Physics2012/Utilities/Dynamics/PhantomBatchMove/hkpPhantomBatchMoveUtil.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Physics2012/Dynamics/Phantom/hkpShapePhantom.h>
#include <Physics2012/Dynamics/Phantom/hkpAabbPhantom.h>

#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseDispatcher.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseHandlePair.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhase.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>


void HK_CALL hkpPhantomBatchMoveUtil::setPositionBatch( hkArray<hkpPhantom*>& phantoms, const hkArray<hkVector4>& positions, hkReal extraTolerance)
{
	int numPhantoms = phantoms.getSize();

	HK_ASSERT2( 0x0593bc95, numPhantoms > 0, "No phantoms!" );
	HK_ASSERT2( 0x83726047, positions.getSize() >= numPhantoms, "More phantoms than positions!" );

	hkpPhantomBatchMoveUtil::setPositionBatch( phantoms.begin(), positions.begin(), numPhantoms, extraTolerance );

}

void HK_CALL hkpPhantomBatchMoveUtil::setPositionBatch( hkpPhantom** phantoms, const hkVector4* positions, int numPhantoms, hkReal extraTolerance)
{
	hkpWorld* world = phantoms[0]->getWorld();
	HK_ASSERT2(0x3b6457e2, world, "All phantoms must be in the world");

	// Assert that the world is not locked
	HK_ASSERT2(0x7395bc06, !world->areCriticalOperationsLockedForPhantoms(), "Can't queue  hkpPhantomUtil::setPositionBatch; aborting.");

	HK_ACCESS_CHECK_OBJECT( world, HK_ACCESS_RW );


	world->lockCriticalOperations();
	{
		hkLocalArray<hkAabb> aabbs(numPhantoms);
		aabbs.setSize(numPhantoms);

		hkLocalArray<hkpBroadPhaseHandle*> handles(numPhantoms);
		handles.setSize(numPhantoms);

		// Tolerance will only apply to shape phantoms (i.e. the tolerance around the shape)
		const hkReal tolerance = 0.5f * world->getCollisionInput()->getTolerance() + extraTolerance;

		for(int i = 0; i < numPhantoms; i++)
		{
			hkAabb& aabb = aabbs[i];
			hkpPhantom* phantom = phantoms[i];
			const hkVector4& position = positions[i];

			HK_ASSERT2(0xa6b7e302, phantom->getWorld() == world, "All phantoms in setPositionBatch must be in the same world");		

			handles[i] = phantom->getCollidableRw()->getBroadPhaseHandle();

			const hkpShape* shape = phantom->getCollidable()->getShape();

			if( shape ) 
			{			
				// Shape phantoms ( Simple or Caching )

				hkpShapePhantom* shapePhantom  = static_cast<hkpShapePhantom*>(phantom);
				HK_ACCESS_CHECK_WITH_PARENT( world, HK_ACCESS_RW, shapePhantom, HK_ACCESS_RW );
				hkTransform& transform = shapePhantom->getMotionState()->getTransform();
				transform.setTranslation(position);
				shape->getAabb( transform, tolerance , aabb );
			}
			else
			{
				// AABB phantoms

				hkAabb oldAabb;
				hkpAabbPhantom* aabbPhantom = static_cast<hkpAabbPhantom*> (phantom);
				oldAabb = aabbPhantom->getAabb();
				hkVector4 midpoint; midpoint.setInterpolate(oldAabb.m_min, oldAabb.m_max, hkSimdReal_Inv2);
				hkVector4 offset; offset.setSub(position, midpoint);
				aabb = oldAabb;
				aabb.m_max.add(offset);
				aabb.m_min.add(offset);

				aabbPhantom->m_aabb = aabb;
			}
		}

		// Perform the actual operation

		hkLocalArray<hkpBroadPhaseHandlePair> newPairs( world->m_broadPhaseUpdateSize );
		hkLocalArray<hkpBroadPhaseHandlePair> delPairs( world->m_broadPhaseUpdateSize );

		world->getBroadPhase()->lock();
		world->getBroadPhase()->updateAabbs( handles.begin(), aabbs.begin(), numPhantoms, newPairs, delPairs );

		// check for changes
		if ( newPairs.getSize() != 0 || delPairs.getSize() != 0)
		{
			hkpTypedBroadPhaseDispatcher::removeDuplicates( newPairs, delPairs );

			world->m_broadPhaseDispatcher->removePairs(static_cast<hkpTypedBroadPhaseHandlePair*>(delPairs.begin()), delPairs.getSize());
			world->m_broadPhaseDispatcher->addPairs( static_cast<hkpTypedBroadPhaseHandlePair*>(newPairs.begin()), newPairs.getSize(),  world->getCollisionFilter() );

			for( int i = 0; i < numPhantoms; i++ )
			{
				phantoms[i]->removeNullOverlapListeners();
			}			
		}

		world->getBroadPhase()->unlock();

		for( int i = 0; i < numPhantoms; i++ )
		{
			phantoms[i]->setBoundingVolumeData(aabbs[i]);
		}
	}
	world->unlockAndAttemptToExecutePendingOperations();	
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
