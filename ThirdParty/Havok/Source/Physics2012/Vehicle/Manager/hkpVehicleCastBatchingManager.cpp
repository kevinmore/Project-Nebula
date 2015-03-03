/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>
#include <Physics2012/Vehicle/Manager/hkpVehicleCastBatchingManager.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Physics2012/Vehicle/WheelCollide/RayCast/hkpVehicleRayCastWheelCollide.h>

#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastOutput.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/hkpRayCastQueryJobs.h>
#include <Common/Base/Thread/Job/ThreadPool/hkJobThreadPool.h>

#include <Physics2012/Collide/Filter/Group/hkpGroupFilter.h>

void hkpVehicleCastBatchingManager::addVehicle( hkpVehicleInstance* vehicle )
{
	hkpVehicleManager::addVehicle( vehicle );
	m_totalNumWheels = m_totalNumWheels + vehicle->getNumWheels();
}


void hkpVehicleCastBatchingManager::removeVehicle( hkpVehicleInstance* vehicle )
{
	hkpVehicleManager::removeVehicle( vehicle );
	HK_ASSERT2( 0xa299bca0, m_totalNumWheels >= vehicle->getNumWheels(), "Removing a vehicle would result in a negative wheel total." );
	m_totalNumWheels = m_totalNumWheels - vehicle->getNumWheels();
}


void hkpVehicleCastBatchingManager::updateBeforeCollisionDetection( hkArray< hkpVehicleInstance* >& activeVehicles )
{
	const int numVehicles = activeVehicles.getSize();
	for ( int i = 0; i < numVehicles; ++i )
	{
		activeVehicles[i]->updateBeforeCollisionDetection();
	}
}


void hkpVehicleCastBatchingManager::stepVehiclesSynchronously( hkpWorld* world, const hkStepInfo& updatedStepInfo, hkJobThreadPool* threadPool, hkJobQueue* jobQueue, int numJobs, hkArray< hkpVehicleInstance* >& activeVehicles )
{
	updateBeforeCollisionDetection( activeVehicles );
	{
		const int bufferSize = getBufferSize( numJobs, activeVehicles );
		char *const buffer = hkAllocateStack<char>( bufferSize, "Wheel casts" );
		{
			{
				hkScopedPtr<hkSemaphoreBusyWait> semaphore;
				numJobs = buildAndAddCastJobs( world, getStandardFilterSize( world ), numJobs, jobQueue, semaphore, buffer, activeVehicles );

				if ( numJobs )
				{
					world->lockReadOnly();

					threadPool->processAllJobs( jobQueue );
					jobQueue->processAllJobs( false );

					// Wait for all the jobs we started to finish.
					threadPool->waitForCompletion();
					semaphore->acquire();

					world->unlockReadOnly();
				}
			}
		}
		stepVehiclesUsingCastResults( updatedStepInfo, numJobs, buffer, activeVehicles );
		hkDeallocateStack( buffer, bufferSize );
	}
}


hkInt32 hkpVehicleCastBatchingManager::getStandardFilterSize( const hkpWorld* world )
{
	HK_ASSERT2( 0x87ba3f1f, world->getCollisionFilter(), "World has no collision filter.");
	if ( world->getCollisionFilter()->m_type == hkpCollisionFilter::HK_FILTER_GROUP )
	{
		return sizeof( hkpGroupFilter );
	}
	else
	{
		return 0;
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
