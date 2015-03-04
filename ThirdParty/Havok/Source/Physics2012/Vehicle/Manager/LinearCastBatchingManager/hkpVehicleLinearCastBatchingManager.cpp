/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>
#include <Physics2012/Vehicle/Manager/LinearCastBatchingManager/hkpVehicleLinearCastBatchingManager.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Physics2012/Vehicle/WheelCollide/LinearCast/hkpVehicleLinearCastWheelCollide.h>

#include <Physics2012/Collide/Query/Collector/PointCollector/hkpRootCdPoint.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobs.h>
#include <Common/Base/Thread/Job/ThreadPool/hkJobThreadPool.h>
#include <Physics2012/Vehicle/Manager/MultithreadedVehicle/hkpVehicleJobs.h>


void hkpVehicleLinearCastBatchingManager::getTotalNumCommands( hkArray< hkpVehicleInstance* >& activeVehicles, int& numCommands, int& numWheels ) const
{
	numCommands = 0;
	numWheels = 0;
	for ( int i = 0; i < activeVehicles.getSize(); ++i )
	{
		HK_ASSERT2( 0x244fea6b, activeVehicles[i]->m_wheelCollide->getType() == hkpVehicleWheelCollide::LINEAR_CAST_WHEEL_COLLIDE, "This manager can only handle vehicles with linear cast wheel collide components." );
		hkpVehicleLinearCastWheelCollide* wheelCollide = static_cast<hkpVehicleLinearCastWheelCollide*>( activeVehicles[i]->m_wheelCollide );
		numCommands += wheelCollide->getTotalNumCommands();
		numWheels += activeVehicles[i]->getNumWheels();
	}
}

int hkpVehicleLinearCastBatchingManager::getBufferSize( int numJobs, hkArray< hkpVehicleInstance* >& activeVehicles )
{
	int totalNumCollidables;
	int totalNumWheels;
	getTotalNumCommands( activeVehicles, totalNumCollidables, totalNumWheels );

	const int collidablesSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpCollidable) * totalNumWheels );
	const int commandsSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpPairLinearCastCommand) * totalNumCollidables );
	const int outputsSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpRootCdPoint) * totalNumCollidables );
	const int headersSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpCollisionQueryJobHeader) * numJobs );

	const int bufferSize = collidablesSize + commandsSize + outputsSize + headersSize;
	return bufferSize;
}


void hkpVehicleLinearCastBatchingManager::getLinearCastBatchFromBuffer( void* buffer, int numJobs, LinearCastBatch& batchOut, hkArray< hkpVehicleInstance* >& activeVehicles )
{
	int totalNumCollidables;
	int totalNumWheels;
	getTotalNumCommands( activeVehicles, totalNumCollidables, totalNumWheels );

	const int collidablesSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpCollidable) * totalNumWheels );
	const int commandsSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpPairLinearCastCommand) * totalNumCollidables );
	const int outputsSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpRootCdPoint) * totalNumCollidables );

	batchOut.m_collidableStorage = (hkpCollidable*) buffer;
	buffer = hkAddByteOffset( buffer, collidablesSize );

	batchOut.m_commandStorage = (hkpPairLinearCastCommand*) buffer;
	buffer = hkAddByteOffset( buffer, commandsSize );

	batchOut.m_outputStorage = (hkpRootCdPoint*) buffer;
	buffer = hkAddByteOffset( buffer, outputsSize );

	batchOut.m_jobHeaders = (hkpCollisionQueryJobHeader*) buffer;
}


int hkpVehicleLinearCastBatchingManager::buildAndAddCastJobs( const hkpWorld* world, hkInt32 filterSize, int numJobs, hkJobQueue* jobQueue, hkSemaphoreBusyWait* semaphore, void* buffer, hkArray< hkpVehicleInstance* >& activeVehicles )
{
	HK_ASSERT2( 0x158fdfb3, numJobs > 0, "numJobs must be greater than zero." );

	world->markForRead();
	const hkpCollisionFilter* filter = world->getCollisionFilter();
	world->unmarkForRead();

	const int numVehicles = activeVehicles.getSize();

	LinearCastBatch batch;
	{
		getLinearCastBatchFromBuffer( buffer, numJobs, batch, activeVehicles );
	}

	//
	// Create the commands.
	//
	int numCommands = 0;
	{
		hkpCollidable* collidablePtr = batch.m_collidableStorage;
		hkpPairLinearCastCommand* commandPtr = batch.m_commandStorage;
		hkpRootCdPoint* outputPtr = batch.m_outputStorage;
		for ( int v = 0; v < numVehicles; ++v )
		{
			hkpVehicleInstance *const vehicle = activeVehicles[v];
			HK_ASSERT2( 0x459abbe2, vehicle->m_wheelCollide->getType() == hkpVehicleWheelCollide::LINEAR_CAST_WHEEL_COLLIDE, "Vehicle has an incompatible WheelCollide object for this manager." );
			hkpVehicleLinearCastWheelCollide* wheelCollide = static_cast<hkpVehicleLinearCastWheelCollide*>( vehicle->m_wheelCollide );

			const int numCommandsCreated = wheelCollide->buildLinearCastCommands( vehicle, filter, collidablePtr, commandPtr, outputPtr );
			
			numCommands += numCommandsCreated;
			// Move the pointers to the next free blocks of storage.
			collidablePtr += vehicle->getNumWheels();
			commandPtr += numCommandsCreated;
			outputPtr += numCommandsCreated;
		}
	}

	// Avoids empty jobs when there are fewer commands than jobs.
	numJobs = hkMath::min2( numJobs, numCommands );
	if ( !numJobs )
	{
		return numJobs;
	}

	// Divide the commands among the number of jobs.
	const int commandsPerJob = numCommands / numJobs;

	HK_ASSERT2( 0x4feabbef, commandsPerJob, "Commands assigned per job must be greater than zero." );

	//
	// Create the jobs.
	//
	{
		const int unevenOffset = numCommands % numJobs;
		hkpPairLinearCastCommand* commandPtr = batch.m_commandStorage;
		hkpCollisionQueryJobHeader* headerPtr = batch.m_jobHeaders;

		for ( int i = 0; i < numJobs; ++i )
		{
			// Set numCommandsThisJob, accounting for uneven division of commands into jobs.
			const int numCommandsThisJob = commandsPerJob + ( i < unevenOffset );

			world->markForRead();

			// The job will be copied, so we can use a local instance.
			hkpPairLinearCastJob pairLinearCastJob( world->getCollisionInput(), headerPtr, commandPtr, numCommandsThisJob, world->getCollisionFilter(), 0.0f, semaphore );

			world->unmarkForRead();

			// Put the job on the queue
			pairLinearCastJob.setRunsOnSpuOrPpu();
			jobQueue->addJob( pairLinearCastJob, hkJobQueue::JOB_LOW_PRIORITY );

			commandPtr += numCommandsThisJob;
			++headerPtr;
		}
	}

	return numJobs;
}


void hkpVehicleLinearCastBatchingManager::stepVehiclesUsingCastResults( const hkStepInfo& updatedStepInfo, int numJobs, void* buffer, hkArray< hkpVehicleInstance* >& activeVehicles )
{
	const int numVehicles = activeVehicles.getSize();

	LinearCastBatch batch;
	{
		getLinearCastBatchFromBuffer( buffer, numJobs, batch, activeVehicles );
	}
	const hkpPairLinearCastCommand* commandPtr = batch.m_commandStorage;
	
	// Use temporary local storage for the collision data.
	hkLocalArray<hkpVehicleWheelCollide::CollisionDetectionWheelOutput> cdInfo( hkpVehicleInstance::s_maxNumLocalWheels );

	for ( int v = 0; v < numVehicles; ++v )
	{
		const hkpVehicleInstance *const vehicle = activeVehicles[v];
		const int numWheels = vehicle->getNumWheels();
		cdInfo.setSize( numWheels );

		hkpVehicleLinearCastWheelCollide *const wheelCollide = static_cast<hkpVehicleLinearCastWheelCollide*>( vehicle->m_wheelCollide );
		
		for ( hkUint8 w = 0; w < numWheels; ++w )
		{
			const hkpRootCdPoint *const nearestHit = wheelCollide->determineNearestHit( w, commandPtr );
			
			if ( nearestHit )
			{
				wheelCollide->getCollisionOutputFromCastResult( vehicle, w, *nearestHit, cdInfo[w] );
			}
			else
			{
				wheelCollide->getCollisionOutputWithoutHit( vehicle, w, cdInfo[w] );
			}
			wheelCollide->wheelCollideCallback( vehicle, w, cdInfo[w] );

			const int numCollidables = wheelCollide->getNumCommands( w );
			commandPtr += numCollidables;
		}
		activeVehicles[v]->stepVehicleUsingWheelCollideOutput( updatedStepInfo, cdInfo.begin() );		
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
