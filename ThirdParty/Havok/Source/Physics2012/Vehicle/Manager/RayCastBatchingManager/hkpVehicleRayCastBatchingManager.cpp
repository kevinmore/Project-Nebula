/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>
#include <Physics2012/Vehicle/Manager/RayCastBatchingManager/hkpVehicleRayCastBatchingManager.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Physics2012/Vehicle/WheelCollide/RayCast/hkpVehicleRayCastWheelCollide.h>

#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastOutput.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/hkpRayCastQueryJobs.h>
#include <Common/Base/Thread/Job/ThreadPool/hkJobThreadPool.h>
#include <Physics2012/Vehicle/Manager/MultithreadedVehicle/hkpVehicleJobs.h>


int hkpVehicleRayCastBatchingManager::getBufferSize( int numJobs, hkArray< hkpVehicleInstance* >& activeVehicles )
{
	int numWheels = 0;
	int numVehicles = activeVehicles.getSize();
	for ( int i = 0; i < numVehicles; ++i )
	{
		numWheels += activeVehicles[i]->getNumWheels();
	}

	const int commandsSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpShapeRayCastCommand) * numWheels );
	const int outputsSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpWorldRayCastOutput) * numWheels );
	const int indexSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkInt8) * numVehicles );
	const int headersSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpCollisionQueryJobHeader) * numJobs );

	const int bufferSize = commandsSize + outputsSize + indexSize + headersSize;
	return bufferSize;
}


void hkpVehicleRayCastBatchingManager::getRaycastBatchFromBuffer( void* raycastBuffer, int numJobs, RaycastBatch& raycastBatch, hkArray< hkpVehicleInstance* >& activeVehicles )
{
	int numWheels = 0;
	int numVehicles = activeVehicles.getSize();
	for ( int i = 0; i < numVehicles; ++i )
	{
		numWheels += activeVehicles[i]->getNumWheels();
	}

	const int commandsSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpShapeRayCastCommand) * numWheels );
	const int outputsSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpWorldRayCastOutput) * numWheels );
	const int indexSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkInt8) * numVehicles );

	void* buffer = raycastBuffer;

	raycastBatch.m_commandStorage = (hkpShapeRayCastCommand*) buffer;
	buffer = hkAddByteOffset( buffer, commandsSize );

	raycastBatch.m_outputStorage = (hkpWorldRayCastOutput*) buffer;
	buffer = hkAddByteOffset( buffer, outputsSize );

	raycastBatch.m_index = (hkUint8*) buffer;
	buffer = hkAddByteOffset( buffer, indexSize );

	raycastBatch.m_jobHeaders = (hkpCollisionQueryJobHeader*) buffer;
}


int hkpVehicleRayCastBatchingManager::buildAndAddCastJobs( const hkpWorld* world, hkInt32 filterSize, int numJobs, hkJobQueue* jobQueue, hkSemaphoreBusyWait* semaphore, void* buffer, hkArray< hkpVehicleInstance* >& activeVehicles )
{
	HK_ASSERT2( 0x158fdfb3, numJobs > 0, "numJobs must be greater than zero." );
	
	world->markForRead();
	const hkpCollisionFilter* filter = world->getCollisionFilter();
	world->unmarkForRead();

	const int numVehicles = activeVehicles.getSize();

	RaycastBatch raycastBatch;
	{
		getRaycastBatchFromBuffer( buffer, numJobs, raycastBatch, activeVehicles );
	}

	//
	// Create the commands.
	//
	int numCommands = 0;
	{
		hkpShapeRayCastCommand* commandPtr = raycastBatch.m_commandStorage;
		hkpWorldRayCastOutput* outputPtr = raycastBatch.m_outputStorage;
		for ( int v = 0; v < numVehicles; ++v )
		{
			hkpVehicleInstance *const vehicle = activeVehicles[v];
			HK_ASSERT2( 0x459abbe2, vehicle->m_wheelCollide->getType() == hkpVehicleWheelCollide::RAY_CAST_WHEEL_COLLIDE, "Vehicle has an incompatible WheelCollide object for this manager." );
			hkpVehicleRayCastWheelCollide* wheelCollide = static_cast<hkpVehicleRayCastWheelCollide*>( vehicle->m_wheelCollide );

			const int numCommandsCreated = wheelCollide->buildRayCastCommands( vehicle, filter, filterSize, commandPtr, outputPtr );
			HK_ASSERT2( 0x9abb42c0, numCommandsCreated < 256, "Too many commands created." );
			if ( numCommandsCreated > 0 )
			{
				raycastBatch.m_index[v] = (hkInt8) numCommandsCreated;
				numCommands += numCommandsCreated;
				// Move the pointers to the next free blocks of storage.
				commandPtr += numCommandsCreated;
				outputPtr += numCommandsCreated;
			}
			else
			{
				raycastBatch.m_index[v] = 0;
			}
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
		hkpShapeRayCastCommand* commandPtr = raycastBatch.m_commandStorage;
		hkpCollisionQueryJobHeader* headerPtr = raycastBatch.m_jobHeaders;
		const int unevenOffset = numCommands % numJobs;
		for ( int i = 0; i < numJobs; ++i )
		{
			// Set numCommandsThisJob, accounting for uneven division of commands into jobs.
			const int numCommandsThisJob = commandsPerJob + ( i < unevenOffset );

			world->markForRead();
			
			// The job will be copied, so we can use a local instance.
			hkpShapeRayCastJob shapeRayCastJob( world->getCollisionInput(), headerPtr, commandPtr, numCommandsThisJob, semaphore );

			world->unmarkForRead();

			// Put the job on the queue
			shapeRayCastJob.setRunsOnSpuOrPpu();
			jobQueue->addJob( shapeRayCastJob, hkJobQueue::JOB_LOW_PRIORITY );

			commandPtr += numCommandsThisJob;
			++headerPtr;
		}
	}

	return numJobs;
}

void hkpVehicleRayCastBatchingManager::stepVehiclesUsingCastResults( const hkStepInfo& updatedStepInfo, int numJobs, void* buffer, hkArray< hkpVehicleInstance* >& activeVehicles )
{
	const int numVehicles = activeVehicles.getSize();

	RaycastBatch raycastBatch;
	{
		getRaycastBatchFromBuffer( buffer, numJobs, raycastBatch, activeVehicles ); 
	}
	const hkpShapeRayCastCommand* commandPtr = raycastBatch.m_commandStorage;

	// Use temporary local storage for the collision data.
	hkLocalArray<hkpVehicleWheelCollide::CollisionDetectionWheelOutput> cdInfo( hkpVehicleInstance::s_maxNumLocalWheels );

	for ( int v = 0; v < numVehicles; ++v )
	{
		const hkpVehicleInstance *const vehicle = activeVehicles[v];
		cdInfo.setSize( vehicle->getNumWheels() );

		HK_ASSERT2( 0x244fea6b, vehicle->m_wheelCollide->getType() == hkpVehicleWheelCollide::RAY_CAST_WHEEL_COLLIDE, "This manager can only handle vehicles with ray cast wheel collide components." );
		hkpVehicleRayCastWheelCollide *const wheelCollide = static_cast<hkpVehicleRayCastWheelCollide*>( vehicle->m_wheelCollide );

		const int numWheels = vehicle->getNumWheels();
		
		for ( hkInt8 w = 0; w < numWheels; ++w )
		{
			const hkBool32 commandWasIssued = raycastBatch.m_index[v];
			// If we issued a command and got results.
			if ( bool(commandWasIssued) && commandPtr->m_numResultsOut )
			{
				wheelCollide->getCollisionOutputFromCastResult( vehicle, w, *commandPtr->m_results, cdInfo[w] );
			}
			else
			{
				wheelCollide->getCollisionOutputWithoutHit( vehicle, w, cdInfo[w] );
			}
			wheelCollide->wheelCollideCallback( vehicle, w, cdInfo[w] );

			if ( commandWasIssued )
			{
				++commandPtr;
			}
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
