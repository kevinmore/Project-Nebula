/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>
#include <Physics2012/Vehicle/Manager/MultithreadedVehicle/hkpMultithreadedVehicleManager.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Thread/Job/ThreadPool/hkJobThreadPool.h>
#include <Physics2012/Vehicle/WheelCollide/LinearCast/hkpVehicleLinearCastWheelCollide.h>
#include <Physics2012/Vehicle/WheelCollide/RayCast/hkpVehicleRayCastWheelCollide.h>
#include <Physics2012/Dynamics/Phantom/hkpAabbPhantom.h>


void hkpMultithreadedVehicleManager::stepVehiclesSynchronously( hkpWorld* world, const hkStepInfo& updatedStepInfo, hkJobThreadPool* threadPool, hkJobQueue* jobQueue, int numJobs )
{
	hkLocalArray< hkpVehicleInstance* > activeVehicles( m_registeredVehicles.getSize() );
	getActiveVehicles( activeVehicles );

	if ( activeVehicles.getSize() )
	{
		const int multithreadSpeedupThreshold = getMultithreadSpeedupThreshold( activeVehicles );

		if ( activeVehicles.getSize() < multithreadSpeedupThreshold )
		{
			stepVehicleArray( activeVehicles, updatedStepInfo );
		}
		else
		{
			stepVehicleArraySynchronously( activeVehicles, world, updatedStepInfo, threadPool, jobQueue, numJobs );
		}
	}
}

void hkpMultithreadedVehicleManager::stepVehicleArraySynchronously( hkArray<hkpVehicleInstance*>& vehicles, hkpWorld* world, const hkStepInfo& updatedStepInfo, hkJobThreadPool* threadPool, hkJobQueue* jobQueue, int numJobs )
{
	const int bufferSize = getBufferSize( vehicles );
	char *const buffer = hkAllocateStack<char>( bufferSize, "Simulate vehicles" );

	//
	// Setup vehicle jobs
	//
	world->markForWrite();

	updateBeforeCollisionDetection( vehicles );

	numJobs = buildAndAddJobs( vehicles, world, updatedStepInfo,  numJobs, jobQueue, buffer );

	world->unmarkForWrite();

	//
	// Process vehicle jobs
	//
	if ( numJobs )
	{
		world->lockReadOnly();

		threadPool->processAllJobs( jobQueue );
		jobQueue->processAllJobs();

		threadPool->waitForCompletion();

		world->unlockReadOnly();

		//
		// Apply results to vehicles
		//
		world->markForWrite();

		stepVehiclesUsingJobResults( vehicles, updatedStepInfo, buffer );

		world->unmarkForWrite();
	}

	hkDeallocateStack( buffer, bufferSize );
}

int hkpMultithreadedVehicleManager::getBufferSize( hkArray<hkpVehicleInstance*>& vehicles )
{
	const int commandsSize = HK_NEXT_MULTIPLE_OF ( 16, sizeof( hkpVehicleCommand ) * vehicles.getSize() );
	const int outputsSize = HK_NEXT_MULTIPLE_OF ( 16, sizeof( hkpVehicleJobResults ) * vehicles.getSize() );

	const int bufferSize = commandsSize + outputsSize;
	return bufferSize;
}

void hkpMultithreadedVehicleManager::updateBeforeCollisionDetection( hkArray<hkpVehicleInstance*>& vehicles )
{
	const int numVehicles = vehicles.getSize();
	for ( int i = 0; i < numVehicles; ++i )
	{
		vehicles[i]->updateBeforeCollisionDetection();
	}
}

void hkpMultithreadedVehicleManager::buildVehicleCommand( hkpVehicleWheelCollide* wheelCollide, const hkpVehicleInstance* vehicle, hkpVehicleCommand* commandStorage, hkpVehicleJobResults* outputStorage )
{
	switch ( wheelCollide->getType() )
	{
	case hkpVehicleWheelCollide::RAY_CAST_WHEEL_COLLIDE:
	default:
		{
			hkpVehicleRayCastWheelCollide* rayCastWheelCollide = static_cast<hkpVehicleRayCastWheelCollide*>( wheelCollide );

			rayCastWheelCollide->m_phantom->ensureDeterministicOrder();
			break;
		}
	case hkpVehicleWheelCollide::LINEAR_CAST_WHEEL_COLLIDE:
		{
			hkpVehicleLinearCastWheelCollide* linearCastWheelCollide = static_cast<hkpVehicleLinearCastWheelCollide*>( wheelCollide );

			const int numWheels = linearCastWheelCollide->m_wheelStates.getSize();
			for ( hkInt8 i = 0; i < numWheels; ++i )
			{
				const hkpVehicleLinearCastWheelCollide::WheelState& wheelState = linearCastWheelCollide->m_wheelStates[i];
				wheelState.m_phantom->ensureDeterministicOrder();
			}
			break;
		}
	}

	commandStorage->m_jobResults = outputStorage;
}

int hkpMultithreadedVehicleManager::buildAndAddJobs( hkArray<hkpVehicleInstance*>& vehicles, const hkpWorld* world, const hkStepInfo& updatedStepInfo, int numJobs, hkJobQueue* jobQueue, void* buffer )
{
	HK_ASSERT2( 0x158fdfb3, numJobs > 0, "numJobs must be greater than zero." );

	VehicleCommandBatch vehicleBatch;
	{
		getVehicleBatchFromBuffer( vehicles, buffer, vehicleBatch );
	}

	//
	// Create the commands.
	//
	int numCommands = 0;
	{
		hkpVehicleCommand* commandPtr = vehicleBatch.m_commandStorage;
		hkpVehicleJobResults* outputPtr = vehicleBatch.m_outputStorage;
		const int numVehicles = vehicles.getSize();
		for ( int v = 0; v < numVehicles; ++v )
		{
			hkpVehicleInstance *const vehicle = vehicles[v];

			buildVehicleCommand( vehicle->m_wheelCollide, vehicle, commandPtr, outputPtr );

			++numCommands;
			// Move the pointers to the next free blocks of storage.
			++commandPtr;
			++outputPtr;
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
		hkpVehicleCommand* commandPtr = vehicleBatch.m_commandStorage;
		const int unevenOffset = numCommands % numJobs;
		int vehicleOffset = 0;
		for ( int i = 0; i < numJobs; ++i )
		{
			// Set numCommandsThisJob, accounting for uneven division of commands into jobs.
			const int numCommandsThisJob = commandsPerJob + ( i < unevenOffset );

			world->markForRead();

			// The job will be copied, so we can use a local instance.
			hkpVehicleIntegrateJob vehicleIntegrateJob( commandPtr, &vehicles[vehicleOffset], numCommandsThisJob, updatedStepInfo );

			world->unmarkForRead();

			// Put the job on the queue
			vehicleIntegrateJob.setRunsOnSpuOrPpu();
			jobQueue->addJob( vehicleIntegrateJob, hkJobQueue::JOB_HIGH_PRIORITY );

			vehicleOffset += numCommandsThisJob;
			commandPtr += numCommandsThisJob;
		}
	}

	return numJobs;
}

void hkpMultithreadedVehicleManager::getVehicleBatchFromBuffer( hkArray<hkpVehicleInstance*>& vehicles, void* vehicleBuffer, VehicleCommandBatch& commandBatch )
{
	const int commandsSize = HK_NEXT_MULTIPLE_OF ( 16, sizeof( hkpVehicleCommand ) * vehicles.getSize() );
	const int outputsSize = HK_NEXT_MULTIPLE_OF ( 16, sizeof( hkpVehicleJobResults ) * vehicles.getSize() );
	void* buffer = vehicleBuffer;

	commandBatch.m_commandStorage = ( hkpVehicleCommand* ) buffer;
	buffer = hkAddByteOffset( buffer, commandsSize );

	commandBatch.m_outputStorage = ( hkpVehicleJobResults* ) buffer;
	buffer = hkAddByteOffset( buffer, outputsSize );
}

void hkpMultithreadedVehicleManager::stepVehiclesUsingJobResults( hkArray<hkpVehicleInstance*>& vehicles, const hkStepInfo& stepInfo, void* buffer )
{
	VehicleCommandBatch raycastBatch;
	{
		getVehicleBatchFromBuffer( vehicles, buffer, raycastBatch );
	}

	const hkpVehicleCommand* commandPtr = raycastBatch.m_commandStorage;
	const int numVehicles = vehicles.getSize();
	for ( int v = 0; v < numVehicles; ++v )
	{		
		hkpVehicleInstance* vehicle = vehicles[v];
		commandPtr->m_jobResults->applyForcesFromStep( *vehicle );

		++commandPtr;
	}
}

int hkpMultithreadedVehicleManager::getMultithreadSpeedupThreshold( hkArray<hkpVehicleInstance*>& vehicles )
{
	HK_ASSERT2( 0x158fdfb3, vehicles.getSize() > 0, "The number of vehicles must be greater than 0" );

	hkpVehicleWheelCollide::WheelCollideType type = vehicles[0]->m_wheelCollide->m_type;
	int threshold;
	switch( type )
	{
	case hkpVehicleWheelCollide::RAY_CAST_WHEEL_COLLIDE:
	default:
		{
#ifdef HK_PLATFORM_XBOX360
			threshold = 5;
#else //HK_PLATFORM_WIN32
			threshold = 12;
#endif
		}
		break;
	case hkpVehicleWheelCollide::LINEAR_CAST_WHEEL_COLLIDE:
		{
#ifdef HK_PLATFORM_XBOX360
			threshold = 3;
#else //HK_PLATFORM_WIN32
			threshold = 6;
#endif
		}
		break;
	}

	return threshold;
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
