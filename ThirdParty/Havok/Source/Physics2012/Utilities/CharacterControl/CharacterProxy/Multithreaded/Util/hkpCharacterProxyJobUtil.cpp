/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/Util/hkpCharacterProxyJobUtil.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseDispatcher.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseHandlePair.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhase.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Phantom/hkpShapePhantom.h>
#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Thread/Job/ThreadPool/hkJobThreadPool.h>

#define HK_DEFAULT_MAX_INTERACTIONS 4
#define HK_DEFAULT_MAX_TRIGGER_VOLUMES 4

hkpCharacterProxyJobUtil::JobData::JobData( const hkStepInfo& stepInfo, hkpWorld* world ) 
	: m_jobQueue( HK_NULL )
	, m_numJobs( 0 )
	, m_world( world )
	, m_characters( HK_NULL )
	, m_maxInteractions( HK_DEFAULT_MAX_INTERACTIONS )
	, m_maxTriggerVolumes( HK_DEFAULT_MAX_TRIGGER_VOLUMES )
	, m_worldGravity( world->getGravity() ) 
	, m_stepInfo( stepInfo ) 
	, m_createCdPointCollectorOnCpuFunc( HK_NULL )
	, m_broadphase( HK_NULL )
{
	m_collisionInput = world->getCollisionInputRw();
}

static inline void removeListeners( const hkArray<hkpCharacterProxy*>& characters, hkpWorld* world )
{
	// Remove listeners from all bodies and phantoms
	world->markForWrite();
	{
		const int numCharacters = characters.getSize();
		for( int k = 0 ; k < numCharacters ; k ++ )
		{
			hkpCharacterProxy* character = characters[k];			
			hkArray<hkpRigidBody*>& bodies = character->m_bodies;
			hkArray<hkpPhantom*>& phantoms = character->m_phantoms;

			const int numBodies = bodies.getSize();
			for (int i=0 ; i < numBodies; i++)
			{
				bodies[i]->removeEntityListener( character );
			}

			const int numPhantoms = phantoms.getSize();
			for (int j=0; j< numPhantoms ; j++)
			{
				phantoms[j]->removePhantomListener( character );		
			}
			bodies.clear();
			phantoms.clear();
		}
	}
	world->unmarkForWrite();
}

static inline void addListeners( const hkArray<hkpCharacterProxy*>& characters, hkpWorld* world )
{
	// Add listeners to all bodies and phantoms
	world->markForWrite();
	{
		const int numCharacters = characters.getSize();
		for( int k = 0 ; k < numCharacters ; k ++ )
		{
			hkpCharacterProxy* character = characters[k];			
			hkArray<hkpRigidBody*>& bodies = character->m_bodies;
			hkArray<hkpPhantom*>& phantoms = character->m_phantoms;
			
			const int numCollidables = character->m_manifold.getSize();
			for ( int i = 0; i < numCollidables; ++i )
			{
				hkpRootCdPoint& cdPoint = character->m_manifold[i];
				hkpRigidBody* body = hkpGetRigidBody( cdPoint.m_rootCollidableB );
				if ( body )
				{
					if ( bodies.indexOf( body ) == -1 )
					{
						body->addEntityListener( character );
						bodies.pushBack( body );
					}
				}
				else
				{
					hkpPhantom* phantom = hkpGetPhantom( cdPoint.m_rootCollidableB );
					HK_ASSERT2( 0x4d9f1a17, phantom, "Collidable in manifold which is neither a body nor a phantom" );
					if ( phantoms.indexOf( phantom ) == -1 )
					{
						phantom->addPhantomListener( character );
						phantoms.pushBack( phantom );
					}
				}
			}
		}
	}
	world->unmarkForWrite();
}

void HK_CALL hkpCharacterProxyJobUtil::getCharacterJobBatchFromBuffer( const hkpCharacterProxyJobUtil::JobData& mtData, void* buffer, CharacterJobBatch& batchOut )
{
	const int numCharacters = mtData.m_characters->getSize();
	const int maxInteractions = mtData.m_maxInteractions;
	const int maxTriggerVolumes = mtData.m_maxTriggerVolumes;
	const int perTriggerVolumeSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpTriggerVolume*) * maxTriggerVolumes );

	const int commandsSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpCharacterProxyIntegrateCommand) * numCharacters );
	const int characterSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpCharacterProxy) * numCharacters );
	const int collidableSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpCollidable) * numCharacters );
	const int objectInteractionSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpCharacterProxyInteractionResults) * numCharacters * maxInteractions );
	const int triggerVolumeSize = perTriggerVolumeSize * numCharacters;

	batchOut.m_commandStorage = reinterpret_cast<hkpCharacterProxyIntegrateCommand*>( buffer );
	buffer = hkAddByteOffset( buffer, commandsSize );
	batchOut.m_characterStorage = reinterpret_cast<hkpCharacterProxy*>( buffer );
	buffer = hkAddByteOffset( buffer, characterSize );
	batchOut.m_collidableStorage = reinterpret_cast<hkpCollidable*>( buffer );
	buffer = hkAddByteOffset( buffer, collidableSize );
	batchOut.m_objectInteractionStorage = reinterpret_cast<hkpCharacterProxyInteractionResults*>( buffer );
	buffer = hkAddByteOffset( buffer, objectInteractionSize );	
	batchOut.m_triggerVolumeStorage = reinterpret_cast<hkpTriggerVolume**>( buffer );
	buffer = hkAddByteOffset( buffer, triggerVolumeSize );
	batchOut.m_jobHeaders = reinterpret_cast<hkpCharacterProxyJobHeader*>( buffer );	
}

static inline void updateAabbsBatch(const hkArray<hkpCharacterProxy*>& characters, hkpWorld* world)
{
	HK_TIMER_BEGIN("updateAabbsBatch", HK_NULL ); 
	
	const int numCharacters = characters.getSize();
	hkArray<hkAabb> aabbs( numCharacters );
	hkArray<hkpBroadPhaseHandle*> handles( numCharacters );

	const hkReal halfTolerance = 0.5f * world->getCollisionInput()->getTolerance();	

	for( int i = 0; i < numCharacters; i++ )
	{
		hkAabb& aabb = aabbs[i];
		hkpCharacterProxy* character = characters[i];			
		hkpShapePhantom* shapePhantom = character->getShapePhantom();

		// Get broadphase handles from shape phantom collidable
		{
			HK_ASSERT2(0x3c6f9e3a, shapePhantom->getWorld() == world, "All phantoms must be in the same world");
			hkpCollidable* col = const_cast<hkpCollidable*>( shapePhantom->getCollidable() );
			handles[i] = col->getBroadPhaseHandle();
		}		

		// Get shape phantom's AABB
		{
			const hkpShape* shape = shapePhantom->getCollidable()->getShape();
			HK_ASSERT2( 0x37b9ea6b, shape, "shape phantom has no shape!");
			HK_ACCESS_CHECK_WITH_PARENT( world, HK_ACCESS_RW, shapePhantom, HK_ACCESS_RW );
			const hkReal tolerance = halfTolerance + character->m_keepDistance + character->m_keepContactTolerance;
			shape->getAabb( shapePhantom->getMotionState()->getTransform(), tolerance, aabb );	
		}

		// Calculated new AABB of phantom based on old displacement.
		{
			const hkVector4& path = character->m_oldDisplacement;

			hkVector4 zero; zero.setZero();
			hkVector4 pathMin; pathMin.setMin( zero, path );
			hkVector4 pathMax; pathMax.setMax( zero, path );

			aabb.m_min.add( pathMin );
			aabb.m_max.add( pathMax );
		}
	}

	// Check if the world is locked, if so bail out
	if( world->areCriticalOperationsLockedForPhantoms() )
	{
		HK_ASSERT2(0x6330489e, false, "Can't queue hkpPhantomUtil::_setPositionBatch aborting.");
		return;
	}

	// Perform the actual operation
	HK_ACCESS_CHECK_OBJECT( world, HK_ACCESS_RW );

	world->lockCriticalOperations();

	hkLocalArray<hkpBroadPhaseHandlePair> newPairs( world->m_broadPhaseUpdateSize );
	hkLocalArray<hkpBroadPhaseHandlePair> delPairs( world->m_broadPhaseUpdateSize );

	world->getBroadPhase()->lock();
	world->getBroadPhase()->updateAabbs( handles.begin(), aabbs.begin(), numCharacters, newPairs, delPairs );

	// check for changes
	if( ( newPairs.getSize() != 0 ) || ( delPairs.getSize() != 0 ) )
	{
		hkpTypedBroadPhaseDispatcher::removeDuplicates( newPairs, delPairs );

		world->m_broadPhaseDispatcher->removePairs(static_cast<hkpTypedBroadPhaseHandlePair*>( delPairs.begin()), delPairs.getSize() );
		world->m_broadPhaseDispatcher->addPairs( static_cast<hkpTypedBroadPhaseHandlePair*>( newPairs.begin()), newPairs.getSize(), world->getCollisionFilter() );
	}

	world->getBroadPhase()->unlock();
	world->unlockAndAttemptToExecutePendingOperations();

	HK_TIMER_END();
}

int HK_CALL hkpCharacterProxyJobUtil::getBufferSize(const JobData& mtData)
{
	const int numJobs = mtData.m_numJobs;
	const int numCharacters = mtData.m_characters->getSize();
	const int maxInteractions = mtData.m_maxInteractions;
	const int maxTriggerVolumes = mtData.m_maxTriggerVolumes;
	const int perTriggerVolumeSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpTriggerVolume*) * maxTriggerVolumes );
	
	const int commandsSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpCharacterProxyIntegrateCommand) * numCharacters );
	const int characterSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpCharacterProxy) * numCharacters );
	const int collidableSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpCollidable) * numCharacters );
	const int objectInteractionSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpCharacterProxyInteractionResults) * numCharacters * maxInteractions );	
	const int triggerVolumeSize = perTriggerVolumeSize * numCharacters;
	const int headersSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpCharacterProxyJobHeader) * numJobs );

	return commandsSize + characterSize + collidableSize + objectInteractionSize + triggerVolumeSize + headersSize;
}

void HK_CALL hkpCharacterProxyJobUtil::buildAndAddJobs(const JobData& mtData, hkSemaphoreBusyWait* semaphore, void* buffer)
{
	HK_TIMER_BEGIN("buildAndAddJobs", HK_NULL ); 

	const hkArray<hkpCharacterProxy*>& characters = *(mtData.m_characters);	
	
	removeListeners( characters, mtData.m_world );
	
	CharacterJobBatch batch;
	{
		getCharacterJobBatchFromBuffer( mtData, buffer, batch );
	}	
	
	// One command per character
	const int numCommands = characters.getSize(); 
	{
		hkpCharacterProxyIntegrateCommand* commandPtr = batch.m_commandStorage;
		hkpCharacterProxyInteractionResults* objectInteractionPtr = batch.m_objectInteractionStorage;
		hkpTriggerVolume** triggerVolumePtr = batch.m_triggerVolumeStorage;
		hkpCharacterProxy* characterPtr = batch.m_characterStorage;
		hkpCollidable* collidablePtr = batch.m_collidableStorage;
		for ( int i = 0; i < numCommands ; ++i )
		{
			// Create the command in the buffer.
			new ( commandPtr ) hkpCharacterProxyIntegrateCommand;

			characters[i]->m_shapePhantom->ensureDeterministicOrder();
			commandPtr->m_objectInteraction = objectInteractionPtr;
			commandPtr->m_triggerVolumeAndFlags = triggerVolumePtr;
			commandPtr->m_character = characters[i];
			commandPtr->m_collidable = characters[i]->m_shapePhantom->getCollidableRw();
			commandPtr->m_maxInteractions = mtData.m_maxInteractions;
			commandPtr->m_maxTriggerVolumes = mtData.m_maxTriggerVolumes;
			
			++commandPtr;
			++characterPtr;
			++collidablePtr;
			objectInteractionPtr += mtData.m_maxInteractions;
			const int perTriggerVolumeSize = HK_NEXT_MULTIPLE_OF (16, sizeof(hkpTriggerVolume*) * mtData.m_maxTriggerVolumes );
			triggerVolumePtr = reinterpret_cast<hkpTriggerVolume**>( hkAddByteOffset( triggerVolumePtr, perTriggerVolumeSize ) );
		}
	}

	// Avoids empty jobs when there are fewer commands than jobs.
	const int numJobs = hkMath::min2( mtData.m_numJobs , numCommands );

	// Divide the commands among the number of jobs.
	const int commandsPerJob = numCommands / numJobs;
	HK_ASSERT2( 0x4feabbef, commandsPerJob, "Commands assigned per job must be greater than zero." );

	//
	// Create the jobs.
	//
	{
		const int unevenOffset = numCommands % numJobs;
		hkpCharacterProxyIntegrateCommand* commandPtr = batch.m_commandStorage;
		hkpCharacterProxyJobHeader* headerPtr = batch.m_jobHeaders;

		for ( int i = 0; i < numJobs; ++i )
		{
			// Set numCommandsThisJob, accounting for uneven division of commands into jobs.
			const int numCommandsThisJob = commandsPerJob + ( i < unevenOffset );

			hkpCharacterProxyIntegrateJob characterProxyJob( mtData.m_collisionInput
				, headerPtr
				, semaphore
				, commandPtr
				, numCommandsThisJob
				, mtData.m_stepInfo.m_deltaTime
				, mtData.m_stepInfo.m_invDeltaTime
				, mtData.m_worldGravity
				, mtData.m_createCdPointCollectorOnCpuFunc
				, mtData.m_broadphase );

			// Put the job on the queue
			characterProxyJob.setRunsOnSpuOrPpu();
			mtData.m_jobQueue->addJob( characterProxyJob, hkJobQueue::JOB_LOW_PRIORITY );

			commandPtr += numCommandsThisJob;
			++headerPtr;
		}
	}
	HK_TIMER_END();
}

void HK_CALL hkpCharacterProxyJobUtil::handleResults(const JobData& mtData , void* buffer)
{	
	HK_TIMER_BEGIN("handleResults", HK_NULL ); 	
	
	CharacterJobBatch batch;
	{
		getCharacterJobBatchFromBuffer( mtData, buffer, batch );
	}

	const hkArray<hkpCharacterProxy*>& characters = *(mtData.m_characters);	
	const int numCharacters = characters.getSize();
	for( int i = 0 ; i < numCharacters ; i ++ )
	{
		hkpCharacterProxyIntegrateCommand& command = batch.m_commandStorage[i];

		HK_TIMER_BEGIN("applyImpulses", HK_NULL ); 
		// Handle interactions with rigid bodies.
		{
			hkpCharacterProxyInteractionResults* output = command.m_objectInteraction;
			int j = 0;
			while ( ( j < mtData.m_maxInteractions ) && ( output->m_collidingBody != HK_NULL ) )
			{				
				output->m_collidingBody->applyPointImpulse( output->m_objectImpulse, output->m_impulsePosition );
				++j;
				++output;
			}
			HK_WARN_ON_DEBUG_IF( ( mtData.m_maxInteractions > 0 ) && ( j == mtData.m_maxInteractions ), 0xc8a991a0, "Interaction results array was full, so some interactions may have been lost during character processing." );
		}
		HK_TIMER_END();

		characters[i]->processTriggerVolumes( command.m_triggerVolumeAndFlags, command.m_maxTriggerVolumes );
		
		characters[i]->getShapePhantom()->setPosition( command.m_position );
	}
	
	addListeners( characters, mtData.m_world );

	updateAabbsBatch( characters, mtData.m_world );

	HK_TIMER_END();
}

void HK_CALL hkpCharacterProxyJobUtil::simulateCharactersSynchronously( hkJobThreadPool* threadPool, const JobData& mtData )
{
	HK_ASSERT2( 0x168f3f42, mtData.m_characters != HK_NULL,	"Did you forget to set the character array in hkCharacterProxyJobCInfo?" );
	HK_ASSERT2( 0x128f3843, mtData.m_numJobs > 0, "There must be at least one job." );
	HK_ASSERT2( 0xc6cfcd4b, mtData.m_jobQueue != HK_NULL , "Did you forget to set the jobQueue in hkCharacterProxyJobCInfo?");
	HK_ASSERT2( 0x1384bf8a, mtData.m_characters->getSize() > 0, "There must be at least one character for this job." );	
	HK_ASSERT2( 0xe3242ee2, mtData.m_world == (*(mtData.m_characters))[0]->getShapePhantom()->getWorld(), "The character's shape phantom hasn't been added to the world!" );	

	hkpWorld* world = mtData.m_world;

	const int bufferSize = getBufferSize( mtData );
	char *const buffer = hkAllocateStack<char>( bufferSize, "Character Proxy MT" );
	{
		{				
			hkScopedPtr<hkSemaphoreBusyWait> semaphore;
			buildAndAddJobs( mtData, semaphore, buffer );
			world->unmarkForWrite();
			world->lockReadOnly();
			{				
				HK_TIMER_BEGIN("processAllJobs", HK_NULL ); 
				threadPool->processAllJobs( mtData.m_jobQueue );
				mtData.m_jobQueue->processAllJobs();
				// Wait for all the jobs we started to finish.
				threadPool->waitForCompletion();
				semaphore->acquire();
				HK_TIMER_END();
			}
			world->unlockReadOnly();
			world->markForWrite();
		}
	}
	handleResults( mtData, buffer );	
	hkDeallocateStack(buffer, bufferSize);
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
