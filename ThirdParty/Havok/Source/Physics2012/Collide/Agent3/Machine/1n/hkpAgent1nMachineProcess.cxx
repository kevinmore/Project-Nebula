/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

// AgentTrack is on SPU, pointer in array points to PPU memory

 // if this breaks check the destructor of hkpProcessCollisionInput to free memory of the correct size

ON_1N_MACHINE( void HK_CALL HK_PROCESS_FUNC_NAME(hkAgent1nMachine_Process)( hkpAgent1nTrack& agentTrack, const hkpAgent3ProcessInput& inputIn, const HK_SHAPE_CONTAINER* shapeContainerB, const hkpShapeKey* newKeys, hkpProcessCollisionOutput& output ) )
ON_NM_MACHINE( void HK_CALL HK_PROCESS_FUNC_NAME(hkAgentNmMachine_Process)( hkpAgent1nTrack& agentTrack, const hkpAgent3ProcessInput& inputIn, const HK_SHAPE_CONTAINER* shapeContainerA, const HK_SHAPE_CONTAINER* shapeContainerB, const hkpShapeKeyPair* newKeys, hkpProcessCollisionOutput& output ) )
{
	//ON_1N_MACHINE(const int mode = HKP_AGENT_MACHINE_MODE_1N);
	//ON_NM_MACHINE(const int mode = HKP_AGENT_MACHINE_MODE_NM);
#define MAX_NUM_SECTORS ( ( (HK_MAX_AGENTS_IN_1N_MACHINE-1) / HK_AGENT3_FEWEST_AGENTS_PER_1N_SECTOR ) + 1 )

#if !defined(HK_PLATFORM_SPU)
	hkMath::prefetch128( agentTrack.m_sectors.begin() );
#else
	const int sectorPointersBufferSize = HK_NEXT_MULTIPLE_OF(128, MAX_NUM_SECTORS*sizeof(hkpAgent1nSector*));
	hkpAgent1nSector** pReadSectorBuffer = hkAllocateStack<hkpAgent1nSector*>(sectorPointersBufferSize, "1n-machine read sector pntrs");
	hkpAgent1nSector** pWriteSectorBuffer = hkAllocateStack<hkpAgent1nSector*>(sectorPointersBufferSize, "1n-machine write sector pntrs");

	HK_SPU_STACK_POINTER_CHECK();

	// bring in the sectors array from main memory
	hkArraySpu agentSectorsOnSpu;
	agentSectorsOnSpu.init( agentTrack.m_sectors, pReadSectorBuffer, MAX_NUM_SECTORS );

	int numWrittenSectors = 0;

#endif


#if defined(HK_1N_MACHINE_SUPPORTS_WELDING)
	hkpProcessCollisionOutput::PotentialInfo *oldPotentialInfo = output.m_potentialContacts;
	hkpProcessCollisionOutput::PotentialInfo potentialContacts;
	if ( !oldPotentialInfo && shapeContainerB->isWeldingEnabled() )
	{
		potentialContacts.reset();
		output.m_potentialContacts = &potentialContacts;
	}
	else
	{
		// no recursive welding
		output.m_potentialContacts = HK_NULL;
	}
#endif
	hkpAgent3ProcessInput input;
	hkString::memCpy16NonEmpty(&input, &inputIn, sizeof(hkpAgent3ProcessInput)/16 );

	ON_1N_MACHINE( HK_CONTAINER_ITERATOR_TYPE containerIterator( shapeContainerB, newKeys ) );
	ON_NM_MACHINE( HK_CONTAINER_ITERATOR_TYPE containerIterator( shapeContainerA, shapeContainerB, newKeys ) );
	
	hkpCollisionDispatcher* dispatcher = input.m_input->m_dispatcher;

	const int numSlotsForTransformedCdBodies = 4;

	hkMotionState transformForTransformedShapeB[numSlotsForTransformedCdBodies];
	ON_NM_MACHINE(hkMotionState transformForTransformedShapeA[numSlotsForTransformedCdBodies]); 
	hkpCdBody cdBodyForTransformedShapeB[numSlotsForTransformedCdBodies];
	ON_NM_MACHINE(hkpCdBody cdBodyForTransformedShapeA[numSlotsForTransformedCdBodies]);


	hkPadSpu<const hkpCdBody*>& modifiedBodyBPtr = input.m_bodyB;
	hkPadSpu<const hkpCdBody*>& modifiedBodyAPtr = input.m_bodyA;

	hkpCdBody childBodyB( inputIn.m_bodyB );
	modifiedBodyBPtr = &childBodyB;
	ON_NM_MACHINE( hkpCdBody childBodyA( inputIn.m_bodyA ) );
	ON_NM_MACHINE( modifiedBodyAPtr = &childBodyA; );

	ON_1N_MACHINE( HK_ASSERT2(0xad348372, modifiedBodyAPtr->getShape()->getType() != hkcdShapeType::TRANSFORM, "The bodyA cannot have a hkpTransformShape!") );


#if 1
	int numTim = 0;
#define ON_DEBUG(X) X
#endif

#if !defined(HK_PLATFORM_SPU)
	hkpAgent1nSector* readSector = agentTrack.m_sectors[0];
	hkMath::forcePrefetch<HK_AGENT3_SECTOR_SIZE>( readSector );
	hkpAgent1nSector* currentSector = (input.m_input->m_spareAgentSector)? input.m_input->m_spareAgentSector : new hkpAgent1nSector;
	input.m_input->m_spareAgentSector = HK_NULL;
	int               currentSectorIndex = 0;
#else
	void* buffer = hkAllocateStack<void>(4 * HK_AGENT3_SECTOR_SIZE, "AgentSector Buffers");

	hkpAgent1nSector* readSector			= (hkpAgent1nSector*)hkAddByteOffset(buffer, 0 * HK_AGENT3_SECTOR_SIZE);
	hkpAgent1nSector* prefetchReadSector = (hkpAgent1nSector*)hkAddByteOffset(buffer, 1 * HK_AGENT3_SECTOR_SIZE);
	hkpAgent1nSector* currentSector		= (hkpAgent1nSector*)hkAddByteOffset(buffer, 2 * HK_AGENT3_SECTOR_SIZE);
	hkpAgent1nSector* postWriteSector	= (hkpAgent1nSector*)hkAddByteOffset(buffer, 3 * HK_AGENT3_SECTOR_SIZE);

	hkSpuDmaManager::waitForDmaCompletion( hk1nMachineDmaGroups::GET_SECTOR_POINTERS_DMA_GROUP );


	// Not used, because hkArraySpu::does a wait for completion
	hkSpuDmaManager::getFromMainMemory( readSector, pReadSectorBuffer[0], sizeof(hkpAgent1nSector), hkSpuDmaManager::READ_COPY, hk1nMachineDmaGroups::GET_SECTOR_DMA_GROUP );
	HK_SPU_DMA_DEFER_FINAL_CHECKS_UNTIL_WAIT( pReadSectorBuffer[0], readSector, sizeof(hkpAgent1nSector) );
	if ( agentTrack.m_sectors.getSize()>1)
	{
		hkSpuDmaManager::getFromMainMemory( prefetchReadSector, pReadSectorBuffer[1], sizeof(hkpAgent1nSector), hkSpuDmaManager::READ_COPY, hk1nMachineDmaGroups::GET_SECTOR_DMA_GROUP+1 );
		HK_SPU_DMA_DEFER_FINAL_CHECKS_UNTIL_WAIT   ( pReadSectorBuffer[1], prefetchReadSector, sizeof(hkpAgent1nSector) );
	}
	HK_COMPILE_TIME_ASSERT( (hk1nMachineDmaGroups::GET_SECTOR_DMA_GROUP & 1) == 0 ); // we alternate this bit when double buffering
	HK_COMPILE_TIME_ASSERT( (hk1nMachineDmaGroups::WRITE_SECTOR_DMA_GROUP & 1) == 0 );
	int prefetchReadDmaGroup = hk1nMachineDmaGroups::GET_SECTOR_DMA_GROUP + 1;
	int postWriteDmaGroup = hk1nMachineDmaGroups::WRITE_SECTOR_DMA_GROUP;
#endif

	//
	//	Some input of the last time step
	//
	bool                 prevInputInitialized = false;
	bool                 prevFlippedInputInitialized = false;

	hkpAgent3Input prevInput;
	hkpAgent3Input prevFlippedInput;

	hkpAgentNmMachineBodyTemp prevTemp;
	hkpAgentNmMachineBodyTemp prevFlippedTemp;

#if defined(HK_PLATFORM_SPU)
	// wait for current sector
	hkSpuDmaManager::waitForDmaCompletion(hk1nMachineDmaGroups::GET_SECTOR_DMA_GROUP);
	hkpAgent1nSector* writeSectorOnPpu = pReadSectorBuffer[0]; // have copy in spu so we can reuse this sector immediately for writing
#endif

	bool inExtensionBuffer = false;								// if set to true, the agent is written to a small local buffer, as we do not know whether it will fit in the current sector
	HK_ALIGN_REAL(hkUchar extensionBuffer[hkAgent3::MAX_SIZE]);	// extra data to hold the last agent for the sector. We do this as we do not know the size of the agent yet

	int nextReadSectorIndex = 1;
	hkpAgentData* readData = readSector->getBegin();
	hkpAgentData* readEnd  = readSector->getEnd();
	hkpAgentData* currentData    = currentSector->getBegin();
	hkpAgentData* currentDataEnd = currentSector->getEndOfCapacity();
	int bytesUsedInCurrentSector = 0;	// only initialized when inExtensionBuffer == true


	while ( 1 )
	{
	#if defined HK_PLATFORM_PS3_SPU
		hkSpuMonitorCache::dmaMonitorDataToMainMemorySpu();
	#endif

		// merge the newKeys
		ON_1N_MACHINE(hkpShapeKeyPairLocal1n shapeKeyPair);
		ON_NM_MACHINE(hkpShapeKeyPairLocalNm shapeKeyPair);

		modifiedBodyBPtr = &childBodyB;
		ON_NM_MACHINE( modifiedBodyAPtr = &childBodyA; );
		input.m_aTb = inputIn.m_aTb;

		if ( !newKeys )
		{
				//
				//	keep and copy the whole agent
				//
			hkpAgent1nMachineEntry* entry = reinterpret_cast<hkpAgent1nMachineEntry*>(readData);
			int entrySize = entry->m_size;
			hkString::memCpy16NonEmpty( currentData, readData, entrySize >> 4 );
			readData = hkAddByteOffset( readData, entrySize );
			shapeKeyPair = entry->m_shapeKeyPair;
		}
		else
		{
			hkpAgent1nMachineEntry* readHeader = reinterpret_cast<hkpAgent1nMachineEntry*>(readData);

			containerIterator.update();
			ON_1N_MACHINE( shapeKeyPair.m_shapeKeyB = containerIterator.getShapeKey() );
			ON_NM_MACHINE( shapeKeyPair = containerIterator.getShapeKeyPair() );

			if ( shapeKeyPair == readHeader->m_shapeKeyPair )
			{
				//
				//	keep and copy the whole agent
				//
				int size = readHeader->m_size;
				hkString::memCpy16NonEmpty( currentData, readData, size >> 4  );
				readData = hkAddByteOffset( readData, size );
				containerIterator.advance();
			}
			else
			{
				if ( shapeKeyPair > readHeader->m_shapeKeyPair )
				{
					//
					// delete the agent
					//
					HK_COMPILE_TIME_ASSERT( hkAgent3::STREAM_CALL_WITH_TIM == (hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED & 0x6)) ;
					HK_COMPILE_TIME_ASSERT( hkAgent3::STREAM_CALL_WITH_TIM == (hkAgent3::STREAM_CALL_WITH_TIM_WITH_TRANSFORM & 0x6)) ;
					HK_COMPILE_TIME_ASSERT( hkAgent3::STREAM_CALL_WITH_TIM == (hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED_WITH_TRANSFORM & 0x6)) ;

					hkpAgentData* agentData = ( (readHeader->m_streamCommand&0x6) == hkAgent3::STREAM_CALL_WITH_TIM )
												? hkAddByteOffset( readHeader, hkSizeOf( hkpAgent1nMachineTimEntry) )
												: hkAddByteOffset( readHeader, hkSizeOf( hkpAgent1nMachinePaddedEntry) );

					readData = hkAddByteOffset( readData, readHeader->m_size );
					dispatcher->getAgent3DestroyFunc( readHeader->m_agentType )( readHeader, agentData, input.m_contactMgr, *output.m_constraintOwner.val(), dispatcher );
					goto checkReadSector;
				}
				else
				{
					//
					// create new agent 
					//
					if ( containerIterator.isCollisionEnabled( input.m_input, inputIn.m_bodyA, inputIn.m_bodyB ) )
					{
						if ( hkMemoryStateIsOutOfMemory(18)  )
						{
							containerIterator.advance();
							goto checkReadSector;
						}

						EXTRACT_CHILD_SHAPES(containerIterator, childBodyA, childBodyB);

						const hkpShape* shapeB = modifiedBodyBPtr->m_shape;
						const hkpShape* shapeA = modifiedBodyAPtr->m_shape;
						hkPadSpu<hkUchar> cdBodyHasTransformFlag = 0;
						{
							// strip hkpTransformShapes
							modifiedBodyBPtr = hkAgentMachine_processTransformedShapes(modifiedBodyBPtr, cdBodyForTransformedShapeB, transformForTransformedShapeB, numSlotsForTransformedCdBodies, cdBodyHasTransformFlag);
							shapeB = modifiedBodyBPtr->m_shape;
							ON_NM_MACHINE( modifiedBodyAPtr = hkAgentMachine_processTransformedShapes(modifiedBodyAPtr, cdBodyForTransformedShapeA, transformForTransformedShapeA, numSlotsForTransformedCdBodies, cdBodyHasTransformFlag) );
							ON_NM_MACHINE( shapeA = modifiedBodyAPtr->m_shape );

							input.m_aTb.setMulInverseMul(modifiedBodyAPtr->getTransform(), modifiedBodyBPtr->getTransform());
						}

						// create a new agent, will be processed immediately
						hkpAgent1nMachineEntry* h = reinterpret_cast<hkpAgent1nMachineEntry*>( currentData );
						shapeKeyPair.writeTo(h->m_shapeKeyPair);

						h->m_agentType = hkUchar(dispatcher->getAgent3Type( shapeA->getType(), shapeB->getType(), input.m_input->m_createPredictiveAgents ));
						h->m_numContactPoints = 0;

						hkpAgent3ProcessInput flippedInput;
						hkpAgent3ProcessInput* in = &input;
						int isFlipped = 0;

						if ( dispatcher->getAgent3Symmetric(h->m_agentType) == hkAgent3::IS_NOT_SYMMETRIC_AND_FLIPPED )
						{
							isFlipped = 1;
							in = &flippedInput;
							hkAgent1nMachine_flipInput( input, flippedInput);
						}

						
						if ( dispatcher->getAgent3SepNormalFunc( h->m_agentType ) != HK_NULL)
						{
							hkpAgent1nMachineTimEntry* entry = reinterpret_cast<hkpAgent1nMachineTimEntry*>( currentData );

							entry->m_streamCommand = hkUchar(hkAgent3::STREAM_CALL_WITH_TIM + isFlipped) | cdBodyHasTransformFlag;
							entry->m_timeOfSeparatingNormal = hkTime(-1.0f);
							entry->m_separatingNormal.setZero();

							hkpAgentData* agentData = reinterpret_cast<hkpAgentData*>(entry+1);
							hkpAgentData* agentEnd  = dispatcher->getAgent3CreateFunc( entry->m_agentType )( *in, entry, agentData );
							hkUlong size = hkGetByteOffset( entry, agentEnd );
							entry->m_size = hkUchar(size);
							HK_ASSERT2( 0xf0100404, size <= hkAgent3::MAX_SIZE, "Your agent's initial size is too big" );
						}
						else
						{
							hkpAgent1nMachinePaddedEntry* entry = reinterpret_cast<hkpAgent1nMachinePaddedEntry*>( currentData );

							entry->m_streamCommand = hkUchar(hkAgent3::STREAM_CALL + isFlipped) | cdBodyHasTransformFlag;

							hkpAgentData* agentData = reinterpret_cast<hkpAgentData*>(entry+1);
							hkpAgentData* agentEnd  = dispatcher->getAgent3CreateFunc( entry->m_agentType )( *in, entry, agentData );
							hkUlong size = hkGetByteOffset( entry, agentEnd );
							entry->m_size = hkUchar(size);
							HK_ASSERT2( 0xf0100405, size <= hkAgent3::MAX_SIZE, "Your agent's initial size is too big" );
						}
					}
					else
					{
						// We create a persistent dummy entry. This is to avoid calling collisionFilters each time this agent track is processed.
						hkpAgent1nMachinePaddedEntry* h = reinterpret_cast<hkpAgent1nMachinePaddedEntry*>( currentData );
						*(int *)h = 0;	// clear the first 4 bytes of h
						shapeKeyPair.writeTo( h->m_shapeKeyPair );
						h->m_size = hkUchar(hkSizeOf(hkpAgent1nMachinePaddedEntry));
					}
					containerIterator.advance();
				}
			}
		}

#if defined(DISPLAY_TRIANGLE_ENABLED)
		if( shapeKeyPair.m_shapeKeyB != HK_INVALID_SHAPE_KEY)
		{
			//hkpRigidBody* body = hkpGetRigidBody(input.m_bodyA->getRootCollidable());
			//hkpWorld* world = body->getWorld();
			//hkUint32 uid = world->m_lastEntityUid;
			//if ( uid == body->getUid()+1)
			{
				hkAgent1nMachine_displayTriangle( input.m_bodyB->getTransform(), shapeContainerB, shapeKeyPair.m_shapeKeyB );
			}
		}
#endif

		{		
			//
			//  We created or copied the entry, and process it below
			//

			bool thisChildHasTransformApplied = false;

			hkAgent3::StreamCommand command = hkAgent3::StreamCommand(reinterpret_cast<hkpAgentEntry*>(currentData)->m_streamCommand);
commandSwitch:
			switch ( command )
			{
				case hkAgent3::STREAM_CALL_AGENT_WITH_TRANSFORM:
				case hkAgent3::STREAM_CALL_WITH_TRANSFORM:
				case hkAgent3::STREAM_CALL_FLIPPED_WITH_TRANSFORM:
				case hkAgent3::STREAM_CALL_WITH_TIM_WITH_TRANSFORM:
				case hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED_WITH_TRANSFORM:
					{
						EXTRACT_CHILD_SHAPES(containerIterator, childBodyA, childBodyB);

						// strip hkpTransformShapes
						hkPadSpu<hkUchar> dummyCdBodyHasTransformFlag = 0;
						modifiedBodyBPtr = hkAgentMachine_processTransformedShapes(modifiedBodyBPtr, cdBodyForTransformedShapeB, transformForTransformedShapeB, numSlotsForTransformedCdBodies, dummyCdBodyHasTransformFlag);
						ON_NM_MACHINE( modifiedBodyAPtr = hkAgentMachine_processTransformedShapes(modifiedBodyAPtr, cdBodyForTransformedShapeA, transformForTransformedShapeA, numSlotsForTransformedCdBodies, dummyCdBodyHasTransformFlag) );

						input.m_aTb.setMulInverseMul(modifiedBodyAPtr->getTransform(), modifiedBodyBPtr->getTransform());

						command = static_cast<hkAgent3::StreamCommand> ( static_cast<hkUchar>(command) & static_cast<hkUchar>(~hkAgent3::TRANSFORM_FLAG) );

						thisChildHasTransformApplied = true;
						prevFlippedInputInitialized = false;
						prevInputInitialized = false;

						goto commandSwitch;

					}

				case hkAgent3::STREAM_CALL_AGENT:
					{	
						hkpAgent1nMachinePaddedEntry* paddedEntry = reinterpret_cast<hkpAgent1nMachinePaddedEntry*>(currentData);
						hkpAgentData* agentData = reinterpret_cast<hkpAgentData*>(paddedEntry+1);
#if !defined(HK_PLATFORM_SPU)
						EXTRACT_CHILD_SHAPES(containerIterator, childBodyA, childBodyB);

						hkpCollisionAgent* agent = hkAgent3Bridge::getChildAgent(agentData);
						agent->processCollision( *modifiedBodyAPtr, *modifiedBodyBPtr, *input.m_input, output );
#else
						HK_ASSERT2( 0xf0345466, 0, "This type of collision is not handled on spu, consider setting m_forceCollideOntoPpu to true on one of the entities involved" );
#endif
						currentData = hkAgent3Bridge::getAgentDataEnd(agentData);
						
						break;
					}

				case hkAgent3::STREAM_CALL:
					{
						//
						//	Get child shape
						//
						hkpAgent1nMachinePaddedEntry* entry = reinterpret_cast<hkpAgent1nMachinePaddedEntry*>(currentData);

						EXTRACT_CHILD_SHAPES(containerIterator, childBodyA, childBodyB);

						//
						//	call agent
						//
						hkpAgentData* agentData = reinterpret_cast<hkpAgentData*>(entry+1);
						currentData = dispatcher->getAgent3ProcessFunc( entry->m_agentType )( input, entry, agentData, HK_NULL, output );
						entry->m_size = hkUchar(hkGetByteOffset( entry, currentData ));
						HK_ASSERT2( 0xf0010405, entry->m_size <= hkAgent3::MAX_SIZE , "Your agent's initial size is too big" );


						break;
					}
				case hkAgent3::STREAM_CALL_FLIPPED:
					{
						hkpAgent3ProcessInput flippedInput;
						hkAgent1nMachine_flipInput( input, flippedInput);

						//
						//	Get child shape
						//
						hkpAgent1nMachinePaddedEntry* entry = reinterpret_cast<hkpAgent1nMachinePaddedEntry*>(currentData);

						EXTRACT_CHILD_SHAPES(containerIterator, childBodyA, childBodyB);

						// Store previous TOI time
						hkTime oldToi = output.m_toi.m_time;

						//
						//	call agent
						//
						hkpAgentData* agentData = reinterpret_cast<hkpAgentData*>(entry+1);
						hkpProcessCdPoint* oldResultBegin = output.m_firstFreeContactPoint;
						currentData = dispatcher->getAgent3ProcessFunc( entry->m_agentType )( flippedInput, entry, agentData, HK_NULL, output );
						entry->m_size = hkUchar(hkGetByteOffset( entry, currentData ));
						HK_ASSERT2( 0xf0011405, entry->m_size <= hkAgent3::MAX_SIZE , "Your agent's initial size is too big" );

						//
						//	Flip all new normals
						//
						while( oldResultBegin < output.m_firstFreeContactPoint)
						{
							oldResultBegin->m_contact.flip();
							oldResultBegin++;
						}

						if( oldToi != output.m_toi.m_time )
						{
							output.m_toi.flip();
						}

						break;
					}
				
				case hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED:
				{
					hkpAgent3ProcessInput flippedInput;
					hkAgent1nMachine_flipInput( input, flippedInput);
					hkpAgent1nMachineTimEntry* entry = reinterpret_cast<hkpAgent1nMachineTimEntry*>(currentData);
					
					//
					//	validate separating plane
					//
					hkSimdReal distAtT1;
					if ( ! (entry->m_timeOfSeparatingNormal == flippedInput.m_input->m_stepInfo.m_startTime) )
					{
						const hkpCollisionQualityInfo& ci = *flippedInput.m_input->m_collisionQualityInfo;
						if ( !ci.m_useContinuousPhysics.val())
						{
							entry->m_timeOfSeparatingNormal = flippedInput.m_input->m_stepInfo.m_endTime;
							distAtT1.load<1>(&ci.m_minSeparation);
							distAtT1.mul(hkSimdReal_Half);
							entry->m_separatingNormal.setXYZ_W(hkVector4::getZero(), distAtT1);
							goto PROCESS_AT_T1_FLIPPED;
						}

						HK_INTERNAL_TIMER_BEGIN("recalcT0", HK_NULL);
						if ( !prevFlippedInputInitialized)
						{
							prevFlippedInputInitialized = !thisChildHasTransformApplied;
							hkAgent1nMachine_initInputAtTime( flippedInput, prevFlippedTemp, prevFlippedInput );
							prevFlippedTemp.m_bodyB.setShape( flippedInput.m_bodyB->getShape(), flippedInput.m_bodyB->m_shapeKey );
						}

						EXTRACT_CHILD_SHAPES(containerIterator, childBodyA, childBodyB);
						prevFlippedTemp.m_bodyA.setShape( modifiedBodyBPtr->m_shape, modifiedBodyBPtr->m_shapeKey );
						ON_NM_MACHINE( prevFlippedTemp.m_bodyB.setShape( modifiedBodyAPtr->m_shape, modifiedBodyAPtr->m_shapeKey ) );

						hkpAgentData* agentData = reinterpret_cast<hkpAgentData*>(entry+1);
						dispatcher->getAgent3SepNormalFunc( entry->m_agentType )( prevFlippedInput, entry, agentData, entry->m_separatingNormal);
						HK_INTERNAL_TIMER_END();
					}
						// optimistically set the separatingNormal time to the end of the step
					entry->m_timeOfSeparatingNormal = flippedInput.m_input->m_stepInfo.m_endTime;

					{
						const hkSimdReal worstCaseDeltaDist  = flippedInput.m_linearTimInfo.dot4xyz1( entry->m_separatingNormal );
						distAtT1 = (entry->m_separatingNormal.getW() - worstCaseDeltaDist);
					}

					//
					//	Solve for traditional tims
					//
					if ( distAtT1.isGreaterEqual(hkSimdReal::fromFloat(flippedInput.m_input->getTolerance())) )
					{
						entry->m_separatingNormal.setW(distAtT1);

						if ( !entry->m_numContactPoints )
						{
							currentData = hkAddByteOffset(currentData, entry->m_size );
						}
						else
						{
							hkpAgentData* agentData = reinterpret_cast<hkpAgentData*>(entry+1);
							currentData = dispatcher->getAgent3CleanupFunc( entry->m_agentType )( entry, agentData, flippedInput.m_contactMgr, *output.m_constraintOwner.val() );
							entry->m_size = hkUchar(hkGetByteOffset( entry, currentData ));
						}
						ON_DEBUG( numTim++ );
						break;
					}
PROCESS_AT_T1_FLIPPED:
					distAtT1.store<1>(&(flippedInput.m_distAtT1.ref()));

					EXTRACT_CHILD_SHAPES(containerIterator, childBodyA, childBodyB);

					hkpProcessCdPoint* oldResultBegin = output.m_firstFreeContactPoint;

					// Store previous TOI time
					hkTime oldToi = output.m_toi.m_time;

					hkpAgentData* agentData = reinterpret_cast<hkpAgentData*>(entry+1);
					currentData = dispatcher->getAgent3ProcessFunc( entry->m_agentType )( flippedInput, entry, agentData, &entry->m_separatingNormal, output );
					entry->m_size = hkUchar(hkGetByteOffset( entry, currentData));
					HK_ASSERT2( 0xf0011405, entry->m_size <= hkAgent3::MAX_SIZE , "Your agent's initial size is too big" );

					//
					//	Flip all normals
					//
					while( oldResultBegin < output.m_firstFreeContactPoint)
					{
						oldResultBegin->m_contact.flip();
						oldResultBegin++;
					}

					if( oldToi != output.m_toi.m_time )
					{
						output.m_toi.flip();
					}

					break;
				}


				case hkAgent3::STREAM_CALL_WITH_TIM:
				{
					hkpAgent1nMachineTimEntry* entry = reinterpret_cast<hkpAgent1nMachineTimEntry*>(currentData);
					
					//
					//	validate separating plane
					//
					hkSimdReal distAtT1;
 					if ( ! (entry->m_timeOfSeparatingNormal == input.m_input->m_stepInfo.m_startTime) )
					{
						if ( entry->m_timeOfSeparatingNormal >= input.m_input->m_stepInfo.m_endTime )
						{
							hkReal backDt = entry->m_timeOfSeparatingNormal - input.m_input->m_stepInfo.m_endTime;
							hkVector4 backTim; 
							hkSweptTransformUtil::calcTimInfo( *input.m_bodyA->getMotionState(), *input.m_bodyB->getMotionState(), backDt, backTim);
							// walk backwards in time
							const hkSimdReal backLinear = backTim.dot<3>( entry->m_separatingNormal );
							const hkSimdReal backAng = backTim.getW();
							distAtT1 = (entry->m_separatingNormal.getW() + backLinear - backAng);
						}
						else
						{
						    const hkpCollisionQualityInfo& ci = *input.m_input->m_collisionQualityInfo;
						    if ( !ci.m_useContinuousPhysics.val() )
						    {
							    entry->m_timeOfSeparatingNormal = input.m_input->m_stepInfo.m_endTime;
							    distAtT1.load<1>(&ci.m_minSeparation);
								distAtT1.mul(hkSimdReal::fromFloat(hkReal(0.1f)));
							    entry->m_separatingNormal.setXYZ_W(hkVector4::getZero(), distAtT1);
							    goto PROCESS_AT_T1;
						    }
						    HK_INTERNAL_TIMER_BEGIN("recalcT0", HK_NULL);
						    if ( !prevInputInitialized)
						    {
							    prevInputInitialized = !thisChildHasTransformApplied;
							    hkAgent1nMachine_initInputAtTime( input, prevTemp, prevInput );
						    }
    
						    EXTRACT_CHILD_SHAPES(containerIterator, childBodyA, childBodyB);
							prevTemp.m_bodyB.setShape( modifiedBodyBPtr->m_shape, modifiedBodyBPtr->m_shapeKey );
							ON_NM_MACHINE( prevTemp.m_bodyA.setShape( modifiedBodyAPtr->m_shape, modifiedBodyAPtr->m_shapeKey ) );
    
    
						    hkpAgentData* agentData = reinterpret_cast<hkpAgentData*>(entry+1);
						    dispatcher->getAgent3SepNormalFunc( entry->m_agentType )( prevInput, entry, agentData, entry->m_separatingNormal);
						    HK_INTERNAL_TIMER_END();
					    }
					}
						// optimistically set the separatingNormal time to the end of the step
					entry->m_timeOfSeparatingNormal = input.m_input->m_stepInfo.m_endTime;

					{
						const hkSimdReal worstCaseDeltaDist  = input.m_linearTimInfo.dot4xyz1( entry->m_separatingNormal );
						distAtT1 = (entry->m_separatingNormal.getW() - worstCaseDeltaDist);
					}

					//
					//	Solve for traditional tims
					//
					if ( distAtT1.isGreaterEqual(hkSimdReal::fromFloat(input.m_input->getTolerance())) )
					{
						entry->m_separatingNormal.setW(distAtT1);

						if ( !entry->m_numContactPoints )
						{
							currentData = hkAddByteOffset(currentData, entry->m_size );
						}
						else
						{
							hkpAgentData* agentData = reinterpret_cast<hkpAgentData*>(entry+1);
							currentData = dispatcher->getAgent3CleanupFunc( entry->m_agentType )( entry, agentData, input.m_contactMgr, *output.m_constraintOwner.val() );
							entry->m_size = hkUchar(hkGetByteOffset( entry, currentData ));
						}
						ON_DEBUG( numTim++ );
						break;
					}
PROCESS_AT_T1:
					distAtT1.store<1>(&(input.m_distAtT1.ref()));

					EXTRACT_CHILD_SHAPES(containerIterator, childBodyA, childBodyB);

					hkpAgentData* agentData = reinterpret_cast<hkpAgentData*>(entry+1);
 					currentData = dispatcher->getAgent3ProcessFunc( entry->m_agentType )( input, entry, agentData, &entry->m_separatingNormal, output );
					entry->m_size = hkUchar(hkGetByteOffset( entry, currentData));
					HK_ASSERT2( 0xf0011405, entry->m_size <= hkAgent3::MAX_SIZE , "Your agent's initial size is too big" );

					break;
				}

			case hkAgent3::STREAM_NULL:
				{
					currentData = hkAddByteOffset( currentData, sizeof( hkpAgent1nMachinePaddedEntry ) );
					//entry->m_size = sizeof( hkpAgent1nMachinePaddedEntry );
					break;
				}

			case hkAgent3::STREAM_END:
				{
					currentData = hkAddByteOffset( currentData, sizeof( hkpAgent1nMachinePaddedEntry ) );
					break;
				}

			default: 
				HK_ASSERT2( 0xf0000001,0, "Unknown command in hkCdAgentStream");
			}
		}
		//
		//	Check whether we still have data in the read buffer
		//
checkReadSector:
		HK_ASSERT(0x2f958137, readData <= readEnd);
		
		if ( readData == readEnd )
		{
#if !defined(HK_PLATFORM_SPU)
			{
				//	Free existing read buffer
				if ( input.m_input->m_spareAgentSector )
				{
					delete readSector;
				}
				else
				{
					input.m_input->m_spareAgentSector = readSector;
				}

				// Get a new one
				if ( nextReadSectorIndex < agentTrack.m_sectors.getSize() )
				{
					readSector = agentTrack.m_sectors[nextReadSectorIndex];
					readData = readSector->getBegin();
					readEnd    = readSector->getEnd();
					nextReadSectorIndex++;
				}
				else
				{
					// finished
					break;
				}
			}
#else
			{
				// Get a new one
				if ( nextReadSectorIndex < agentTrack.m_sectors.getSize() )
				{
					// swap current and prefetch
					prefetchReadDmaGroup ^= 1;
					hkAlgorithm::swap( readSector, prefetchReadSector );

					if ( nextReadSectorIndex+1 < agentTrack.m_sectors.getSize() ) // prefetch next if not at end
					{
						hkSpuDmaManager::getFromMainMemory( prefetchReadSector, pReadSectorBuffer[ nextReadSectorIndex+1 ], sizeof(hkpAgent1nSector), hkSpuDmaManager::READ_COPY, prefetchReadDmaGroup );
						HK_SPU_DMA_DEFER_FINAL_CHECKS_UNTIL_WAIT( pReadSectorBuffer[ nextReadSectorIndex+1 ], prefetchReadSector, sizeof(hkpAgent1nSector) );
					}

					// now wait for the working buffer
					hkSpuDmaManager::waitForDmaCompletion( prefetchReadDmaGroup ^1 );
					DEALLOCATE_SECTOR( pReadSectorBuffer[ nextReadSectorIndex ] ); // have spu copy
					
					readData = readSector->getBegin();
					readEnd    = readSector->getEnd();
					nextReadSectorIndex += 1;
				}
				else
				{
					// finished
					break;
				}
			}
#endif
			HK_ASSERT( 0xf0346fde, nextReadSectorIndex <= MAX_NUM_SECTORS );
		}

		//
		//	Check write buffer
		//

		if ( hkAddByteOffset( currentData, hkAgent3::MAX_SIZE ) <= currentDataEnd )
		{
			continue;
		}

		if ( !inExtensionBuffer )
		{
			inExtensionBuffer = true;
			bytesUsedInCurrentSector = int(hkGetByteOffset( currentSector->getBegin(), currentData ));
			HK_ASSERT2( 0xf0100405, bytesUsedInCurrentSector <= hkpAgent1nSector::NET_SECTOR_SIZE, "Overflow" );

			currentData = &extensionBuffer[0];
			currentDataEnd = hkAddByteOffset( &extensionBuffer[0], hkAgent3::MAX_SIZE );
			continue;
		}
		//	else
		
		int bytesUsedInExtensionBuffer = int(hkGetByteOffset( &extensionBuffer[0], currentData ));
		{
			// if the extension fits into the normal sector, copy it there
			{
				int freeBytes = currentSector->getCapacity() - bytesUsedInCurrentSector;
				if ( bytesUsedInExtensionBuffer <= freeBytes )
				{
					// great, it fits, use it
					void* dst = hkAddByteOffset( currentSector->getBegin(), bytesUsedInCurrentSector);
					hkString::memCpy16NonEmpty( dst, &extensionBuffer[0], bytesUsedInExtensionBuffer>>4 );
					bytesUsedInCurrentSector += bytesUsedInExtensionBuffer;

#if defined(HK_1N_MACHINE_SUPPORTS_WELDING)
					fixupPotentialContactPointers( extensionBuffer, dst, bytesUsedInExtensionBuffer, output.m_potentialContacts );
#endif

					// if there is still space try to fill a new sector
					if ( bytesUsedInCurrentSector <	currentSector->getCapacity() )	
					{
						// reset to the start of the extension
						currentData = &extensionBuffer[0];
						continue;
					}
					inExtensionBuffer = false;
					HK_ON_DEBUG( currentData = 0 );
					HK_ON_DEBUG( currentDataEnd = 0 );
				}
			}
		}


		//
		//	Sector full, store
		//
		currentSector->m_bytesAllocated = bytesUsedInCurrentSector;

#if !defined(HK_PLATFORM_SPU)
		{
				// Store in existing sector file
			{
				if ( currentSectorIndex >= nextReadSectorIndex )
				{
					agentTrack.m_sectors.expandAt( nextReadSectorIndex, 1 );
					nextReadSectorIndex++;
				}
				agentTrack.m_sectors[currentSectorIndex] = currentSector;
				HK_ASSERT( 0xf0346fde, currentSectorIndex < MAX_NUM_SECTORS );

				currentSectorIndex++;
			}
			
			// Get a new one
			{
				currentSector = (input.m_input->m_spareAgentSector)? input.m_input->m_spareAgentSector : new hkpAgent1nSector;
				input.m_input->m_spareAgentSector = HK_NULL;
				currentData = currentSector->getBegin();
				currentDataEnd = currentSector->getEndOfCapacity();
			}
		}
#else
		{
			pWriteSectorBuffer[ numWrittenSectors ] = writeSectorOnPpu;

			hkSpuDmaManager::putToMainMemory( writeSectorOnPpu, currentSector, sizeof(hkpAgent1nSector), hkSpuDmaManager::WRITE_NEW, postWriteDmaGroup^1 );
			HK_SPU_DMA_DEFER_FINAL_CHECKS_UNTIL_WAIT( writeSectorOnPpu, currentSector, sizeof(hkpAgent1nSector) );

			if( numWrittenSectors != 0 ) // wait for previous write to complete
			{
				hkSpuDmaManager::waitForDmaCompletion( postWriteDmaGroup );
			}


			// now write sector is free to use
			postWriteDmaGroup ^= 1;
			hkAlgorithm::swap( currentSector, postWriteSector );			
			currentData = currentSector->getBegin();
			currentDataEnd = currentSector->getEndOfCapacity();
			numWrittenSectors += 1;
			HK_ASSERT( 0xf0346fde, numWrittenSectors < MAX_NUM_SECTORS );

			// get a new write sector for the next sector
			writeSectorOnPpu = hkAllocateChunk<hkpAgent1nSector>(1, HK_MEMORY_CLASS_DYNAMICS);
		}
#endif
		// put overflow of last iteration to the start of our new sector
		if ( inExtensionBuffer )
		{
			hkString::memCpy16NonEmpty( currentData, &extensionBuffer[0], bytesUsedInExtensionBuffer>>4 );

#if defined(HK_1N_MACHINE_SUPPORTS_WELDING)
			fixupPotentialContactPointers( extensionBuffer, currentData, bytesUsedInExtensionBuffer, output.m_potentialContacts );
#endif

			currentData = hkAddByteOffset( currentData, bytesUsedInExtensionBuffer );
			inExtensionBuffer = false;
		}
	}	// while(1)

	if ( !inExtensionBuffer )
	{
		bytesUsedInCurrentSector = int(hkGetByteOffset( currentSector->getBegin(), currentData ));
	}
	else
	{
#ifdef HK_REAL_IS_DOUBLE
		//Edge condition-- this occurs often on the double precision build
		//when the sector buffer fills just before a STREAM_END command
		if( bytesUsedInCurrentSector == hkpAgent1nSector::NET_SECTOR_SIZE )
		{
			{
				currentSector->m_bytesAllocated = bytesUsedInCurrentSector;

				if ( currentSectorIndex >= nextReadSectorIndex )
				{
					agentTrack.m_sectors.expandAt( nextReadSectorIndex, 1 );
					nextReadSectorIndex++;
				}

				agentTrack.m_sectors[currentSectorIndex] = currentSector;
				HK_ASSERT( 0xf0346fde, currentSectorIndex < MAX_NUM_SECTORS );
				currentSectorIndex++;
			}
			
			// Get a new one
			{
				currentSector = (input.m_input->m_spareAgentSector)? input.m_input->m_spareAgentSector : new hkpAgent1nSector;
				input.m_input->m_spareAgentSector = HK_NULL;
				bytesUsedInCurrentSector = 0;
			}
		}
#endif

		int bytesUsedInExtensionBuffer = int(hkGetByteOffset( &extensionBuffer[0], currentData ));
		HK_ASSERT( 0xf034defe, bytesUsedInExtensionBuffer==sizeof(hkpAgent1nMachinePaddedEntry) ); // end marker
		void* dst = hkAddByteOffset( currentSector->getBegin(), bytesUsedInCurrentSector);
		hkString::memCpy16NonEmpty( dst, &extensionBuffer[0], bytesUsedInExtensionBuffer>>4 );

#if defined(HK_1N_MACHINE_SUPPORTS_WELDING)
		// Update the agent entry and data for potential contacts.

		hkpProcessCollisionOutput::PotentialInfo* potentialInfo = output.m_potentialContacts;
		
		if ( potentialInfo )
		{
			// Is there a potential contact?
			if ( potentialInfo->m_firstFreePotentialContact > &potentialInfo->m_potentialContacts[0] )
			{
				// If there is, make sure the agent pointers of the last potential contact correctly point
				// to the repositioned agent.
				hkpProcessCollisionOutput::ContactRef* lastContactRef = potentialInfo->m_firstFreePotentialContact - 1;
				hkLong offsetInExtensionBuffer = hkGetByteOffset( &extensionBuffer[0], lastContactRef->m_agentEntry );
				// Check that the pointer points into the extension buffer.
				if ( 0 <= offsetInExtensionBuffer && offsetInExtensionBuffer < bytesUsedInExtensionBuffer )
				{
					HK_ASSERT2(0xad000000, false, "We should never hit this ?");
					lastContactRef->m_agentEntry = (hkpAgentEntry*) hkAddByteOffset( dst, offsetInExtensionBuffer );
					lastContactRef->m_agentData = (hkpAgentData*) hkAddByteOffset( dst, hkGetByteOffset( &extensionBuffer[0], lastContactRef->m_agentData ) );
				}
			}
		}
#endif

		bytesUsedInCurrentSector += bytesUsedInExtensionBuffer;
	}
	HK_ASSERT(0x2d141c6, bytesUsedInCurrentSector <= hkpAgent1nSector::NET_SECTOR_SIZE );
	currentSector->m_bytesAllocated = bytesUsedInCurrentSector;

	//
	//	Store very last sector and finalize agentTrack
	//
	{
#if !defined(HK_PLATFORM_SPU)
		agentTrack.m_sectors.setSize( currentSectorIndex+1 );
		agentTrack.m_sectors[currentSectorIndex] = currentSector;
		HK_ASSERT( 0xf0346fde, currentSectorIndex <= MAX_NUM_SECTORS );
#else
		pWriteSectorBuffer[ numWrittenSectors++ ] = writeSectorOnPpu;
		HK_ASSERT( 0xf0346fde, numWrittenSectors <= MAX_NUM_SECTORS );

		hkSpuDmaManager::putToMainMemory( writeSectorOnPpu, currentSector, sizeof(hkpAgent1nSector), hkSpuDmaManager::WRITE_NEW, hk1nMachineDmaGroups::WRITE_SECTOR_DMA_GROUP );
		HK_SPU_DMA_DEFER_FINAL_CHECKS_UNTIL_WAIT( writeSectorOnPpu, currentSector, sizeof(hkpAgent1nSector) );

		// write back sectors array
		agentSectorsOnSpu.overwriteData(pWriteSectorBuffer, numWrittenSectors);
		agentSectorsOnSpu.putArrayToMainMemoryNotInplace( agentTrack.m_sectors, hk1nMachineDmaGroups::WRITE_SECTOR_DMA_GROUP );
#endif
	}

	if ( numTim ) { HK_MONITOR_ADD_VALUE( "numTim", float(numTim), HK_MONITOR_TYPE_INT); }

#if defined(HK_1N_MACHINE_SUPPORTS_WELDING)
	//	revert the input
	HK_ASSERT2(0xad983433, inputIn.m_bodyB == childBodyB.getParent(), "Wrong body in input.");
	input.m_bodyB = inputIn.m_bodyB;
	input.m_aTb = inputIn.m_aTb;

	//
	//	Welding, but only if we own the potential information
	//
	if ( output.m_potentialContacts )
	{
		if ( output.m_potentialContacts->m_firstFreePotentialContact > &output.m_potentialContacts->m_potentialContacts[0])
		{
			HK_INTERNAL_TIMER_BEGIN("Welding", HK_NULL);
			hkAgent1nMachine_Weld( input, shapeContainerB, output );
			HK_INTERNAL_TIMER_END();
		}
	}
	output.m_potentialContacts = oldPotentialInfo;
#endif

#if defined(HK_PLATFORM_SPU)
	hkSpuDmaManager::waitForAllDmaCompletion();
	hkDeallocateStack(buffer);
	hkDeallocateStack(pReadSectorBuffer);
	hkDeallocateStack(pWriteSectorBuffer);
#endif
}


ON_1N_MACHINE( hkpAgentData* hkAgent1nMachine_UpdateShapeCollectionFilterVisitor(hkpAgent1nMachine_VisitorInput& vin, hkpAgent1nMachineEntry* entry, hkpAgentData* agentData ) )
ON_NM_MACHINE( hkpAgentData* hkAgentNmMachine_UpdateShapeCollectionFilterVisitor(hkpAgent1nMachine_VisitorInput& vin, hkpAgent1nMachineEntry* entry, hkpAgentData* agentData ) )
{
	HK_ASSERT2(0xad876ddd, entry->m_streamCommand != hkAgent3::STREAM_NULL, "entry's command cannot be STREAM_NULL");
	hkpShapeKeyPair shapeKeyPair = entry->m_shapeKeyPair;

	ON_1N_MACHINE( if ( vin.m_input->m_filter->isCollisionEnabled( *vin.m_input, *vin.m_bodyA, *vin.m_collectionBodyB, *vin.m_containerShapeB, shapeKeyPair.m_shapeKeyB ) ) )
	ON_NM_MACHINE( if ( vin.m_input->m_filter->isCollisionEnabled( *vin.m_input, *vin.m_bodyA, *vin.m_collectionBodyB, *vin.m_containerShapeA, *vin.m_containerShapeB, shapeKeyPair.m_shapeKeyA, shapeKeyPair.m_shapeKeyB ) ) )
	{
		// no filter change, run the agent's handler and just return the full size
		hkAgent3::UpdateFilterFunc func = vin.m_input->m_dispatcher->getAgent3UpdateFilterFunc( entry->m_agentType );
		if (func)
		{
			hkpShapeBuffer shapeBufferB;
			const hkpShape* shapeB = vin.m_containerShapeB->getChildShape( shapeKeyPair.m_shapeKeyB, shapeBufferB);
			ON_NM_MACHINE( hkpShapeBuffer shapeBufferA );
			ON_NM_MACHINE( const hkpShape* shapeA = vin.m_containerShapeA->getChildShape( shapeKeyPair.m_shapeKeyA, shapeBufferA) );

			hkpCdBody childCdBodyB(vin.m_collectionBodyB);
			childCdBodyB.setShape(shapeB, shapeKeyPair.m_shapeKeyB);
			ON_NM_MACHINE( hkpCdBody childCdBodyA(vin.m_bodyA) );
			ON_NM_MACHINE( childCdBodyA.setShape(shapeA, shapeKeyPair.m_shapeKeyA) );

			hkMotionState transformForTransformedShapeB[4];
			ON_NM_MACHINE(hkMotionState transformForTransformedShapeA[4]);
			hkpCdBody cdBodyForTransformedShapeB[4];
			ON_NM_MACHINE(hkpCdBody cdBodyForTransformedShapeA[4]);

			const hkpCdBody* firstNonTransformBodyB = &childCdBodyB;
			ON_1N_MACHINE( const hkpCdBody* firstNonTransformBodyA = vin.m_bodyA );
			ON_NM_MACHINE( const hkpCdBody* firstNonTransformBodyA = &childCdBodyA );

			if (hkAgent3::TRANSFORM_FLAG & entry->m_streamCommand)
			{
				hkPadSpu<hkUchar> dummyCdBodyHasTransformFlag = 0;
				firstNonTransformBodyB = hkAgentMachine_processTransformedShapes(firstNonTransformBodyB, cdBodyForTransformedShapeB, transformForTransformedShapeB, 4, dummyCdBodyHasTransformFlag);
				ON_NM_MACHINE( firstNonTransformBodyA = hkAgentMachine_processTransformedShapes(firstNonTransformBodyA, cdBodyForTransformedShapeA, transformForTransformedShapeA, 4, dummyCdBodyHasTransformFlag) );
			}

			HK_COMPILE_TIME_ASSERT( (hkAgent3::STREAM_CALL | 0x1) == (hkAgent3::STREAM_CALL_FLIPPED)) ;
			HK_COMPILE_TIME_ASSERT( (hkAgent3::STREAM_CALL_WITH_TIM | 0x1) == (hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED)) ;

			HK_COMPILE_TIME_ASSERT( hkAgent3::STREAM_CALL_FLIPPED == (hkAgent3::STREAM_CALL_FLIPPED_WITH_TRANSFORM & 0x7)) ;
			HK_COMPILE_TIME_ASSERT( hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED == (hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED_WITH_TRANSFORM & 0x7)) ;

			const bool flipped = (entry->m_streamCommand & 0x7) == hkAgent3::STREAM_CALL_FLIPPED || (entry->m_streamCommand & 0x7) == hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED;

			if (!flipped)
			{
				func( entry, agentData, *firstNonTransformBodyA, *firstNonTransformBodyB, *vin.m_input, vin.m_contactMgr, *vin.m_constraintOwner );
			}
			else
			{
				func( entry, agentData, *firstNonTransformBodyB, *firstNonTransformBodyA, *vin.m_input, vin.m_contactMgr, *vin.m_constraintOwner );
			}
		}
		
		return hkAddByteOffset( entry, entry->m_size );
	}
	else
	{
		// now the filter is changed, just delete the agent, it will be properly recreated the next time
		vin.m_input->m_dispatcher->getAgent3DestroyFunc( entry->m_agentType )( entry, agentData, vin.m_contactMgr, *vin.m_constraintOwner, vin.m_input->m_dispatcher );
		return entry;
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
