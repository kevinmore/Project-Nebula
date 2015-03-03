/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Monitor/hkMonitorStream.h>

#if defined(HK_PLATFORM_SPU)
#include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#endif

#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Spu/hkpSpuConfig.h>
#include <Physics2012/Internal/Collide/Gjk/hkpGsk.h>

#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>

#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>

#include <Physics2012/Collide/Agent/hkpCollisionAgentConfig.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>

#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nTrack.h>
#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachine.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>

#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>

#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>

// Causes dep on hkvisualize. Only include if you use it in external builds.
// #include <hkvisualize/hkDebugDisplay.h>

HK_COMPILE_TIME_ASSERT( sizeof(hkpAgentEntry) == 4 );

HK_COMPILE_TIME_ASSERT( (sizeof(hkpAgent1nMachinePaddedEntry) % 16) == 0 );
HK_COMPILE_TIME_ASSERT( sizeof(hkpAgent1nMachinePaddedEntry) == HK_REAL_ALIGNMENT ); // see hkpAgent3::SIZE_OF_HEADER
HK_COMPILE_TIME_ASSERT( (sizeof(hkpAgent1nMachineTimEntry) % 32) == 0 );
HK_COMPILE_TIME_ASSERT( sizeof(hkpAgent1nMachineTimEntry) == (2*HK_REAL_ALIGNMENT) );  // see hkpAgent3::MAX_NET_SIZE
HK_COMPILE_TIME_ASSERT(sizeof( hkpAgent1nSector().m_data ) == hkpAgent1nSector::NET_SECTOR_SIZE );

#define MAX_NUM_SECTORS HK_MAX_AGENTS_IN_1N_MACHINE / HK_AGENT3_FEWEST_AGENTS_PER_1N_SECTOR

void hkAgent1nMachine_Destroy( hkpAgent1nTrack& agentTrack, hkpCollisionDispatcher* dispatcher, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner )
{
#if !defined(HK_PLATFORM_SPU)
	hkpAgent1nSector** readSectors = agentTrack.m_sectors.begin(); 
	hkpAgent1nSector* readSector   = readSectors[0];
#else
	hkpAgent1nSector** readSectors = hkAllocateStack<hkpAgent1nSector*>(MAX_NUM_SECTORS, "1n-machine read sector pntrs");

	HK_SPU_STACK_POINTER_CHECK();

	int numSectors = agentTrack.m_sectors.getSize();
	hkpAgent1nSector** sectorsPPu = agentTrack.m_sectors.begin();
	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( readSectors, sectorsPPu, HK_NEXT_MULTIPLE_OF(16, sizeof(void*)*numSectors), hkSpuDmaManager::READ_ONLY );
	HK_ALIGN16(hkpAgent1nSector sectorBuffer);
	hkpAgent1nSector* readSector = &sectorBuffer;
	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( &sectorBuffer, readSectors[0], sizeof(hkpAgent1nSector), hkSpuDmaManager::READ_COPY );
	HK_SPU_DMA_PERFORM_FINAL_CHECKS( readSectors[0], &sectorBuffer, sizeof(hkpAgent1nSector) );
#endif

	hkpAgentData*     readData   = readSector->getBegin();
	hkpAgentData*     readEnd    = readSector->getEnd();
	int     readSectorIndex = 0;

	while(1)
	{

		hkpAgent1nMachineEntry* entry = reinterpret_cast<hkpAgent1nMachineEntry*>(readData);
		hkAgent3::StreamCommand command = hkAgent3::StreamCommand(entry->m_streamCommand);

		switch ( command )
		{
			case hkAgent3::STREAM_CALL_AGENT:
			case hkAgent3::STREAM_CALL_AGENT_WITH_TRANSFORM:
			case hkAgent3::STREAM_CALL:
			case hkAgent3::STREAM_CALL_FLIPPED:
			case hkAgent3::STREAM_CALL_WITH_TRANSFORM:
			case hkAgent3::STREAM_CALL_FLIPPED_WITH_TRANSFORM:
			{
				hkpAgentData* agentData = hkAddByteOffset( readData, hkSizeOf( hkpAgent1nMachinePaddedEntry ) );
				readData = hkAddByteOffset( readData, entry->m_size );
				dispatcher->getAgent3DestroyFunc( entry->m_agentType )( entry, agentData, mgr, constraintOwner, dispatcher );
				break;
			}
			case hkAgent3::STREAM_CALL_WITH_TIM:
			case hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED:
			case hkAgent3::STREAM_CALL_WITH_TIM_WITH_TRANSFORM:
			case hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED_WITH_TRANSFORM:
			{
				hkpAgentData* agentData = hkAddByteOffset( readData, hkSizeOf( hkpAgent1nMachineTimEntry ) );
				readData = hkAddByteOffset( readData, entry->m_size );
				dispatcher->getAgent3DestroyFunc( entry->m_agentType )( entry, agentData, mgr, constraintOwner, dispatcher);
				break;
			}

			case hkAgent3::STREAM_NULL:
			{
				readData = hkAddByteOffset( readData, entry->m_size );
				break;
			}
			case hkAgent3::STREAM_END:
			{
				goto endOfWhileLoop;
			}

			default: 
				HK_ASSERT2( 0xf0000001,0, "Unknown command in hkCdAgentStream");
		}

		//
		//	Check for read buffer empty, delete the old and get a new one
		//
		if ( readData >= readEnd )
		{
			delete readSectors[readSectorIndex++]; // works on ppu and spu

#if !defined(HK_PLATFORM_SPU)
			readSector = agentTrack.m_sectors[readSectorIndex];
#else
			hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( &sectorBuffer, readSectors[readSectorIndex], sizeof(hkpAgent1nSector), hkSpuDmaManager::READ_COPY );
			HK_SPU_DMA_PERFORM_FINAL_CHECKS( readSectors[readSectorIndex], &sectorBuffer, sizeof(hkpAgent1nSector) );
			readSector = &sectorBuffer;
#endif

			readData = readSector->getBegin();
			readEnd    = readSector->getEnd();
		}

	}
endOfWhileLoop:;
	delete readSectors[readSectorIndex]; // works on ppu and spu
#if defined(HK_PLATFORM_SPU)
	HK_SPU_DMA_PERFORM_FINAL_CHECKS( sectorsPPu, readSectors, 0 );
	hkDeallocateStack(readSectors);
#endif
	agentTrack.m_sectors.clearAndDeallocate();
}

#if !defined(HK_PLATFORM_SPU)
void hkAgent1nMachine_InvalidateTim( hkpAgent1nTrack& agentTrack, const hkpCollisionInput& input )
{
	hkpAgent1nSector* readSector = agentTrack.m_sectors[0];
	hkpAgentData*     readData   = readSector->getBegin();
	hkpAgentData*     readEnd    = readSector->getEnd();
	int     nextReadSectorIndex = 1;

	while(1)
	{
		hkpAgent1nMachineEntry* entry = reinterpret_cast<hkpAgent1nMachineEntry*>(readData);
		hkAgent3::StreamCommand command = hkAgent3::StreamCommand(entry->m_streamCommand);

		switch ( command )
		{
			case hkAgent3::STREAM_CALL_AGENT:
			case hkAgent3::STREAM_CALL_AGENT_WITH_TRANSFORM:

			case hkAgent3::STREAM_CALL:
			case hkAgent3::STREAM_CALL_FLIPPED:
			case hkAgent3::STREAM_CALL_WITH_TRANSFORM:
			case hkAgent3::STREAM_CALL_FLIPPED_WITH_TRANSFORM:
			{
				hkpAgentData* agentData = hkAddByteOffset( readData, hkSizeOf( hkpAgent1nMachinePaddedEntry ) );
				readData = hkAddByteOffset( readData, entry->m_size );
				hkAgent3::InvalidateTimFunc func = input.m_dispatcher->getAgent3InvalidateTimFunc( entry->m_agentType );
				if (func)
				{
					func(entry, agentData, input);
				}
				break;
			}
			case hkAgent3::STREAM_CALL_WITH_TIM:
			case hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED:
			case hkAgent3::STREAM_CALL_WITH_TIM_WITH_TRANSFORM:
			case hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED_WITH_TRANSFORM:
			{
				hkpAgentData* agentData = hkAddByteOffset( readData, hkSizeOf( hkpAgent1nMachineTimEntry ) );
				hkpAgent1nMachineTimEntry* timEntry = static_cast<hkpAgent1nMachineTimEntry*>(readData);
				timEntry->m_timeOfSeparatingNormal = hkTime(-1.0f);
				timEntry->m_separatingNormal.setZero();
				readData = hkAddByteOffset( readData, entry->m_size );
				hkAgent3::InvalidateTimFunc func = input.m_dispatcher->getAgent3InvalidateTimFunc( entry->m_agentType );
				if (func)
				{
					func(entry, agentData, input);
				}
				break;
			}

			case hkAgent3::STREAM_NULL:
			{
				readData = hkAddByteOffset( readData, entry->m_size );
				break;
			}
			case hkAgent3::STREAM_END:
			{
				goto endOfWhileLoop;
			}

			default: 
				HK_ASSERT2( 0xf0000001,0, "Unknown command in hkCdAgentStream");
		}

		//
		//	Check for read buffer empty and get a new one
		//
		if ( readData >= readEnd )
		{
			readSector = agentTrack.m_sectors[nextReadSectorIndex++];
			readData   = readSector->getBegin();
			readEnd    = readSector->getEnd();
		}

	}
endOfWhileLoop:;

}
#endif

#if !defined(HK_PLATFORM_SPU)
void hkAgent1nMachine_WarpTime( hkpAgent1nTrack& agentTrack, hkTime oldTime, hkTime newTime, const hkpCollisionInput& input )
{
	hkpAgent1nSector* readSector = agentTrack.m_sectors[0];
	hkpAgentData* readData = readSector->getBegin();
	hkpAgentData* readEnd    = readSector->getEnd();
	int             nextReadSectorIndex = 1;

	while(1)
	{

		hkpAgent1nMachineEntry* entry = reinterpret_cast<hkpAgent1nMachineEntry*>(readData);
		hkAgent3::StreamCommand command = hkAgent3::StreamCommand(entry->m_streamCommand);

		switch ( command )
		{
			case hkAgent3::STREAM_CALL_AGENT:
			case hkAgent3::STREAM_CALL_AGENT_WITH_TRANSFORM:

			case hkAgent3::STREAM_CALL:
			case hkAgent3::STREAM_CALL_FLIPPED:
			case hkAgent3::STREAM_CALL_WITH_TRANSFORM:
			case hkAgent3::STREAM_CALL_FLIPPED_WITH_TRANSFORM:
			{
				hkpAgentData* agentData = hkAddByteOffset( readData, hkSizeOf( hkpAgent1nMachinePaddedEntry ) );
				readData = hkAddByteOffset( readData, entry->m_size );
				hkAgent3::WarpTimeFunc func = input.m_dispatcher->getAgent3WarpTimeFunc( entry->m_agentType );
				if (func)
				{
					func(entry, agentData, oldTime, newTime, input);
				}
				break;
			}
			case hkAgent3::STREAM_CALL_WITH_TIM:
			case hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED:
			case hkAgent3::STREAM_CALL_WITH_TIM_WITH_TRANSFORM:
			case hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED_WITH_TRANSFORM:
			{
				hkpAgentData* agentData = hkAddByteOffset( readData, hkSizeOf( hkpAgent1nMachineTimEntry ) );
				hkpAgent1nMachineTimEntry* timEntry = static_cast<hkpAgent1nMachineTimEntry*>(readData);
				if (timEntry->m_timeOfSeparatingNormal == oldTime)
				{
					timEntry->m_timeOfSeparatingNormal = newTime;
				}
				else
				{
					timEntry->m_timeOfSeparatingNormal = hkTime(-1.0f);
				}
				readData = hkAddByteOffset( readData, entry->m_size );
				hkAgent3::WarpTimeFunc func = input.m_dispatcher->getAgent3WarpTimeFunc( entry->m_agentType );
				if (func)
				{
					func(entry, agentData, oldTime, newTime, input);
				}
				break;
			}

			case hkAgent3::STREAM_NULL:
			{
				readData = hkAddByteOffset( readData, entry->m_size );
				break;
			}
			case hkAgent3::STREAM_END:
			{
				goto endOfWhileLoop;
			}

		default: 
			HK_ASSERT2( 0xf0000001,0, "Unknown command in hkCdAgentStream");
		}

		//
		//	Check for read buffer empty
		//
		if ( readData >= readEnd )
		{
			//
			// Get a new one
			//
			readSector = agentTrack.m_sectors[nextReadSectorIndex];
			readData = readSector->getBegin();
			readEnd    = readSector->getEnd();
			nextReadSectorIndex++;
		}

	}
endOfWhileLoop:;

}
#endif

#if !defined(HK_PLATFORM_SPU)
hkpAgent1nMachineEntry* hkAgent1nMachine_FindAgent( hkpAgent1nTrack& agentTrack, hkpShapeKeyPair keyPair, hkpAgentData** agentDataOut)
{
	hkpAgent1nSector* readSector = agentTrack.m_sectors[0];
	hkpAgentData* readData = readSector->getBegin();
	hkpAgentData* readEnd    = readSector->getEnd();
	int             nextReadSectorIndex = 1;

	while(1)
	{

		hkpAgent1nMachineEntry* entry = reinterpret_cast<hkpAgent1nMachineEntry*>(readData);
		hkAgent3::StreamCommand command = hkAgent3::StreamCommand(entry->m_streamCommand);

		switch ( command )
		{
		case hkAgent3::STREAM_CALL_AGENT:
		case hkAgent3::STREAM_CALL:
		case hkAgent3::STREAM_CALL_FLIPPED:
		case hkAgent3::STREAM_CALL_AGENT_WITH_TRANSFORM:
		case hkAgent3::STREAM_CALL_WITH_TRANSFORM:
		case hkAgent3::STREAM_CALL_FLIPPED_WITH_TRANSFORM:
			{
				hkpAgentData* agentData = hkAddByteOffset( readData, hkSizeOf( hkpAgent1nMachinePaddedEntry ) );
				if ( entry->m_shapeKeyPair == keyPair )
				{
					agentDataOut[0] = agentData;
					return entry;
				}

				readData = hkAddByteOffset( readData, entry->m_size );
				break;
			}
		case hkAgent3::STREAM_CALL_WITH_TIM:
		case hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED:
		case hkAgent3::STREAM_CALL_WITH_TIM_WITH_TRANSFORM:
		case hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED_WITH_TRANSFORM:
			{
				hkpAgentData* agentData = hkAddByteOffset( readData, hkSizeOf( hkpAgent1nMachineTimEntry ) );
				if ( entry->m_shapeKeyPair == keyPair )
				{
					agentDataOut[0] = agentData;
					return entry;
				}
				readData = hkAddByteOffset( readData, entry->m_size );
				break;
			}

		case hkAgent3::STREAM_NULL:
			{
				readData = hkAddByteOffset( readData, entry->m_size );
				break;
			}
		case hkAgent3::STREAM_END:
			{
				goto endOfWhileLoop;
			}

		default: 
			HK_ASSERT2( 0xf0000001,0, "Unknown command in hkCdAgentStream");
		}

		//
		//	Check for read buffer empty and get a new one
		//
		if ( readData >= readEnd )
		{
			readSector = agentTrack.m_sectors[nextReadSectorIndex];
			readData = readSector->getBegin();
			readEnd    = readSector->getEnd();
			nextReadSectorIndex++;
		}

	}
endOfWhileLoop:;
	return HK_NULL;
}
#endif


#if defined(HK_PLATFORM_SPU)
void hkAgent1nMachine_Create( hkpAgent1nTrack& agentTrack )
{

	// Allocate the sector in main memory
	HK_ALIGN16(hkpAgent1nSector* sector) = hkAllocateChunk<hkpAgent1nSector>(1, HK_MEMORY_CLASS_AGENT);

	// Allocate the array data
	hkpAgent1nSector** arrayData = hkAllocateChunk<hkpAgent1nSector*>(4, HK_MEMORY_CLASS_AGENT);

	// Setup the local array to point to the allocated array data
	agentTrack.m_sectors.setDataAutoFree(arrayData, 1, 4);

	// Write the first pointer in the array data to point to the allocated sector
	hkSpuDmaManager::putToMainMemorySmall( arrayData, &sector, 4, hkSpuDmaManager::WRITE_NEW );
	
	
	HK_ALIGN16(hkpAgent1nSector localSector);

	hkpAgent1nMachineEntry* entry = reinterpret_cast<hkpAgent1nMachineEntry*>(localSector.getBegin());
	entry->m_size = hkSizeOf( hkpAgent1nMachinePaddedEntry );
	entry->m_streamCommand = hkAgent3::STREAM_END;
	entry->m_shapeKeyPair.m_shapeKeyB   = HK_INVALID_SHAPE_KEY;
	entry->m_shapeKeyPair.m_shapeKeyA   = HK_INVALID_SHAPE_KEY;
	localSector.m_bytesAllocated = hkSizeOf( hkpAgent1nMachinePaddedEntry );

	hkSpuDmaManager::putToMainMemory( sector, &localSector, sizeof(hkpAgent1nMachinePaddedEntry) + 16, hkSpuDmaManager::WRITE_NEW );
	hkSpuDmaManager::waitForAllDmaCompletion();
	hkSpuDmaManager::performFinalChecks( arrayData, &sector, 4 );
	hkSpuDmaManager::performFinalChecks( sector, &localSector, sizeof(hkpAgent1nMachinePaddedEntry) + 16 );

}

#else
void hkAgent1nMachine_Create( hkpAgent1nTrack& agentTrack )
{
	agentTrack.m_sectors.clear();
	agentTrack.m_sectors.reserve(4);
	agentTrack.m_sectors.expandOne();

	hkpAgent1nSector* sector = new hkpAgent1nSector;
	agentTrack.m_sectors[0] = sector;

	hkpAgent1nMachineEntry* entry = reinterpret_cast<hkpAgent1nMachineEntry*>(sector->getBegin());
	entry->m_size = hkSizeOf( hkpAgent1nMachinePaddedEntry );
	entry->m_streamCommand = hkAgent3::STREAM_END;
	entry->m_shapeKeyPair.m_shapeKeyB   = HK_INVALID_SHAPE_KEY;
	entry->m_shapeKeyPair.m_shapeKeyA   = HK_INVALID_SHAPE_KEY;
	sector->m_bytesAllocated = hkSizeOf( hkpAgent1nMachinePaddedEntry );
}
#endif

#if !defined(HK_PLATFORM_SPU)
// A factor of the collision tolerance.
// A contact point is only accepted if it is closer than existing point - tolerance * WELDING_POINT_CLOSER_DIST_RELATIVE_TOLERANCE
#define WELDING_POINT_CLOSER_DIST_RELATIVE_TOLERANCE 0.05f

enum hkContactPairCompare
{
	HK_CONTACT_PAIR_OK = 0,
	HK_CONTACT_PAIR_REJECT = 1,
	HK_CONTACT_PAIR_CONFLICT
};

static HK_FORCE_INLINE hkContactPairCompare HK_CALL 
contactPointIsValid( hkReal radiusA, const hkContactPoint& pointToCheck, const hkpCollisionQualityInfo& qi, hkReal radiusB, const hkpProcessCdPoint& referenceCdPoint )
{
	const hkContactPoint& referencePoint = referenceCdPoint.m_contact;

		// early out if the candidate point is penetrating
		// not used because of confict solving
//	if( pointToCheck.getDistance() < MAX_WELDING_PENETRATION )
//	{
//		return true;
//	}

	hkVector4 diff;	diff.setSub( referencePoint.getPosition(), pointToCheck.getPosition() );	

	hkVector4 dots; hkVector4Util::dot3_3vs3( pointToCheck.getNormal(), diff, referencePoint.getNormal(), diff, pointToCheck.getNormal(), referencePoint.getNormal(), dots);

	// d0 is the distance of our pointToCheck to the referencePlane (d1 the other point)
	hkSimdReal d0  = -dots.getComponent<1>();
	hkSimdReal d1  =  dots.getComponent<0>();
	const hkSimdReal n0n1 = dots.getComponent<2>();

	const hkSimdReal radiusAsr = hkSimdReal::fromFloat(radiusA);

	//
	//	Virtually move the radius of object A to object B
	//
	{
		hkSimdReal dx; dx.setMul(radiusAsr, n0n1 - hkSimdReal_1);
		d0.add(dx);
		d1.add(dx);
	}

	const hkSimdReal radiusSum = radiusAsr + hkSimdReal::fromFloat(radiusB);

	// If the new point is above the plane of the reference point, it is ok from the perspective of the reference point
	if ( d0.isGreater(radiusSum) )
	{
		return HK_CONTACT_PAIR_OK;
	}

	//
	// Accept if the normal is the same. but only if the second point is not a potential one
	//
	if ( n0n1.isGreater(hkSimdReal::fromFloat(hkReal(0.9999f))) && (referenceCdPoint.m_contactPointId == HK_INVALID_CONTACT_POINT) )
	{
		return HK_CONTACT_PAIR_OK;
	}

	//
	//	Check for conflicts between new points
	//
	{
		const hkSimdReal dist0 = pointToCheck.getDistanceSimdReal();
		const hkSimdReal dist1 = referenceCdPoint.m_contact.getDistanceSimdReal();

#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
		hkVector4Comparison c0 = n0n1.less(hkSimdReal::fromFloat(hkReal(0.8f)));
		hkVector4Comparison c1 = (d0+d1).less(hkSimdReal::fromFloat(hkReal(0.1f)));
		hkVector4Comparison c2 = (dist0+dist1).less(-(radiusSum+radiusSum));
		c0.setAnd(c0,c1);
		c0.setAnd(c0,c2);
		if ( (referenceCdPoint.m_contactPointId == HK_INVALID_CONTACT_POINT ) && c0.anyIsSet() )
#else
		if  (
			  (referenceCdPoint.m_contactPointId == HK_INVALID_CONTACT_POINT )
			&& n0n1.isLess(hkSimdReal::fromFloat(hkReal(0.8f)))	             // different normals
			&& (d0 + d1).isLess(hkSimdReal::fromFloat(hkReal(0.1f)))         // semi convex situation
			&& (dist0 + dist1).isLess(-(radiusSum+radiusSum))       // convex core is penetrating
			)
#endif
		{
			return HK_CONTACT_PAIR_CONFLICT;
		}
	}

	//
	//	Check distance only 
	//
	hkSimdReal minDeltaDistanceToAcceptNewPoint; minDeltaDistanceToAcceptNewPoint.setZero();
	// if the reference point is a valid one, we only create a new contact
	// point is it is seriously closer than the valid reference point
	// fixes HVK-2168
	if ( referenceCdPoint.m_contactPointId != HK_INVALID_CONTACT_POINT )
	{
		minDeltaDistanceToAcceptNewPoint.load<1>(&qi.m_keepContact);
		minDeltaDistanceToAcceptNewPoint.mul(hkSimdReal::fromFloat(hkReal(WELDING_POINT_CLOSER_DIST_RELATIVE_TOLERANCE)));
	}

	if ( pointToCheck.getDistanceSimdReal().isLessEqual(referencePoint.getDistanceSimdReal() - minDeltaDistanceToAcceptNewPoint) )
	{
		return HK_CONTACT_PAIR_OK;
	}
	return HK_CONTACT_PAIR_REJECT;


//	if ( d0 > -WELDING_ACCEPT_EPS  )
//	{
//		return d0 >= d1;
//	}
//
//	//
//	//	Now we have a real conflict, let's do a hack for now
//	//
//	return d0 > d1;
}
#if defined(HK_1N_MACHINE_SUPPORTS_WELDING)
void hkAgent1nMachine_Weld( hkpAgent3Input& input, const hkpShapeContainer* shapeCollection, hkpProcessCollisionOutput& output )
{
	hkpShapeBuffer shapeBuffer;

   	int numConflictContacts = 0;
	hkContactPointId newIds[ HK_MAX_CONTACT_POINT ];
	hkUint16 conflictContacts[ HK_MAX_CONTACT_POINT ];

	hkpProcessCollisionOutput::PotentialInfo& potentialContacts = *output.m_potentialContacts;
	
	{
		const hkpConvexShape* shapeA = reinterpret_cast<const hkpConvexShape*>(input.m_bodyA->getShape());
		hkpProcessCollisionOutput::ContactRef* contactRefPtr = &potentialContacts.m_potentialContacts[0];
		for ( int i = 0; contactRefPtr < potentialContacts.m_firstFreePotentialContact; i++, contactRefPtr++ )
		{
			newIds[i] = HK_INVALID_CONTACT_POINT;

			hkpProcessCdPoint* check = contactRefPtr->m_contactPoint;

			//
			//	Check other potential contacts
			//
			{
				bool hasConflict = false;
				for ( hkpProcessCdPoint** pref = &potentialContacts.m_representativeContacts[0]; pref < potentialContacts.m_firstFreeRepresentativeContact; pref++ )
				{
					hkpProcessCdPoint* ref = *pref;
					if ( check != ref )
					{
						hkReal radiusB = hkReal(0);
						hkContactPairCompare contactPairCheckResult = contactPointIsValid( shapeA->getRadius(), check->m_contact, *(input.m_input->m_collisionQualityInfo), radiusB, *ref );
						if ( contactPairCheckResult != HK_CONTACT_PAIR_OK)
						{
							if ( contactPairCheckResult == HK_CONTACT_PAIR_REJECT)
							{
								goto tryNextPoint;
							}
							hasConflict = true;
						}
					}
				}
				if ( hasConflict )
				{
					conflictContacts[ numConflictContacts++ ] = hkUint16(i);
				}
			}

			//
			// try to get an id for new and ok+conflict contacts
			//
			{
				hkpProcessCollisionOutput::ContactRef& toCommit = potentialContacts.m_potentialContacts[i];
				hkpAgent1nMachineEntry* entry = static_cast<hkpAgent1nMachineEntry*>(toCommit.m_agentEntry);

				const hkpShape* shape = shapeCollection->getChildShape( entry->m_shapeKeyPair.m_shapeKeyB, shapeBuffer );
				hkpCdBody modifiedBodyB( input.m_bodyB );
				modifiedBodyB.setShape( shape, entry->m_shapeKeyPair.m_shapeKeyB );

				newIds[i] = input.m_contactMgr->addContactPoint( *input.m_bodyA, modifiedBodyB, *input.m_input, output, HK_NULL, toCommit.m_contactPoint->m_contact );
				if ( newIds[i] != HK_INVALID_CONTACT_POINT )
				{
					input.m_contactMgr->reserveContactPoints(-1);	// release reserved point
				}
			}
tryNextPoint:	continue;
		}
	}

	//
	// Convert conflicts into zombies
	//
	{
		hkpCollisionDispatcher* dis = input.m_input->m_dispatcher;
		for (int a = 0; a < numConflictContacts; a++)
		{
			int i = conflictContacts[a];
			hkpProcessCollisionOutput::ContactRef& cr = potentialContacts.m_potentialContacts[ i ]; 
			// todo: think about multiple new points per agent
			dis->getAgent3CreateZombieFunc( cr.m_agentEntry->m_agentType)( cr.m_agentEntry, cr.m_agentData, HK_INVALID_CONTACT_POINT );
		}
	}

	//
	// Solve conflicts (in this version using pairs)
	//
	while ( numConflictContacts >= 2)
	{
		HK_TIMER_BEGIN("Conflicts", HK_NULL);

		// search two points with the most opposing normal
		HK_ALIGN16(int p0);
		HK_ALIGN16(int p1);
		{
			hkSimdReal minDot = hkSimdReal_2;
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
			const hkIntVector countOne = hkIntVector::getConstant<HK_QUADINT_1>();
			hkIntVector a_counter; a_counter.setZero();
			hkIntVector p0_idx; p0_idx.setZero();
			hkIntVector p1_idx = countOne;
#else
			p0 = 0; p1 = 1;
#endif
			for (int a = 0; a < numConflictContacts; a++)
			{
				hkpProcessCollisionOutput::ContactRef& refA = potentialContacts.m_potentialContacts[ conflictContacts[a] ]; 
				const hkVector4& normalA = refA.m_contactPoint->m_contact.getNormal();
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
				hkIntVector b_counter; b_counter.setAddS32(a_counter, countOne);
#endif
				for (int b = a+1; b < numConflictContacts; b++)
				{
					hkpProcessCollisionOutput::ContactRef& refB = potentialContacts.m_potentialContacts[ conflictContacts[b] ]; 
					const hkSimdReal d = normalA.dot<3>( refB.m_contactPoint->m_contact.getNormal() );
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
					hkVector4Comparison dLTmin = d.less(minDot);
					p0_idx.setSelect(dLTmin, a_counter, p0_idx);
					p1_idx.setSelect(dLTmin, b_counter, p1_idx);
					b_counter.setAddS32(b_counter, countOne);
#else
					if ( d.isLess(minDot) )
					{
						p0 = a;	p1 = b;
					}
#endif
					minDot.setMin(d, minDot);
				}
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
				a_counter.setAddS32(a_counter, countOne);
#endif
			}
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
			p0_idx.store<1>((hkUint32*)&p0);
			p1_idx.store<1>((hkUint32*)&p1);
#endif			

			//
			//	Solve conflict if we have 3 convex shapes
			//
			{
				hkpShapeBuffer shapeBuffer2;

				hkpProcessCollisionOutput::ContactRef& ref0 = potentialContacts.m_potentialContacts[ conflictContacts[p0] ]; 
				hkpProcessCollisionOutput::ContactRef& ref1 = potentialContacts.m_potentialContacts[ conflictContacts[p1] ]; 

				hkpAgent1nMachineEntry* header0 = static_cast<hkpAgent1nMachineEntry*>(ref0.m_agentEntry);
				hkpAgent1nMachineEntry* header1 = static_cast<hkpAgent1nMachineEntry*>(ref1.m_agentEntry);

				const hkpConvexShape* shapeA =  reinterpret_cast<const hkpConvexShape*>( input.m_bodyA->getShape() );
				if ( !input.m_input->m_dispatcher->hasAlternateType( shapeA->getType(), hkcdShapeType::CONVEX )) { break; }
				const hkpConvexShape* shape0 = reinterpret_cast<const hkpConvexShape*> ( shapeCollection->getChildShape( header0->m_shapeKeyPair.m_shapeKeyB, shapeBuffer ));
				if ( !input.m_input->m_dispatcher->hasAlternateType( shape0->getType(), hkcdShapeType::CONVEX )) { break; }
				const hkpConvexShape* shape1 = reinterpret_cast<const hkpConvexShape*> ( shapeCollection->getChildShape( header1->m_shapeKeyPair.m_shapeKeyB, shapeBuffer2 ));
				if ( !input.m_input->m_dispatcher->hasAlternateType( shape1->getType(), hkcdShapeType::CONVEX )) { break; }


				const hkpConvexShape* shapesB[2] = { shape0, shape1 };
				hkContactPoint* points[2] = {  &ref0.m_contactPoint->m_contact, &ref1.m_contactPoint->m_contact }; 

				//hkResult conflictResolved = 
				hkCalcMultiPenetrationDepth( input.m_bodyA->getTransform(), shapeA, &shapesB[0], 2, input.m_aTb, &points[0]  );
			}

			// Remove from conflict array
			conflictContacts[ p1 ] = conflictContacts[ --numConflictContacts ];
			conflictContacts[ p0 ] = conflictContacts[ --numConflictContacts ];
		}
		HK_TIMER_END();
	}

	// kill remaining conflict
	if (numConflictContacts)
	{
		if ( newIds[ conflictContacts[0] ] != HK_INVALID_CONTACT_POINT )
		{
			input.m_contactMgr->removeContactPoint( newIds[ conflictContacts[0] ], *output.m_constraintOwner.val() );
			newIds[ conflictContacts[0] ] = HK_INVALID_CONTACT_POINT;
			input.m_contactMgr->reserveContactPoints(1);	// reserve so we can release later
		}
	}


	//
	//	Commit points or delete points (in reverse order, so we can use the quick delete in the output.m_contactPoints)
	//
	{
		hkpCollisionDispatcher* dis = input.m_input->m_dispatcher;

		hkpProcessCollisionOutput::ContactRef* cr = potentialContacts.m_firstFreePotentialContact;
		int numPotential = int(cr - &potentialContacts.m_potentialContacts[0]);

		for ( int i = numPotential-1; i >=0 ; i-- )
		{
			cr--;
			hkContactPointId newId = newIds[i];

			if ( newId != HK_INVALID_CONTACT_POINT)
			{
				dis->getAgent3CommitPotentialFunc( cr->m_agentEntry->m_agentType)( cr->m_agentEntry, cr->m_agentData, newId );
				cr->m_contactPoint->m_contactPointId = newId;
			}
			else
			{
				input.m_contactMgr->reserveContactPoints(-1);
				dis->getAgent3RemovePointFunc( cr->m_agentEntry->m_agentType )( cr->m_agentEntry, cr->m_agentData, newId );
				output.m_firstFreeContactPoint = output.m_firstFreeContactPoint-1;
				*cr->m_contactPoint = *output.m_firstFreeContactPoint;
			}
		}
	}

	// You can run this check for hkpSimpleConstraintContactMgr but not e.g. for hkpReportContactMgr
	// hkpSimpleConstraintContactMgr* scmgr = reinterpret_cast<hkpSimpleConstraintContactMgr*>( input.m_contactMgr );
	// HK_ASSERT2( 0xad67933, scmgr->m_reservedContactPoints == 0, "Internal error: Welding: reserved contact points counter corrupted." );
}
#endif //HK_1N_MACHINE_SUPPORTS_WELDING

#endif // if !defined HK_NM_MACHINE_CODE_PATH


#if !defined(HK_PLATFORM_SPU)
void HK_CALL hkAgent1nMachine_VisitAllAgents( hkpAgent1nTrack& agentTrack, hkpAgent1nMachine_VisitorInput& vin,  hkAgent1nMachine_VisitorCallback visitor )
{
	int             currentSectorIndex = 0;
	hkpAgent1nSector* readSector = agentTrack.m_sectors[0];
	hkpAgent1nSector* spareSector = HK_NULL;	
	hkpAgent1nSector* currentSector = new hkpAgent1nSector;
	int             nextReadSectorIndex = 1;
	
	bool inExtensionBuffer = false;								// if set to true, the agent is written to a small local buffer, as we do not know whether it will fit in the current sector
	HK_ALIGN_REAL(hkUchar extensionBuffer[hkAgent3::MAX_SIZE]);	// extra data to hold the last agent for the sector. We do this as we do not know the size of the agent yet

	hkpAgentData* readData = readSector->getBegin();
	hkpAgentData* readEnd    = readSector->getEnd();
	hkpAgentData* currentData   = currentSector->getBegin();
	hkpAgentData* currentDataEnd = currentSector->getEndOfCapacity();
	int bytesUsedInCurrentSector = 0;	// only initialized when inExtensionBuffer == true


	while ( 1 )
	{
		// merge the newKeys

			//
			//	keep and copy the whole agent
			//
		{
			hkpAgent1nMachineEntry* entry = reinterpret_cast<hkpAgent1nMachineEntry*>(readData);
			HK_ASSERT2(0XAD875DAA, entry->m_size % 16 == 0, "hkpAgentEntry size is expected to be a multiple of 16.");
			hkString::memCpy16NonEmpty( currentData, readData, entry->m_size>>4 );
			readData = hkAddByteOffset( readData, entry->m_size );
		}

		hkpAgent1nMachine_VisitorInput vmod = vin;
		{		
			hkAgent3::StreamCommand command = hkAgent3::StreamCommand(reinterpret_cast<hkpAgentEntry*>(currentData)->m_streamCommand);
			int sizeofentry = sizeof( hkpAgent1nMachinePaddedEntry );

			command = static_cast<hkAgent3::StreamCommand> ( static_cast<hkUchar>(command) & static_cast<hkUchar>(~hkAgent3::TRANSFORM_FLAG) );

			switch ( command )
			{
				case hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED:
				case hkAgent3::STREAM_CALL_WITH_TIM:
						sizeofentry = sizeof( hkpAgent1nMachineTimEntry );
				case hkAgent3::STREAM_CALL_FLIPPED:
				case hkAgent3::STREAM_CALL:
				case hkAgent3::STREAM_CALL_AGENT:
				{	
					//
					//	call the visitor
					//
					hkpAgent1nMachineEntry* entry = reinterpret_cast<hkpAgent1nMachineEntry*>(currentData);
					hkpAgentData* agentData =  hkAddByteOffset( entry, sizeofentry );

					currentData = visitor( vmod, entry, agentData );
					break;
				}
				case hkAgent3::STREAM_NULL:
				{
					//do nothing
					break;
				}
				case hkAgent3::STREAM_END:
				{
					currentData = hkAddByteOffset( currentData, sizeof( hkpAgent1nMachinePaddedEntry ) );
					break;
				}
				default: 
					HK_ASSERT2( 0xf0000111,0, "Unknown command in hkCdAgentStream");
			}
		}
		//
		//	Check whether we still have data in the read buffer
		//
		HK_ASSERT(0x2612f888, readData <= readEnd);
		
		if ( readData == readEnd )
		{
			//	Free existing read buffer
			if ( spareSector )	{	delete spareSector;		}
			spareSector = readSector;

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
					hkString::memCpy16NonEmpty( hkAddByteOffset( currentSector->getBegin(), bytesUsedInCurrentSector), &extensionBuffer[0], bytesUsedInExtensionBuffer>>4 );
					bytesUsedInCurrentSector += bytesUsedInExtensionBuffer;

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
			
				// Get a new one
			{
				if ( spareSector )
				{
					currentSector = spareSector;
					spareSector = HK_NULL;
				}
				else
				{
					currentSector = new hkpAgent1nSector;
				}
				currentData = currentSector->getBegin();
				currentDataEnd = currentSector->getEndOfCapacity();
			}
		}
		// put overflow of last iteration to the start of our new sector
		if ( inExtensionBuffer )
		{
			hkString::memCpy16NonEmpty( currentData, &extensionBuffer[0], bytesUsedInExtensionBuffer>>4 );
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
		int bytesUsedInExtensionBuffer = int(hkGetByteOffset( &extensionBuffer[0], currentData ));
		HK_ASSERT( 0xf034defe, bytesUsedInExtensionBuffer==16 ); // end marker
		hkString::memCpy16NonEmpty( hkAddByteOffset( currentSector->getBegin(), bytesUsedInCurrentSector), &extensionBuffer[0], bytesUsedInExtensionBuffer>>4 );
		bytesUsedInCurrentSector += bytesUsedInExtensionBuffer;
	}
	HK_ASSERT(0x441076d0, bytesUsedInCurrentSector <= hkpAgent1nSector::NET_SECTOR_SIZE );
	currentSector->m_bytesAllocated = bytesUsedInCurrentSector;

	//
	//	Store very last sector and finalize agentTrack
	//
	{
		agentTrack.m_sectors.setSize( currentSectorIndex+1 );
		agentTrack.m_sectors[currentSectorIndex] = currentSector;
	}

	//
	//	revert the vin
	//
	if ( spareSector )
	{
		delete spareSector;
	}
}
#endif

#define ON_1N_MACHINE(code) code
#define ON_NM_MACHINE(code)

void HK_CALL hkAgent1nMachine_initInputAtTime( hkpAgent3Input& in, hkpAgentNmMachineBodyTemp& temp, hkpAgent3Input& out) // used 2times
{
	hkSweptTransformUtil::lerp2( in.m_bodyA->getMotionState()->getSweptTransform(), in.m_input->m_stepInfo.m_startTime, temp.m_transA );
	hkSweptTransformUtil::lerp2( in.m_bodyB->getMotionState()->getSweptTransform(), in.m_input->m_stepInfo.m_startTime, temp.m_transB );

	out.m_bodyA = &temp.m_bodyA;
	out.m_bodyB = &temp.m_bodyB;
	out.m_contactMgr = in.m_contactMgr;
	out.m_input = in.m_input;

	temp.m_bodyA.setShape( in.m_bodyA->getShape(), in.m_bodyA->getShapeKey() );

	new (&temp.m_bodyA) hkpCdBody( in.m_bodyA, &temp.m_transA );
	new (&temp.m_bodyB) hkpCdBody( in.m_bodyB, &temp.m_transB );
	out.m_aTb.setMulInverseMul(temp.m_transA, temp.m_transB);
}

void HK_CALL hkAgent1nMachine_flipInput( hkpAgent3ProcessInput& in, hkpAgent3ProcessInput& out ) //used 3times
{
	out.m_bodyA = in.m_bodyB;
	out.m_bodyB = in.m_bodyA;
	out.m_input = in.m_input;
	out.m_contactMgr = in.m_contactMgr;
	out.m_linearTimInfo.setNeg<3>( in.m_linearTimInfo );
	out.m_aTb.setInverse( in.m_aTb );
}

#if 0

#include <Common/Visualize/hkDebugDisplay.h>
#define DISPLAY_TRIANGLE_ENABLED
static inline void HK_CALL hkAgent1nMachine_displayTriangle( const hkTransform& transform, const hkpShapeContainer* collection, hkpShapeKey key )
{
	hkpShapeBuffer shapeBuffer;

	const hkpShape* shape = collection->getChildShape( key, shapeBuffer );
	if ( shape->getType() != hkcdShapeType::TRIANGLE)
	{
		return;
	}

	const hkpTriangleShape* t = static_cast<const hkpTriangleShape*>(shape);

	hkVector4 a; a.setTransformedPos(transform, t->getVertex(0));
	hkVector4 b; b.setTransformedPos(transform, t->getVertex(1));
	hkVector4 c; c.setTransformedPos(transform, t->getVertex(2));

	hkVector4 center; center.setAdd4( a, b);
	center.add4( c);
	center.mul4( 1.0f/ 3.0f);


	HK_DISPLAY_LINE( a, b, hkColor::YELLOW );
	HK_DISPLAY_LINE( a, c, hkColor::YELLOW );
	HK_DISPLAY_LINE( b, c, hkColor::YELLOW );
}
#endif

#if !defined(HK_PLATFORM_SPU)

void HK_CALL hkAgent1nMachine_UpdateShapeCollectionFilter( hkpAgent1nTrack& agentTrack, hkpAgent1nMachine_VisitorInput& vin )
{
	hkAgent1nMachine_VisitAllAgents( agentTrack, vin, hkAgent1nMachine_UpdateShapeCollectionFilterVisitor );
}

void HK_CALL hkAgentNmMachine_UpdateShapeCollectionFilter( hkpAgent1nTrack& agentTrack, hkpAgent1nMachine_VisitorInput& vin )
{
	hkAgent1nMachine_VisitAllAgents( agentTrack, vin, hkAgentNmMachine_UpdateShapeCollectionFilterVisitor );
}

#endif

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
