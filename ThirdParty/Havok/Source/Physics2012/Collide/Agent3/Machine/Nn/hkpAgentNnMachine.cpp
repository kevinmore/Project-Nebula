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
#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>

#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpLinkedCollidable.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>

#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>

#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>

#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnTrack.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.inl>
#if defined(HK_PLATFORM_SPU)
#	include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgrSpu.inl> // include this after we include the actual contact manager!
#endif

HK_COMPILE_TIME_ASSERT( sizeof(hkpAgentNnMachinePaddedEntry) % 16 == 0 );

// Internal helper functions

static HK_FORCE_INLINE hkpAgentNnEntry* HK_CALL hkAgentNnMachine_getLastEntry( hkpAgentNnTrack& track, int forcePpuEntry )
{
#if !defined(HK_PLATFORM_HAS_SPU)
	HK_ASSERT2(0Xad000191, track.m_bytesUsedInLastSector, "Internal error: never have empty sectors");
	return hkAddByteOffset( track.m_sectors.back()->getBegin(), track.m_bytesUsedInLastSector - track.getAgentSize() );
#else
	if ( !forcePpuEntry )
	{
		// spu
		HK_ASSERT(0Xf0ed4e35, track.m_spuBytesUsedInLastSector );
		return hkAddByteOffset( track.m_sectors[track.m_spuNumSectors-1]->getBegin(), track.m_spuBytesUsedInLastSector - track.getAgentSize() );
	}
	else
	{
		// ppu
		HK_ASSERT(0Xf0ed4e34, track.m_ppuBytesUsedInLastSector );
		return hkAddByteOffset( track.m_sectors.back()->getBegin(), track.m_ppuBytesUsedInLastSector - track.getAgentSize() );
	}
#endif
}

#if ! defined (HK_PLATFORM_SPU)
static HK_FORCE_INLINE void HK_CALL hkAgentNnMachine_RelinkEntry( hkpAgentNnEntry* entry )
{
	HK_ON_DEBUG(hkpRigidBody* bodyA = hkpGetRigidBody(entry->m_collidable[0]));
	HK_ON_DEBUG(hkpRigidBody* bodyB = hkpGetRigidBody(entry->m_collidable[1]));
	HK_ACCESS_CHECK_WITH_PARENT( bodyA->getWorld(), HK_ACCESS_RW, bodyA->getSimulationIsland(), HK_ACCESS_RW );
	HK_ACCESS_CHECK_WITH_PARENT( bodyB->getWorld(), HK_ACCESS_RW, bodyB->getSimulationIsland(), HK_ACCESS_RW );

	hkObjectIndex idx0 = entry->m_agentIndexOnCollidable[0];
	hkObjectIndex idx1 = entry->m_agentIndexOnCollidable[1];
	entry->m_collidable[0]->getCollisionEntriesNonDeterministic()[idx0].m_agentEntry = entry;
	entry->m_collidable[1]->getCollisionEntriesNonDeterministic()[idx1].m_agentEntry = entry;
}
#endif

#if ! defined (HK_PLATFORM_SPU)
static hkpAgentNnEntry* HK_CALL hkAgentNnMachine_AllocateEntry( hkpAgentNnTrack& track, int forcePpuEntry )
{
	hkpAgentNnSector* sector;
	hkpAgentNnEntry* entry;
#if !defined(HK_PLATFORM_HAS_SPU)
	if (track.m_bytesUsedInLastSector < HK_AGENT3_SECTOR_SIZE)
	{
		sector = track.m_sectors.back();
		entry = hkAddByteOffset(sector->getBegin(), track.m_bytesUsedInLastSector );
		track.m_bytesUsedInLastSector = track.m_bytesUsedInLastSector + hkUint16( track.getAgentSize() );
	}
	else
	{
		sector = static_cast<hkpAgentNnSector*>( hkAllocateChunk<void>( HK_AGENT3_SECTOR_SIZE, HK_MEMORY_CLASS_CDINFO ));
		entry = sector->getBegin();
		track.m_sectors.pushBack(sector);
		track.m_bytesUsedInLastSector = hkUint16( track.getAgentSize() );
	}
#else
	if ( !forcePpuEntry )
	{
			// spu
		if (track.m_spuBytesUsedInLastSector < HK_AGENT3_SECTOR_SIZE)
		{
			sector = track.m_sectors[track.m_spuNumSectors-1];
			entry = hkAddByteOffset(sector->getBegin(), track.m_spuBytesUsedInLastSector );

			track.m_spuBytesUsedInLastSector += hkUint16( track.getAgentSize() );
		}
		else
		{
			sector = static_cast<hkpAgentNnSector*>( hkAllocateChunk<void>( HK_AGENT3_SECTOR_SIZE, HK_MEMORY_CLASS_CDINFO ));
			entry = sector->getBegin();

			track.m_sectors.insertAt(track.m_spuNumSectors, sector);
			track.m_spuNumSectors++;
			track.m_spuBytesUsedInLastSector = hkUint16( track.getAgentSize() );
		}
	}
	else
	{
			// ppu 
		if (track.m_ppuBytesUsedInLastSector < HK_AGENT3_SECTOR_SIZE)
		{
			sector = track.m_sectors.back();
			entry = hkAddByteOffset(sector->getBegin(), track.m_ppuBytesUsedInLastSector );

			track.m_ppuBytesUsedInLastSector += hkUint16( track.getAgentSize() );
		}
		else
		{
			sector = static_cast<hkpAgentNnSector*>( hkAllocateChunk<void>( HK_AGENT3_SECTOR_SIZE, HK_MEMORY_CLASS_CDINFO ));
			entry = sector->getBegin();

			track.m_sectors.pushBack(sector);
			track.m_ppuBytesUsedInLastSector = hkUint16( track.getAgentSize() );
		}
	}
	entry->m_forceCollideOntoPpu = hkUchar(forcePpuEntry);
#endif
	return entry;
}
#endif

#if !defined(HK_PLATFORM_SPU)
static HK_FORCE_INLINE void HK_CALL hkAgentNnMachine_DeallocateLastEntry( hkpAgentNnTrack& track, int ppuEntry)
{
#if !defined(HK_PLATFORM_HAS_SPU)

	track.m_bytesUsedInLastSector = track.m_bytesUsedInLastSector - hkUint16( track.getAgentSize() );
	if (!track.m_bytesUsedInLastSector)
	{
		hkpAgentNnSector* sector = track.m_sectors.back();
		track.m_sectors.popBack();
		hkDeallocateChunk<void>( sector, HK_AGENT3_SECTOR_SIZE, HK_MEMORY_CLASS_CDINFO );
		track.m_bytesUsedInLastSector = HK_AGENT3_SECTOR_SIZE;
	}
#else
	if ( !ppuEntry )
	{
		// spu
		track.m_spuBytesUsedInLastSector -= hkUint16( track.getAgentSize() );

		hkpAgentNnSector* sector = track.m_sectors[ track.m_spuNumSectors-1 ];

		if (!track.m_spuBytesUsedInLastSector)
		{
			track.m_spuNumSectors--;

				// we cannot use removeAt as we last ppu sector must always be the last ppu sector
			track.m_sectors.removeAtAndCopy(track.m_spuNumSectors );
			track.m_spuBytesUsedInLastSector = HK_AGENT3_SECTOR_SIZE;

			hkDeallocateChunk<void>( sector, HK_AGENT3_SECTOR_SIZE, HK_MEMORY_CLASS_CDINFO );
		}
#if defined(HK_DEBUG)
		else
		{
			hkString::memSet( hkAddByteOffset(sector,track.m_spuBytesUsedInLastSector), 0xcf, track.getAgentSize() );
		}
#endif
	}
	else
	{
		// ppu
		track.m_ppuBytesUsedInLastSector -= hkUint16( track.getAgentSize() );
		if (!track.m_ppuBytesUsedInLastSector)
		{
			hkpAgentNnSector* sector = track.m_sectors.back();
			track.m_sectors.popBack();
			track.m_ppuBytesUsedInLastSector = HK_AGENT3_SECTOR_SIZE;

			hkDeallocateChunk<void>( sector, HK_AGENT3_SECTOR_SIZE, HK_MEMORY_CLASS_CDINFO );
		}
	}
#endif
	track.m_sectors.optimizeCapacity(4);
}
#endif

#if !defined(HK_PLATFORM_SPU)
void hkAgentNnMachine_GetAgentType( const hkpCdBody* cdBodyA, const hkpCdBody* cdBodyB, const hkpProcessCollisionInput& input, int& agentTypeOut, int& isFlippedOut )
{
	int agentType; 
	{
		hkpCollisionDispatcher* dispatcher = input.m_dispatcher;
		hkpShapeType shapeTypeA = cdBodyA->getShape()->getType();
		hkpShapeType shapeTypeB = cdBodyB->getShape()->getType();
		agentType = dispatcher->getAgent3Type( shapeTypeA, shapeTypeB, input.m_createPredictiveAgents );
		int flip = ( dispatcher->getAgent3Symmetric(agentType) == hkAgent3::IS_NOT_SYMMETRIC_AND_FLIPPED );
		if ( agentType == HK_AGENT_TYPE_BRIDGE )
		{
			flip = dispatcher->getIsFlipped( shapeTypeA, shapeTypeB );
		}
		isFlippedOut = flip;

		if ( flip )
		{
			agentType = dispatcher->getAgent3Type( shapeTypeB, shapeTypeA, input.m_createPredictiveAgents );
		}
	}
	agentTypeOut = agentType;
}
#endif

#if !defined(HK_PLATFORM_SPU)
hkpAgentNnEntry* HK_CALL hkAgentNnMachine_CreateAgent( hkpAgentNnTrack& track, hkpLinkedCollidable* collA, const hkpCdBody* firstNonTransformBodyA, hkpLinkedCollidable* collB, const hkpCdBody* firstNonTransformBodyB, hkUchar cdBodyHasTransformFlag, int agentType, const hkpProcessCollisionInput& input, hkpContactMgr* mgr )
{
	// handle flipping
	hkpCollisionDispatcher* dispatcher = input.m_dispatcher;

	// linkage
	// allocate new entry space
	int forceCollideOntoPpu = 0;
#if defined(HK_PLATFORM_HAS_SPU)
	{
		forceCollideOntoPpu = collA->m_forceCollideOntoPpu | collB->m_forceCollideOntoPpu;

		// send bridge agents to PPU
		if ( agentType == HK_AGENT_TYPE_BRIDGE && !forceCollideOntoPpu )
		{
			HK_WARN_ONCE( 0xf0235457, "This Collision Agent is is not supported on SPU. Are you using a hkpTransformShape? Or check your hkpAgentRegisterUtil.");
			forceCollideOntoPpu |= true;
		}

		// send report contact managers to PPU
		forceCollideOntoPpu |= ( mgr->m_type != hkpContactMgr::TYPE_SIMPLE_CONSTRAINT_CONTACT_MGR );
	}
#endif

	hkpAgentNnEntry* entry = hkAgentNnMachine_AllocateEntry(track, forceCollideOntoPpu);
	{
		entry->m_contactMgr = mgr;
		entry->m_nnTrackType = track.m_nnTrackType;

		entry->m_collidable[0] = collA;
		entry->m_collidable[1] = collB;
		entry->m_agentIndexOnCollidable[0] = hkObjectIndex(collA->getCollisionEntriesNonDeterministic().getSize()); 
		entry->m_agentIndexOnCollidable[1] = hkObjectIndex(collB->getCollisionEntriesNonDeterministic().getSize()); 
		
		hkpLinkedCollidable::CollisionEntry& centry0 = collA->getCollisionEntriesNonDeterministic().expandOne();
		hkpLinkedCollidable::CollisionEntry& centry1 = collB->getCollisionEntriesNonDeterministic().expandOne();
		centry0.m_agentEntry = entry;
		centry1.m_agentEntry = entry;
		centry0.m_partner = collB;
		centry1.m_partner = collA;
	}

	// create agent
	{
		hkpAgent3Input in3;
		{
			in3.m_bodyA = firstNonTransformBodyA;
			in3.m_bodyB = firstNonTransformBodyB;
			in3.m_contactMgr = entry->m_contactMgr;
			in3.m_input = &input;

			const hkMotionState* msA = firstNonTransformBodyA->getMotionState();
			const hkMotionState* msB = firstNonTransformBodyB->getMotionState();

			//	hkSweptTransformUtil::calcTimInfo( *msA, *msB, input.m_stepInfo.m_deltaTime, in3.m_linearTimInfo); this is not a process input
			in3.m_aTb.setMulInverseMul(msA->getTransform(), msB->getTransform());
		}
		
		hkpCollidableQualityType qTypeA = collA->getQualityType();
		hkpCollidableQualityType qTypeB = collB->getQualityType();
		entry->m_collisionQualityIndex  = hkUchar( input.m_dispatcher->getCollisionQualityIndex( qTypeA, qTypeB ) );
		input.m_collisionQualityInfo = input.m_dispatcher->getCollisionQualityInfo( entry->m_collisionQualityIndex );
		entry->m_agentType = hkUchar(agentType);
			 
		hkpAgentData* agentData;
		if ( dispatcher->getAgent3SepNormalFunc( agentType ) != HK_NULL )
		{
			hkpAgentNnMachineTimEntry* e = reinterpret_cast<hkpAgentNnMachineTimEntry*>( entry );

			e->m_streamCommand = hkUchar(hkAgent3::STREAM_CALL_WITH_TIM | cdBodyHasTransformFlag);
			e->m_timeOfSeparatingNormal = hkTime(-1.0f);
			e->m_separatingNormal.setZero();

			agentData = reinterpret_cast<hkpAgentData*>(e+1);
		}
		else
		{
			hkpAgentNnMachinePaddedEntry* e = reinterpret_cast<hkpAgentNnMachinePaddedEntry*>( entry );

			e->m_streamCommand = hkUchar(hkAgent3::STREAM_CALL | cdBodyHasTransformFlag);
			agentData = reinterpret_cast<hkpAgentData*>(e+1);
		}

		hkAgent3::CreateFunc createFunc = dispatcher->getAgent3CreateFunc( entry->m_agentType );
		HK_ON_DEBUG(hkpAgentData* agentEnd  =)
			createFunc( in3, entry, agentData );
		HK_ON_DEBUG(hkUlong size = hkGetByteOffset( entry, agentEnd ));
		HK_ASSERT2( 0xf0100404, size <= track.getAgentSize(), "Your agent's initial size is too big" );
		entry->m_size = hkUchar( track.getAgentSize() );
	}
	CHECK_TRACK( track );
	return entry;
}
#endif

#if !defined(HK_PLATFORM_SPU)
void HK_CALL hkAgentNnMachine_DestroyAgent( hkpAgentNnTrack& track, hkpAgentNnEntry* entry, hkpCollisionDispatcher* dispatcher, hkCollisionConstraintOwner& constraintOwner )
{
	HK_ASSERT2( 0xf45ba902, track.m_nnTrackType == entry->m_nnTrackType, "The entry's track type does not match the track's track type" );
	
	// destroy the agent
	{
		hkAgent3::StreamCommand command = hkAgent3::StreamCommand(entry->m_streamCommand);

		switch ( command )
		{
			case hkAgent3::STREAM_CALL:
			case hkAgent3::STREAM_CALL_AGENT:
			case hkAgent3::STREAM_CALL_WITH_TRANSFORM:
			case hkAgent3::STREAM_CALL_AGENT_WITH_TRANSFORM:
			{
				hkpAgentData* agentData = hkAddByteOffset( entry, hkSizeOf( hkpAgentNnMachinePaddedEntry ) );
				dispatcher->getAgent3DestroyFunc( entry->m_agentType )( entry, agentData, entry->m_contactMgr, constraintOwner, dispatcher );
				break;
			}
			case hkAgent3::STREAM_CALL_WITH_TIM:
			case hkAgent3::STREAM_CALL_WITH_TIM_WITH_TRANSFORM:
			{
				hkpAgentData* agentData = hkAddByteOffset( entry, hkSizeOf( hkpAgentNnMachineTimEntry ) );
				dispatcher->getAgent3DestroyFunc( entry->m_agentType )( entry, agentData, entry->m_contactMgr, constraintOwner, dispatcher );
				break;
			}
			case hkAgent3::STREAM_NULL:
				break;
		default: 
			HK_ASSERT2( 0xf0000001,0, "Unknown command in hkCdAgentStream");
		}

	}
	
	// unlink the agent from collisionEntry lists on both hkpLinkedCollidable's
	{
		for (int i = 0; i < 2; i++)
		{
			hkpLinkedCollidable* collidable = entry->m_collidable[i];
			int idxToDelete  = entry->m_agentIndexOnCollidable[i];

			collidable->getCollisionEntriesNonDeterministic().removeAt( idxToDelete );

			if (idxToDelete < collidable->getCollisionEntriesNonDeterministic().getSize())
			{
				hkpAgentNnEntry* otherEntry = collidable->getCollisionEntriesNonDeterministic()[idxToDelete].m_agentEntry;
				int idx = otherEntry->m_collidable[0] == collidable ? 0 : 1;
				otherEntry->m_agentIndexOnCollidable[idx] = hkObjectIndex(idxToDelete);
			}
			collidable->getCollisionEntriesNonDeterministic().optimizeCapacity( 4 );
		}
	}
	hkAgentNnMachine_InternalDeallocateEntry(track, entry);
	CHECK_TRACK( track );
}
#endif

#if !defined(HK_PLATFORM_SPU)
bool HK_CALL hkAgentNnMachine_IsEntryOnTrack(hkpAgentNnTrack& track, hkpAgentNnEntry* entry)
{
	HK_ASSERT2( 0xf45ba902, track.m_nnTrackType == entry->m_nnTrackType, "The entry's track type does not match the track's track type" );
	for (int i = 0; i < track.m_sectors.getSize(); i++)
	{
		hkpAgentNnSector* sector = track.m_sectors[i];
		hkpAgentNnEntry* sectorEnd = hkAddByteOffset( sector->getBegin(), track.getSectorSize( i ) );
		if ( sector->getBegin() <= entry && entry < sectorEnd )
		{
			return true;
		}
	}
	return false;
}
#endif
#if !defined(HK_PLATFORM_SPU)
void HK_CALL hkAgentNnMachine_InternalDeallocateEntry(hkpAgentNnTrack& track, hkpAgentNnEntry* entry)
{
	// check whether the entry actually resides in the specified track
	HK_ASSERT2(0xf0ff0083, hkAgentNnMachine_IsEntryOnTrack(track, entry), "Internal Error: delallocating an entry from an incorrect track.");

	// check if is not in the last sector
	int forceCollideOntoPpu = entry->m_forceCollideOntoPpu;
	hkpAgentNnEntry* lastEntry = hkAgentNnMachine_getLastEntry(track, forceCollideOntoPpu ); 

	// check if the agent is not the last track entry
	if (entry != lastEntry)
	{
		//copy the last entry
		hkString::memCpy16NonEmpty(entry, lastEntry, track.getAgentSize() / 16 );

		// relink the last agent
		hkAgentNnMachine_RelinkEntry(entry);
	}

	// deallocate last entry space
	hkAgentNnMachine_DeallocateLastEntry(track, forceCollideOntoPpu );
}
#endif
#if !defined(HK_PLATFORM_SPU)
void HK_CALL hkAgentNnMachine_UpdateShapeCollectionFilter( hkpAgentNnEntry* entry, const hkpCollisionInput& input, hkCollisionConstraintOwner& constraintOwner )
{
	hkpCdBody modBodyA[4];
	hkpCdBody modBodyB[4];
	hkMotionState modMotionA[4];
	hkMotionState modMotionB[4];

	const hkpCdBody* firstNonTransformBodyA = entry->m_collidable[0];
	const hkpCdBody* firstNonTransformBodyB = entry->m_collidable[1];

	hkAgent3::StreamCommand command = hkAgent3::StreamCommand(entry->m_streamCommand);

	hkpAgentData* agentData;
commandSwitch:
	switch ( command )
	{
	case hkAgent3::STREAM_CALL_WITH_TRANSFORM:
	case hkAgent3::STREAM_CALL_AGENT_WITH_TRANSFORM:
	case hkAgent3::STREAM_CALL_WITH_TIM_WITH_TRANSFORM:
		{
			hkPadSpu<hkUchar> dummyCdBodyHasTransformFlag = 0;
			firstNonTransformBodyA = hkAgentMachine_processTransformedShapes(firstNonTransformBodyA, modBodyA, modMotionA, 4, dummyCdBodyHasTransformFlag);
			firstNonTransformBodyB = hkAgentMachine_processTransformedShapes(firstNonTransformBodyB, modBodyB, modMotionB, 4, dummyCdBodyHasTransformFlag);

			command = static_cast<hkAgent3::StreamCommand> ( static_cast<hkUchar>(command) & static_cast<hkUchar>(~hkAgent3::TRANSFORM_FLAG) );
			goto commandSwitch;
		}
	case hkAgent3::STREAM_CALL:
	case hkAgent3::STREAM_CALL_AGENT:
		{
			agentData = hkAddByteOffset( entry, hkSizeOf( hkpAgentNnMachinePaddedEntry ) );
			goto callChildAgent;
		}
	case hkAgent3::STREAM_CALL_WITH_TIM:
		{
			agentData = hkAddByteOffset( entry, hkSizeOf( hkpAgentNnMachineTimEntry ) );
callChildAgent:
			hkAgent3::UpdateFilterFunc func = input.m_dispatcher->getAgent3UpdateFilterFunc( entry->m_agentType );
			if (func)
			{
				func( entry, agentData, *firstNonTransformBodyA, *firstNonTransformBodyB, input, entry->m_contactMgr, constraintOwner );
			}
			break;
		}
	case hkAgent3::STREAM_NULL:
		break;
	default: 
		HK_ASSERT2( 0xf0000001,0, "Unknown command in hkAgentNnMachine_UpdateShapeCollectionFilter");
	}
}
#endif


//
// SweptTranform update upon island [de]activation
//
#if !defined(HK_PLATFORM_SPU)
void HK_CALL hkAgentNnMachine_InvalidateTimInAgent( hkpAgentNnEntry* entry, const hkpCollisionInput& input )
{
	hkAgent3::StreamCommand command = hkAgent3::StreamCommand(entry->m_streamCommand);

	switch ( command )
	{
		case hkAgent3::STREAM_CALL:
		case hkAgent3::STREAM_CALL_AGENT:
		case hkAgent3::STREAM_CALL_WITH_TRANSFORM:
		case hkAgent3::STREAM_CALL_AGENT_WITH_TRANSFORM:
		{
			hkpAgentData* agentData = hkAddByteOffset( entry, hkSizeOf( hkpAgentNnMachinePaddedEntry ) );
			hkAgent3::InvalidateTimFunc func = input.m_dispatcher->getAgent3InvalidateTimFunc( entry->m_agentType );
			if (func)
			{
				func( entry, agentData, input );
			}
			break;
		}
		case hkAgent3::STREAM_CALL_WITH_TIM:
		case hkAgent3::STREAM_CALL_WITH_TIM_WITH_TRANSFORM:
		{
			hkpAgentData* agentData = hkAddByteOffset( entry, hkSizeOf( hkpAgentNnMachineTimEntry ) );
			hkpAgentNnMachineTimEntry* timEntry = static_cast<hkpAgentNnMachineTimEntry*>(entry);
			timEntry->m_timeOfSeparatingNormal = hkTime(-1.0f);
			timEntry->m_separatingNormal.setZero();
			hkAgent3::InvalidateTimFunc func = input.m_dispatcher->getAgent3InvalidateTimFunc( entry->m_agentType );
			if (func)
			{
				func( entry, agentData, input );
			}
			break;
		}
		case hkAgent3::STREAM_NULL:
			break;
	default: 
		HK_ASSERT2( 0xf0000001,0, "Unknown command in hkCdAgentStream");
	}
}
#endif

#if !defined(HK_PLATFORM_SPU)
void HK_CALL hkAgentNnMachine_WarpTimeInAgent( hkpAgentNnEntry* entry, hkTime oldTime, hkTime newTime, const hkpCollisionInput& input )
{
	hkAgent3::StreamCommand command = hkAgent3::StreamCommand(entry->m_streamCommand);

	switch ( command )
	{
		case hkAgent3::STREAM_CALL:
		case hkAgent3::STREAM_CALL_AGENT:
		case hkAgent3::STREAM_CALL_WITH_TRANSFORM:
		case hkAgent3::STREAM_CALL_AGENT_WITH_TRANSFORM:
		{
			hkpAgentData* agentData = hkAddByteOffset( entry, hkSizeOf( hkpAgentNnMachinePaddedEntry ) );
			hkAgent3::WarpTimeFunc func = input.m_dispatcher->getAgent3WarpTimeFunc( entry->m_agentType );
			if (func)
			{
				func( entry, agentData, oldTime, newTime, input );
			}
			break;
		}
		case hkAgent3::STREAM_CALL_WITH_TIM:
		case hkAgent3::STREAM_CALL_WITH_TIM_WITH_TRANSFORM:
		{
			hkpAgentData* agentData = hkAddByteOffset( entry, hkSizeOf( hkpAgentNnMachineTimEntry ) );
			hkpAgentNnMachineTimEntry* timEntry = static_cast<hkpAgentNnMachineTimEntry*>(entry);
			if (timEntry->m_timeOfSeparatingNormal == oldTime)
			{
				timEntry->m_timeOfSeparatingNormal = newTime;
			}
			else
			{
				//HK_ASSERT2(0xad000280, timEntry->m_timeOfSeparatingNormal == hkTime(-1.0f), "Internal Error: this value should be preset to -1.0 for all new agents.");
				timEntry->m_timeOfSeparatingNormal = hkTime(-1.0f);
			}
			hkAgent3::WarpTimeFunc func = input.m_dispatcher->getAgent3WarpTimeFunc( entry->m_agentType );
			if (func)
			{
				func( entry, agentData, oldTime, newTime, input );
			}
			break;
		}
		case hkAgent3::STREAM_NULL:
			break;
	default: 
		HK_ASSERT2( 0xf0000001,0, "Unknown command in hkCdAgentStream");
	}
}
#endif

//
// Merging and splitting tracks
//
#if !defined(HK_PLATFORM_SPU)
void HK_CALL hkAgentNnMachine_AppendTrack( hkpAgentNnTrack& track, hkpAgentNnTrack& appendee)
{
	HK_ASSERT2( 0xf45ba903, track.m_nnTrackType == appendee.m_nnTrackType, "Cannot append tracks of different types." );
#if !defined(HK_PLATFORM_HAS_SPU)

		// entries needed to fill up the last track.sector
	while (track.m_bytesUsedInLastSector != HK_AGENT3_SECTOR_SIZE && appendee.m_sectors.getSize())
	{
		const int unusedParam = 0;
		hkpAgentNnEntry* lastAppendee = hkAgentNnMachine_getLastEntry(appendee, unusedParam); 
		hkAgentNnMachine_CopyAndRelinkAgentEntry( track, lastAppendee);

		hkAgentNnMachine_DeallocateLastEntry(appendee, unusedParam);
	}
		// copy all remaining sectors
	if (appendee.m_sectors.getSize())
	{
		track.m_sectors.insertAt(track.m_sectors.getSize(), appendee.m_sectors.begin(), appendee.m_sectors.getSize());
		track.m_bytesUsedInLastSector = appendee.m_bytesUsedInLastSector;

		appendee.m_sectors.clear();
		appendee.m_bytesUsedInLastSector = HK_AGENT3_SECTOR_SIZE;
	}
#else
	// fill spu entries
	while (track.m_spuBytesUsedInLastSector != HK_AGENT3_SECTOR_SIZE && appendee.m_spuNumSectors )
	{
		hkpAgentNnEntry* lastAppendee = hkAgentNnMachine_getLastEntry(appendee, 0); 
		hkAgentNnMachine_CopyAndRelinkAgentEntry( track, lastAppendee);

		hkAgentNnMachine_DeallocateLastEntry(appendee,0);
	}

	// fill ppu entries
	while (track.m_ppuBytesUsedInLastSector!= HK_AGENT3_SECTOR_SIZE && appendee.getNumPpuSectors())
	{
		hkpAgentNnEntry* lastAppendee = hkAgentNnMachine_getLastEntry(appendee, 1); 
		hkAgentNnMachine_CopyAndRelinkAgentEntry( track, lastAppendee);

		hkAgentNnMachine_DeallocateLastEntry(appendee,1);
	}

	// copy all remaining sectors
	// spu first
	if (appendee.m_spuNumSectors)
	{
		track.m_sectors.insertAt(track.m_spuNumSectors, appendee.m_sectors.begin(), appendee.m_spuNumSectors );
		track.m_spuNumSectors = track.m_spuNumSectors + appendee.m_spuNumSectors;
		track.m_spuBytesUsedInLastSector = appendee.m_spuBytesUsedInLastSector;
	}

	// ppu second
	int numPpuSectors = appendee.getNumPpuSectors();
	if ( numPpuSectors)
	{
		track.m_sectors.insertAt(track.m_sectors.getSize(), appendee.m_sectors.begin() + appendee.m_spuNumSectors, numPpuSectors );
		track.m_ppuBytesUsedInLastSector = appendee.m_ppuBytesUsedInLastSector;
	}

	appendee.m_sectors.clear();
	appendee.m_ppuBytesUsedInLastSector = HK_AGENT3_SECTOR_SIZE;
	appendee.m_spuNumSectors = 0;
	appendee.m_spuBytesUsedInLastSector = HK_AGENT3_SECTOR_SIZE;
#endif
	CHECK_TRACK( track );
}
#endif

#if ! defined (HK_PLATFORM_SPU)
hkpAgentNnEntry* HK_CALL hkAgentNnMachine_CopyAndRelinkAgentEntry( hkpAgentNnTrack& destTrack, hkpAgentNnEntry* entry )
{
	HK_ASSERT2( 0xf45ba902, destTrack.m_nnTrackType == entry->m_nnTrackType, "The entry's track type does not match the track's track type" );

		// allocate new entry space
	hkpAgentNnEntry* newEntry = hkAgentNnMachine_AllocateEntry(destTrack, entry->m_forceCollideOntoPpu );

		// copy and relink
	hkString::memCpy16NonEmpty(newEntry, entry, destTrack.getAgentSize() / 16 );
	hkAgentNnMachine_RelinkEntry(newEntry);

	return newEntry;
}
#endif

//
// Destruction
//

#if ! defined (HK_PLATFORM_SPU)
void HK_CALL hkAgentNnMachine_DestroyTrack( hkpAgentNnTrack& track, hkpCollisionDispatcher* dispatcher, hkCollisionConstraintOwner& constraintOwner )
{
	HK_ASSERT(0xf0ff0002, !track.m_sectors.getSize() );

	// and if you really want it
#if !defined(HK_PLATFORM_HAS_SPU)
	while(track.m_sectors.getSize())
	{
		hkpAgentNnEntry* lastEntry = hkAgentNnMachine_getLastEntry(track,0);
		hkAgentNnMachine_DestroyAgent(track, lastEntry ,dispatcher, constraintOwner);
	}
#else
	// PPU first
	{
		while(track.m_sectors.getSize() > track.m_spuNumSectors )
		{
			hkpAgentNnEntry* lastEntry = hkAgentNnMachine_getLastEntry(track, 1);
			hkAgentNnMachine_DestroyAgent(track, lastEntry ,dispatcher, constraintOwner);
		}
	}
	// SPU next
	{
		while(track.m_sectors.getSize() )
		{
			hkpAgentNnEntry* lastEntry = hkAgentNnMachine_getLastEntry(track, 0);
			hkAgentNnMachine_DestroyAgent(track, lastEntry ,dispatcher, constraintOwner);
		}
	}
#endif
}
#endif

void HK_CALL hkAgentNnMachine_ProcessAgent( hkpAgentNnEntry* entry, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& output, hkpContactMgr* mgr  )
{
	int numTim = 0;
	hkAgentNnMachine_InlineProcessAgent( entry, input, numTim, output, mgr );
}

#if ! defined (HK_PLATFORM_SPU)
void HK_CALL hkAgentNnMachine_ProcessTrack( hkpConstraintOwner* owner, hkpAgentNnTrack& agentTrack, const hkpProcessCollisionInput& input  )
{
	int numTim = 0;
	HK_ALIGN16(typedef hkpProcessCollisionOutput hkAlignedCollisionOutput);
	hkAlignedCollisionOutput processOutput(owner);

	HK_FOR_ALL_AGENT_ENTRIES_BEGIN( agentTrack, entry )
	{
		hkpCollidable* collA = entry->getCollidableA();
		hkpCollidable* collB = entry->getCollidableB();
		{
			processOutput.reset();
			hkMath::prefetch128( entry->m_contactMgr );
			hkMath::prefetch128( hkAddByteOffset(entry, 128) );

			hkAgentNnMachine_InlineProcessAgent( entry, input, numTim, processOutput, entry->m_contactMgr );

			if (hkMemoryStateIsOutOfMemory(19)  )
			{
				return;
			}

			if ( !processOutput.isEmpty() )
			{
				entry->m_contactMgr->processContact( *collA, *collB, input, processOutput );
			}
		}
	}
	HK_FOR_ALL_AGENT_ENTRIES_END;
	HK_MONITOR_ADD_VALUE( "numTim", float(numTim), HK_MONITOR_TYPE_INT);
}
#endif
//
// External Utilities
//

#if ! defined (HK_PLATFORM_SPU)
hkpAgentNnEntry* HK_CALL hkAgentNnMachine_FindAgent( const hkpLinkedCollidable* collA, const hkpLinkedCollidable* collB )
{

	// Get the agent and its island (the agent may be in either collidable's island (and those may not be yet merged))
	const hkpLinkedCollidable* collidable;
	const hkpLinkedCollidable* otherCollidable;
	if (collA->getCollisionEntriesNonDeterministic().getSize() <= collB->getCollisionEntriesNonDeterministic().getSize())
	{
		collidable = collA;
		otherCollidable = collB;
	}
	else
	{
		collidable = collB;
		otherCollidable = collA;
	}

	const hkArray<hkpLinkedCollidable::CollisionEntry>& entries = collidable->getCollisionEntriesNonDeterministic();
	int i;
	for (i = 0; i < entries.getSize(); i++)
	{
		if (entries[i].m_partner == otherCollidable)
		{
			return entries[i].m_agentEntry;
		}
	}

	return HK_NULL;
}


//
// Debugging
//

void HK_CALL hkAgentNnMachine_AssertTrackValidity( hkpAgentNnTrack& track )
{
	// go through all agents
	HK_FOR_ALL_AGENT_ENTRIES_BEGIN(track, entry)
	{
		HK_ASSERT2( 0xf45ba902, track.m_nnTrackType == entry->m_nnTrackType, "The entry's track type does not match the track's track type" );
		//HK_ON_DEBUG(const hkCollidablePair* entry = &island->m_collisionAgents[a]);

		HK_ON_DEBUG(hkObjectIndex idx0 = entry->m_agentIndexOnCollidable[0]);
		HK_ON_DEBUG(hkObjectIndex idx1 = entry->m_agentIndexOnCollidable[1]);
		// verify that onEntityList indices are ok for both bodies

		// verify that back-pointers on both entities are ok
		HK_ASSERT(0xf0ff0084, entry->m_collidable[0]->getCollisionEntriesNonDeterministic()[idx0].m_agentEntry == entry);
		HK_ASSERT(0xf0ff0087, entry->m_collidable[1]->getCollisionEntriesNonDeterministic()[idx1].m_agentEntry == entry);

		// verify inter-entity pointers on both entities
		HK_ASSERT(0xf0ff0085, entry->m_collidable[0]->getCollisionEntriesNonDeterministic()[idx0].m_partner == entry->m_collidable[1]);
		HK_ASSERT(0xf0ff0086, entry->m_collidable[1]->getCollisionEntriesNonDeterministic()[idx1].m_partner == entry->m_collidable[0]);

		// verify that collidables do not have doubled pointers to agentEntries

#if defined(HK_PLATFORM_HAS_SPU)
		HK_ON_DEBUG( int forcePpu = entry->m_collidable[0]->m_forceCollideOntoPpu | entry->m_collidable[1]->m_forceCollideOntoPpu; )
		HK_ON_DEBUG( if ( forcePpu ){HK_ASSERT( 0xf04e3456, entry->m_forceCollideOntoPpu ); } )
		HK_ASSERT( 0xf04e3476, (entry->m_forceCollideOntoPpu > 0) ==  (HKLOOP_sectorIndex > track.m_spuNumSectors) );
#endif
#if defined(HK_DEBUG)
		if (true)
		{
			for (int e = 0; e < 2; e++)
			{
				int agentPresent = false;
				hkArray<hkpLinkedCollidable::CollisionEntry>& entries = entry->m_collidable[e]->getCollisionEntriesNonDeterministic();
				for (int i = 0; i < entries.getSize(); i++)
				{
					if (entries[i].m_agentEntry == entry)
					{
						HK_ASSERT2(0xf0ff0002, !agentPresent, "Duplicated Pntr to an agent entry found");
						agentPresent = true;
					}
				}
				HK_ASSERT2(0xf0ff0004, agentPresent, "Cannot find an agent");
			}
		}
#endif
	}
	HK_FOR_ALL_AGENT_ENTRIES_END;
}
#endif

const hkpCdBody* hkAgentMachine_processTransformedShapes(const hkpCdBody* cdBody, hkpCdBody* newCdBodies, hkMotionState* newMotionStates, int numSlots, hkPadSpu<hkUchar>& cdBodyHasTransformFlag)
{
	const hkpShape* shape = cdBody->m_shape;
	while(shape->getType() == hkcdShapeType::TRANSFORM)
	{
		HK_ASSERT2(0xad348329, numSlots, "Too many hkpTransformShapes in a row. The max supported number is 4. Will crash now.");

		const hkpTransformShape* transformShape = static_cast<const hkpTransformShape*>(shape);

		const hkMotionState& oldMotion = *cdBody->getMotionState();
		hkMotionState& newMotion = *newMotionStates;

		// Calculate new combined hkMotionState
		{
			//	Calc transform
			//
			// inline this: newMotion.getTransform().setMul( oldMotion.getTransform(), transformShape->getTransform());
			{
				hkTransform& thisT = newMotion.getTransform();
				const hkTransform& aTb = oldMotion.getTransform();
				const hkTransform& bTc = transformShape->getTransform();

				HK_ASSERT(0x4763da71,  &thisT != &aTb );
				hkVector4Util::rotatePoints( aTb.getRotation(), &bTc.getRotation().getColumn<0>(), 4, &thisT.getRotation().getColumn(0) );
				thisT.getTranslation().add( aTb.getTranslation());
			}

			//	Calc swept transform
			//
			{
				hkSweptTransform& newSwept = newMotion.getSweptTransform();
				const hkSweptTransform& oldSwept = oldMotion.getSweptTransform();

				newSwept.m_centerOfMass0 = oldSwept.m_centerOfMass0;
				newSwept.m_centerOfMass1 = oldSwept.m_centerOfMass1;

				newSwept.m_rotation0.setMul( oldSwept.m_rotation0, transformShape->getRotation() );
				newSwept.m_rotation1.setMul( oldSwept.m_rotation1, transformShape->getRotation() );

				newSwept.m_centerOfMassLocal._setTransformedInversePos( transformShape->getTransform(), oldSwept.m_centerOfMassLocal );
			}
			newMotion.m_deltaAngle = oldMotion.m_deltaAngle;
			newMotion.m_objectRadius = oldMotion.m_objectRadius;

		}

		// Create new hkpCdBody
		new (newCdBodies) hkpCdBody( cdBody, newMotionStates );
		newCdBodies->setShape(transformShape->getChildShape(), 0);

		// Move pointers
		cdBody = newCdBodies;
		newCdBodies++;
		newMotionStates++;

		// Assign shape to perform loop test
		shape = cdBody->m_shape;

		HK_ON_DEBUG(numSlots--);

		cdBodyHasTransformFlag = hkAgent3::TRANSFORM_FLAG;
	}

	return cdBody;

}

void HK_CALL hkAgentNnMachine_TouchAgent( hkpAgentEntry* entry,	const hkpProcessCollisionInput& input ) {}

// If this ceases to be true, please edit hkpAgentNnTrack::getAgentSize.
HK_COMPILE_TIME_ASSERT( HK_AGENT3_MIDPHASE_AGENT_SIZE * 2 == HK_AGENT3_NARROWPHASE_AGENT_SIZE );
HK_COMPILE_TIME_ASSERT( HK_AGENT3_SECTOR_SIZE % HK_AGENT3_MIDPHASE_AGENT_SIZE == 0 );
HK_COMPILE_TIME_ASSERT( HK_AGENT3_SECTOR_SIZE % HK_AGENT3_NARROWPHASE_AGENT_SIZE == 0 );

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
