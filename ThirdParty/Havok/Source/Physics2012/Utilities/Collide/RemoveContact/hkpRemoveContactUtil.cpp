/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Collide/RemoveContact/hkpRemoveContactUtil.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent3/PredGskAgent3/hkpPredGskAgent3.h>
#include <Physics2012/Collide/Agent3/ConvexList3/hkpConvexListAgent3.h>
#include <Physics2012/Collide/Agent/hkpCollisionAgent.h>

// Agents
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>
#include <Physics2012/Collide/Agent/ConvexAgent/SphereCapsule/hkpSphereCapsuleAgent.h>
#include <Physics2012/Collide/Agent/HeightFieldAgent/hkpHeightFieldAgent.h>
#include <Physics2012/Collide/Agent3/BoxBox/hkpBoxBoxAgent3.h>
#include <Physics2012/Collide/Agent3/CapsuleTriangle/hkpCapsuleTriangleAgent3.h>
#include <Physics2012/Collide/Agent3/PredGskAgent3/hkpPredGskAgent3.h>
#include <Physics2012/Collide/Agent3/PredGskCylinderAgent3/hkpPredGskCylinderAgent3.h>
#include <Physics2012/Collide/Agent3/List3/hkpListAgent3.h>
#include <Physics2012/Collide/Agent3/ConvexList3/hkpConvexListAgent3.h>
#include <Physics2012/Collide/Agent3/CollectionCollection3/hkpCollectionCollectionAgent3.h>
#include <Physics2012/Collide/Agent3/Collection3/hkpCollectionAgent3.h>
#include <Physics2012/Collide/Agent3/BvTree3/hkpBvTreeAgent3.h>


#include <Common/Base/Reflection/hkClass.h>

void HK_CALL hkpRemoveContactUtil::removeContactPoint( hkpAgentNnEntry* entry, const hkpWorld* world, hkContactPointId idToRemove )
{
	removeCollisionEntry( entry, world->getCollisionInput(), idToRemove );
}

void HK_CALL hkpRemoveContactUtil::removeContactPoint( hkpAgentNnEntry* entry, const hkpProcessCollisionInput* collisionInput, hkContactPointId idToRemove )
{
	removeCollisionEntry( entry, collisionInput, idToRemove );
}

void HK_CALL hkpRemoveContactUtil::removeContactPoints( const hkpEntity** entities, int numEntities )
{
	if (numEntities == 0)
	{
		return;
	}

	hkpWorld* world = entities[0]->getWorld();
	HK_ASSERT2(0x45cc964e, world, "Attempting to remove contact points of an hkpEntity that's not in an hkpWorld.");

	// Build map of entities we're interested in
	hkMap<hkUlong> processedCollidables( numEntities );

	for (int e = 0; e < numEntities; e++)
	{
		HK_ASSERT2(0x12e6cdd0, entities[e]->getWorld() == world, "The entities in this list must belong to the same world" );

		processedCollidables.insert( hkUlong(entities[e]->getLinkedCollidable()), 0 );

		const hkArray<hkpLinkedCollidable::CollisionEntry>& collisionEntries = entities[e]->getLinkedCollidable()->getCollisionEntriesDeterministicUnchecked();

		for (int c = 0; c < collisionEntries.getSize(); c++)
		{
			const hkpLinkedCollidable::CollisionEntry& entry = collisionEntries[c];
			hkMap<hkUlong>::Iterator it = processedCollidables.findKey( hkUlong(entry.m_partner) );
			if (!processedCollidables.isValid(it))
			{
				hkpContactMgr *const cmgr = collisionEntries[c].m_agentEntry->m_contactMgr;
				if ( cmgr->m_type == hkpContactMgr::TYPE_SIMPLE_CONSTRAINT_CONTACT_MGR )
				{
					hkpSimpleConstraintContactMgr *const mgr = static_cast< hkpSimpleConstraintContactMgr* > ( cmgr );

					hkArray<hkContactPointId> contactPointIds;
					mgr->getAllContactPointIds( contactPointIds );

					// Remove all contact points...
					for (int i = 0; i < contactPointIds.getSize(); ++i)
					{
						removeContactPoint( entry.m_agentEntry, world->getCollisionInput(), contactPointIds[i] );
					}
				}
			}
		}
	}
}

//
// Removes the contact point id from the agent involved in this collision
// Only works for manifold contacts
//
hkResult HK_CALL hkpRemoveContactUtil::removeCollisionEntry( hkpAgentNnEntry* entry, const hkpProcessCollisionInput* collisionInput, hkContactPointId idToRemove )
{
	if (((hkpDynamicsContactMgr*)(entry->m_contactMgr))->getType() == hkpContactMgr::TYPE_SIMPLE_CONSTRAINT_CONTACT_MGR &&
		((hkpSimpleConstraintContactMgr*)(entry->m_contactMgr))->m_contactConstraintData.m_atom->m_numContactPoints > 0 )
	{
		int sizeOfThisEntry = -1;
		const hkBool nnEntry = true;

		// Try to remove the contact point from the entry (recursive)
		if (HK_SUCCESS == removeEntry(entry, nnEntry, collisionInput, idToRemove, sizeOfThisEntry ))
		{
			// Finally, remove the contact point id from the manager
			hkpSimpleConstraintContactMgr *const mgr = static_cast< hkpSimpleConstraintContactMgr* > ( entry->m_contactMgr );
			mgr->removeContactPoint( idToRemove, *mgr->m_constraint.getOwner() );

			return HK_SUCCESS;
		}
		else
		{
			return HK_FAILURE;
		}
	}

	// Contact manager is not hkpContactMgr::TYPE_SIMPLE_CONSTRAINT_CONTACT_MGR or no contact points
	return HK_FAILURE;
}

//
// Find appropriate agentData and call the removePoint method
// For shape collections this is done recursively
//
hkResult HK_CALL hkpRemoveContactUtil::removeEntry( hkpAgentEntry* entry, hkBool isNnEntry, const hkpProcessCollisionInput* input, const hkContactPointId& idToRemove, int& sizeOfThisEntryOut )
{
	hkpAgentData* agentData;
	hkAgent3::StreamCommand command = static_cast<hkAgent3::StreamCommand>(entry->m_streamCommand);
	AgentType agentType;

	switch ( command )
	{
		// The non-stream agent should be removed
	case hkAgent3::STREAM_CALL_FLIPPED:
	case hkAgent3::STREAM_CALL_FLIPPED_WITH_TRANSFORM:
	case hkAgent3::STREAM_CALL_AGENT:
	case hkAgent3::STREAM_CALL_AGENT_WITH_TRANSFORM:
		{
			// Agent 2, use hkAgent3Bridge
			agentData = hkAddByteOffset( entry, isNnEntry ? hkSizeOf( hkpAgentNnMachinePaddedEntry ) : hkSizeOf( hkpAgent1nMachinePaddedEntry ) ); 
			hkAgent3Bridge::removePoint( entry, agentData, idToRemove );
			return HK_SUCCESS;
		}
	case hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED:
	case hkAgent3::STREAM_CALL_WITH_TIM_FLIPPED_WITH_TRANSFORM:
	case hkAgent3::STREAM_CALL_WITH_TIM:
	case hkAgent3::STREAM_CALL_WITH_TIM_WITH_TRANSFORM:
		{
			agentData = hkAddByteOffset( entry, isNnEntry ? hkSizeOf( hkpAgentNnMachineTimEntry ) : hkSizeOf( hkpAgent1nMachineTimEntry ) ); 
			goto continueConvertEntryToSerialized;
		}
	case hkAgent3::STREAM_CALL:
	case hkAgent3::STREAM_CALL_WITH_TRANSFORM:
		{
			agentData = hkAddByteOffset( entry, isNnEntry ? hkSizeOf( hkpAgentNnMachinePaddedEntry ) : hkSizeOf( hkpAgent1nMachinePaddedEntry ) );
continueConvertEntryToSerialized:		

			hkAgent3::ProcessFunc func = input->m_dispatcher->getAgent3ProcessFunc( entry->m_agentType );
			agentType = getAgentType(func);

			switch(agentType)
			{
			case hkpSerializedAgentNnEntry::BOX_BOX_AGENT3:
			case hkpSerializedAgentNnEntry::CAPSULE_TRIANGLE_AGENT3:
			case hkpSerializedAgentNnEntry::PRED_GSK_AGENT3:
			case hkpSerializedAgentNnEntry::PRED_GSK_CYLINDER_AGENT3:
				{
					agentType = getAgentType(func);
					// moved up hkString::memCpy( entryOut.m_nnEntryData, entry, HK_AGENT3_AGENT_SIZE);
					HK_ASSERT2(0xad7644dd, !isNnEntry || ( entry->m_size == hkpAgentNnTrack::getAgentSize( static_cast<hkpAgentNnEntry*>( entry )->m_nnTrackType ) ), "Top level (nn) entry of wrong size." );
					sizeOfThisEntryOut = entry->m_size;

					// Simple type, remove this contact point and exit
					hkAgent3::RemovePointFunc removeFunc = input->m_dispatcher->getAgent3RemovePointFunc( entry->m_agentType );
					removeFunc( entry, agentData, idToRemove );

					return HK_SUCCESS;
				}
			case hkpSerializedAgentNnEntry::CONVEX_LIST_AGENT3:
				{
					agentType = getAgentType(func);
					// moved up hkString::memCpy( entryOut.m_nnEntryData, entry, HK_AGENT3_AGENT_SIZE);
					HK_ASSERT2(0xad7644dd, !isNnEntry || ( entry->m_size == hkpAgentNnTrack::getAgentSize( static_cast<hkpAgentNnEntry*>( entry )->m_nnTrackType ) ), "Top level (nn) entry of wrong size." );

					sizeOfThisEntryOut = entry->m_size;
					if (!hkPredGskAgent3::getGskFlag(agentData, hkpGskCache::GSK_FLAGS_CONVEX_LIST_IN_GSK_MODE))
					{
						// List shape; recursively check sub tracks until contact is found and removed
						const hkpAgent1nTrack* agent1nTrack = hkConvexListAgent3::getAgent1nTrack(entry, agentData);
						return processSubTracks(agent1nTrack->m_sectors, input, idToRemove);
					}
					return HK_SUCCESS;
				}
			case hkpSerializedAgentNnEntry::LIST_AGENT3:
			case hkpSerializedAgentNnEntry::BV_TREE_AGENT3:
			case hkpSerializedAgentNnEntry::COLLECTION_COLLECTION_AGENT3:
			case hkpSerializedAgentNnEntry::COLLECTION_AGENT3:
				{
					sizeOfThisEntryOut = entry->m_size; 
					agentType = getAgentType(func);
					const hkpAgent1nTrack* agent1nTrack = reinterpret_cast<const hkpAgent1nTrack*>(agentData);
					return processSubTracks(agent1nTrack->m_sectors, input, idToRemove);
				}
			default:
				{
					if (func == hkAgent3Bridge::process)
					{
						hkAgent3Bridge::removePoint( entry, agentData, idToRemove );
						return HK_SUCCESS;
					}

					HK_ASSERT2(0xad54baa1, false, "Unsupported agent3 type. Unregister some agent3 agents.");
					return HK_FAILURE;
				}
			}
		}

	case hkAgent3::STREAM_NULL:
	case hkAgent3::STREAM_END:
		{
			HK_ASSERT2(0xad7644dd, !isNnEntry, "Internal error. Stream null & stream end commands not supported for NnEntries.");
			sizeOfThisEntryOut = sizeof( hkpAgent1nMachinePaddedEntry );
			return HK_SUCCESS;
		}
	case hkAgent3::TRANSFORM_FLAG:
		{
			// This case is included to avoid warnings about enumeration value not handled in a switch.
			break;
		}
	default:
		{
			HK_ASSERT2(0x5ed1a61b, false, "Unhandled command in stream.");
			return HK_FAILURE;
		}

	}

	return HK_SUCCESS;

}

//
// The agent is a shape collection, so read the sub tracks
// and recursively process to reach the leaf agents
//
hkResult HK_CALL hkpRemoveContactUtil::processSubTracks( const hkArray<hkpAgent1nSector*>& sectorsIn, const hkpProcessCollisionInput* input, const hkContactPointId& idToRemove )
{
	for (int i = 0; i < sectorsIn.getSize(); ++i)
	{
		hkpAgent1nSector* readSector = sectorsIn[i];
		hkpAgentData* readData = readSector->getBegin();
		hkpAgentData* readEnd = readSector->getEnd();

		// Make sure the sector is ok
		while (readData < readEnd)
		{
			hkpAgentEntry* entry = reinterpret_cast<hkpAgentEntry*>(readData);
			const hkBool nonNnEntry = false;

			int sizeOfThisEntry = 0;

			const hkUchar numContacts = entry->m_numContactPoints;

			if ( HK_SUCCESS == removeEntry(entry, nonNnEntry, input, idToRemove, sizeOfThisEntry ) )
			{
				// If we removed a contact, then we're done;
				// otherwise, go on to the next agent data

				// This check won't work for Agent2 types
				if (numContacts != entry->m_numContactPoints)
				{
					return HK_SUCCESS;
				}
				else
				{
					readData = hkAddByteOffset(readData, sizeOfThisEntry);
				}
			}
			else
			{
				return HK_FAILURE;
			}
		}
	}

	return HK_SUCCESS;
}

hkpRemoveContactUtil::AgentType HK_CALL hkpRemoveContactUtil::getAgentType(hkAgent3::ProcessFunc func)
{
	if ( func == hkBoxBoxAgent3::process )						{ return hkpRemoveContactUtil::BOX_BOX_AGENT3; } 
	else if ( func == hkCapsuleTriangleAgent3::process )			{ return hkpRemoveContactUtil::CAPSULE_TRIANGLE_AGENT3; }
	else if ( func == hkPredGskAgent3::process )					{ return hkpRemoveContactUtil::PRED_GSK_AGENT3; }
	else if ( func == hkPredGskCylinderAgent3::process )			{ return hkpRemoveContactUtil::PRED_GSK_CYLINDER_AGENT3; }
	else if ( func == hkConvexListAgent3::process )					{ return hkpRemoveContactUtil::CONVEX_LIST_AGENT3; }
	else if ( func == hkListAgent3::process )						{ return hkpRemoveContactUtil::LIST_AGENT3; }
	else if ( func == hkBvTreeAgent3::process )						{ return hkpRemoveContactUtil::BV_TREE_AGENT3; }
	else if ( func == hkpCollectionCollectionAgent3::process )		{ return hkpRemoveContactUtil::COLLECTION_COLLECTION_AGENT3; }
	else if ( func == hkpCollectionAgent3::process )				{ return hkpRemoveContactUtil::COLLECTION_AGENT3; }
	else 
	{
		return hkpRemoveContactUtil::INVALID_AGENT_TYPE;
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
