/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Dispatch/Agent3Bridge/hkpAgent3Bridge.h>
#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>


int hkAgent3Bridge::registerAgent3( hkpCollisionDispatcher* dispatcher )
{
	hkpCollisionDispatcher::Agent3Funcs f;
	f.m_commitPotentialFunc = commitPotential;
	f.m_createZombieFunc    = createZombie;
	f.m_removePointFunc     = removePoint;

	f.m_createFunc   = create;
	f.m_processFunc  = process;
	f.m_sepNormalFunc = HK_NULL; //sepNormalFunc;
	f.m_cleanupFunc  = HK_NULL;
	f.m_destroyFunc  = destroy;
	f.m_updateFilterFunc = updateFilter;
	f.m_invalidateTimFunc = invalidateTim;
	f.m_warpTimeFunc = warpTime;
	f.m_isPredictive = true;
	int id = dispatcher->registerAgent3( f, hkcdShapeType::ALL_SHAPE_TYPES, hkcdShapeType::ALL_SHAPE_TYPES );
	return id;
}

hkpAgentData* hkAgent3Bridge::create  ( const hkpAgent3Input& input, hkpAgentEntry* entry, hkpAgentData* agentData )
{
	const hkUchar cdBodyHasTransformFlag = static_cast<hkUchar>( entry->m_streamCommand & hkAgent3::TRANSFORM_FLAG );
	entry->m_streamCommand = static_cast<hkUchar>(hkAgent3::STREAM_CALL_AGENT | cdBodyHasTransformFlag);
	getChildAgent(agentData) = input.m_input->m_dispatcher->getNewCollisionAgent( input.m_bodyA[0], input.m_bodyB[0], input.m_input[0], input.m_contactMgr );
	entry->m_numContactPoints = hkUchar(-1);
	return getAgentDataEnd(agentData);
}


hkpAgentData* hkAgent3Bridge::process ( const hkpAgent3ProcessInput& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkVector4* separatingNormalOut, hkpProcessCollisionOutput& result)
{
	HK_WARN_ONCE(0xf0ff00b0, "hkAgent3Bridge::process should never be called" ); // should never be called, as the stuff is inlined anyway
	hkpCollisionAgent* agent = getChildAgent(agentData);
	agent->processCollision( input.m_bodyA[0], input.m_bodyB[0], input.m_input[0], result );
	return getAgentDataEnd(agentData);
}


void hkAgent3Bridge::destroy ( hkpAgentEntry* entry, hkpAgentData* agentData, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner, hkpCollisionDispatcher* dispatcher )
{
	hkpCollisionAgent* agent = getChildAgent(agentData);
	agent->cleanup(constraintOwner);
}

void hkAgent3Bridge::updateFilter(hkpAgentEntry* entry, hkpAgentData* agentData, const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner)
{
	hkpCollisionAgent* agent = getChildAgent(agentData);
	agent->updateShapeCollectionFilter(bodyA, bodyB, input, constraintOwner);
}

void hkAgent3Bridge::invalidateTim(hkpAgentEntry* entry, hkpAgentData* agentData, const hkpCollisionInput& input)
{
	hkpCollisionAgent* agent = getChildAgent(agentData);
	agent->invalidateTim(input);
}

void hkAgent3Bridge::warpTime(hkpAgentEntry* entry, hkpAgentData* agentData, hkTime oldTime, hkTime newTime, const hkpCollisionInput& input)
{
	hkpCollisionAgent* agent = getChildAgent(agentData);
	agent->warpTime(oldTime, newTime, input);
}


void HK_CALL hkAgent3Bridge::removePoint( hkpAgentEntry* entry, hkpAgentData* agentData, hkContactPointId idToRemove )
{
	hkpCollisionAgent* agent = getChildAgent(agentData);
	agent->removePoint( idToRemove );
}

void HK_CALL hkAgent3Bridge::commitPotential( hkpAgentEntry* entry, hkpAgentData* agentData, hkContactPointId newId )
{
	hkpCollisionAgent* agent = getChildAgent(agentData);
	agent->commitPotential( newId );
}

void HK_CALL hkAgent3Bridge::createZombie( hkpAgentEntry* entry, hkpAgentData* agentData, hkContactPointId idTobecomeZombie )
{
	hkpCollisionAgent* agent = getChildAgent(agentData);
	agent->createZombie( idTobecomeZombie );
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
