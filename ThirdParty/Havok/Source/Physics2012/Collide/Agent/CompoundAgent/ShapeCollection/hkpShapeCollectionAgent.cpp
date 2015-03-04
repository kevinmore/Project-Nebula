/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Shape/hkpShapeContainer.h>

#include <Physics2012/Collide/Agent/CompoundAgent/ShapeCollection/hkpShapeCollectionAgent.h>
#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>

void HK_CALL hkpShapeCollectionAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createListBAgent;
		af.m_getPenetrationsFunc  = hkpSymmetricAgent<hkpShapeCollectionAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc = hkpSymmetricAgent<hkpShapeCollectionAgent>::staticGetClosestPoints;
		af.m_linearCastFunc      = hkpSymmetricAgent<hkpShapeCollectionAgent>::staticLinearCast;
		af.m_isFlipped           = true;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::ALL_SHAPE_TYPES, hkcdShapeType::COLLECTION );
		dispatcher->registerCollisionAgent(af, hkcdShapeType::ALL_SHAPE_TYPES, hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION );
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createListAAgent;
		af.m_getPenetrationsFunc  = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::COLLECTION, hkcdShapeType::ALL_SHAPE_TYPES );
		dispatcher->registerCollisionAgent(af, hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION, hkcdShapeType::ALL_SHAPE_TYPES );
	}
}


hkpCollisionAgent* HK_CALL hkpShapeCollectionAgent::createListAAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, 
									const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpShapeCollectionAgent* agent = new hkpShapeCollectionAgent(bodyA, bodyB, input, mgr);

	return agent;
}


hkpCollisionAgent* HK_CALL hkpShapeCollectionAgent::createListBAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, 
									const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpShapeCollectionAgent* agent = new hkpSymmetricAgent<hkpShapeCollectionAgent>(bodyA, bodyB, input, mgr);
	
	return agent;
}


hkpShapeCollectionAgent::hkpShapeCollectionAgent(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
: hkpCollisionAgent( mgr )
{
	hkpCdBody newOperandA( &bodyA );

	//
	// initialize all the new child agents
	//
	const hkpShapeContainer* shapeContainer = bodyA.getShape()->getContainer();
	HK_ASSERT2(0x75845342, shapeContainer != HK_NULL, "Shape collection agent called where bodyA is not a shape container");

	int numChildren = shapeContainer->getNumChildShapes();
	m_agents.reserve( numChildren );
	
	hkpShapeBuffer shapeBuffer;
	{
	    for ( hkpShapeKey key = shapeContainer->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = shapeContainer->getNextKey(key) )
	    {
		    newOperandA.setShape( shapeContainer->getChildShape( key, shapeBuffer), key );
		    if ( input.m_filter->isCollisionEnabled( input, bodyB, bodyA, *shapeContainer , key ) )
		    {
			    KeyAgentPair& ap = *m_agents.expandByUnchecked(1);
			    ap.m_agent = input.m_dispatcher->getNewCollisionAgent(newOperandA, bodyB, input, mgr);
			    ap.m_key = key;
		    }
	    }
    }
}


void hkpShapeCollectionAgent::cleanup( hkCollisionConstraintOwner& info)
{
	for (int i = 0; i < m_agents.getSize(); ++i)
	{
		m_agents[i].m_agent->cleanup( info );
	}
	delete this;
}

void hkpShapeCollectionAgent::invalidateTim( const hkpCollisionInput& input )
{
	for (int i = 0; i < m_agents.getSize(); i++)
	{
		m_agents[i].m_agent->invalidateTim(input);
	}
}

void hkpShapeCollectionAgent::warpTime( hkTime oldTime, hkTime newTime, const hkpCollisionInput& input )
{
	for (int i = 0; i < m_agents.getSize(); i++)
	{
		m_agents[i].m_agent->warpTime(oldTime, newTime, input);
	}
}

void hkpShapeCollectionAgent::processCollision(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, 
									const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0x1a969a32,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );
	HK_ON_DEBUG( if (bodyA.getShape()->getType() != hkcdShapeType::CONVEX_LIST &&
        bodyA.getShape()->getContainer() && 
        bodyA.getShape()->getContainer()->getNumChildShapes() > 10) HK_WARN_ONCE(0x5607bb49,  "hkpShapeCollection used without an hkpBvTreeShape, possible huge performance loss"););

	HK_TIMER_BEGIN( "ShapeCollection", HK_NULL );
	
	const hkpShapeContainer* shapeContainer = bodyA.getShape()->getContainer();
	HK_ASSERT2( 0xefee9806, shapeContainer != HK_NULL, "hkpShapeCollectionAgent called on a shape which is not a shape container");

	//
	// call collision agents for shapeB against all shapeAs
	//
	
	hkpCdBody newOperandA( &bodyA );

	KeyAgentPair* agentPair = m_agents.begin();
	hkpShapeBuffer shapeBuffer;

	for ( int i = m_agents.getSize() -1; i>=0; i-- )
	{
		newOperandA.setShape( shapeContainer->getChildShape( agentPair->m_key, shapeBuffer), agentPair->m_key);
		agentPair->m_agent->processCollision(newOperandA, bodyB, input, result);
		agentPair++;
	}


	HK_TIMER_END();
}


		
void hkpShapeCollectionAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector )
{
	HK_TIMER_BEGIN( "ShapeCollection", HK_NULL );

	const hkpShapeContainer* shapeContainer = bodyA.getShape()->getContainer();
	HK_ASSERT2( 0xefee9806, shapeContainer != HK_NULL, "hkpShapeCollectionAgent called on a shape which is not a shape container");

	//
	// call collision agents for shapeB against all shapeAs
	//
	
	hkpCdBody newOperandA( &bodyA );
	KeyAgentPair* agentPair = m_agents.begin();
	hkpShapeBuffer shapeBuffer;

	for ( int i = m_agents.getSize() -1; i>=0; i-- )
	{
		newOperandA.setShape( shapeContainer->getChildShape( agentPair->m_key, shapeBuffer), agentPair->m_key);
		agentPair->m_agent->getClosestPoints(newOperandA, bodyB, input, collector);
		agentPair++;
	}

	HK_TIMER_END();
}


void hkpShapeCollectionAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector )
{
	HK_TIMER_BEGIN( "ShapeCollection", HK_NULL );

	const hkpShapeContainer* shapeContainer = bodyA.getShape()->getContainer();
	HK_ASSERT2( 0xefee9806, shapeContainer != HK_NULL, "hkpShapeCollectionAgent called on a shape which is not a shape container");

	//
	// call collision agents for shapeB against all shapeAs
	//
	
	hkpCdBody newOperandA( &bodyA );

	hkpShapeType typeB = bodyB.getShape()->getType();

	hkpShapeBuffer shapeBuffer;

	for ( hkpShapeKey key = shapeContainer->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = shapeContainer->getNextKey(key) )
	{
		if ( input.m_filter->isCollisionEnabled( input, bodyB, bodyA, *shapeContainer , key) )
		{
			newOperandA.setShape( shapeContainer->getChildShape( key, shapeBuffer), key);
			hkpShapeType typeA = newOperandA.getShape()->getType();
			hkpCollisionDispatcher::GetClosestPointsFunc getClosestPointFunc = input.m_dispatcher->getGetClosestPointsFunc( typeA, typeB );

			getClosestPointFunc(newOperandA, bodyB, input, collector);
		}
	}

	HK_TIMER_END();
}


void hkpShapeCollectionAgent::linearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	HK_TIMER_BEGIN( "ShapeCollection", HK_NULL );

	const hkpShapeContainer* shapeContainer = bodyA.getShape()->getContainer();	
	HK_ASSERT2( 0xefee9806, shapeContainer != HK_NULL, "hkpShapeCollectionAgent called on a shape which is not a shape container");
	//
	// call collision agents for shapeB against all shapeAs
	//
	
	hkpCdBody newOperandA( &bodyA );

	KeyAgentPair* agentPair = m_agents.begin();
	hkpShapeBuffer shapeBuffer;

	for ( int i = m_agents.getSize() -1; i>=0; i-- )
	{
		newOperandA.setShape( shapeContainer->getChildShape( agentPair->m_key, shapeBuffer), agentPair->m_key);
		agentPair->m_agent->linearCast(newOperandA, bodyB, input, collector, startCollector );
		agentPair++;
	}

	HK_TIMER_END();
}

void hkpShapeCollectionAgent::staticLinearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	HK_TIMER_BEGIN( "ShapeCollection", HK_NULL );

	const hkpShapeContainer* shapeContainer = bodyA.getShape()->getContainer();
	HK_ASSERT2( 0xefee9806, shapeContainer != HK_NULL, "hkpShapeCollectionAgent called on a shape which is not a shape container");
	//
	// call collision agents for shapeB against all shapeAs
	//
	
	hkpCdBody newOperandA( &bodyA );

	hkpShapeType typeB = bodyB.getShape()->getType();

	hkpShapeBuffer shapeBuffer;

	for ( hkpShapeKey key = shapeContainer->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = shapeContainer->getNextKey(key) )
	{
		if ( input.m_filter->isCollisionEnabled( input, bodyB, bodyA, *shapeContainer, key) )
		{
			newOperandA.setShape( shapeContainer->getChildShape( key, shapeBuffer), key);
			hkpShapeType typeA = newOperandA.getShape()->getType();
			hkpCollisionDispatcher::LinearCastFunc linearCastFunc = input.m_dispatcher->getLinearCastFunc( typeA, typeB );
			linearCastFunc(newOperandA, bodyB, input, collector, startCollector );
		}
	}

	HK_TIMER_END();
}

void hkpShapeCollectionAgent::getPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN( "ShapeCollection", HK_NULL );

	const hkpShapeContainer* shapeContainer = bodyA.getShape()->getContainer();
	HK_ASSERT2( 0xefee9806, shapeContainer != HK_NULL, "hkpShapeCollectionAgent called on a shape which is not a shape container");
	
	hkpCdBody newOperandA( &bodyA );

	KeyAgentPair* agentPair = m_agents.begin();
	
	hkpShapeBuffer shapeBuffer;

	for ( int i = m_agents.getSize() -1; i>=0; i-- )
	{
		newOperandA.setShape( shapeContainer->getChildShape( agentPair->m_key, shapeBuffer), agentPair->m_key);
		HK_ASSERT2(0x24bd34bf, newOperandA.getShape() != HK_NULL , "No shape exists for corresponding agent");		
		agentPair->m_agent->getPenetrations(newOperandA, bodyB, input, collector );
		if ( collector.getEarlyOut() )
		{
			break;
		}
		agentPair++;
	}

	HK_TIMER_END();
}

void hkpShapeCollectionAgent::staticGetPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN( "ShapeCollection", HK_NULL );

	const hkpShapeContainer* shapeContainer = bodyA.getShape()->getContainer();
	HK_ASSERT2( 0xefee9806, shapeContainer != HK_NULL, "hkpShapeCollectionAgent called on a shape which is not a shape container");

	hkpCdBody newOperandA( &bodyA );

	hkpShapeType typeB = bodyB.getShape()->getType();

	hkpShapeBuffer shapeBuffer;

	for ( hkpShapeKey key = shapeContainer->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = shapeContainer->getNextKey(key) )
	{
		if ( input.m_filter->isCollisionEnabled( input, bodyB, bodyA, *shapeContainer , key) )
		{
			newOperandA.setShape( shapeContainer->getChildShape( key, shapeBuffer), key);
			hkpShapeType typeA = newOperandA.getShape()->getType();
			hkpCollisionDispatcher::GetPenetrationsFunc getPenetrationsFunc = input.m_dispatcher->getGetPenetrationsFunc( typeA, typeB );
			getPenetrationsFunc(newOperandA, bodyB, input, collector );
			if( collector.getEarlyOut() )
			{
				break;
			}
		}
	}

	HK_TIMER_END();
}

inline int hkpShapeCollectionAgent::getAgentIndex( hkpShapeKey key )
{
	for ( int index = 0; index < m_agents.getSize(); index++ )
	{
		if ( m_agents[index].m_key == key )
		{
			return index;
		}
	}
	return -1;
}


void hkpShapeCollectionAgent::updateShapeCollectionFilter( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkCollisionConstraintOwner& constraintOwner )
{
	hkpCdBody newOperandA( &bodyA );

	//
	// initialize all the new child agents
	//
	const hkpShapeContainer* shapeContainer = bodyA.getShape()->getContainer();
	HK_ASSERT2( 0xefee9806, shapeContainer != HK_NULL, "hkpShapeCollectionAgent called on a shape which is not a shape container");


	hkpShapeBuffer shapeBuffer;
	{
	    for ( hkpShapeKey key = shapeContainer->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = shapeContainer->getNextKey(key) )
	    {
		    int index = getAgentIndex( key );
		    if ( input.m_filter->isCollisionEnabled( input, bodyB, bodyA, *shapeContainer , key ) )
		    {
			    newOperandA.setShape( shapeContainer->getChildShape( key, shapeBuffer), key );
    
			    if ( index == -1 )
			    {
				    // A new agent needs to be created and added to the list
				    KeyAgentPair& newPair = m_agents.expandOne();
				    newPair.m_key = key;
				    newPair.m_agent = input.m_dispatcher->getNewCollisionAgent(newOperandA, bodyB, input, m_contactMgr);
			    }
			    else
			    {
				    m_agents[index].m_agent->updateShapeCollectionFilter( newOperandA, bodyB, input, constraintOwner );
			    }
		    }
		    else
		    {
			    if ( index != -1 )
			    {
				    m_agents[index].m_agent->cleanup(constraintOwner);
				    m_agents.removeAt(index);
			    }
		    }
    
	    }
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
