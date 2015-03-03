/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Agent/Deprecated/MultiSphere/hkpMultiSphereAgent.h>
#include <Physics2012/Collide/Shape/Deprecated/MultiSphere/hkpMultiSphereShape.h>

void HK_CALL hkpMultiSphereAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createListBAgent;
		af.m_getPenetrationsFunc  = hkpSymmetricAgent<hkpMultiSphereAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc = hkpSymmetricAgent<hkpMultiSphereAgent>::staticGetClosestPoints;
		af.m_linearCastFunc      = hkpSymmetricAgent<hkpMultiSphereAgent>::staticLinearCast;
		af.m_isFlipped           = true;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::ALL_SHAPE_TYPES, hkcdShapeType::MULTI_SPHERE );
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createListAAgent;
		af.m_getPenetrationsFunc  = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::MULTI_SPHERE, hkcdShapeType::ALL_SHAPE_TYPES );
	}
}


hkpCollisionAgent* HK_CALL hkpMultiSphereAgent::createListAAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, 
									const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpMultiSphereAgent* agent = new hkpMultiSphereAgent(bodyA, bodyB, input, mgr);

	return agent;
}


hkpCollisionAgent* HK_CALL hkpMultiSphereAgent::createListBAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, 
									const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpMultiSphereAgent* agent = new hkpSymmetricAgent<hkpMultiSphereAgent>(bodyA, bodyB, input, mgr);
	
	return agent;
}


hkpMultiSphereAgent::hkpMultiSphereAgent(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
: hkpCollisionAgent( mgr )
{

	//
	// initialize all the new child agents
	//
	const hkpMultiSphereShape* MultiSphere = static_cast<const hkpMultiSphereShape*>(bodyA.getShape());

	int numChildren = MultiSphere->getNumSpheres();
	m_agents.reserve( numChildren );
	
	hkpSphereShape sphereShape(0.0f);

	hkMotionState ms = *bodyA.getMotionState();
	hkpCdBody newOperandA( &bodyA, &ms );
	
	const hkVector4* spheres = MultiSphere->getSpheres();
	for (int i = 0; i < numChildren; i++ )
	{
		hkVector4 offsetWs;
		offsetWs._setRotatedDir( ms.getTransform().getRotation(), spheres[0] );
		ms.getTransform().getTranslation().setAdd( bodyA.getTransform().getTranslation(), offsetWs );
		ms.getSweptTransform().m_centerOfMass0.setAdd( bodyA.getMotionState()->getSweptTransform().m_centerOfMass0, offsetWs );
		ms.getSweptTransform().m_centerOfMass1.setAdd( bodyA.getMotionState()->getSweptTransform().m_centerOfMass1, offsetWs );

		sphereShape.setRadius( spheres[0](3) );
		newOperandA.setShape( &sphereShape, i );

		{
			KeyAgentPair& ap = *m_agents.expandByUnchecked(1);
			ap.m_agent = input.m_dispatcher->getNewCollisionAgent(newOperandA, bodyB, input, mgr);
			ap.m_key = i;
		}
		spheres++;
	}
	
}


void hkpMultiSphereAgent::cleanup( hkCollisionConstraintOwner& info )
{
	for (int i = 0; i < m_agents.getSize(); ++i)
	{
		m_agents[i].m_agent->cleanup( info );
	}
	delete this;
}


void hkpMultiSphereAgent::processCollision(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, 
									const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0x2cadb41e,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_INTERNAL_TIMER_BEGIN( "MultiSphere", this );
	
	const hkpMultiSphereShape* MultiSphere = static_cast<const hkpMultiSphereShape*>(bodyA.getShape());
	
	//
	// call collision agents for shapeB against all shapeAs
	//
	
	hkMotionState ms = *bodyA.getMotionState();
	hkpSphereShape sphereShape(0.0f);
	hkpCdBody newOperandA( &bodyA, &ms );

	KeyAgentPair* agentPair = m_agents.begin();

	for ( int i = m_agents.getSize() -1; i>=0; i-- )
	{
		const hkVector4& sphere = MultiSphere->getSpheres()[ agentPair->m_key ];

		hkVector4 offsetWs;
		offsetWs._setRotatedDir( ms.getTransform().getRotation(), sphere );
		ms.getTransform().getTranslation().setAdd( bodyA.getTransform().getTranslation(), offsetWs );
		ms.getSweptTransform().m_centerOfMass0.setAdd( bodyA.getMotionState()->getSweptTransform().m_centerOfMass0, offsetWs );
		ms.getSweptTransform().m_centerOfMass1.setAdd( bodyA.getMotionState()->getSweptTransform().m_centerOfMass1, offsetWs );

		sphereShape.setRadius( sphere(3) );
		newOperandA.setShape( &sphereShape, i );

		agentPair->m_agent->processCollision(newOperandA, bodyB, input, result);
		agentPair++;
	}


	HK_INTERNAL_TIMER_END();
}


		
void hkpMultiSphereAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector )
{
	HK_INTERNAL_TIMER_BEGIN( "MultiSphere", this );

	const hkpMultiSphereShape* MultiSphere = static_cast<const hkpMultiSphereShape*>(bodyA.getShape());
	
	//
	// call collision agents for shapeB against all shapeAs
	//
	
	hkMotionState ms = *bodyA.getMotionState();
	hkpSphereShape sphereShape(0.0f);
	hkpCdBody newOperandA( &bodyA, &ms );
	

	KeyAgentPair* agentPair = m_agents.begin();

	for ( int i = m_agents.getSize() -1; i>=0; i-- )
	{
		const hkVector4& sphere = MultiSphere->getSpheres()[ agentPair->m_key ];
		hkVector4 off;	off._setRotatedDir( ms.getTransform().getRotation(), sphere );
		ms.getTransform().getTranslation().setAdd( bodyA.getTransform().getTranslation(), off );

		sphereShape.setRadius( sphere(3) );
		newOperandA.setShape( &sphereShape, i );
		agentPair->m_agent->getClosestPoints(newOperandA, bodyB, input, collector);
		agentPair++;
	}

	HK_INTERNAL_TIMER_END();
}


void hkpMultiSphereAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector )
{
	HK_INTERNAL_TIMER_BEGIN( "MultiSphere", this );

	const hkpMultiSphereShape* MultiSphere = static_cast<const hkpMultiSphereShape*>(bodyA.getShape());
	
	//
	// call collision agents for shapeB against all shapeAs
	//

	hkMotionState ms = *bodyA.getMotionState();
	hkpSphereShape sphereShape(0.0f);
	hkpCdBody newOperandA( &bodyA, &ms );
	
	hkpShapeType typeB = bodyB.getShape()->getType();

	for ( int key = 0; key < MultiSphere->getNumSpheres(); key++ )
	{
		const hkVector4& sphere = MultiSphere->getSpheres()[ key ];
		hkVector4 off;	off._setRotatedDir( ms.getTransform().getRotation(), sphere );
		ms.getTransform().getTranslation().setAdd( bodyA.getTransform().getTranslation(), off );
		sphereShape.setRadius( sphere(3) );
		newOperandA.setShape( &sphereShape, key );

		hkpShapeType typeA = sphereShape.getType();
		hkpCollisionDispatcher::GetClosestPointsFunc getClosestPointFunc = input.m_dispatcher->getGetClosestPointsFunc( typeA, typeB );

		getClosestPointFunc(newOperandA, bodyB, input, collector);
	}

	HK_INTERNAL_TIMER_END();
}


void hkpMultiSphereAgent::linearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	HK_INTERNAL_TIMER_BEGIN( "MultiSphere", this );

	const hkpMultiSphereShape* MultiSphere = static_cast<const hkpMultiSphereShape*>(bodyA.getShape());
	
	//
	// call collision agents for shapeB against all shapeAs
	//
	
	hkMotionState ms = *bodyA.getMotionState();
	hkpSphereShape sphereShape(0.0f);
	hkpCdBody newOperandA( &bodyA, &ms );
	
	KeyAgentPair* agentPair = m_agents.begin();

	for ( int i = m_agents.getSize() -1; i>=0; i-- )
	{
		const hkVector4& sphere = MultiSphere->getSpheres()[ i ];
		hkVector4 off;	off._setRotatedDir( ms.getTransform().getRotation(), sphere );
		ms.getTransform().getTranslation().setAdd( bodyA.getTransform().getTranslation(), off );

		sphereShape.setRadius( sphere(3) );
		newOperandA.setShape( &sphereShape, i );
		agentPair->m_agent->linearCast(newOperandA, bodyB, input, collector, startCollector );
	}

	HK_INTERNAL_TIMER_END();
}

void hkpMultiSphereAgent::staticLinearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	HK_INTERNAL_TIMER_BEGIN( "MultiSphere", this );

	const hkpMultiSphereShape* MultiSphere = static_cast<const hkpMultiSphereShape*>(bodyA.getShape());
	
	//
	// call collision agents for shapeB against all shapeAs
	//
	
	hkMotionState ms = *bodyA.getMotionState();
	hkpSphereShape sphereShape(0.0f);
	hkpCdBody newOperandA( &bodyA, &ms );
	
	hkpShapeType typeB = bodyB.getShape()->getType();

	for ( int key = 0; key < MultiSphere->getNumSpheres(); key++ )
	{
		const hkVector4& sphere = MultiSphere->getSpheres()[ key ];
		hkVector4 off;	off._setRotatedDir( ms.getTransform().getRotation(), sphere );
		ms.getTransform().getTranslation().setAdd( bodyA.getTransform().getTranslation(), off );
		sphereShape.setRadius( sphere(3) );
		newOperandA.setShape( &sphereShape, key );

		hkpShapeType typeA = sphereShape.getType();
		hkpCollisionDispatcher::LinearCastFunc linearCastFunc = input.m_dispatcher->getLinearCastFunc( typeA, typeB );
		linearCastFunc(newOperandA, bodyB, input, collector, startCollector );
	}

	HK_INTERNAL_TIMER_END();
}

void hkpMultiSphereAgent::getPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_INTERNAL_TIMER_BEGIN( "MultiSphere", this );

	const hkpMultiSphereShape* MultiSphere = static_cast<const hkpMultiSphereShape*>(bodyA.getShape());
	
	hkMotionState ms = *bodyA.getMotionState();
	hkpSphereShape sphereShape(0.0f);
	hkpCdBody newOperandA( &bodyA, &ms );
	

	KeyAgentPair* agentPair = m_agents.begin();

	for ( int i = m_agents.getSize() -1; i>=0; i-- )
	{
		const hkVector4& sphere = MultiSphere->getSpheres()[ agentPair->m_key ];
		hkVector4 off;	off._setRotatedDir( ms.getTransform().getRotation(), sphere );
		ms.getTransform().getTranslation().setAdd( bodyA.getTransform().getTranslation(), off );

		sphereShape.setRadius( sphere(3) );
		newOperandA.setShape( &sphereShape, i );

		agentPair->m_agent->getPenetrations(newOperandA, bodyB, input, collector );
		if ( collector.getEarlyOut() )
		{
			break;
		}
		agentPair++;
	}

	HK_INTERNAL_TIMER_END();
}

void hkpMultiSphereAgent::staticGetPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_INTERNAL_TIMER_BEGIN( "MultiSphere" , this);

	const hkpMultiSphereShape* MultiSphere = static_cast<const hkpMultiSphereShape*>(bodyA.getShape());
	
	hkMotionState ms = *bodyA.getMotionState();
	hkpSphereShape sphereShape(0.0f);
	hkpCdBody newOperandA( &bodyA, &ms );
	
	hkpShapeType typeB = bodyB.getShape()->getType();

	for ( int key = 0; key < MultiSphere->getNumSpheres(); key++ )
	{
		const hkVector4& sphere = MultiSphere->getSpheres()[ key ];
		hkVector4 off;	off._setRotatedDir( ms.getTransform().getRotation(), sphere );
		ms.getTransform().getTranslation().setAdd( bodyA.getTransform().getTranslation(), off );
		sphereShape.setRadius( sphere(3) );
		newOperandA.setShape( &sphereShape, key );

		hkpShapeType typeA = sphereShape.getType();
		hkpCollisionDispatcher::GetPenetrationsFunc getPenetrationsFunc = input.m_dispatcher->getGetPenetrationsFunc( typeA, typeB );
		getPenetrationsFunc(newOperandA, bodyB, input, collector );
		if( collector.getEarlyOut() )
		{
			break;
		}
	}

	HK_INTERNAL_TIMER_END();
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
