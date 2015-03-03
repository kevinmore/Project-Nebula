/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>

#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpMoppAgent.h>
#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>
#if defined HK_PLATFORM_SPU
#	include <Physics2012/Collide/Filter/Spu/hkpSpuCollisionFilterUtil.h>
#endif


#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppAabbCastVirtualMachine.h>


#ifdef HK_MOPP_DEBUGGER_ENABLED
#	include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>
#endif


#if !defined(HK_PLATFORM_SPU)

hkpMoppAgent::hkpMoppAgent( hkpContactMgr* mgr )
:	hkpBvTreeAgent( mgr )
{
}

void HK_CALL hkpMoppAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createBvTreeShapeAgent;
		af.m_getPenetrationsFunc  = hkpSymmetricAgent<hkpMoppAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc = hkpSymmetricAgent<hkpMoppAgent>::staticGetClosestPoints;
		af.m_linearCastFunc      = hkpSymmetricAgent<hkpMoppAgent>::staticLinearCast;
		af.m_isFlipped           = true;
		af.m_isPredictive		 = false;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::MOPP, hkcdShapeType::ALL_SHAPE_TYPES );
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createShapeBvAgent;
		af.m_getPenetrationsFunc  = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = false;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::ALL_SHAPE_TYPES, hkcdShapeType::MOPP );
	}

	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createBvBvAgent;
		af.m_getPenetrationsFunc  = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::MOPP, hkcdShapeType::MOPP );
	}
}

hkpCollisionAgent* HK_CALL hkpMoppAgent::createBvBvAgent(	const hkpCdBody& bodyA, const hkpCdBody& bodyB,
													   const hkpCollisionInput& input,	hkpContactMgr* mgr )
{
	const hkpMoppBvTreeShape* bvA = static_cast<const hkpMoppBvTreeShape*>( bodyA.getShape() );
	const hkpMoppBvTreeShape* bvB = static_cast<const hkpMoppBvTreeShape*>( bodyB.getShape() );

	// This is where a dodgy MOPP gets caught if it is added to a dispatcher on load from
	// the serialization.
	HK_ASSERT2( 0xec6c2e4d, bvA->getMoppCode() && bvB->getMoppCode(), "No MOPP Code in a MoppBvTreeShape.");

	int sizeA = bvA->getMoppCode()->m_data.getSize();
	int sizeB = bvB->getMoppCode()->m_data.getSize();

	// we should call getAabb only on the smaller MOPP tree, or
	// we risk to tall getAabb on a big landscape.
	// so if radiusA is smaller than radiusB it is allowed
	// to call bodyA->getAabb(). So we want to collide bodyA with MOPP of bodyB
	if ( sizeA < sizeB)
	{
		hkpBvTreeAgent* agent = new hkpMoppAgent( mgr );
		return agent;
	}
	else
	{
		hkpBvTreeAgent* agent = new hkpSymmetricAgent<hkpMoppAgent>( mgr );
		return agent;
	}
}

#else

void HK_CALL hkpMoppAgent::initAgentFunc(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc = staticGetClosestPoints;
	af.m_linearCastFunc	 = staticLinearCast;
}


void HK_CALL hkpMoppAgent::initAgentFuncInverse(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc  = hkpSymmetricAgent<hkpMoppAgent>::staticGetClosestPoints;
	af.m_linearCastFunc	  = hkpSymmetricAgent<hkpMoppAgent>::staticLinearCast;
}

#endif


void hkpMoppAgent::staticLinearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{

#if defined(HK_MOPP_DEBUGGER_ENABLED)
	hkpClosestCdPointCollector debugCollector;
	{
		hkAabb aabb;	calcAabbLinearCast( bodyA, bodyB, input, aabb );

		const hkpMoppBvTreeShape* bvB = static_cast<const hkpMoppBvTreeShape*>( bodyB.getShape() );

		//
		// query the BvTreeShape
		//
		hkInplaceArray<hkpShapeKey,128> hitList;
		{
			bvB->queryAabb( aabb, hitList );
		}

		{
			hkpShapeType typeA = bodyA.getShape()->getType();

			hkArray<hkpShapeKey>::iterator itr = hitList.begin();
			hkArray<hkpShapeKey>::iterator end = hitList.end();

			hkpCdBody modifiedBodyB( &bodyB );

			hkpShapeBuffer( shapeStorage, shapeBuffer );
#if !defined (HK_PLATFORM_SPU)
			const hkpShapeCollection* shapeCollection = bvB->getShapeCollection();
#else
			hkpShapeBuffer buffer;
			const hkpShapeCollection* shapeCollection = bvB->getShapeCollection(buffer);
#endif

			while ( itr != end )
			{
				const hkpShape* shape = shapeCollection->getChildShape( *itr, shapeBuffer );
				modifiedBodyB.setShape( shape, *itr );
				hkpShapeType typeB = shape->getType();
				hkpCollisionDispatcher::LinearCastFunc linCastFunc = input.m_dispatcher->getLinearCastFunc( typeA, typeB );

				linCastFunc( bodyA, modifiedBodyB, input, debugCollector, HK_NULL );
				itr++;
			}
		}
		hkpMoppDebugger::getInstance().initDisabled();
		if ( debugCollector.hasHit() )
		{
			hkpShapeKey hitKey = debugCollector.getHit().m_shapeKeyB;
			hkpMoppDebugger::getInstance().initUsingCodeAndTri( bvB->getMoppCode(), hitKey );
		}

	}
#endif


	HK_TIMER_BEGIN( "Mopp", HK_NULL );

		// get the AABB
	hkAabb aabb;
	{
		hkTransform bTa;
		{
			const hkTransform& wTb = bodyB.getTransform();
			const hkTransform& wTa = bodyA.getTransform();
			bTa.setMulInverseMul( wTb, wTa );
		}
		bodyA.getShape()->getAabb( bTa, input.m_tolerance, aabb );
	}
	//
	//	expand the AABB
	//
	hkVector4 pathB; pathB._setRotatedInverseDir( bodyB.getTransform().getRotation(), input.m_path );


#if defined(HK_PLATFORM_SPU)
	hkpMoppBvTreeShape* bvShapeB = (hkpMoppBvTreeShape*) bodyB.getShape();
	HK_DECLARE_ALIGNED_LOCAL_PTR( hkpMoppCode, codePtr, 16 );
	hkpMoppCode::CodeInfo& codeInfo = (hkpMoppCode::CodeInfo&)bvShapeB->m_codeInfoCopy;
	codePtr->initialize( codeInfo, bvShapeB->m_moppData, bvShapeB->m_moppDataSize );
	bvShapeB->m_code = codePtr;
#endif

	hkpMoppAabbCastVirtualMachine::hkpAabbCastInput ai;
	ai.m_castBody = &bodyA;
	ai.m_moppBody = &bodyB;
	ai.m_from.setInterpolate( aabb.m_min, aabb.m_max, hkSimdReal_Half );
	ai.m_to.setAdd( ai.m_from, pathB );
	ai.m_extents.setSub( aabb.m_max, aabb.m_min );
	ai.m_extents.mul( hkSimdReal_Half );
	hkVector4 tol4; tol4.setAll( input.getTolerance() ); tol4.zeroComponent<3>();
	ai.m_extents.add( tol4 );
	ai.m_collisionInput = &input;

	hkpMoppAabbCastVirtualMachine machine;
	machine.aabbCast( ai, collector, startCollector );
	HK_TIMER_END();
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
