/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Physics2012/Internal/Collide/Gjk/hkpGsk.h>

#include <Physics2012/Collide/Shape/Convex/hkpConvexShape.h>

#include <Physics2012/Collide/Agent/hkpCollisionAgentConfig.h>

#include <Physics2012/Collide/Agent/ConvexAgent/Gjk/hkpGskBaseAgent.h>

//HK_COMPILE_TIME_ASSERT( sizeof( hkpGskBaseAgent ) == 24 /* 12=base + 12=cache */ );


#if !defined(HK_PLATFORM_SPU)
void hkpGskBaseAgent::processCollision(	const hkpCdBody& bodyA,						const hkpCdBody& bodyB, 
										const hkpProcessCollisionInput& input,		hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0xf0040345,  0, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );
}
#endif


#if !defined(HK_PLATFORM_SPU)
hkpGskBaseAgent::hkpGskBaseAgent(	const hkpCdBody& bodyA,	const hkpCdBody& bodyB, hkpContactMgr* mgr ): hkpIterativeLinearCastAgent( mgr )
{
	const hkpConvexShape* shapeA = static_cast<const hkpConvexShape*>(bodyA.getShape());
	const hkpConvexShape* shapeB = static_cast<const hkpConvexShape*>(bodyB.getShape());
	hkTransform t; t.setMulInverseMul( bodyA.getTransform(), bodyB.getTransform() );
	if ( shapeB->getType() == hkcdShapeType::TRIANGLE )
	{
		m_cache.initTriangle( shapeA, reinterpret_cast<const hkpTriangleShape*>(shapeB), t );
	}
	else
	{
		m_cache.init( shapeA, shapeB, t );
	}
	m_separatingNormal.setZero();
	m_separatingNormal(3) = -1.f;
	m_timeOfSeparatingNormal = hkTime(-1.0f);
	hkReal maxPenA = bodyA.getRootCollidable()->m_allowedPenetrationDepth;
	hkReal maxPenB = bodyB.getRootCollidable()->m_allowedPenetrationDepth;
	m_allowedPenetration = hkMath::min2( maxPenA, maxPenB );

}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpGskBaseAgent::invalidateTim( const hkpCollisionInput& input)
{
	m_separatingNormal.setZero();
	m_timeOfSeparatingNormal = hkTime(-1.0f);
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpGskBaseAgent::cleanup( hkCollisionConstraintOwner& constraintOwner )
{
	delete this;
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpGskBaseAgent::warpTime( hkTime oldTime, hkTime newTime, const hkpCollisionInput& input )
{
	if (m_timeOfSeparatingNormal == oldTime)
	{
		m_timeOfSeparatingNormal = newTime;
	}
	else
	{
		m_timeOfSeparatingNormal = hkTime(-1.0f);
		m_separatingNormal.setZero();
	}
}
#endif


#if !defined(HK_PLATFORM_SPU)

#if defined HK_COMPILER_MSVC
	// C4701: local variable 'output' may be used without having been initialized
#	pragma warning(disable: 4701)
#endif

hkBool hkpGskBaseAgent::_getClosestPoint(	const hkpCdBody& bodyA, const hkpCdBody& bodyB, 
										const hkpCollisionInput& input, struct hkpExtendedGskOut& output )
{
	const hkpConvexShape* shapeA = static_cast<const hkpConvexShape*>(bodyA.getShape());
	const hkpConvexShape* shapeB = static_cast<const hkpConvexShape*>(bodyB.getShape());


	{
		// Get the relative transform for the two bodies for the collision detector
		hkTransform aTb;	aTb.setMulInverseMul( bodyA.getTransform(), bodyB.getTransform());

		// Call the collision detector
		{
			hkpGsk gsk;
			hkVector4 separatingNormal;
			gsk.init( shapeA, shapeB, m_cache );
			gsk.getClosestFeature(shapeA, shapeB, aTb, separatingNormal);
			gsk.checkForChangesAndUpdateCache( m_cache );
			gsk.convertFeatureToClosestDistance( separatingNormal, output );
		}

		// convert contact normal to world space...
		output.m_normalInWorld._setRotatedDir( bodyA.getTransform().getRotation(), output.m_normalInA);
		output.m_unweldedNormalInWorld = output.m_normalInWorld;

		const hkReal dist = output.m_distance - shapeA->getRadius() - shapeB->getRadius();
		output.m_distance = dist;

		if( output.m_distance < input.getTolerance() )
		{
			// adjust the contact points by the radius
			output.m_pointAinA.subMul(hkSimdReal::fromFloat(shapeA->getRadius()), output.m_normalInA);
			hkVector4 pointBinA; pointBinA.setAddMul( output.m_pointAinA, output.m_normalInA, hkSimdReal::fromFloat(-dist) );
			output.m_pointBinB._setTransformedInversePos(aTb, pointBinA);
			return true;
		}
	}
	return false;
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpGskBaseAgent::calcSeparatingNormal( const hkpCdBody& bodyA, const hkpCdBody& bodyB, hkReal earlyOutTolerance, hkpGsk& gsk, hkVector4& separatingNormalOut )
{
	const hkpConvexShape* shapeA = static_cast<const hkpConvexShape*>(bodyA.getShape());
	const hkpConvexShape* shapeB = static_cast<const hkpConvexShape*>(bodyB.getShape());

	// Get the relative transform for the two bodies for the collision detector
	hkTransform aTb;	aTb.setMulInverseMul( bodyA.getTransform(), bodyB.getTransform());

		// Call the collision detector
	hkVector4 separatingNormal; gsk.getClosestFeature(shapeA, shapeB, aTb, separatingNormal);

	separatingNormalOut._setRotatedDir( bodyA.getTransform().getRotation(), separatingNormal);
	separatingNormalOut(3) = separatingNormal(3) - shapeA->getRadius() - shapeB->getRadius();
}
#endif


HK_FORCE_INLINE hkBool hkpGskBaseAgent::staticGetClosestPoint(	const hkpCdBody& bodyA,	const hkpCdBody& bodyB,
														   const hkTransform& aTb, const hkpCollisionInput& input,
														   hkpGskCache& cache, struct hkpExtendedGskOut& output)

{

	HK_INTERNAL_TIMER_SPLIT_LIST( "Gsk" );

	const hkpConvexShape* shapeA = static_cast<const hkpConvexShape*>(bodyA.getShape());
	const hkpConvexShape* shapeB = static_cast<const hkpConvexShape*>(bodyB.getShape());

	{
		// Call the collision detector
		hkVector4 separatingNormal;
		hkpGsk gsk;
		gsk.init( shapeA, shapeB, cache );
		const hkpGskStatus gskStatus = gsk.getClosestFeature(shapeA, shapeB, aTb, separatingNormal);
		
		if(gskStatus == HK_GSK_OK )
		{
			gsk.convertFeatureToClosestDistance( separatingNormal, output );

			// convert contact normal to world space...
			output.m_normalInWorld._setRotatedDir( bodyA.getTransform().getRotation(), output.m_normalInA);
			output.m_unweldedNormalInWorld = output.m_normalInWorld;
			
			const hkReal dist = output.m_distance - shapeA->getRadius() - shapeB->getRadius();
			output.m_distance = dist;

			if(output.m_distance < input.getTolerance())
			{
				// adjust the contact points by the radius
				output.m_pointAinA.addMul(hkSimdReal::fromFloat(-shapeA->getRadius()), output.m_normalInA);
				hkVector4 pointBinA; pointBinA.setAddMul( output.m_pointAinA, output.m_normalInA, hkSimdReal::fromFloat(-dist) );
				output.m_pointBinB._setTransformedInversePos(aTb, pointBinA);

				// weld closest point normal
				if ( shapeB->getType() == hkcdShapeType::TRIANGLE &&  input.m_weldClosestPoints.val() )
				{
					gsk.checkForChangesAndUpdateCache( cache );
					hkVector4 weldedNormal = output.m_normalInWorld;
					hkVector4 cpInWs; cpInWs._setTransformedPos( bodyB.getTransform(), output.m_pointBinB );
					hkpConvexShape::WeldResult weldResult = (hkpConvexShape::WeldResult)shapeB->weldContactPoint(	&cache.m_vertices[cache.m_dimA], cache.m_dimB, cpInWs, 
						&bodyB.getTransform(), shapeA, &bodyA.getTransform(), weldedNormal );
					
					if (!input.m_forceAcceptContactPoints.val() && (weldResult == hkpConvexShape::WELD_RESULT_REJECT_CONTACT_POINT))
					{						
						return false;
					}
					else if (weldResult == hkpConvexShape::WELD_RESULT_ACCEPT_CONTACT_POINT_MODIFIED)
					{
						output.m_normalInWorld = weldedNormal;
					}
				}
				return true;
			}
		}
	}
	return false;
}

HK_DISABLE_OPTIMIZATION_VS2008_X64
void hkpGskBaseAgent::staticGetClosestPoints( const hkpCdBody& bodyA , const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector  )
{
	HK_TIMER_BEGIN( "Gsk", HK_NULL );

	hkpExtendedGskOut output;

	hkTransform aTb; 
	hkpGskCache cache;
	{
		const hkpConvexShape* sA = static_cast<const hkpConvexShape*>(bodyA.getShape());
		const hkpConvexShape* sB = static_cast<const hkpConvexShape*>(bodyB.getShape());
		aTb.setMulInverseMul( bodyA.getTransform(), bodyB.getTransform() );
		if ( sB->getType() == hkcdShapeType::TRIANGLE )
		{
			cache.initTriangle( sA, reinterpret_cast<const hkpTriangleShape*>(sB), aTb );
		}
		else
		{
			cache.init( sA, sB, aTb );
		}
	}

	if( staticGetClosestPoint(bodyA, bodyB, aTb, input, cache, output) )
	{
		hkContactPoint contact;
		hkVector4 cpPos; cpPos._setTransformedPos( bodyB.getTransform(), output.m_pointBinB );
		contact.setPosition(cpPos);
		contact.setSeparatingNormal( output.m_normalInWorld, output.m_distance );
		
		hkpCdPoint event( bodyA, bodyB, contact );
		event.setUnweldedNormal( output.m_unweldedNormalInWorld );
		
		collector.addCdPoint( event );
	}	

	HK_TIMER_END();
}
HK_RESTORE_OPTIMIZATION_VS2008_X64

#if !defined(HK_PLATFORM_SPU)
void hkpGskBaseAgent::getPenetrations( const hkpCdBody& bodyA,	const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN_LIST( "Gsk", "tim" );

	// we cannot use tims in here as we might not have a motion state
	//	if(0)
	//	{
	//		hkVector4 timInfo;
	//		hkSweptTransformUtil::calcTimInfo( *bodyA.getMotionState(), *bodyB.getMotionState(), input.m_stepInfo.m_deltaTime, timInfo);
	//		if ( m_separatingNormal(3) > input.getTolerance() )
	//		{
	//			m_separatingNormal(3) -= timInfo.dot4xyz1( m_separatingNormal );
	//			if ( m_separatingNormal(3) > input.getTolerance() )
	//			{
	//				goto END;
	//			}
	//		}
	//	}


	HK_TIMER_SPLIT_LIST( "SepNormal");
	{
		const hkpConvexShape* shapeA = static_cast<const hkpConvexShape*>(bodyA.getShape());
		const hkpConvexShape* shapeB = static_cast<const hkpConvexShape*>(bodyB.getShape());
		

		// Get the relative transform for the two bodies for the collision detector
		hkTransform aTb; aTb.setMulInverseMul( bodyA.getTransform(), bodyB.getTransform());

		// Call the collision detector
		hkpGsk gsk;
		gsk.m_doNotHandlePenetration = true;

		hkVector4 separatingNormalInA;
		gsk.init( shapeA, shapeB, m_cache );
		hkpGskStatus gjkStatus = gsk.getClosestFeature(shapeA, shapeB, aTb, separatingNormalInA);
		gsk.checkForChangesAndUpdateCache( m_cache );

		if(gjkStatus == HK_GSK_OK)
		{
			const hkReal dist = separatingNormalInA(3) - shapeA->getRadius() - shapeB->getRadius();
			m_separatingNormal._setRotatedDir( bodyA.getTransform().getRotation(), separatingNormalInA );
			m_separatingNormal(3) = dist;
			if ( dist < 0.0f )
			{
				collector.addCdBodyPair( bodyA, bodyB );
			}
		}
		else
		{
			m_separatingNormal.setZero();
			collector.addCdBodyPair( bodyA, bodyB );
		}
	}
//END:;
	HK_TIMER_END_LIST();
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpGskBaseAgent::staticGetPenetrations( const hkpCdBody& bodyA,	const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector  )
{

	HK_TIMER_BEGIN( "Gsk", HK_NULL );

	const hkpConvexShape* shapeA = static_cast<const hkpConvexShape*>(bodyA.getShape());
	const hkpConvexShape* shapeB = static_cast<const hkpConvexShape*>(bodyB.getShape());
	

	// Get the relative transform for the two bodies for the collision detector
	hkTransform aTb;
	aTb.setMulInverseMul( bodyA.getTransform(), bodyB.getTransform());

	hkpGskCache cache;
	{
		if ( shapeB->getType() == hkcdShapeType::TRIANGLE )
		{
			cache.initTriangle( shapeA, reinterpret_cast<const hkpTriangleShape*>(shapeB), aTb );
		}
		else
		{
			cache.init( shapeA, shapeB, aTb );
		}
	}

	// Call the collision detector
	hkpGsk gsk;
	gsk.m_doNotHandlePenetration = true;

	gsk.init(shapeA, shapeB, cache);
	hkVector4 separatingNormal;
	const hkpGskStatus gjkStatus = gsk.getClosestFeature(shapeA, shapeB, aTb, separatingNormal);

	HK_TIMER_END();

	if( gjkStatus == HK_GSK_OK )
	{
		const hkReal dist = separatingNormal(3) - shapeA->getRadius() - shapeB->getRadius();
		if ( dist < 0.0f )
		{
			collector.addCdBodyPair( bodyA, bodyB );
		}
	}
	else
	{
		collector.addCdBodyPair( bodyA, bodyB );
	}
}
#endif


#if !defined(HK_PLATFORM_SPU)
hkBool hkpGskBaseAgent::getClosestPoint(	const hkpCdBody& bodyA, const hkpCdBody& bodyB, 
										const hkpCollisionInput& input, struct hkpExtendedGskOut& output )
{
	return _getClosestPoint( bodyA, bodyB, input, output );
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpGskBaseAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB,
									   const hkpCollisionInput& input, hkpCdPointCollector& collector  )
{
	HK_TIMER_BEGIN( "Gsk", HK_NULL );

	hkpGsk::GetClosesetPointInput gskInput;
	hkTransform aTb;	aTb.setMulInverseMul( bodyA.getTransform(), bodyB.getTransform());
	{
		gskInput.m_shapeA = static_cast<const hkpConvexShape*>(bodyA.getShape());
		gskInput.m_shapeB = static_cast<const hkpConvexShape*>(bodyB.getShape());
		gskInput.m_aTb = &aTb;
		gskInput.m_transformA = &bodyA.getTransform();
		gskInput.m_collisionTolerance = input.getTolerance();
	}
	
	hkVector4 separatingNormal;
	hkVector4 pointOnB;
	if( hkpGsk::getClosestPoint( gskInput, m_cache, separatingNormal, pointOnB ) == HK_SUCCESS )
	{
		hkContactPoint contact;
		contact.setPosition(pointOnB);
		contact.setSeparatingNormal( separatingNormal );
		hkpCdPoint event( bodyA, bodyB, contact );
		collector.addCdPoint( event );
	}
	HK_TIMER_END();
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
