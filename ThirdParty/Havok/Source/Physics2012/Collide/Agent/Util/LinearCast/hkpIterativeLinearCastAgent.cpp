/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Agent/Util/LinearCast/hkpIterativeLinearCastAgent.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpSimpleClosestContactCollector.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>
#include <Physics2012/Collide/Agent/Query/hkpLinearCastCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpCollisionAgentConfig.h>
#include <Physics2012/Collide/Shape/Convex/hkpConvexShape.h>

#if !defined(HK_PLATFORM_SPU)

void hkpIterativeLinearCastAgent::linearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	hkpClosestCdPointCollector startPointCollector;

	hkpCollisionInput in2 = input;
	in2.m_tolerance = in2.m_tolerance + input.m_cachedPathLength;
	// Tell the agents not to reject contact points based on welding
	// We need the same hit points for welded as unwelded, better to
	// just not weld points that would be rejected
	in2.m_forceAcceptContactPoints = true;

	getClosestPoints( bodyA, bodyB, in2, startPointCollector );

	if ( !startPointCollector.hasHit() )
	{
		return;
	}

	HK_ASSERT2(0x48e7a51c, startPointCollector.getUnweldedNormal().isOk<3>(), "Invalid unwelded normal");

	hkContactPoint contact;

	{
		hkSimdReal dist = startPointCollector.getHitContact().getDistanceSimdReal();
		contact.setPositionNormalAndDistance(startPointCollector.getHitContact().getPosition(), startPointCollector.getUnweldedNormal(), dist);

		hkSimdReal inputTol; inputTol.load<1>(&(input.m_tolerance.ref()));
		if (dist.isLess(inputTol) && startCollector )
		{
			hkpCdPoint event(bodyA, bodyB, contact);
			startCollector->addCdPoint( event );
		}
	}	

	const hkVector4& path = input.m_path;

	{
		const hkSimdReal startDistance = contact.getDistanceSimdReal();
		const hkSimdReal pathProjected = contact.getNormal().dot<3>( path );
		const hkSimdReal endDistance   = startDistance + pathProjected;

		//
		//	Check whether we could move the full distance
		//
		{
			if ( endDistance.isGreaterZero() )
			{
				// not hitting at all
				return;
			}

			HK_ASSERT2(0x35e26c9c,  input.m_maxExtraPenetration >= 0.f, "You have to set the  m_maxExtraPenetration to something bigger than 0");
			if ( (pathProjected + hkSimdReal::fromFloat(input.m_maxExtraPenetration)).isGreaterEqualZero() )
			{
				// we are not moving closer than m_maxExtraPenetration
				return;
			}
		}

		//
		// check for early outs
		//
		const hkSimdReal castFraction = startDistance / ( startDistance - endDistance);
		if ( startDistance <= hkSimdReal::fromFloat(input.m_config->m_iterativeLinearCastEarlyOutDistance) )
		{
			if ( startDistance.isGreaterZero() )
			{
				if ( castFraction > hkSimdReal::fromFloat(collector.getEarlyOutDistance()) )
				{
					return;
				}
				contact.setDistanceSimdReal( castFraction );
			}
			else
			{
				// we are hitting immediately
				contact.setDistanceSimdReal( hkSimdReal_0 );
			}

			// early out, because we are already very close
			hkpCdPoint event(bodyA, bodyB, contact);
			collector.addCdPoint( event );
			return;
		}

		// now endDistance is negative and startDistance position, so this division is allowed
		contact.setDistanceSimdReal( castFraction );
	}


	//
	// now find precise collision point
	//
	hkpClosestCdPointCollector checkPointCollector;
	hkTransform bodyACopyTransform = bodyA.getTransform();
	hkpCdBody bodyACopy( &bodyA, &bodyACopyTransform);
	{
		bodyACopy.setShape( bodyA.getShape() , HK_INVALID_SHAPE_KEY); // This cdbody is not passed back in the collision event, so we don't worry about the hkpShapeKey

		for ( int i = input.m_config->m_iterativeLinearCastMaxIterations-1; i>=0; i-- )
		{

			//
			//	Move bodyA along the path and recheck the collision 
			//
			{
				checkPointCollector.reset();
				//
				// Move the object along the path
				//
				const hkVector4& oldPosition  =  bodyA.getTransform().getTranslation();
				hkVector4 newPosition;	newPosition.setAddMul( oldPosition, path, contact.getDistanceSimdReal() );
				bodyACopyTransform.setTranslation( newPosition );

				getClosestPoints( bodyACopy, bodyB , in2, checkPointCollector);
				HK_ASSERT2(0x48e7a51c, checkPointCollector.getUnweldedNormal().isOk<3>(), "Invalid unwelded normal");

				
				HK_ASSERT2(0x3ea49f9b,  checkPointCollector.hasHit(), "The collision agent reports no hit when queried a second time (the second time is closer than the first time");
			}

			//
			// redo the checks
			//
			{
				const hkContactPoint& checkPoint = checkPointCollector.getHitContact();
				const hkVector4& normal = checkPointCollector.getUnweldedNormal();

				hkSimdReal pathProjected2 = normal.dot<3>( path );
				if ( pathProjected2.isGreaterEqualZero() )
				{
					return;	// normal points away
				}
				pathProjected2 = -pathProjected2;


				const hkSimdReal startDistance2 = checkPoint.getDistanceSimdReal();

				//
				//	pre distance is the negative already traveled distance relative to the new normal
				//
				const hkSimdReal preDistance = pathProjected2 * contact.getDistanceSimdReal();
				HK_ASSERT2(0x730fe223,  preDistance.isGreaterEqualZero(), "Numerical accuracy problem in linearCast" );

				if ( startDistance2 + preDistance > pathProjected2 )
				{
					// endDistance + preDistance = realEndDistance;
					// if realEndDistance > 0, than endplane is not penetrated, so no hit
					return;
				}

				// now we know that pathProjected2 < 0
				// so the division is safe
				const hkSimdReal castFraction = contact.getDistanceSimdReal() + (startDistance2 / pathProjected2);
				if ( castFraction > hkSimdReal::fromFloat(collector.getEarlyOutDistance()) )
				{
					return;
				}

				contact.setPositionNormalAndDistance( checkPoint.getPosition(), normal, castFraction );

				if ( startDistance2 <= hkSimdReal::fromFloat(input.m_config->m_iterativeLinearCastEarlyOutDistance) )
				{
					// early out, because we are already very close
					break;
				}
			}
		}
	}

	// Select either the welded or unwelded contact normal to return
	if(in2.m_weldClosestPoints.val())
	{
		contact.setNormalOnly(checkPointCollector.getHitContact().getNormal());
	}
	hkpCdPoint event(bodyA, bodyB, contact);
	collector.addCdPoint( event );
}
#endif

void hkpIterativeLinearCastAgent::staticLinearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	hkpClosestCdPointCollector startPointCollector;

	hkpCollisionInput in2 = input;
	in2.m_tolerance = in2.m_tolerance + input.m_cachedPathLength;
	// Tell the agents not to reject contact points based on welding
	// We need the same hit points for welded as unwelded, better to
	// just not weld points that would be rejected
	in2.m_forceAcceptContactPoints = true;

	hkpShapeType typeA = bodyA.getShape()->getType();
	hkpShapeType typeB = bodyB.getShape()->getType();

	HK_ASSERT2(0xad904281, input.m_dispatcher->hasAlternateType(typeA, hkcdShapeType::CONVEX), "Iterative agent processes a non-convex (probably non-terminal) shape.");
	HK_ASSERT2(0xad904281, input.m_dispatcher->hasAlternateType(typeB, hkcdShapeType::CONVEX), "Iterative agent processes a non-convex (probably non-terminal) shape.");


	#if !defined(HK_PLATFORM_SPU)
		hkpCollisionDispatcher::GetClosestPointsFunc getClosestPoints = input.m_dispatcher->getGetClosestPointsFunc( typeA, typeB );
	#else
	hkpSpuCollisionQueryDispatcher::GetClosestPointsFunc getClosestPoints = input.m_queryDispatcher->getGetClosestPointsFunc( typeA, typeB );
	#endif

	getClosestPoints( bodyA, bodyB, in2, startPointCollector );

	if ( !startPointCollector.hasHit() )
	{
		return;
	}
	HK_ASSERT2(0x48e7a51c, startPointCollector.getUnweldedNormal().isOk<3>(), "Invalid unwelded normal");

	hkContactPoint contact;
	const hkContactPoint& startPoint = startPointCollector.getHitContact();
	{
		contact = startPoint;
		const hkSimdReal dist = startPoint.getDistanceSimdReal();
		hkSimdReal inputTol; inputTol.load<1>(&(input.m_tolerance.ref()));
		if (dist.isLess(inputTol) && startCollector )
		{
			hkpCdPoint event(bodyA, bodyB, contact);
			startCollector->addCdPoint( event );
		}
	}	

	const hkVector4& path = input.m_path;

	hkSimdReal currentFraction; currentFraction.setZero();
	{
		const hkSimdReal startDistance = contact.getDistanceSimdReal();
		const hkSimdReal pathProjected = startPointCollector.getUnweldedNormal().dot<3>( path );
		const hkSimdReal endDistance   = startDistance + pathProjected;

		//
		//	Check whether we could move the full distance
		//
		{
			if ( endDistance.isGreaterZero() )
			{
				// not hitting at all
				return;
			}

			HK_ASSERT2(0x74a9aa26,  input.m_maxExtraPenetration >= 0.f, "You have to set the  m_maxExtraPenetration to something bigger than 0");
			if ( (pathProjected + hkSimdReal::fromFloat(input.m_maxExtraPenetration)).isGreaterEqualZero() )
			{
				// we are not moving closer than m_maxExtraPenetration
				return;
			}
		}

		//
		// check for early outs
		//
		if ( startDistance <= hkSimdReal::fromFloat(input.m_config->m_iterativeLinearCastEarlyOutDistance) )
		{
			if ( startDistance.isGreaterZero() )
			{
					// early out if our endpoint is overshooting our early out distance
				if ( startDistance > ( hkSimdReal::fromFloat(collector.getEarlyOutDistance()) * ( startDistance - endDistance) ) )
				{
					return;
				}
			}
			contact.setDistanceSimdReal(hkSimdReal_0);

			// early out, because we are already very close
			hkpCdPoint event(bodyA, bodyB, contact);
			collector.addCdPoint( event );
			return;
		}

		// now endDistance is negative and startDistance position, so this division is allowed
		currentFraction = startDistance / ( startDistance - endDistance);
	}


	//
	// now find precise collision point
	//
	hkpClosestCdPointCollector checkPointCollector;
	hkTransform bodyACopyTransform = bodyA.getTransform();
	hkpCdBody bodyACopy( &bodyA, &bodyACopyTransform);

	{
		bodyACopy.setShape( bodyA.getShape(), HK_INVALID_SHAPE_KEY );  // This cdbody is not passed back in the collision event, so we don't worry about the hkpShapeKey
																	   // also this is a terminal agent, so no collision filtering callbacks are done.

		for ( int i = input.m_config->m_iterativeLinearCastMaxIterations-1; i>=0; i-- )
		{

			//
			//	Move bodyA along the path and recheck the collision 
			//
			{
				checkPointCollector.reset();
				//
				// Move the object along the path
				//
				const hkVector4& oldPosition  =  bodyA.getTransform().getTranslation();
				hkVector4 newPosition;	newPosition.setAddMul( oldPosition, path, currentFraction );
				bodyACopyTransform.setTranslation( newPosition );

				getClosestPoints( bodyACopy, bodyB , in2, checkPointCollector);

				if ( !checkPointCollector.hasHit() )
				{
					return;
				}
				HK_ASSERT2(0x48e7a51c, checkPointCollector.getUnweldedNormal().isOk<3>(), "Invalid unwelded normal");

			}

			//
			// redo the checks
			//
			{
				const hkContactPoint& checkPoint = checkPointCollector.getHitContact();
				const hkVector4& normal = checkPointCollector.getUnweldedNormal();
				contact.setPositionNormalAndDistance( checkPoint.getPosition(), normal, currentFraction );

				hkSimdReal pathProjected2 = normal.dot<3>( path );
				if ( pathProjected2.isGreaterEqualZero() )
				{
					return;	// normal points away
				}
				pathProjected2 = -pathProjected2;


				const hkSimdReal startDistance2 = checkPoint.getDistanceSimdReal();

				//
				//	pre distance is the negative already traveled distance relative to the new normal
				//
				const hkSimdReal preDistance = pathProjected2 * contact.getDistanceSimdReal();
				HK_ASSERT2(0x573be33d,  preDistance.isGreaterEqualZero(), "Numerical accuracy problem in linearCast" );

				if ( startDistance2 + preDistance > pathProjected2 )
				{
					// endDistance + preDistance = realEndDistance;
					// if realEndDistance > 0, than endplane is not penetrated, so no hit
					return;
				}

				if ( startDistance2 <= hkSimdReal::fromFloat(input.m_config->m_iterativeLinearCastEarlyOutDistance) )
				{
					// early out, because we are already very close
					break;
				}

				// now we know that pathProjected2 < 0
				// so the division is safe
				currentFraction = currentFraction + (startDistance2 / pathProjected2);
				if ( currentFraction > hkSimdReal::fromFloat(collector.getEarlyOutDistance()) )
				{
					// the next currentFraction would already be beyond the earlyOutDistance, so no more hits possible
					return;
				}
			}
		}
	}

	// Select either the welded or unwelded contact normal to return
	if(in2.m_weldClosestPoints.val())
	{
		contact.setNormalOnly(checkPointCollector.getHitContact().getNormal());
	}
	hkpCdPoint event(bodyA, bodyB, contact);
	collector.addCdPoint( event );
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
