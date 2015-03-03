/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>

#include <Physics2012/Internal/Collide/Gjk/hkpGsk.h>

#include <Physics2012/Collide/Agent/hkpCollisionAgentConfig.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>

#include <Physics2012/Collide/Agent/hkpCollisionQualityInfo.h>

#include <Physics2012/Collide/Agent3/PredGskAgent3/hkpPredGskAgent3.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Physics2012/Internal/Collide/Gjk/GskManifold/hkpGskManifoldUtil.h>
#include <Physics2012/Internal/Collide/Gjk/Agent/hkpGskAgentUtil.h>
#include <Physics2012/Internal/Collide/Gjk/Continuous/hkpContinuousGsk.h>



void HK_CALL hkPredGskAgent3::initAgentFunc(hkpCollisionDispatcher::Agent3Funcs& f)
{
	f.m_createFunc   = create;
	f.m_processFunc  = process;
	f.m_sepNormalFunc = sepNormal;
	f.m_cleanupFunc  = cleanup;
#if !defined(HK_PLATFORM_SPU)
	f.m_removePointFunc  = removePoint;
	f.m_commitPotentialFunc  = commitPotential;
	f.m_createZombieFunc  = createZombie;
#endif
	f.m_destroyFunc  = destroy;
	f.m_isPredictive = true;
}

#if !defined(HK_PLATFORM_SPU)
void HK_CALL hkPredGskAgent3::registerAgent3(hkpCollisionDispatcher* dispatcher, hkpShapeType typeA, hkpShapeType typeB)
{
	hkpCollisionDispatcher::Agent3Funcs f;
	initAgentFunc(f);
	dispatcher->registerAgent3( f, typeA, typeB );
}
#endif

hkpAgentData* HK_CALL hkPredGskAgent3::create  ( const hkpAgent3Input& input, hkpAgentEntry* entry, hkpAgentData* agentData )
{
	hkpGskCache* gskCache = reinterpret_cast<hkpGskCache*>( agentData );


	const hkpConvexShape* shapeA = static_cast<const hkpConvexShape*>(input.m_bodyA->getShape());
	const hkpConvexShape* shapeB = static_cast<const hkpConvexShape*>(input.m_bodyB->getShape());

	if ( shapeB->getType() == hkcdShapeType::TRIANGLE )
	{
		gskCache->initTriangle( shapeA, reinterpret_cast<const hkpTriangleShape*>(shapeB), input.m_aTb );
	}
	else
	{
		gskCache->init( shapeA, shapeB, input.m_aTb );
	}
	
	entry->m_numContactPoints = 0;
	setGskFlagToFalse(agentData, hkpGskCache::GSK_FLAGS_DISABLE_CONTACT_TIMS);

#if !defined(HK_PLATFORM_SPU)
	hkpGskManifold* gskManifold = reinterpret_cast<hkpGskManifold*>(gskCache+1);
	gskManifold->init();
	int sizeofGskAgent3 = sizeof(hkpGskCache) + gskManifold->getTotalSizeInBytes();
#else
	hkGskManifoldPpu* gskManifold = reinterpret_cast<hkGskManifoldPpu*>(gskCache+1);
	gskManifold->init();
	int sizeofGskAgent3 = sizeof(hkpGskCache) + gskManifold->getTotalSizeInBytes();
#endif

	HK_ASSERT(0x44ff9920,  HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeofGskAgent3) <= hkAgent3::MAX_NET_SIZE);

	return hkAddByteOffset( agentData, HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeofGskAgent3) );
}


void    HK_CALL hkPredGskAgent3::sepNormal( const hkpAgent3Input& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkVector4& separatingNormalOut )
{
	hkpGskCache* gskCache = reinterpret_cast<hkpGskCache*>( agentData );

	const hkpConvexShape* shapeA = static_cast<const hkpConvexShape*>(input.m_bodyA->getShape());
	const hkpConvexShape* shapeB = static_cast<const hkpConvexShape*>(input.m_bodyB->getShape());

		// Call the collision detector
	hkpGsk gsk;
	gsk.init( shapeA, shapeB, *gskCache );
	hkVector4 separatingNormal; gsk.getClosestFeature(shapeA, shapeB, input.m_aTb, separatingNormal);
	gsk.checkForChangesAndUpdateCache( *gskCache );

	separatingNormalOut._setRotatedDir( input.m_bodyA->getTransform().getRotation(), separatingNormal);
#if defined(HK_PLATFORM_PS3_SPU)
	separatingNormalOut(3) = separatingNormal(3) - (shapeA->getRadius() + shapeB->getRadius());
#else
	separatingNormalOut.setW( separatingNormal.getW() - hkSimdReal::fromFloat(shapeA->getRadius() + shapeB->getRadius()) );
#endif
}



#if !defined(HK_PLATFORM_SPU)
void    HK_CALL hkPredGskAgent3::removePoint ( hkpAgentEntry* entry, hkpAgentData* agentData, hkContactPointId idToRemove )
{
	hkpGskCache* gskCache = reinterpret_cast<hkpGskCache*>( agentData );
	hkpGskManifold* gskManifold = reinterpret_cast<hkpGskManifold*>(gskCache+1);
	for ( int i = 0; i < gskManifold->m_numContactPoints; i++)
	{
		if ( gskManifold->m_contactPoints[i].m_id == idToRemove)
		{
			hkGskManifold_removePoint( *gskManifold, i );
			break;
		}
	}
}

void    HK_CALL hkPredGskAgent3::commitPotential( hkpAgentEntry* entry, hkpAgentData* agentData, hkContactPointId idToCommit )
{
	hkpGskCache* gskCache = reinterpret_cast<hkpGskCache*>( agentData );
	hkpGskManifold* gskManifold = reinterpret_cast<hkpGskManifold*>(gskCache+1);
	for ( int i = 0; i < gskManifold->m_numContactPoints; i++)
	{
		if ( gskManifold->m_contactPoints[i].m_id == HK_INVALID_CONTACT_POINT)
		{
			gskManifold->m_contactPoints[i].m_id = idToCommit;
			return;
		}
	}
	HK_ASSERT2( 0xf0de2ead, 0, "Cannot find contact point in agent, maybe the memory is corrupt");
}

void	HK_CALL hkPredGskAgent3::createZombie( hkpAgentEntry* entry, hkpAgentData* agentData, hkContactPointId idToConvert )
{
	hkpGskCache* gskCache = reinterpret_cast<hkpGskCache*>( agentData );
	hkpGskManifold* gskManifold = reinterpret_cast<hkpGskManifold*>(gskCache+1);
	for ( int i = 0; i < gskManifold->m_numContactPoints; i++)
	{
		hkpGskManifold::ContactPoint& cp = gskManifold->m_contactPoints[i];
		if ( cp.m_id == idToConvert)
		{
			cp.m_dimA = 0;
			cp.m_dimB = 0;
			break;
		}
	}
}
#endif



hkpAgentData* HK_CALL hkPredGskAgent3::cleanup ( hkpAgentEntry* entry, hkpAgentData* agentData, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner )
{
	hkpGskCache* gskCache = reinterpret_cast<hkpGskCache*>( agentData );
#if !defined(HK_PLATFORM_SPU)
	hkpGskManifold* gskManifold = reinterpret_cast<hkpGskManifold*>(gskCache+1);
	hkGskManifold_cleanup( *gskManifold, mgr, constraintOwner );
	int manifoldSize = gskManifold->getTotalSizeInBytes();
#else
	hkGskManifoldPpu* gskManifold = reinterpret_cast<hkGskManifoldPpu*>(gskCache+1);
	hkpGskManifold spuManifold; spuManifold.loadFromPacked( *gskManifold );
	hkGskManifold_cleanup( spuManifold, mgr, constraintOwner );
	gskManifold->init();
	int manifoldSize = gskManifold->getTotalSizeInBytes();
#endif

	HK_ASSERT(0x44ff9921,  HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeof(hkpGskCache) + manifoldSize) <= hkAgent3::MAX_NET_SIZE);

	entry->m_numContactPoints = 0;
	return hkAddByteOffset( agentData, HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeof(hkpGskCache) + manifoldSize) );
}

void  HK_CALL hkPredGskAgent3::destroy ( hkpAgentEntry* entry, hkpAgentData* agentData, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner, hkpCollisionDispatcher* dispatcher )
{
	cleanup( entry, agentData, mgr, constraintOwner );
}


/*
static void HK_CALL hkPredGskAgent3_calcSeparatingNormal( const hkpCdBody& bodyA, const hkpCdBody& bodyB, hkReal earlyOutTolerance, hkpGsk& gsk, hkVector4& separatingNormalOut )
{
	const hkpConvexShape* shapeA = static_cast<const hkpConvexShape*>(bodyA.getShape());
	const hkpConvexShape* shapeB = static_cast<const hkpConvexShape*>(bodyB.getShape());

	// Get the relative transform for the two bodies for the collision detector
	hkTransform aTb;	aTb.setMulInverseMul( bodyA.getTransform(), bodyB.getTransform());

		// Call the collision detector
	hkVector4 separatingNormal; gsk.getClosestFeature(shapeA, shapeB, aTb, separatingNormal);

	separatingNormalOut._setRotatedDir( bodyA.getTransform().getRotation(), separatingNormal);
	separatingNormalOut(3) = separatingNormal(3) - (shapeA->getRadius() + shapeB->getRadius());
}
*/

#if !defined(HK_PLATFORM_SPU)
	HK_COMPILE_TIME_ASSERT( sizeof(hkpGskCache) + sizeof(hkpGskManifold) == 16*5 );
#else
	HK_COMPILE_TIME_ASSERT( sizeof(hkpGskCache) + sizeof(hkGskManifoldPpu) == 16*5 );
#endif

hkpAgentData* HK_CALL hkPredGskAgent3::process( const hkpAgent3ProcessInput& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkVector4* separatingNormal, hkpProcessCollisionOutput& output)
{
	//
	//	Get material
	//
	HK_TIME_CODE_BLOCK("PredGskf3", HK_NULL );
	HK_INTERNAL_TIMER_BEGIN_LIST("intern" , "init" );

	//
	//	Calc the relative movement for this timestep
	//

	hkpGskCache* gskCache = reinterpret_cast<hkpGskCache*>( agentData );
	hkpAgentData* nextData = gskCache+1;
#if !defined(HK_PLATFORM_SPU)
	hkSimdReal distAtT1; distAtT1.load<1>(&(input.m_distAtT1.ref()));
	hkpGskManifold& gskManifold = *reinterpret_cast<hkpGskManifold*>(nextData);
#else
	hkReal distAtT1 = input.m_distAtT1;
	hkGskManifoldPpu* ppuManifold = reinterpret_cast<hkGskManifoldPpu*>(nextData);
	hkpGskManifold gskManifold; gskManifold.loadFromPacked( *ppuManifold );
#endif
	hkCheckDeterminismUtil::checkMt( 0xf00001c0, 0xadadadad);
	hkCheckDeterminismUtil::checkMt( 0xf00001c0, gskManifold.m_numContactPoints);
	hkCheckDeterminismUtil::checkMtCrc( 0xf00001c1, &input.m_bodyA->getTransform(),1);
	hkCheckDeterminismUtil::checkMtCrc( 0xf00001c2, &input.m_bodyB->getTransform(),1);
	hkCheckDeterminismUtil::checkMtCrc( 0xf00001c3, &input.m_linearTimInfo,1);
	hkCheckDeterminismUtil::checkMt( 0xf00001c4, input.m_input->m_collisionQualityInfo->m_minSeparation);


		//
		// tim early out for manifolds
		// We only want to generate new points in the manifold if the worst case estimated
		// distance is less than ci.m_manifoldTimDistance.
		// If not, we simply grep the points from the manifold 
		//
	int explicitlyAllowNewPoint = 0;
	const hkpCollisionQualityInfo& qi = *input.m_input->m_collisionQualityInfo;
	hkCheckDeterminismUtil::checkMt( 0xf00001c5, qi.m_useContinuousPhysics.val());
	if ( qi.m_useContinuousPhysics.val() )
	{
#if defined(HK_PLATFORM_PS3_SPU)
		const hkReal allowedPenetrationA = input.m_bodyA->getRootCollidable()->m_allowedPenetrationDepth;
		const hkReal allowedPenetrationB = input.m_bodyB->getRootCollidable()->m_allowedPenetrationDepth;
		const hkReal allowedPenetrationDepth = hkMath::min2( allowedPenetrationA, allowedPenetrationB );
		HK_ASSERT2(0xad56dbbf, allowedPenetrationDepth > 0.0f, "hkpCollidable::m_allowedPenetrationDepth must always be set to a positive non-zero value.");
		const hkReal distance = (*separatingNormal)(3);
		const hkReal minSeparation  = hkMath::min2( qi.m_minSeparation * allowedPenetrationDepth, distance + qi.m_minExtraSeparation * allowedPenetrationDepth );
		if (distAtT1 >= minSeparation)
		{
			goto QUICK_VERIFY_MANIFOLD;
		}
		const hkReal toiSeparation = hkMath::min2( qi.m_toiSeparation * allowedPenetrationDepth, distance + qi.m_toiExtraSeparation * allowedPenetrationDepth );

		HK_INTERNAL_TIMER_SPLIT_LIST("toi");
		hk4dGskCollideCalcToi( input, hkSimdReal::fromFloat(allowedPenetrationDepth), hkSimdReal::fromFloat(minSeparation), hkSimdReal::fromFloat(toiSeparation), *gskCache, *separatingNormal, output );

#else
		//
		//  Check if our worst case distance is below our minSeparation,
		//  if it is, we have to check for TOIs
		//
		hkSimdReal allowedPenetrationA; allowedPenetrationA.load<1>(&HK_PADSPU_REF(input.m_bodyA->getRootCollidable()->m_allowedPenetrationDepth));
		hkSimdReal allowedPenetrationB; allowedPenetrationB.load<1>(&HK_PADSPU_REF(input.m_bodyB->getRootCollidable()->m_allowedPenetrationDepth));
		hkSimdReal allowedPenetrationDepth; allowedPenetrationDepth.setMin( allowedPenetrationA, allowedPenetrationB );

		HK_ASSERT2(0xad56dbbf, allowedPenetrationDepth.isGreaterZero(), "hkpCollidable::m_allowedPenetrationDepth must always be set to a positive non-zero value.");

		const hkSimdReal distance = separatingNormal->getW();
		hkSimdReal minSep; minSep.load<1>(&qi.m_minSeparation);
		hkSimdReal minExtraSep; minExtraSep.load<1>(&qi.m_minExtraSeparation);
		hkSimdReal minSeparation; minSeparation.setMin( minSep * allowedPenetrationDepth, distance + minExtraSep * allowedPenetrationDepth );
		if (distAtT1.isGreaterEqual(minSeparation))
		{
			goto QUICK_VERIFY_MANIFOLD;
		}
		hkSimdReal toiSep; toiSep.load<1>(&qi.m_toiSeparation);
		hkSimdReal toiExtraSep; toiExtraSep.load<1>(&qi.m_toiExtraSeparation);
		hkSimdReal toiSeparation; toiSeparation.setMin( toiSep * allowedPenetrationDepth, distance + toiExtraSep * allowedPenetrationDepth );

		HK_INTERNAL_TIMER_SPLIT_LIST("toi");
		hk4dGskCollideCalcToi( input, allowedPenetrationDepth, minSeparation, toiSeparation, *gskCache, *separatingNormal, output );
#endif
	}
	else
	{	// conditions ok, to not use continuous physics (no cont-phys or tims ok)
QUICK_VERIFY_MANIFOLD:
		//HK_MONITOR_ADD_VALUE("NumContacts", float(gskManifold.m_numContactPoints), HK_MONITOR_TYPE_INT );

#if defined(HK_PLATFORM_PS3_SPU)
		if ( (distAtT1 > qi.m_manifoldTimDistance) && ( ! getGskFlag(agentData, hkpGskCache::GSK_FLAGS_DISABLE_CONTACT_TIMS) ) )
		{
			(*separatingNormal)(3) = distAtT1;
#else
		if ( distAtT1.isGreater(hkSimdReal::fromFloat(qi.m_manifoldTimDistance)) && ( ! getGskFlag(agentData, hkpGskCache::GSK_FLAGS_DISABLE_CONTACT_TIMS) ) )
		{
			separatingNormal->setW(distAtT1);
#endif
			if ( gskManifold.m_numContactPoints )
			{
				HK_INTERNAL_TIMER_SPLIT_LIST("getPoints");
				hkpGskManifoldWork work;
				hkGskManifold_init( gskManifold, (*separatingNormal), *input.m_bodyA, *input.m_bodyB, input.m_input->getTolerance(), work );
				explicitlyAllowNewPoint |= hkGskManifold_verifyAndGetPoints( gskManifold, work, 0, output, input.m_contactMgr ); 

				if (0 == explicitlyAllowNewPoint || !(gskCache->m_gskFlags & hkpGskCache::GSK_FLAGS_ALLOW_QUICKER_CONTACT_POINT_RECREATION))
				{
					// mark the first contact as a representative contact
#if defined(HK_1N_MACHINE_SUPPORTS_WELDING)
					if ( gskManifold.m_numContactPoints && output.m_potentialContacts )
					{
						*(output.m_potentialContacts->m_firstFreeRepresentativeContact++) = output.m_firstFreeContactPoint - gskManifold.m_numContactPoints;
					}
#endif
					goto END_OF_FUNCTION;
				}
				else
				{
					// abort all confirmed points
					output.uncommitContactPoints(gskManifold.m_numContactPoints);
				}
			}
			else
			{
				goto END_OF_FUNCTION;
			}
		}
	}

	HK_INTERNAL_TIMER_SPLIT_LIST("process");
	{
		// Warning: the following line assumes that gskCache == agentData
		hkGskAgentUtil_processCollisionNoTim( input, entry, gskCache, *gskCache, gskManifold, *separatingNormal, explicitlyAllowNewPoint, output );
	}

END_OF_FUNCTION:
	entry->m_numContactPoints = hkUchar(gskManifold.m_numContactPoints);
	hkCheckDeterminismUtil::checkMt( 0xf00001c6, gskManifold.m_numContactPoints);
	hkCheckDeterminismUtil::checkMtCrc( 0xf00002e6, gskCache,1);
	HK_INTERNAL_TIMER_END_LIST();
#if !defined(HK_PLATFORM_SPU)
	HK_ASSERT(0x44ff9922,  HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeof(hkpGskCache) + gskManifold.getTotalSizeInBytes()) <= hkAgent3::MAX_NET_SIZE);
	return hkAddByteOffset( agentData, HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeof(hkpGskCache) + gskManifold.getTotalSizeInBytes()) );
#else
	gskManifold.storeToPacked( *ppuManifold );
	HK_ASSERT(0x44ff9923,  HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeof(hkpGskCache) + ppuManifold->getTotalSizeInBytes()) <= hkAgent3::MAX_NET_SIZE);
	return hkAddByteOffset( agentData, HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeof(hkpGskCache) + ppuManifold->getTotalSizeInBytes()) );
#endif
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
