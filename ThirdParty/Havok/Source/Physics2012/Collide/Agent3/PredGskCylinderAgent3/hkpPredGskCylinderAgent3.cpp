/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Physics2012/Internal/Collide/Gjk/hkpGsk.h>

#include <Physics2012/Collide/Agent/hkpCollisionAgentConfig.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>

#include <Physics2012/Collide/Agent/hkpCollisionQualityInfo.h>

#include <Physics2012/Collide/Agent3/PredGskCylinderAgent3/hkpPredGskCylinderAgent3.h>
#include <Physics2012/Collide/Agent3/PredGskAgent3/hkpPredGskAgent3.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Physics2012/Internal/Collide/Gjk/GskManifold/hkpGskManifoldUtil.h>
#include <Physics2012/Internal/Collide/Gjk/Agent/hkpGskAgentUtil.h>
#include <Physics2012/Internal/Collide/Gjk/Continuous/hkpContinuousGsk.h>

#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>

HK_COMPILE_TIME_ASSERT(hkpGskCache::GSK_FLAGS_REPLACE_SHAPE_A_WITH_CAPSULE << 1 == hkpGskCache::GSK_FLAGS_REPLACE_SHAPE_B_WITH_CAPSULE);


void HK_CALL hkPredGskCylinderAgent3::registerAgent3(hkpCollisionDispatcher* dispatcher)
{
	hkpCollisionDispatcher::Agent3Funcs f;
	f.m_createFunc   = create;
	f.m_processFunc  = process;
	f.m_sepNormalFunc = hkPredGskAgent3::sepNormal;
	f.m_cleanupFunc  = hkPredGskAgent3::cleanup;
	f.m_removePointFunc  = hkPredGskAgent3::removePoint;
	f.m_commitPotentialFunc  = hkPredGskAgent3::commitPotential;
	f.m_createZombieFunc  = hkPredGskAgent3::createZombie;
	f.m_destroyFunc  = hkPredGskAgent3::destroy;
	f.m_isPredictive = true;
	f.m_ignoreSymmetricVersion = true;
	dispatcher->registerAgent3( f, hkcdShapeType::CYLINDER, hkcdShapeType::CONVEX );
	f.m_reusePreviousEntry = true;
	dispatcher->registerAgent3( f, hkcdShapeType::CONVEX, hkcdShapeType::CYLINDER );
}


static HK_FORCE_INLINE void replaceCylindersWithCapsulesInCdBodies(hkPadSpu<const hkpCdBody*>** cdBodies, hkPadSpu<hkBool32>* isReplacedWithCapsule, hkpCdBody* temporaryCdBodies, hkPadSpu<const hkpCdBody*>* originalCdBodies)


{
	for (int i = 0; i < 2; i++)
	{
		if (isReplacedWithCapsule[i].val())
		{

			const hkpCylinderShape* cylinder = static_cast<const hkpCylinderShape*>((*cdBodies[i])->getShape());

			//
			// It is important to add a little bit of extra padding in the cylinder -- here 0.002f helps.
			// (Look at TooManyToiEventsForCylindersDemo.) With out this padding the shape generates *loads*
			// (hundreds/thousands) of redundant TOIs. With this padding (at this value) it generates none.
			//
			hkpCapsuleShape* capsule = new hkpCapsuleShape(cylinder->getVertex<0>(), cylinder->getVertex<1>(), cylinder->getCylinderRadius() + cylinder->getRadius() + 0.002f);

			new (&temporaryCdBodies[i]) hkpCdBody((*cdBodies[i])/*->getParent()*/, (*cdBodies[i])->getMotionState()); // must have current body as parent, if is the current body is the rootCollidable
			
			temporaryCdBodies[i].setShape(capsule, HK_INVALID_SHAPE_KEY);

			originalCdBodies[i] = (*cdBodies[i]).val();
			(*cdBodies[i]) = &temporaryCdBodies[i];
		}
		else
		{
			originalCdBodies[i] = HK_NULL;
		}
	}
}

static HK_FORCE_INLINE void restoreOriginalShapesInCdBodies(hkPadSpu<const hkpCdBody*>** cdBodies, hkPadSpu<const hkpCdBody*>* originalCdBodies)
{
	for (int i = 0; i < 2; i++)
	{
		if (originalCdBodies[i])
		{
			(*cdBodies[i])->getShape()->removeReference(); 
			(*cdBodies[i]) = originalCdBodies[i];
		}
	}
}

static HK_FORCE_INLINE void resetGskManifold(hkpGskManifold* gskManifold, hkpAgentEntry* entry, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner)
{
	hkGskManifold_cleanup( *gskManifold, mgr, constraintOwner );
	entry->m_numContactPoints = gskManifold->m_numContactPoints;
	
	// Inlining this assert causes an Internal compiler error on Mac
#if !( defined(HK_PLATFORM_MAC386) || defined(HK_PLATFORM_MACPPC) )
	HK_ASSERT2(0xad324443, unsigned(entry->m_numContactPoints) == 0, "Internal check");
#endif
}


// Convert from cylinder to capsule: convert vertexId's + remove doubled/tripled points
static HK_FORCE_INLINE void convertCylinderCacheToCapsuleCache(hkpGskCache& gskCache, hkPadSpu<hkBool32>* isReplacedWithCapsule)
{
	if (isReplacedWithCapsule[0].val())
	{
		gskCache.m_vertices[0] = hkpVertexId((1-((gskCache.m_vertices[ 0 ] >> hkpCylinderShape::VERTEX_ID_ENCODING_IS_BASE_A_SHIFT) & 1)) * 16);

		if (gskCache.m_dimA > 1)
		{
			gskCache.m_vertices[0] = 16;
			gskCache.m_vertices[1] = 0;

			// Capsule cannot have 3 verts!
			if (gskCache.m_dimA == 3)
			{
				gskCache.m_dimA = 2;
				gskCache.m_vertices[2] = gskCache.m_vertices[3];
			}
		}
	}

	if (isReplacedWithCapsule[1].val())
	{
		gskCache.m_vertices[gskCache.m_dimA] = hkpVertexId((1-((gskCache.m_vertices[ gskCache.m_dimA ] >> hkpCylinderShape::VERTEX_ID_ENCODING_IS_BASE_A_SHIFT) & 1)) * 16);

		if (gskCache.m_dimB > 1)
		{
			gskCache.m_vertices[gskCache.m_dimA] = 16;
			gskCache.m_vertices[gskCache.m_dimA+1] = 0;

			// Capsule cannot have 3 verts!
			if (gskCache.m_dimB == 3)
			{
				gskCache.m_dimB = 2;
			}
		}
	}
}


hkpAgentData* HK_CALL hkPredGskCylinderAgent3::create  ( const hkpAgent3Input& input, hkpAgentEntry* entry, hkpAgentData* agentData )
{
	hkpAgentData* returnValue = hkPredGskAgent3::create(input, entry, agentData);

	hkPredGskAgent3::setGskFlagToFalse(agentData, hkpGskCache::GSK_FLAGS_CYLINDER_AGENT_FLAGS);
	if (input.m_bodyA->getShape()->getType() == hkcdShapeType::CYLINDER) 
	{
		hkPredGskAgent3::setGskFlagToTrue(agentData, hkpGskCache::GSK_FLAGS_SHAPE_A_IS_CYLINDER); 
		hkPredGskAgent3::setGskFlagToTrue(agentData, hkpGskCache::GSK_FLAGS_REPLACE_SHAPE_A_WITH_CAPSULE); 
	}
	if (input.m_bodyB->getShape()->getType() == hkcdShapeType::CYLINDER) 
	{
		hkPredGskAgent3::setGskFlagToTrue(agentData, hkpGskCache::GSK_FLAGS_SHAPE_B_IS_CYLINDER); 
		hkPredGskAgent3::setGskFlagToTrue(agentData, hkpGskCache::GSK_FLAGS_REPLACE_SHAPE_B_WITH_CAPSULE); 
	}

	return returnValue;
}



//
// Calculate separating distance here, with capsules
//
static hkSimdReal separatingDistanceUsingCapsuleRepresentations( const hkpAgent3ProcessInput& input, hkpAgentEntry*entry, hkpAgentData* agentData )
{
	hkpAgent3ProcessInput inputLocal = input;
	hkPadSpu<const hkpCdBody*>* cdBodiesLocal[2] = { const_cast<hkPadSpu<const hkpCdBody* > * >(&inputLocal.m_bodyA), 
		const_cast<hkPadSpu<const hkpCdBody* > *> (&inputLocal.m_bodyB) };
	// Always replace
	hkPadSpu<hkBool32> isReplacedWithCapsuleLocal[2] = { hkPredGskAgent3::getGskFlag(agentData, hkpGskCache::GSK_FLAGS_SHAPE_A_IS_CYLINDER),
		hkPredGskAgent3::getGskFlag(agentData, hkpGskCache::GSK_FLAGS_SHAPE_B_IS_CYLINDER) };
	hkPadSpu<const hkpCdBody*> originalCdBodiesLocal[2] = { HK_NULL, HK_NULL };
	hkpCdBody temporaryCdBodiesLocal[2];

	replaceCylindersWithCapsulesInCdBodies(cdBodiesLocal, isReplacedWithCapsuleLocal, temporaryCdBodiesLocal, originalCdBodiesLocal );

	const hkpConvexShape* shapeA = static_cast<const hkpConvexShape*>(inputLocal.m_bodyA->getShape());
	const hkpConvexShape* shapeB = static_cast<const hkpConvexShape*>(inputLocal.m_bodyB->getShape());
	hkpGskCache emptyCache; emptyCache.init(shapeA, shapeB, input.m_aTb);
	hkVector4 checkForCapsuleSeparatingDistance;
	hkPredGskAgent3::sepNormal(inputLocal, entry, reinterpret_cast<hkpAgentData*>(&emptyCache), checkForCapsuleSeparatingDistance);

	restoreOriginalShapesInCdBodies(cdBodiesLocal, originalCdBodiesLocal);

	return checkForCapsuleSeparatingDistance.getW();
}




//
// This agent uses cylinder representation for tims, and returns separating normal for the cylinder representation.
// Basing on the cylinder's separating normal it determines whether it can use a capsule representation.
// If so it clears the manifold and then updates it with capsule's vertices.
// Cache is never cleared it always holds data for the cylinder representation.
// The same cache is used for manifold calculation. For input the cache is modified and they cylinder's points are transformed
// to capsule's points, also their number is limited to 2. The gsk information from manifold update is not exported 
// if the agent is in the cylinder mode.
//
hkpAgentData* HK_CALL hkPredGskCylinderAgent3::process( const hkpAgent3ProcessInput& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkVector4* separatingNormal, hkpProcessCollisionOutput& output)
{
	//
	//	Get material
	//
	HK_TIMER_BEGIN("PredGskf3", HK_NULL );
	HK_INTERNAL_TIMER_BEGIN_LIST("intern" , "init" );

	//
	//	Calc the relative movement for this timestep
	//

	hkSimdReal distAtT1; distAtT1.load<1>(&(input.m_distAtT1.ref()));

	hkpGskCache* gskCache = reinterpret_cast<hkpGskCache*>( agentData );
	hkpAgentData* nextData = gskCache+1;
	hkpGskManifold* gskManifold = reinterpret_cast<hkpGskManifold*>(nextData);


	//
	// Init + determine currently used  representation for cylinders
	// 

	hkPadSpu<const hkpCdBody*>* cdBodies[2] = { const_cast<hkPadSpu<const hkpCdBody* > * >(&input.m_bodyA), 
												const_cast<hkPadSpu<const hkpCdBody* > *> (&input.m_bodyB) };
	
	hkPadSpu<hkBool32> isCylinder[2] = { hkPredGskAgent3::getGskFlag(agentData, hkpGskCache::GSK_FLAGS_SHAPE_A_IS_CYLINDER),
										 hkPredGskAgent3::getGskFlag(agentData, hkpGskCache::GSK_FLAGS_SHAPE_B_IS_CYLINDER) };

	hkPadSpu<hkBool32> isReplacedWithCapsule[2] = {	hkPredGskAgent3::getGskFlag(agentData, hkpGskCache::GSK_FLAGS_REPLACE_SHAPE_A_WITH_CAPSULE), 
													hkPredGskAgent3::getGskFlag(agentData, hkpGskCache::GSK_FLAGS_REPLACE_SHAPE_B_WITH_CAPSULE) };

	hkPadSpu<const hkpCdBody*> originalCdBodies[2] = { HK_NULL, HK_NULL };
	hkpCdBody temporaryCdBodies[2];

	//
	// Checking whether current representation is ok for the angles
	//
	hkVector4 checkForCylinderSeparatingNormal;	
	{

		//
		// Calculate separating normal here, with original cylinders
		//
		hkPredGskAgent3::sepNormal(input, entry, agentData, checkForCylinderSeparatingNormal);

		for (int i = 0; i < 2; i++)
		{
			if (isCylinder[i].val())
			{
				const hkpCylinderShape* cylinder = static_cast<const hkpCylinderShape*>((*cdBodies[i])->getShape());
				hkVector4 axis; axis.setSub(cylinder->getVertex<1>(), cylinder->getVertex<0>());
				axis.normalize<3>();
				axis._setRotatedDir( (*cdBodies[i])->getTransform().getRotation(), axis);
				hkSimdReal absDot; absDot.setAbs(axis.dot<3>(checkForCylinderSeparatingNormal));

				const hkSimdReal distAsCapsule = separatingDistanceUsingCapsuleRepresentations(input, entry, agentData);

				if (isReplacedWithCapsule[i].val())
				{
					//we only switch back when normal is not perpendicular to axis, and when we have penetration
					if (absDot.isGreater(hkSimdReal::fromFloat(hkReal(0.1f))) && distAsCapsule.isLessZero())
					{
						// switch to cylinder
 						isReplacedWithCapsule[i] = false;
						hkPredGskAgent3::setGskFlagToFalse(agentData, hkpGskCache::GskFlagValues(hkpGskCache::GSK_FLAGS_REPLACE_SHAPE_A_WITH_CAPSULE << i));
   						resetGskManifold(gskManifold, entry, input.m_contactMgr, *output.m_constraintOwner.val() );
					}
				}
				else
				{
					// check the angle and distance to revert back to cylinder ?? 
					// this will break in deeply penetrating cases ?
					if (absDot.isLess(hkSimdReal::fromFloat(hkReal(0.05f))) || distAsCapsule.isGreater(hkSimdReal::fromFloat(hkReal(0.01f))))
					{
						// switch to capsule
						isReplacedWithCapsule[i] = true;
						hkPredGskAgent3::setGskFlagToTrue(agentData,hkpGskCache::GskFlagValues(hkpGskCache::GSK_FLAGS_REPLACE_SHAPE_A_WITH_CAPSULE << i)); 
						resetGskManifold(gskManifold, entry, input.m_contactMgr, *output.m_constraintOwner.val());
					}
				}

			}
		}
	}

		//
		// tim early out for manifolds
		// We only want to generate new points in the manifold if the worst case estimated
		// distance is less than ci.m_manifoldTimDistance.
		// If not, we simply grep the points from the manifold 
		//
	int explicitlyAllowNewPoint = 0;
	bool shapesReplaced = false;
	const hkpCollisionQualityInfo& qi = *input.m_input->m_collisionQualityInfo;
	if ( qi.m_useContinuousPhysics.val() )
	{
		//
		//  Check if our worst case distance is below our minSeparation,
		//  if it is, we have to check for TOIs
		//
		hkSimdReal allowedPenetrationA; allowedPenetrationA.load<1>(&input.m_bodyA->getRootCollidable()->m_allowedPenetrationDepth);
		hkSimdReal allowedPenetrationB; allowedPenetrationB.load<1>(&input.m_bodyB->getRootCollidable()->m_allowedPenetrationDepth);
		hkSimdReal allowedPenetrationDepth; allowedPenetrationDepth.setMin( allowedPenetrationA, allowedPenetrationB );

		HK_ASSERT2(0xad56dbbe, allowedPenetrationDepth.isGreaterZero(), "hkpCollidable::m_allowedPenetrationDepth must always be set to a positive non-zero value.");

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
	}
	else
	{	// conditions ok, to not use continuous physics (no cont-phys or tims ok)
QUICK_VERIFY_MANIFOLD:
		if ( distAtT1.isGreater(hkSimdReal::fromFloat(qi.m_manifoldTimDistance)) && ( ! hkPredGskAgent3::getGskFlag(agentData, hkpGskCache::GSK_FLAGS_DISABLE_CONTACT_TIMS ) ) && checkForCylinderSeparatingNormal.getW().isLess(hkSimdReal::fromFloat(input.m_input->m_tolerance)) )
		{
			separatingNormal->setW(distAtT1);
			if ( gskManifold->m_numContactPoints )
			{
				HK_INTERNAL_TIMER_SPLIT_LIST("getPoints");

				//
				//
				//
				{
					replaceCylindersWithCapsulesInCdBodies(cdBodies, isReplacedWithCapsule, temporaryCdBodies, originalCdBodies );
					shapesReplaced = true;
				}
				// NO CACHE NEEDED

				hkpGskManifoldWork work;
				hkGskManifold_init( (*gskManifold), (*separatingNormal), *input.m_bodyA, *input.m_bodyB, input.m_input->getTolerance(), work );
				explicitlyAllowNewPoint |= hkGskManifold_verifyAndGetPoints( (*gskManifold), work, 0, output, input.m_contactMgr ); 

				if (0 == explicitlyAllowNewPoint || !(gskCache->m_gskFlags & hkpGskCache::GSK_FLAGS_ALLOW_QUICKER_CONTACT_POINT_RECREATION))
				{
					// mark the first contact as a representative contact
#if defined(HK_1N_MACHINE_SUPPORTS_WELDING)
					if ( gskManifold->m_numContactPoints && output.m_potentialContacts )
					{
						*(output.m_potentialContacts->m_firstFreeRepresentativeContact++) = output.m_firstFreeContactPoint - gskManifold->m_numContactPoints;
					}
#endif
					goto END_OF_FUNCTION;
				}
				else
				{
					// abort all confirmed points
					output.uncommitContactPoints(gskManifold->m_numContactPoints);
				}
			}
			else
			{
				goto END_OF_FUNCTION;
			}
		}
	}

	HK_INTERNAL_TIMER_SPLIT_LIST("process");
	if(checkForCylinderSeparatingNormal.getW().isLess(hkSimdReal::fromFloat(input.m_input->m_tolerance)))
	{
		//
		//
		//
		if (!shapesReplaced)
		{
			replaceCylindersWithCapsulesInCdBodies(cdBodies, isReplacedWithCapsule, temporaryCdBodies, originalCdBodies );
			shapesReplaced = true;
		}

		hkpGskCache* capsuleGskCache = gskCache;
		hkpGskCache tmpGskCache;
		if (isReplacedWithCapsule[0].val() | isReplacedWithCapsule[1].val())
		{
			capsuleGskCache = &tmpGskCache;
			tmpGskCache = *gskCache;
			convertCylinderCacheToCapsuleCache(tmpGskCache, isReplacedWithCapsule);
		}

		hkGskAgentUtil_processCollisionNoTim( input, entry, agentData, *capsuleGskCache, *gskManifold, *separatingNormal, explicitlyAllowNewPoint, output );
	}
	else
	{
		//cleanup the agent
		if ( gskManifold->m_numContactPoints)
		{
			hkGskManifold_cleanup( *gskManifold, input.m_contactMgr, *output.m_constraintOwner.val() );
		}
	}
	// override normal from our cylinder cylinder code
	*separatingNormal = checkForCylinderSeparatingNormal;

END_OF_FUNCTION:

	//
	// Revert to original shapes.
	//
	{
		restoreOriginalShapesInCdBodies(cdBodies, originalCdBodies);
	}

	entry->m_numContactPoints = gskManifold->m_numContactPoints;
	HK_INTERNAL_TIMER_END_LIST();
	HK_TIMER_END();
	
	HK_ASSERT(0x44ff9924,  HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeof(hkpGskCache) + gskManifold->getTotalSizeInBytes()) <= hkAgent3::MAX_NET_SIZE);
	return hkAddByteOffset( agentData, HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeof(hkpGskCache) + gskManifold->getTotalSizeInBytes()) );
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
