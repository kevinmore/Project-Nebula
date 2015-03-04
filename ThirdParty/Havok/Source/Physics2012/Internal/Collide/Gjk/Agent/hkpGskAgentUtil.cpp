/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Internal/hkpInternal.h>
#include <Physics2012/Collide/Util/Welding/hkpMeshWeldingUtility.h>
#include <Physics2012/Internal/Collide/Gjk/hkpGsk.h>

#include <Physics2012/Collide/Shape/Convex/hkpConvexShape.h>

#include <Physics2012/Collide/Agent/hkpCollisionAgentConfig.h>
#include <Physics2012/Collide/Agent/hkpCollisionQualityInfo.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>

#include <Physics2012/Collide/Agent/ConvexAgent/Gjk/hkpGskfAgent.h>
#include <Physics2012/Internal/Collide/Gjk/GskManifold/hkpGskManifoldUtil.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>

#include <Physics2012/Collide/Agent3/PredGskAgent3/hkpPredGskAgent3.h>

#include <Common/Visualize/hkDebugDisplay.h>

#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>

#include <Physics2012/Internal/Collide/Gjk/Agent/hkpGskAgentUtil.h>

#if defined(HK_PLATFORM_SPU)
#	include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgrSpu.inl> // include this after we include the actual contact manager!
#endif


// hkVector4& separatingNormal is output only.
void hkGskAgentUtil_processCollisionNoTim(const hkpAgent3Input& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkpGskCache& gskCache, hkpGskManifold& gskManifold, hkVector4& separatingNormal, int explicitlyAllowNewPoint, hkpProcessCollisionOutput& output )
{
	hkVector4 pointOnB;
	hkpGsk::GetClosesetPointInput gskInput;
	{
		gskInput.m_shapeA = static_cast<const hkpConvexShape*>(input.m_bodyA->getShape());
		gskInput.m_shapeB = static_cast<const hkpConvexShape*>(input.m_bodyB->getShape());
		gskInput.m_aTb = &input.m_aTb;
		gskInput.m_transformA = &input.m_bodyA->getTransform();
		gskInput.m_collisionTolerance = input.m_input->getTolerance();
	}

	hkPredGskAgent3::setGskFlagToFalse(agentData, hkpGskCache::GSK_FLAGS_DISABLE_CONTACT_TIMS);

	if( hkpGsk::getClosestPoint( gskInput, gskCache, separatingNormal, pointOnB ) == HK_FAILURE )
	{
		if ( gskManifold.m_numContactPoints)
		{
			hkGskManifold_cleanup( gskManifold, input.m_contactMgr, *output.m_constraintOwner.val() );
		}
	}
	else
	{

		hkVector4 weldedNormal = separatingNormal;
		
		hkpConvexShape::WeldResult weldResult;

		if (gskInput.m_shapeB->getType() == hkcdShapeType::TRIANGLE )
		{
			
			weldResult = (hkpConvexShape::WeldResult)gskInput.m_shapeB->weldContactPoint(	&gskCache.m_vertices[gskCache.m_dimA], gskCache.m_dimB, pointOnB, 
																							&input.m_bodyB->getTransform(), gskInput.m_shapeA, &input.m_bodyA->getTransform(), weldedNormal );

EXAMINE_WELD_RESULTS:
			if (weldResult == hkpConvexShape::WELD_RESULT_REJECT_CONTACT_POINT)
			{
				// If the closest point is rejected, reject all other points
				// If we do not do this then an existing edge collision in the manifold can take on the
				// un-welded separating normal.
				// If we cache the info of the first manifold point, we could then weld the manifold in this case
				// rejectAddedPointByWelding = true;
				hkGskManifold_cleanup( gskManifold, input.m_contactMgr, *output.m_constraintOwner.val() );
				return;
			}
			else if (weldResult == hkpConvexShape::WELD_RESULT_ACCEPT_CONTACT_POINT_MODIFIED)
			{
				// This disables manifold tims when the closest point is welded
				hkPredGskAgent3::setGskFlagToTrue(agentData, hkpGskCache::GSK_FLAGS_DISABLE_CONTACT_TIMS);
			}

		}
		else if ( gskInput.m_shapeA->getType() == hkcdShapeType::TRIANGLE )		// code added to fix HVK-4509: Welding does not work with new collection collection agent
		{
			const hkpTriangleShape* triangleShape = static_cast<const hkpTriangleShape*>(gskInput.m_shapeA.val());
			if ( triangleShape->getWeldingType() != hkpWeldingUtility::WELDING_TYPE_NONE)
			{
				hkVector4 pointOnA; pointOnA.setAddMul( pointOnB, separatingNormal, separatingNormal.getW() );
				weldedNormal.setNeg<3>(weldedNormal);
				weldResult = (hkpConvexShape::WeldResult)gskInput.m_shapeA->weldContactPoint(	&gskCache.m_vertices[0], gskCache.m_dimA, pointOnB, 
																								&input.m_bodyA->getTransform(), gskInput.m_shapeB, &input.m_bodyB->getTransform(), weldedNormal );
				weldedNormal.setNeg<3>(weldedNormal);
				goto EXAMINE_WELD_RESULTS;
			}
		}

		// This returns 1 if the new point is in the manifold (it will be the first point)
		hkpGskManifoldPointExistsFlags flags = hkGskManifold_doesPointExistAndResort( gskManifold, gskCache );
		int closestPointInManifold = flags & HK_GSK_MANIFOLD_POINT_IN_MANIFOLD;
		explicitlyAllowNewPoint |= flags & HK_GSK_MANIFOLD_FEATURE_WITHIN_KEEP_DISTANCE_REMOVED;


		// Get a pointer to the current output pointer
		hkpProcessCdPoint* resultPointArray = output.getEnd();

		//
		// if there are other points in the manifold, get all those points first
		//
		if ( int(gskManifold.m_numContactPoints) > closestPointInManifold)
		{
			hkpGskManifoldWork work;
			hkGskManifold_init( gskManifold, weldedNormal, *input.m_bodyA, *input.m_bodyB, input.m_input->getTolerance(), work );
			explicitlyAllowNewPoint |= hkGskManifold_verifyAndGetPoints( gskManifold, work, closestPointInManifold, output, input.m_contactMgr ); 
		}

		//
		//	Handle the first (== closest point) specially
		//
		//HK_INTERNAL_TIMER_SPLIT_LIST("convert1st");
		{
			hkpProcessCdPoint* ccp = output.reserveContactPoints(1);
			ccp->m_contact.setPosition(pointOnB);
			ccp->m_contact.setSeparatingNormal( weldedNormal );
	
			if ( closestPointInManifold )
			{
				//
				//	use existing contact point id, as this point was already in the manifold
				//
				ccp->m_contactPointId = gskManifold.m_contactPoints[0].m_id;							
				output.commitContactPoints(1);
			}
			else //if (!rejectAddedPointByWelding)
			{ 
				//
				// try to add the contact point to the manifold
				//
				const hkpCollisionQualityInfo& sq = *input.m_input->m_collisionQualityInfo;
				int dim = gskCache.m_dimA + gskCache.m_dimB;
				hkSimdReal createContactRangeMax; createContactRangeMax.load<1>( (dim==4) ? &sq.m_create4dContact : &sq.m_createContact );

				if ( separatingNormal.getW().isLess(createContactRangeMax) || (0 != explicitlyAllowNewPoint) )
				{
					//	add point to manifold, the new point will aways be point 0
					
					hkpGskManifoldUtilMgrHandling useDeprecatedWelding = input.m_input->m_enableDeprecatedWelding ? HK_GSK_MANIFOLD_NO_ID_FOR_POTENTIALS : HK_GSK_MANIFOLD_CREATE_ID_ALWAYS;
					
					hkpGskManifoldAddStatus addStatus = hkGskManifold_addPoint( gskManifold, *input.m_bodyA, *input.m_bodyB, *input.m_input, output, gskCache, ccp, resultPointArray, input.m_contactMgr, useDeprecatedWelding );

					// really added
					if ( addStatus == HK_GSK_MANIFOLD_POINT_ADDED )
					{
						// take new point and check whether the new point just is a potential point
						if ( ccp->m_contactPointId != HK_INVALID_CONTACT_POINT)
						{
							output.commitContactPoints(1);
							//	try to create a second contact point
							//if ( gskManifold.m_numContactPoints == 1 )
							//{
							//	hkGskAgentUtil_tryToAddSecondPoint( input, gskInput, gskCache, gskManifold, separatingNormal, pointOnB, output );
							//}
						}
						else
						{
							// If old style welding is enabled, add the point to a potential list
#if defined(HK_1N_MACHINE_SUPPORTS_WELDING)
							if ( output.m_potentialContacts && entry && agentData)
							{
								if ( input.m_contactMgr->reserveContactPoints(1) == HK_SUCCESS )
								{
									hkpProcessCollisionOutput::ContactRef& contactRef = *(output.m_potentialContacts->m_firstFreePotentialContact++);
									contactRef.m_contactPoint = ccp;
									contactRef.m_agentEntry = entry;
									contactRef.m_agentData   = agentData;
								}
								else
								{
									goto removeAndRejectNewPoint;
								}
							}
							else
#endif
							{
								ccp->m_contactPointId = input.m_contactMgr->addContactPoint( *input.m_bodyA, *input.m_bodyB, *input.m_input, output, &gskCache, ccp->m_contact );
								if ( ccp->m_contactPointId == HK_INVALID_CONTACT_POINT )
								{
#if defined(HK_1N_MACHINE_SUPPORTS_WELDING)
					removeAndRejectNewPoint:
#endif
									HK_ASSERT( 0xf043daed, gskManifold.m_contactPoints[0].m_id == HK_INVALID_CONTACT_POINT );
									hkGskManifold_removePoint( gskManifold, 0 );
									goto rejectNewPoint;
								}
								gskManifold.m_contactPoints[0].m_id = hkContactPointId(ccp->m_contactPointId);

							}
							output.commitContactPoints(1);
						}
					}
					else if (addStatus == HK_GSK_MANIFOLD_POINT_REJECTED )
					{
						// take first point
rejectNewPoint:
						ccp = resultPointArray; 
						output.abortContactPoints(1);
					}
					else if ( addStatus == HK_GSK_MANIFOLD_TWO_POINT2_REJECTED )
					{
							// remove last point in the output array
						output.commitContactPoints(-1);
						output.abortContactPoints(1);
						ccp = resultPointArray; 
					}
					else // replaced
					{
						// take new point
						ccp = resultPointArray + ( addStatus - HK_GSK_MANIFOLD_POINT_REPLACED0 );
						output.abortContactPoints(1);
					}
				}
			}
			// fixes HVK-2168: added '&& ccp < output.m_firstFreeContactPoint'
			// because we can only report a representativeContact if we have a contact
#if defined(HK_1N_MACHINE_SUPPORTS_WELDING)
			if ( output.m_potentialContacts && ccp < output.m_firstFreeContactPoint )
			{
				*output.m_potentialContacts->m_firstFreeRepresentativeContact = ccp;	
				output.m_potentialContacts->m_firstFreeRepresentativeContact++;
			}
#endif
		}
	}
}


//
// Not currently used
//

inline void hkGskAgentUtil_tryToAddSecondPoint( const hkpAgent3Input&input, hkpGsk::GetClosesetPointInput& gskInput, hkpGskCache& gskCache, hkpGskManifold& gskManifold, const hkVector4& separatingNormal, const hkVector4& pointOnB, hkpProcessCollisionOutput& output )
{
	// do a simple collision restitution of first body
	hkTransform aTb2;
	gskInput.m_aTb = &aTb2;
	{
		hkVector4 normalInA; normalInA._setRotatedInverseDir( input.m_bodyA->getTransform().getRotation(), separatingNormal);
		hkVector4 pointInA;  pointInA._setTransformedInversePos(input.m_bodyA->getTransform(), pointOnB );

		hkVector4 mcr; mcr.setSub( pointInA, input.m_bodyA->getMotionState()->getSweptTransform().m_centerOfMassLocal );
		hkVector4 arm; arm.setCross( mcr, normalInA );
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
		hkSimdReal armLen = arm.lengthSquared<3>();
		hkVector4 normArm; normArm.mul(armLen.sqrtInverse());
		arm.setSelect(armLen.greaterZero(), normArm, hkVector4::getConstant<HK_QUADREAL_0100>());
#else
		if ( arm.lengthSquared<3>().isGreaterZero() )
		{
			arm.normalize<3>();
		}
		else
		{
			arm = hkVector4::getConstant<HK_QUADREAL_0100>();
		}
#endif
		const hkSimdReal onePercent = hkSimdReal::fromFloat(hkReal(0.01f));

		hkQuaternion q; q.setAxisAngle( arm, onePercent );

		hkSimdReal d0; d0.setMax(separatingNormal.getW(), hkSimdReal_0);
		hkSimdReal r; r.load<1>(&input.m_bodyA->getMotionState()->m_objectRadius); r.mul(onePercent);
		hkVector4 t; t.setMul(normalInA, d0 + r);

		hkTransform n; n.set(q, t);
		aTb2.setMul( n, input.m_aTb );
	}

	hkVector4 separatingNormal2;
	hkVector4 pointOnB2;
	if( hkpGsk::getClosestPoint( gskInput, gskCache, separatingNormal2, pointOnB2 ) != HK_FAILURE )
	{
		hkpGskManifoldPointExistsFlags flags = hkGskManifold_doesPointExistAndResort( gskManifold, gskCache );
		int closestPointInManifold = flags & HK_GSK_MANIFOLD_POINT_IN_MANIFOLD;

		if ( !closestPointInManifold )
		{
			hkpProcessCdPoint* ccp = output.reserveContactPoints(1);
			ccp->m_contact.setPosition(pointOnB2);
			ccp->m_contact.setSeparatingNormal( separatingNormal );
			hkpProcessCdPoint* resultPointArray = output.getEnd() - gskManifold.m_numContactPoints;
			hkpGskManifoldAddStatus addStatus = hkGskManifold_addPoint( gskManifold, *input.m_bodyA, *input.m_bodyB, *input.m_input, output, gskCache, ccp, resultPointArray, input.m_contactMgr, HK_GSK_MANIFOLD_CREATE_ID_ALWAYS );
			if ( addStatus == HK_GSK_MANIFOLD_POINT_ADDED )
			{
				output.commitContactPoints(1);
			}
			else
			{
				output.abortContactPoints(1);
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
