/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKP_BV_TREE_AGENT_INL
#define HKP_BV_TREE_AGENT_INL

//
//	This inl file is implementing local function for the hkpBvTreeAgent
//

#ifdef HK_DEBUG
//#	define HK_BV_TREE_DISPLAY_AABB
#endif

#if defined(HK_BV_TREE_DISPLAY_AABB)
#	include <Common/Visualize/hkDebugDisplay.h>
#endif

hkResult hkpBvTreeAgent::calcAabbAndQueryTree( const hkpCdBody& bodyA,	const hkpCdBody& bodyB, const hkTransform& bTa,
											 const hkVector4& linearTimInfo, const hkpProcessCollisionInput& input,
											 hkAabb* cachedAabb, hkArray<hkpShapeKey>& hitListOut )
{
	//
	// Calc the AABB extruded by the relative movement
	//
	hkAabb aabb;
#ifdef HK_BV_TREE_DISPLAY_AABB
	hkAabb baseAabb;
#endif
	{
		//added an early out so if the AABB is the same, don't query the MOPP and don't sort, nor call the dispatch/agent
		{
			const hkMotionState* msA = bodyA.getMotionState();
			const hkMotionState* msB = bodyB.getMotionState();

				// if using continuous physics, expand the AABB backwards
				// rotate tim into the bvTree space
			hkVector4 timInfo;	timInfo._setRotatedInverseDir( bodyB.getTransform().getRotation(), linearTimInfo );
			hkSimdReal inputTol; inputTol.load<1>(&(input.m_tolerance.ref())); inputTol.mul(hkSimdReal_Half);

			hkVector4 aabbExtents; 
			if ( input.m_collisionQualityInfo->m_useContinuousPhysics.val() )
			{
				hkSimdReal objRadiusA; objRadiusA.load<1>(&msA->m_objectRadius);

					// object A rotates within object B with the diff of both angular velocities
				hkSimdReal secOrdErrA; secOrdErrA.setMul(objRadiusA, msA->m_deltaAngle.getW() + msB->m_deltaAngle.getW());

					// The angular velocity gets correctly calculated into the trajectory of object A
					// we only need to calculate the maximum error. So we use the square of the error B
				hkSimdReal secOrdErrB; secOrdErrB.load<1>(&msB->m_objectRadius); secOrdErrB.mul(msB->m_deltaAngle.getW() * msB->m_deltaAngle.getW());

				hkSimdReal checkEpsilon; checkEpsilon.setAdd(secOrdErrA + secOrdErrB, inputTol);
				bodyA.getShape()->getAabb( bTa, checkEpsilon.getReal(), aabb );


				// restrict the size of the AABB to the worst case radius size
				hkVector4 massCenterAinB;
				{
					hkVector4 radius4; radius4.setAll( objRadiusA + inputTol + secOrdErrB ); radius4.zeroComponent<3>();
					massCenterAinB._setTransformedInversePos(bodyB.getTransform(), msA->getSweptTransform().m_centerOfMass1 );
					hkVector4 maxR; maxR.setAdd( massCenterAinB, radius4 );
					hkVector4 minR; minR.setSub( massCenterAinB, radius4 );
					aabb.m_min.setMax( aabb.m_min, minR );
					aabb.m_max.setMin( aabb.m_max, maxR );
				}

				// export the size of the base AABB
				aabbExtents.setSub( aabb.m_max, aabb.m_min );

				// expand the AABB backwards
				{
					// correct the timInfo if we have a rotating tree
					if (msB->m_deltaAngle.getComponent<3>().isGreaterZero())
					{
						hkVector4 relPos; relPos.setSub( massCenterAinB, msB->getSweptTransform().m_centerOfMassLocal );
						hkVector4 offsetOut; offsetOut.setCross( relPos, msB->m_deltaAngle );
						const hkSimdReal f = hkSimdReal::fromFloat(input.m_stepInfo.m_deltaTime) * msB->getSweptTransform().getInvDeltaTimeSr();
						timInfo.addMul( f, offsetOut );
					}

					hkVector4 zero;		zero.setZero();
					hkVector4 minPath; 	minPath.setMin( zero, timInfo );
					hkVector4 maxPath;	maxPath.setMax( zero, timInfo );

#ifdef HK_BV_TREE_DISPLAY_AABB
					baseAabb = aabb;
					//baseAabb.m_min.add( timInfo );
					//baseAabb.m_max.add( timInfo );
#endif
					
					aabb.m_min.add( minPath );
					aabb.m_max.add( maxPath );
				}
			}
			else
			{
				bodyA.getShape()->getAabb( bTa, inputTol.getReal(), aabb );
				aabbExtents.setSub( aabb.m_max, aabb.m_min );
#ifdef HK_BV_TREE_DISPLAY_AABB
				baseAabb = aabb;
#endif
			}

			//
			//	Try to do some AABB caching to reduce the number of calls to the bounding volume structure
			//
			if (cachedAabb)
			{
				if ( cachedAabb->contains( aabb ))
				{
					return HK_FAILURE;
				}

				hkVector4 zero; zero.setZero();
				hkVector4 minPath;minPath.setMin( zero, timInfo );
				hkVector4 maxPath;maxPath.setMax( zero, timInfo );


				// expand AABB so we have a higher chance of a hit next frame
				// we expand it by half of our tolerance
				hkVector4 expand4; expand4.setAll( inputTol ); expand4.zeroComponent<3>();
				aabb.m_min.sub( expand4 );
				aabb.m_max.add( expand4 );

				// expand along our path linearly at least 2 frames ahead
				// but a maximum of 40% of the original AABB
				const hkSimdReal maxExpand = hkSimdReal::fromFloat(0.4f);
				const hkSimdReal framesLookAhead = hkSimdReal::fromFloat(-2.0f);

				hkVector4 minExtentPath; minExtentPath.setMul( framesLookAhead, maxPath );
				hkVector4 maxExtentPath; maxExtentPath.setMul( framesLookAhead, minPath );

				hkVector4 maxExpand4; maxExpand4.setMul( maxExpand, aabbExtents );
				maxExtentPath.setMin( maxExtentPath, maxExpand4 );
				hkVector4 minExpand4; minExpand4.setNeg<4>(maxExpand4);
				minExtentPath.setMax( minExtentPath, minExpand4 );

				aabb.m_min.add( minExtentPath );
				aabb.m_max.add( maxExtentPath );
				*cachedAabb = aabb;
			}
		}
	}

	//
	// display the AABB and the cached AABB
	//
#ifdef HK_BV_TREE_DISPLAY_AABB
	{
		hkAabb* bb = &baseAabb; 
		hkColor::Argb color = hkColor::YELLOW;
		for ( int a = 0; a < 2; a ++)
		{
			for ( int x = 0; x < 2; x ++ )
			{	for ( int y = 0; y < 2; y ++ )
				{	for ( int z = 0; z < 2; z ++ )
					{
						hkVector4 a; a.set( (&bb->m_min)[x](0), (&bb->m_min)[y](1), (&bb->m_min)[z](2) );
						a.setTransformedPos( bodyB.getTransform(), a );
						hkVector4 b;

						b.set( (&bb->m_min)[1-x](0), (&bb->m_min)[y](1), (&bb->m_min)[z](2) );
						b.setTransformedPos( bodyB.getTransform(), b );
						HK_DISPLAY_LINE( a, b, color );
						b.set( (&bb->m_min)[x](0), (&bb->m_min)[1-y](1), (&bb->m_min)[z](2) );
						b.setTransformedPos( bodyB.getTransform(), b );
						HK_DISPLAY_LINE( a, b, color );
						b.set( (&bb->m_min)[x](0), (&bb->m_min)[y](1), (&bb->m_min)[1-z](2) );
						b.setTransformedPos( bodyB.getTransform(), b );
						HK_DISPLAY_LINE( a, b, color );
			}	}	}
			color = hkColor::BLUE;
			bb = cachedAabb;
			if (!bb) 
			{
				break;
			}
		}
	}
#endif
	//
	// query the BvTreeShape
	//
	const hkpBvTreeShape* bvB = static_cast<const hkpBvTreeShape*>( bodyB.getShape() );

	bvB->queryAabb( aabb, hitListOut );
	return HK_SUCCESS;
}

#endif // HKP_BV_TREE_AGENT_INL

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
