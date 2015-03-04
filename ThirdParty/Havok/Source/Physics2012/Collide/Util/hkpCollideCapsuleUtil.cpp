/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Geometry/Internal/Algorithms/ClosestPoint/hkcdClosestPointCapsuleCapsule.h>

#include <Physics2012/Collide/Util/hkpCollideCapsuleUtil.h>



#if !defined(HK_PLATFORM_SPU)
hkResult hkCollideCapsuleUtilClostestPointCapsVsCaps( const  hkVector4* capsA, const hkReal& capsARadius, const hkVector4* capsB, const hkReal& capsBRadius,	hkReal collisionTolerance, hkContactPoint& cpoint )
{
	hkVector4 cpPos = cpoint.getPosition();
	hkVector4 cpNormal = cpoint.getSeparatingNormal();
	hkResult result = hkcdClosestPointCapsuleCapsule( capsA[0], capsA[1], hkSimdReal::fromFloat(capsARadius), capsB[0], capsB[1], hkSimdReal::fromFloat(capsBRadius), hkSimdReal::fromFloat(collisionTolerance), &cpPos, &cpNormal );
	cpoint.setPosition(cpPos);
	cpoint.setSeparatingNormal(cpNormal);
	return result;
}
#endif


static HK_FORCE_INLINE void HK_CALL hkCollideCapsuleUtil_calcTrianglePlaneDirections( const hkVector4* tri, const hkpCollideTriangleUtil::PointTriangleDistanceCache& cache, hkTransform& planeEquationsOut, hkVector4& normalOut )
{
	hkVector4 E0; E0.setSub(tri[2], tri[1]);      
	hkVector4 E1; E1.setSub(tri[0], tri[2]);
	hkVector4 E2; E2.setSub(tri[1], tri[0]);

	hkVector4 invE; invE.load<4,HK_IO_NATIVE_ALIGNED>(&cache.m_invEdgeLen[0]); 
	hkSimdReal cacheNLen; cacheNLen.load<1,HK_IO_NATIVE_ALIGNED>(&cache.m_normalLen);

	hkVector4 triNormal; triNormal.setCross( E0, E1 );
	triNormal.mul( invE.getComponent<3>() );
	normalOut = triNormal;

	E0.mul( invE.getComponent<0>() );
	E1.mul( invE.getComponent<1>() );
	E2.mul( invE.getComponent<2>() );

	hkVector4& m0 = planeEquationsOut.getColumn(0);
	hkVector4& m1 = planeEquationsOut.getColumn(1);
	hkVector4& m2 = planeEquationsOut.getColumn(2);
	hkVector4& m3 = planeEquationsOut.getColumn(3);

	m0.setCross( E0, triNormal );
	m1.setCross( E1, triNormal );
	m2.setCross( E2, triNormal );
	m3 = triNormal;

	HK_TRANSPOSE4(m0,m1,m2,m3);

	m3.setZero();
	m3.setComponent<0>( -invE.getComponent<0>() * cacheNLen );
}


// note the searchManifold parameter had to be made int due to an Internal Compiler Error in gcc 2.95.3 when using hkBool
void hkCollideCapsuleUtilCapsVsTri( const  hkVector4* capsAPoints, hkReal capsARadius, const hkVector4* triVertices, hkReal triBRadius,
								   const hkpCollideTriangleUtil::PointTriangleDistanceCache& cache, hkReal collisionTolerance, int searchManifold, hkContactPoint* points, hkpFeatureOutput* featuresOut)
{
	//
	// if allocated, featuresOut will be used for welding the closest point normals 
	//
	if( featuresOut )
	{
		for( int k = 0 ; k < 3 ; k ++ )
		{
			featuresOut[k].numFeatures = 0;
		}
	}	


	//
	//	Project the start and endpoint into the triangle space
	//
	hkTransform planeDirections;
	hkVector4 normal;
	hkCollideCapsuleUtil_calcTrianglePlaneDirections( triVertices, cache, planeDirections, normal );

	//
	// calc distances to triangle planes
	//
	hkVector4 distances[2];
	{
		distances[0].setSub( capsAPoints[0], triVertices[0] );
		distances[1].setSub( capsAPoints[1], triVertices[0] );
		hkVector4Util::mul4xyz1Points( planeDirections, distances, 2, distances );
	}


	const hkReal radiusSum = capsARadius + triBRadius;
	const hkSimdReal radiusSumSr = hkSimdReal::fromFloat(radiusSum);
	const hkReal checkRadius = radiusSum + 	collisionTolerance;

	//
	//	Set a bit if we are outside (positive distance)
	//


	//
	//	Calculate the contact points
	//
	{
		const hkSimdReal d = hkSimdReal::fromFloat(HK_REAL_HIGH) - (radiusSumSr+radiusSumSr);
		points[0].setDistanceSimdReal( d );
		points[1].setDistanceSimdReal( d );
		points[2].setDistanceSimdReal( d );
	}

	//
	//	Try to do early outs
	//
	{
		
		hkVector4 checkRadius4; checkRadius4.setAll( checkRadius );
		hkVector4Comparison bothPos; bothPos.setAnd( checkRadius4.less( distances[0] ), checkRadius4.less( distances[1] ) );
		checkRadius4.setNeg<4>(checkRadius4);
		hkVector4Comparison bothNeg; bothNeg.setAnd( distances[0].less( checkRadius4 ), distances[1].less( checkRadius4 ) );
		if ( bothPos.anyIsSet() | bothNeg.anyIsSet<hkVector4ComparisonMask::MASK_W>() )
		{
			return;
		}
	}


	//
	//	Now set the bits if we are inside
	//
	hkPadSpu<int> mask[2];
	mask[0] = distances[0].lessZero().getMask();
	mask[1] = distances[1].lessZero().getMask();

	//
	// Check for complete penetration
	//
	{
		if ( (mask[0] ^ mask[1]) & hkVector4ComparisonMask::MASK_W )
		{
			//
			//	Points on different side of the triangle, check for complete penetration
			//
			hkSimdReal f = distances[0].getW() / ( distances[0].getW() - distances[1].getW() );
			hkVector4 collidingDist; collidingDist.setInterpolate( distances[0], distances[1], f );

			if ( collidingDist.allLessZero<3>() )
			{
				//
				//	Search the smallest penetration, start with the two endpoints
				//
				hkContactPoint* point = &points[1];
				point->setDistanceSimdReal(hkSimdReal_MinusMax);
				for (int i = 0; i < 2; i++ )
				{
					if ( (mask[i] & hkVector4ComparisonMask::MASK_XYZ) == hkVector4ComparisonMask::MASK_XYZ )
					{
						const hkSimdReal Di = distances[i].getW();
						hkSimdReal absDi; absDi.setAbs(Di);
						const hkSimdReal dist = -absDi - radiusSumSr;
						if ( dist > point->getDistanceSimdReal() )
						{
							if ( mask[i] & hkVector4ComparisonMask::MASK_W )
							{
								point->setSeparatingNormal(normal);
								hkVector4 cpPos; cpPos.setAddMul( capsAPoints[i], normal, hkSimdReal::fromFloat(triBRadius) - Di );
								point->setPosition(cpPos);
							}
							else
							{
								hkVector4 cpN; cpN.setNeg<4>( normal );
								hkVector4 cpPos; cpPos.setAddMul( capsAPoints[i], normal, -hkSimdReal::fromFloat(triBRadius) - Di );
								point->setSeparatingNormal(cpN);
								point->setPosition(cpPos);
							}
							point->setDistanceSimdReal( dist );
						}
					}
				}

				hkVector4 dir; dir.setSub( distances[1], distances[0] );
				//
				//	Check the three edges
				//
				for (int e = 0; e < 3; e++)
				{
					hkSimdReal dirW = dir.getW();
					hkSimdReal dirE = dir.getComponent(e);
					hkSimdReal lenSquared = dirW * dirW + dirE * dirE;
					hkSimdReal invLenSquared; invLenSquared.setReciprocal(lenSquared);

					hkSimdReal dist    = dirE * distances[0].getW() - dirW * distances[0].getComponent(e);
					hkSimdReal distSquared = dist*dist * invLenSquared;
					hkSimdReal contactDist = point->getDistanceSimdReal() + radiusSumSr;

					if ( distSquared < contactDist * contactDist )
					{
						hkSimdReal project = dirE * distances[0].getComponent(e) + dirW * distances[0].getW();
						project.mul(-invLenSquared);
						if ( project > hkSimdReal::fromFloat(-.0001f) && project < hkSimdReal::fromFloat(1.0001f) )
						{
							hkVector4 triDir; planeDirections.getRotation().getRow( e, triDir );
							const hkVector4Comparison dirWlt0 = dirW.lessZero();
							{
								dirW.setFlipSign(dirW,dirWlt0);
								dirE.setFlipSign(dirE,dirWlt0);
							}
							triDir.mul( dirW );
							hkVector4 cpN; cpN.setAddMul( triDir, normal, -dirE );

							// this normalize is safe:
							//   - normal.dot3( triDir ) = 0
							//   - normal.getLength() > 0
							//   - triDir.getLength() == 1
							//   -> m_contact.normal().lengthsqrd = normal.lengthsqrd + triDir.lengthSqrd
							cpN.normalize<3>();
							point->setSeparatingNormal(cpN);

							hkVector4 pos; pos.setInterpolate( capsAPoints[0], capsAPoints[1], project );
							hkSimdReal d = -distSquared.sqrt();
							hkVector4 cpPos; cpPos.setAddMul( pos, point->getNormal(), hkSimdReal::fromFloat(triBRadius) - d);
							point->setPosition(cpPos);

							point->setDistanceSimdReal(  d - radiusSumSr );
						}
					}
				}
				//
				//	break loop
				//

				for(int  i = 0 ; i < 3 ; i ++)
				{
					if( featuresOut )
					{
						featuresOut[i].numFeatures = 3;							
						featuresOut[i].featureIds[0] = 0;
						featuresOut[i].featureIds[1] = 1;
						featuresOut[i].featureIds[2] = 2;
					}
				}			

				return;
			}
		}
	}

	//
	//	Check if the capsuleTriangle check can be reduced to a capsuleCapsule check
	//
	int i = 0;
	hkVector4 diffcapsAPoints; diffcapsAPoints.setSub( capsAPoints[1], capsAPoints[0] );
	hkContactPoint* point = points;
	bool checkSingle = false;

	if ( (hkVector4ComparisonMask::MASK_XYZ & (mask[0] | mask[1])) != hkVector4ComparisonMask::MASK_XYZ )
	{
		if ( searchManifold )
		{
			int orMask = mask[0] | mask[1];
			hkSimdReal d = hkSimdReal::fromFloat(collisionTolerance);
			points[0].setDistanceSimdReal( d );
			points[1].setDistanceSimdReal( d );
			points[2].setDistanceSimdReal( d );
			hkContactPoint prevCp; prevCp.setPositionNormalAndDistance(hkVector4::getZero(), hkVector4::getZero(),hkSimdReal_Max);
			hkVector4 pB[2];
			int maskJ = hkVector4ComparisonMask::MASK_X;
			for ( int j = 0; j < 3; maskJ = ( hkVector4ComparisonMask::MASK_Y > hkVector4ComparisonMask::MASK_X) ? maskJ<<1: maskJ>>1, j++)
			{
				if ( (maskJ & orMask) )
				{
					continue;
				}

				int p0 = hkpCollideTriangleUtil::getNextModulo3( j );
				int p1 = hkpCollideTriangleUtil::getPrevModulo3( j );

				pB[0] = triVertices[ p0 ];
				pB[1] = triVertices[ p1 ];

				int whichCapsCapsPoints = hkLineSegmentUtil::capsuleCapsuleManifold(capsAPoints, capsARadius, pB, triBRadius, points);
				
				if( featuresOut )
				{
					for(int k = 0 ; k < 3 ; k ++)
					{
						featuresOut[k].featureIds[0] = (hkpVertexId)p0;

						if( whichCapsCapsPoints & ( hkLineSegmentUtil::CLSLS_POINTB_START | hkLineSegmentUtil::CLSLS_POINTB_END ) )
						{
							featuresOut[k].numFeatures = 1;
						}
						else
						{
							featuresOut[k].numFeatures = 2;
							featuresOut[k].featureIds[1] = (hkpVertexId)p1;
						}
					}					
				}
				
				// Store/restore the separating normal of the closest point. (HVK-2735)
				if (points[0].getDistanceSimdReal() > prevCp.getDistanceSimdReal())
				{
					
					points[0] = prevCp;
				}
				else
				{
					
					prevCp = points[0];
				}
			}
			return;
		}
		else
		{
			checkSingle = true;
			goto checkSingleSecondPoint;
		}
	}

	//
	//	Calculate the end point info
	//
	for ( ; i < 2; point++,i++)
	{
		// check if completely inside
		if ( (mask[i]&hkVector4ComparisonMask::MASK_XYZ) == hkVector4ComparisonMask::MASK_XYZ )
		{
			//
			//	Completely inside, simply create contact point
			//
			hkVector4 cpPos; 
			if ( mask[i] & hkVector4ComparisonMask::MASK_W )
			{
				cpPos.setAddMul( capsAPoints[i], normal, -hkSimdReal::fromFloat(triBRadius) - distances[i].getW());
				hkVector4 cpN; cpN.setNeg<4>( normal );
				point->setSeparatingNormal(cpN, -distances[i].getW() - radiusSumSr );
			}
			else
			{
				cpPos.setAddMul( capsAPoints[i], normal, hkSimdReal::fromFloat(triBRadius) - distances[i].getW());
				point->setSeparatingNormal(normal, distances[i].getW() - radiusSumSr );
			}
			point->setPosition(cpPos);

			if( featuresOut )
			{
				featuresOut[i].numFeatures = 3;							
				featuresOut[i].featureIds[0] = 0;
				featuresOut[i].featureIds[1] = 1;
				featuresOut[i].featureIds[2] = 2;
			}
		}
		else
		{
			//
			//	Outside, now check each side
			//
checkSingleSecondPoint:
			int m = hkVector4ComparisonMask::MASK_X;
			for (int e = 0; e < 3; m = ( hkVector4ComparisonMask::MASK_Y > hkVector4ComparisonMask::MASK_X ) ? (m<<1) : (m>>1), e++ )
			{
				if (!checkSingle)
				{
					if ( (mask[i] & m) )
					{
						//	Do not check an edge if this point is inside that edge
						continue;
					}
				}
				else
				{
					if ( (mask[0] & mask[1] & m) )
					{
						//	Do not check an edge if we are completely inside that edge
						continue;
					}
				}

				int n = hkpCollideTriangleUtil::getNextModulo3( e );
				int nn = hkpCollideTriangleUtil::getPrevModulo3( e );
			
				const hkVector4& P = triVertices[n];
				hkVector4 edge; edge.setSub( triVertices[ nn ] , P );
				hkLineSegmentUtil::ClosestLineSegLineSegResult result;

				int whichPoints = hkLineSegmentUtil::closestLineSegLineSeg( capsAPoints[0], diffcapsAPoints, P, edge, result );

				if( featuresOut )
				{
					featuresOut[i].featureIds[0] = (hkpVertexId)n;
					
					if( whichPoints & ( hkLineSegmentUtil::CLSLS_POINTB_START | hkLineSegmentUtil::CLSLS_POINTB_END ) )
					{
						featuresOut[i].numFeatures = 1;
					}
					else
					{
						featuresOut[i].numFeatures = 2;
						featuresOut[i].featureIds[1] = (hkpVertexId)nn;
					}
				}				

				const hkSimdReal refDist = point->getDistanceSimdReal() + radiusSumSr;
				if ( hkSimdReal::fromFloat(result.m_distanceSquared) <  refDist * refDist )
				{
					//
					//	Check for other point
					//
					if ( whichPoints & (1<<i) )
					{
						if (checkSingle)
						{
							i = 1;
						}
						else
						{
							continue;
						}
					}

					//
					//	Check whether we we collide with the inner part of our capsule
					//
					hkVector4 normalE;

					if ( whichPoints == 0 )
					{
						//
						//	Calculate normal
						//
						normalE.setCross( diffcapsAPoints, edge );
					}
					else
					{
						const hkSimdReal epsSqrd = hkSimdReal_EpsSqrd;
						if ( hkSimdReal::fromFloat(result.m_distanceSquared) <= epsSqrd)
						{
							planeDirections.getRotation().getRow( e, result.m_closestAminusClosestB );
							normalE.setCross( diffcapsAPoints, edge );
							if (normalE.lengthSquared<3>() <= epsSqrd )
							{
								normalE = result.m_closestAminusClosestB;
							}
						}
						else
						{
							//
							//	Now we collide with our endpoint
							//
							normalE = result.m_closestAminusClosestB;
						}
					}
					normalE.normalize<3>();

					hkSimdReal dist = normalE.dot<3>(result.m_closestAminusClosestB);
					normalE.setFlipSign(normalE, dist);
					dist.setFlipSign(dist, dist);

					hkVector4 cpPos; cpPos.setAddMul( result.m_closestPointA, normalE, hkSimdReal::fromFloat(triBRadius) - dist);
					point->setPosition(cpPos);
					point->setSeparatingNormal( normalE, dist - radiusSumSr );
					continue;
				}
			}
		} // for (i = 0..1 )
		if ( checkSingle )
		{
			return;
		}
	}
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
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
