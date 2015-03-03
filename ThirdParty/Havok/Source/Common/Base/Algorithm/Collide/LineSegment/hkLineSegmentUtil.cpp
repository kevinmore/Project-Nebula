/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Algorithm/Collide/LineSegment/hkLineSegmentUtil.h>
#include <Common/Base/Types/Physics/ContactPoint/hkContactPoint.h>

int HK_CALL hkLineSegmentUtil::closestLineSegLineSeg( hkVector4Parameter A, hkVector4Parameter dA, hkVector4Parameter B, hkVector4Parameter dB, ClosestLineSegLineSegResult& result )
{
	hkVector4 d12; d12.setSub(B,A);

	const hkSimdReal R = dA.dot<3>(dB);
	const hkSimdReal S1 = dA.dot<3>(d12);
	const hkSimdReal S2 = dB.dot<3>(d12);

	const hkSimdReal D1 = dA.lengthSquared<3>();
	const hkSimdReal D2 = dB.lengthSquared<3>();

	hkSimdReal t,u;
	int returnValue;

	// Step 1, compute D1, D2, R and the denominator.
	// The cases (a), (b) and (c) are covered in Steps 2, 3 and 4 as
	// checks for division by zero.

	hkSimdReal denom; denom.setAbs(D1*D2-R*R);	// we use fabs here to avoid negative numbers

		// Step 2
	{
 		t=(S1*D2-S2*R);

			// If t not in range, modify
		if(t * denom >= denom * denom)
		{
			t=hkSimdReal_1;
			returnValue = CLSLS_POINTA_START;
		}
		else if(t.isLessEqualZero())
		{
			t.setZero();
			returnValue = CLSLS_POINTA_END;
		}
		else
		{
			const hkSimdReal eps = (D1*D2 + R*R) * (hkSimdReal_Eps * hkSimdReal_8);
			if ( denom > eps )
			{
				t.div(denom);
				returnValue = 0;
			}
			else
			{
				t = hkSimdReal_1;
				returnValue = CLSLS_POINTA_START;
			}
		}
	}

	// Step 3
	{
		u=(t*R-S2);

			// If u not in range, modify
		if(u>=D2)
		{
			u=hkSimdReal_1;
			returnValue = CLSLS_POINTB_START;
		}
		else if(u.isLessEqualZero())
		{
			u.setZero();
			returnValue = CLSLS_POINTB_END;
		}
		else
		{
			u.div(D2);
			goto end;
		}
	}

		// Step 4
	{
		t=(u*R+S1);
		// If t not in range, modify
		if(t.isLessEqualZero())
		{
			t.setZero();
			returnValue |=  CLSLS_POINTA_END;
		}
		else if(t>=D1)	
		{
			t = hkSimdReal_1;
			returnValue |= CLSLS_POINTA_START;
		}
		else
		{
			t.div(D1);
		}
	}
end:
	
	result.m_t = t;
	result.m_u = u;
	hkVector4 closestPointA; closestPointA.setAddMul( A, dA, t );
	hkVector4 closestPointB; closestPointB.setAddMul( B, dB, u );

	hkVector4 AminusB; AminusB.setSub( closestPointA, closestPointB );
	result.m_closestAminusClosestB = AminusB;
	result.m_closestPointA = closestPointA;
	const hkSimdReal lenSqrd = AminusB.lengthSquared<3>();
	lenSqrd.store<1>( &(hkReal&)result.m_distanceSquared );
	return returnValue;
}

int hkLineSegmentUtil::capsuleCapsuleManifold( const  hkVector4* capsAPointsIn, hkReal capsARadius, const hkVector4* capsBPoints, hkReal capsBRadius,	hkContactPoint* points )
{
	hkVector4 dA; dA.setSub( capsAPointsIn[1], capsAPointsIn[0] );
	hkVector4 dB; dB.setSub( capsBPoints[1], capsBPoints[0] );

	hkSimdReal R = dA.dot<3>(dB);
	hkVector4 capsAPoints[2];
	if ( R.isLessZero() )
	{
		R = -R;
		dA.setNeg<4>( dA );
		capsAPoints[0] = capsAPointsIn[1];
		capsAPoints[1] = capsAPointsIn[0];
	}
	else
	{
		capsAPoints[0] = capsAPointsIn[0];
		capsAPoints[1] = capsAPointsIn[1];
	}

	const hkVector4& A = capsAPoints[0];
	const hkVector4& B = capsBPoints[0];

	//
	//	the next lines are a precise copy of hkLineSegmentUtil::closestLineSegLineSeg
	//
	hkSimdReal denom;
	hkSimdReal D1;
	hkSimdReal D2;
	hkSimdReal t;
	hkSimdReal u;
	hkSimdReal S1;
	hkSimdReal S2;

	int returnValue;
	{
		hkVector4 d12; d12.setSub(B,A);

		S1 = dA.dot<3>(d12);
		S2 = dB.dot<3>(d12);

		D1 = dA.lengthSquared<3>();
		D2 = dB.lengthSquared<3>();


		// Step 1, compute D1, D2, R and the denominator.
		// The cases (a), (b) and (c) are covered in Steps 2, 3 and 4 as
		// checks for division by zero.

		hkSimdReal d1d2 = D1*D2;
		denom.setAbs( d1d2-R*R );
		d1d2.setAbs(d1d2);
		hkSimdReal eps = (d1d2 + R*R) * (hkSimdReal_Eps * hkSimdReal_8);

		// Step 2
		{
			t=(S1*D2-S2*R);

			// If t not in range, modify
			if(t * denom >= denom * denom)
			{
				t=hkSimdReal_1;
				returnValue = hkLineSegmentUtil::CLSLS_POINTA_START;
			}
			else if(t.isLessEqualZero())
			{
				t.setZero();
				returnValue = hkLineSegmentUtil::CLSLS_POINTA_END;
			}
			else
			{
				if ( denom > eps )
				{
					t = t/denom;
					returnValue = 0;
				}
				else
				{
					t=hkSimdReal_1;
					returnValue = hkLineSegmentUtil::CLSLS_POINTA_START;
				}
			}
		}

		// Step 3
		{
			u=(t*R-S2);

			// If u not in range, modify
			if(u>=D2)
			{
				u=hkSimdReal_1;
				returnValue = hkLineSegmentUtil::CLSLS_POINTB_START;
			}
			else if(u.isLessEqualZero())
			{
				u.setZero();
				returnValue = hkLineSegmentUtil::CLSLS_POINTB_END;
			}
			else
			{
				u.div(D2);
				goto end;
			}
		}

		// Step 4
		{
			t=(u*R+S1);
			// If t not in range, modify
			if(t.isLessEqualZero())
			{
				t.setZero();
				returnValue |=  hkLineSegmentUtil::CLSLS_POINTA_END;
			}
			else if(t>=D1)	
			{
				t=hkSimdReal_1;
				returnValue |= hkLineSegmentUtil::CLSLS_POINTA_START;
			}
			else
			{
				t.div(D1);
			}
		}
end:;
	}

	//
	//	Copy of "hkLineSegmentUtil::closestLineSegLineSeg" end
	//

	//
	// calculate the closest point
	//
	{ 
		hkVector4 a; a.setAddMul( A, dA, t );
		hkVector4 b; b.setAddMul( B, dB, u );
		hkVector4 aMinusB; aMinusB.setSub( a,b );

		hkContactPoint& cpoint = points[0];

		const hkReal radiusSum = capsARadius + capsBRadius;
		hkSimdReal refDist; refDist.setMax( hkSimdReal_Eps, cpoint.getDistanceSimdReal() );

		hkSimdReal dist = aMinusB.length<3>();
		const hkSimdReal contactDist = dist - hkSimdReal::fromFloat(radiusSum);
		cpoint.setSeparatingNormal(aMinusB);

		if ( contactDist < refDist )
		{
			hkVector4 sepNormal;
			if ( dist.isLess(hkSimdReal_Eps) )
			{
				dist.setZero();
				hkVector4Util::calculatePerpendicularVector( dA, sepNormal );
				// Note: if the following line throws an assertion it probably means that one of the 
				// capsules is degenerate (both vertices are the same).
				sepNormal.normalize<3>();
			}
			else
			{
				hkSimdReal lengthInv; lengthInv.setReciprocal(dist);
				sepNormal = cpoint.getSeparatingNormal();
				sepNormal.mul( lengthInv );
			}
			cpoint.setSeparatingNormal(sepNormal);

			hkVector4 cpPos; cpPos.setAddMul( a, cpoint.getNormal(), hkSimdReal::fromFloat(capsBRadius) - dist );
			cpoint.setPosition(cpPos);
			cpoint.setDistanceSimdReal( contactDist );				

		}
		else
		{
			//
			//	No need to check for more contact points, as the closest is already too far away
			//  just calculate the separating plane
			//

			hkSimdReal lengthInv; lengthInv.setReciprocal(dist);
			hkVector4 sepNormal = cpoint.getSeparatingNormal();
			sepNormal.mul( lengthInv );
			cpoint.setSeparatingNormal(sepNormal, contactDist);
			return returnValue;
		}
	}

	//
	//	Check if the to lines are parallel and if not return
	//
	{
		// denom / D1*D2 < sin(angle)
		if ( denom > hkSimdReal_Inv5 * D1 * D2 )
		{			
			return returnValue;
		}
	}

	hkSimdReal invD12; invD12.setReciprocal(D1 * D2);
	const hkSimdReal invD1 = D2 * invD12;
	const hkSimdReal invD2 = D1 * invD12;

	//
	//	Now try to extend line in direction of point 0
	//  only if non of the endpoints is used
	//
	{
		hkContactPoint* point = points+1;
		hkVector4 closestA = capsAPoints[0];
		hkVector4 closestB = capsBPoints[0];
		hkSimdReal pAonB = -S2;
		hkSimdReal pBonA = S1;
		int i = 0;
		int mask = hkLineSegmentUtil::CLSLS_POINTA_END | hkLineSegmentUtil::CLSLS_POINTB_END;
		while(1)
		{
			if ( (returnValue & mask) == 0 )
			{
				//
				//	Project each endPoint onto the other line
				//
				{
					if ( pAonB.isGreaterZero() )
					{
						if ( pAonB > D2 )
						{
							goto nextPoint;
						}

						if ( pBonA.isGreaterZero() )
						{
							if ( pBonA > D1 )
							{
								goto nextPoint;
							}

							// find the bigger distance from the endpoint
							// pA0onB / sqrt(D2) > pB0onA / sqrt(D1)    | ^2
							if ( pAonB * pAonB * invD2 > pBonA * pBonA * invD1 )
							{
								closestA.addMul( pBonA * invD1, dA );
							}
							else
							{
								closestB.addMul( pAonB * invD2, dB );
							}
						}
						else
						{
							closestB.addMul( pAonB * invD2, dB );
						}
					}
					else
					{
						if ( pBonA.isGreaterZero() )
						{
							if ( pBonA > D1 )
							{
								goto nextPoint;
							}
							closestA.addMul( pBonA * invD1, dA );
						}
					}
				}

				hkVector4 aMinusB; aMinusB.setSub( closestA, closestB );
				hkContactPoint& cpoint = *point;

				const hkSimdReal radiusSum = hkSimdReal::fromFloat(capsARadius + capsBRadius);
				const hkSimdReal refDist = radiusSum + cpoint.getDistanceSimdReal();

				const hkSimdReal distSquared = aMinusB.lengthSquared<3>();
				if ( distSquared < refDist * refDist )
				{

					const hkSimdReal dist = distSquared.sqrt();
					hkVector4 sepNormal = aMinusB;
					if ( distSquared  > hkSimdReal_Eps )
					{
						sepNormal.normalize<3>();
					}
					else
					{
						sepNormal = points->getSeparatingNormal();
					}
					cpoint.setSeparatingNormal(sepNormal);

					hkVector4 cpPos; cpPos.setAddMul( closestA, cpoint.getNormal(), hkSimdReal::fromFloat(capsBRadius) - dist );
					cpoint.setPosition(cpPos);
					cpoint.setDistanceSimdReal( dist - radiusSum );
				}
			}
nextPoint:
			if ( i == 1 )
			{
				break;
			}

			closestA = capsAPoints[1];
			closestB = capsBPoints[1];
			pAonB = D2 - pAonB - R;
			pBonA = D1 - pBonA - R;
			dA.setNeg<4>( dA );
			dB.setNeg<4>( dB );

			point++;
			i++;
			mask = hkLineSegmentUtil::CLSLS_POINTA_START | hkLineSegmentUtil::CLSLS_POINTB_START;
		}
	}
	return returnValue;
}



int HK_CALL hkLineSegmentUtil::closestPointLineSeg( hkVector4Parameter A, hkVector4Parameter B, hkVector4Parameter B2, ClosestPointLineSegResult& result )
{
	hkVector4 d12; d12.setSub( A, B );
	hkVector4 dB;  dB.setSub( B2, B );

	hkSimdReal S2 = dB.dot<3>(d12);
	hkSimdReal D2 = dB.dot<3>(dB);


		// If u not in range, modify
	if(S2.isLessEqualZero())
	{
		result.m_pointOnEdge = B;
		return CLSLS_POINTB_END;
	}
	else
	{
		if(S2>=D2)
		{
			result.m_pointOnEdge = B2;
			return CLSLS_POINTB_START;
		}
		else
		{
			S2.div(D2);
			result.m_pointOnEdge.setAddMul(B, dB, S2);
			return 0;
		}
	}
}

hkResult HK_CALL hkLineSegmentUtil::closestPointInfLineInfLine( const hkVector4& A, const hkVector4& dA, const hkVector4& B, const hkVector4& dB, ClosestPointInfLineInfLineResult& result )
{
	hkVector4 d12; d12.setSub(B,A);
	const hkSimdReal R=dA.dot<3>(dB);

	const hkSimdReal S1=dA.dot<3>(d12);
	const hkSimdReal S2=dB.dot<3>(d12);

	const hkSimdReal D1=dA.dot<3>(dA);
	const hkSimdReal D2=dB.dot<3>(dB);
	//HK_ASSERT2(0x13635242,  D1 != 0.0f, "Length of segment A is zero");
	//HK_ASSERT2(0x494dab88,  D2 != 0.0f, "Length of segment B is zero");


	// Step 1, compute D1, D2, R and the denominator.
	// The cases (a), (b) and (c) are covered in Steps 2, 3 and 4 as
	// checks for division by zero.

	const hkSimdReal denom = D1*D2-R*R;
	hkSimdReal absD1D2; absD1D2.setAbs(D1*D2);
	hkSimdReal eps = (absD1D2 + R*R) * (hkSimdReal_Eps * hkSimdReal_8);

	hkSimdReal t, u;
	hkResult returnVal;
	hkSimdReal absDenom; absDenom.setAbs(denom);
	if( absDenom > eps )
	{
		t=(S1*D2-S2*R)/denom;
		returnVal = HK_SUCCESS;
	}
	else
	{
		t.setZero();
		returnVal = HK_FAILURE;
	}

	// Step 2
	u=(t*R-S2)/D2;

	hkVector4 a; a.setAddMul(A, dA, t );
	hkVector4 b; b.setAddMul(B, dB, u );
	hkVector4 diff; diff.setSub( a,b );

	result.m_closestPointA = a;
	result.m_closestPointB = b;
	diff.lengthSquared<3>().store<1>(&(hkReal&)result.m_distanceSquared);
	t.store<1>(&(hkReal&)result.m_fractionA);
	u.store<1>(&(hkReal&)result.m_fractionB);

	return returnVal;
}

hkResult HK_CALL hkLineSegmentUtil::intersectionInfLinePlane( const hkVector4& A, const hkVector4& dA, const hkVector4& planeEquation, IntersectionInfLinePlaneResult& result)
{
	hkSimdReal denom = dA.dot<3>(planeEquation);
	hkSimdReal denomAbs;
	denomAbs.setAbs(denom);

	hkSimdReal t;
	hkResult returnVal;
	// check for divide by zero
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	const hkVector4Comparison denomBad = denomAbs.less(hkSimdReal_Eps);

	t = -planeEquation.dot4xyz1(A) * denom.reciprocal();
	t.zeroIfTrue(denomBad);

	returnVal = denomBad.anyIsSet() ? HK_FAILURE : HK_SUCCESS;
#else
	if (denomAbs < hkSimdReal_Eps)
	{
		t.setZero();
		returnVal = HK_FAILURE;
	}
	else
	{
		t = -planeEquation.dot4xyz1(A) / denom;
		returnVal = HK_SUCCESS;
	}
#endif

	result.m_pointOnPlane.setAddMul(A, dA, t);
	t.store<1>((hkReal*)&result.m_fractionA);

	return returnVal;
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
