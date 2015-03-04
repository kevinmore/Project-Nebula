/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Physics2012/Collide/BoxBox/hkpBoxBoxCollisionDetection.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>

#include <Physics2012/Collide/BoxBox/hkpBoxBoxManifold.h>
#include <Physics2012/Collide/Agent/ConvexAgent/BoxBox/hkpBoxBoxAgent.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>

#include <Physics2012/Collide/Agent/hkpCollisionAgentConfig.h>
#if defined(HK_PLATFORM_SPU)
#	include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgrSpu.inl> // include this after we include the actual contact manager!
#endif


#if defined HK_COMPILER_MSVC
	// C4100 conversion from 'int' to 'unsigned short', possible loss of data
#	pragma warning(disable: 4244)
#endif


#define INTERNAL_ERROR_STR "internal error, please report to havok"

const hkReal hkpBoxBoxCollisionDetection::m_manifoldConsistencyCheckAngularCosTolerance = 0.95f;
const hkReal hkpBoxBoxCollisionDetection::m_coplanerAngularCosTolerance = 0.98f;
const hkReal hkpBoxBoxCollisionDetection::m_contactNormalAngularCosTolerance = 0.95f;
// 0.001f was too small.  Should calc what hkReal error margin is.
const hkReal hkpBoxBoxCollisionDetection::m_edgeEndpointTolerance = HK_REAL_EPSILON;
// a valid point should be found within 8 "closest" features. so find closest features.
// max possible value is 15
const int hkpBoxBoxCollisionDetection::m_maxFeaturesToReject = 8;

static HK_FORCE_INLINE void selectIfGT3( hkVector4& result, hkSimdRealParameter selectIf, hkVector4Parameter greaterV, hkSimdRealParameter than )
{
	hkVector4 than4; than4.setAll( than );
	hkVector4 selectIf4; selectIf4.setAll( selectIf );
	result.setSelect( greaterV.greater(than4), selectIf4, result );
}

// featureBitSet  bits 0x  [
HK_COMPILE_TIME_ASSERT( hkVector4ComparisonMask::MASK_XYZ == 0x07 );

static HK_FORCE_INLINE void getAxisMapFromFeatureBitSet( hkInt32 bitSet, hkVector4& v )
{
	int bits = (bitSet & 0x0070) >> 4;
	hkVector4Comparison mask; mask.set((hkVector4ComparisonMask::Mask)bits);
	v.setFlipSign(hkVector4::getConstant<HK_QUADREAL_1>(), mask);
}

static HK_FORCE_INLINE void getNormalSignFromFeatureBitSet( hkUint8 bitSet, hkVector4Comparison& signMask )
{
	if (bitSet & 0x08)
	{
		signMask.set<hkVector4ComparisonMask::MASK_XYZW>();
	}
	else
	{
		signMask.set<hkVector4ComparisonMask::MASK_NONE>();
	}
}

static HK_FORCE_INLINE void setEdgeFeatureBitSetFromAxisMap( hkVector4Parameter axisMapAp, hkVector4Parameter axisMapBp, int featureIndexA, int featureIndexB, hkpFeatureContactPoint& fcp )
{
	hkVector4 axisMapA = axisMapAp;
	hkVector4 axisMapB = axisMapBp;

	// need to choose a side for consistency, otherwise edgeZ with -Z Axis map = -edgeZ with +Z axis map, but their bitSets will be different
	axisMapA.zeroComponent( featureIndexA );
	axisMapB.zeroComponent( featureIndexB );

	int maskA = axisMapA.lessZero().getMask<hkVector4ComparisonMask::MASK_XYZ>();
	int maskB = axisMapB.lessZero().getMask<hkVector4ComparisonMask::MASK_XYZ>() ^ hkVector4ComparisonMask::MASK_XYZ;

	fcp.m_featureIdA = (hkUint16)((hkVector4ComparisonMask::MASK_W<<4) | featureIndexA | (maskA << 4));
	fcp.m_featureIdB = (hkUint16)(featureIndexB + (maskB << 4));
}


static HK_FORCE_INLINE hkUint16 calcFaceFeatureBitSetFromAxisMap( hkVector4Parameter axisMap, hkVector4ComparisonParameter normalSign )
{
	const int maskA = axisMap.lessZero().getMask<hkVector4ComparisonMask::MASK_XYZ>();
	int fBitSet = maskA << 4;
	if (normalSign.anyIsSet()) fBitSet |= 8;
	return fBitSet;
}

static HK_FORCE_INLINE void setTerminalsFromVertex( int& bitSetEdgeTerminals, int bitSet, int planeAxis )
{
	bitSet = bitSet >> 4;
	int xFactor, yFactor, zFactor;
	xFactor = ( bitSet & hkVector4ComparisonMask::MASK_X ) ? 0 : 1;
	yFactor = ( bitSet & hkVector4ComparisonMask::MASK_Y ) ? 0 : 1;
	zFactor = ( bitSet & hkVector4ComparisonMask::MASK_Z ) ? 0 : 1;

	// each edge has two bits in bitSetEdgeTerminals.  The high bit is on if there are
	// more than 1 terminals... therefore it is terminated.  the low bit is on if there
	// is only 1 terminal.

	// any edge that has been started will be terminated by vertex if available
	// x-edge
	if( planeAxis != 0 )
	{
		const int addBit = 1<<(( yFactor + 2*zFactor )*2);
		if( (( addBit << 1 ) & bitSetEdgeTerminals) == 0 )
		{
			bitSetEdgeTerminals += addBit;
		}
	}
	// y-edge
	if( planeAxis != 1 )
	{
		const int addBit = 1<<(( 4 + xFactor + 2*zFactor )*2);
		if( (( addBit << 1 ) & bitSetEdgeTerminals) == 0 )
		{
			bitSetEdgeTerminals += addBit;
		}
	}
	// z-edge
	if( planeAxis != 2 )
	{
		const int addBit = 1<<(( 8 + xFactor + 2*yFactor )*2);
		if( (( addBit << 1 ) & bitSetEdgeTerminals) == 0 )
		{
			bitSetEdgeTerminals += addBit;
		}
	}

}


static HK_FORCE_INLINE void setTerminalsFromEdge( int& bitSetEdgeTerminals, int bitSet )
{
	int edgeAxis = (bitSet & 0x000f);
	bitSet = bitSet >> 4;
	int xFactor, yFactor, zFactor;
	xFactor = ( bitSet & hkVector4ComparisonMask::MASK_X ) ? 0 : 1;
	yFactor = ( bitSet & hkVector4ComparisonMask::MASK_Y ) ? 0 : 1;
	zFactor = ( bitSet & hkVector4ComparisonMask::MASK_Z ) ? 0 : 1;

	// each edge has two bits in bitSetEdgeTerminals.  The high bit is on if there are
	// more than 1 terminals... therefore it is terminated.  the low bit is on if there
	// is only 1 terminal.

	// any edge that has been started will be terminated by vertex if available
	// x-edge
	if( edgeAxis == 0 )
	{
		const int addBit = 1<<(( yFactor + 2*zFactor )*2);
		if( (( addBit << 1 ) & bitSetEdgeTerminals) == 0 )
		{
			bitSetEdgeTerminals += addBit;
		}
	}
	// y-edge
	else if( edgeAxis == 1 )
	{
		const int addBit = 1<<(( 4 + xFactor + 2*zFactor )*2);
		if( (( addBit << 1 ) & bitSetEdgeTerminals) == 0 )
		{
			bitSetEdgeTerminals += addBit;
		}
	}
	// z-edge
	else
	{
		const int addBit = 1<<(( 8 + xFactor + 2*yFactor )*2);
		if( (( addBit << 1 ) & bitSetEdgeTerminals) == 0 )
		{
			bitSetEdgeTerminals += addBit;
		}
	}

}


void hkpBoxBoxCollisionDetection::checkCompleteness( hkpBoxBoxManifold& manifold, int planeMaskA, int planeMaskB ) const
{

	// early exit. need 3 for completeness, unless we look at tim
	if( manifold.getNumPoints() < 3 )
		return;

	// simple check.
	if( manifold.m_faceVertexFeatureCount >= 4 )
	{
		manifold.setComplete( true );
		return;
	}

	// find the faces the manifolds are on
	int planeAxisA = hkVector4Comparison_maskToFirstIndex[planeMaskA >> 4];
	int planeAxisB = hkVector4Comparison_maskToFirstIndex[planeMaskB >> 4];


	// check edges.
	// an edge is part of a complete manifold if it has 2 terminal points in it's local body.
	// 0 = x-edge -y,-z: 1 = x-edge y,-z: 2 = x-edge -y,z: 3 = x-edge y,z: 4 = y-edge -x,-z

	int bitSetEdgeTerminalsA = 0;
	int bitSetEdgeTerminalsB = 0;
	{
		// find any edge points
		int i;
		for( i = 0; i < manifold.getNumPoints(); i++ )
		{
			const hkpFeatureContactPoint& fcpMan = manifold[i];

			if( fcpMan.m_featureIdA <= HK_BOXBOX_FACE_A_Z )
			{
				setTerminalsFromVertex( bitSetEdgeTerminalsB, fcpMan.m_featureIdB, planeAxisB );
			}
			else if( fcpMan.m_featureIdA <= HK_BOXBOX_FACE_B_Z )
			{
				setTerminalsFromVertex( bitSetEdgeTerminalsA, fcpMan.m_featureIdB, planeAxisA );
			}
			else //if( fcpMan.m_featureIdA > HK_BOXBOX_FACE_B_Z )
			{
				setTerminalsFromEdge( bitSetEdgeTerminalsA, fcpMan.m_featureIdA );
				setTerminalsFromEdge( bitSetEdgeTerminalsB, fcpMan.m_featureIdB );
			}
		}
	}

	// 0x00555555 = mask for every odd bit for 12 records of 2 bits each

	const hkBool isNotComplete = ((bitSetEdgeTerminalsA & 0x00555555) > 0 ) ||
		((bitSetEdgeTerminalsB & 0x00555555) > 0 );

	// we made it here, so all edges have 2 terminals in manifold, therefore manifold is complete
	manifold.setComplete( !isNotComplete );
}


HK_FORCE_INLINE void hkpBoxBoxCollisionDetection::removePoint( hkpBoxBoxManifold& manifold, int index ) const
{
	// change completeness criteria
	if( manifold[index].m_featureIdA <= HK_BOXBOX_FACE_B_Z )
	{
		manifold.m_faceVertexFeatureCount--;
	}

	m_contactMgr->removeContactPoint( manifold.m_contactPoints[index].m_contactPointId, *m_result->m_constraintOwner.val() );
	manifold.removePoint( index );
}


HK_FORCE_INLINE void hkpBoxBoxCollisionDetection::faceAVertexBValidationDataFromFeatureId( hkpFeaturePointCache& fpp, const hkpFeatureContactPoint &fcp ) const
{
// 	HK_TIME_CODE_BLOCK("faceAVertexBValidationDataFromFeatureId",0);
	fpp.m_featureIndexA = fcp.m_featureIdA;		// axis
	getNormalSignFromFeatureBitSet( fcp.m_featureIdB, fpp.m_normalIsFlipped );
	hkVector4 axisMap; getAxisMapFromFeatureBitSet( fcp.m_featureIdB, axisMap );
	fpp.m_vB.setMul( m_radiusB, axisMap );
	fpp.m_vA._setTransformedPos( m_aTb, fpp.m_vB );
}

// normal points from B to A
HK_FORCE_INLINE void hkpBoxBoxCollisionDetection::faceAVertexBContactPointFromFeaturePointCache( hkpProcessCdPoint& ccpOut, const hkpFeatureContactPoint &fcp, const hkpFeaturePointCache& fpp ) const
{
// 	HK_TIME_CODE_BLOCK("faceAVertexBContactPointFromFeaturePointCache",0);
	const int featureIndex = fcp.m_featureIdA;
	ccpOut.m_contactPointId = fcp.m_contactPointId;

	hkVector4 cpPos; cpPos._setTransformedPos( m_wTb, fpp.m_vB );
	ccpOut.m_contact.setPosition(cpPos);
	hkVector4Comparison notFlipped; notFlipped.setNot(fpp.m_normalIsFlipped);
	hkVector4 cpN; cpN.setFlipSign( m_wTa.getRotation().getColumn( featureIndex ), notFlipped );
	ccpOut.m_contact.setSeparatingNormal(cpN, fpp.m_distance);
}


HK_FORCE_INLINE void hkpBoxBoxCollisionDetection::faceBVertexAValidationDataFromFeatureId( hkpFeaturePointCache& fpp, const hkpFeatureContactPoint &fcp ) const
{
// 	HK_TIME_CODE_BLOCK("faceBVertexAValidationDataFromFeatureId",0);
	fpp.m_featureIndexA = fcp.m_featureIdA;		// axis
	getNormalSignFromFeatureBitSet( fcp.m_featureIdB, fpp.m_normalIsFlipped );
	hkVector4 axisMap; getAxisMapFromFeatureBitSet( fcp.m_featureIdB, axisMap );
	fpp.m_vA.setMul( m_radiusA, axisMap );
	fpp.m_vB._setTransformedInversePos( m_aTb, fpp.m_vA );

}

// normal points from B to A
HK_FORCE_INLINE void hkpBoxBoxCollisionDetection::faceBVertexAContactPointFromFeaturePointCache( hkpProcessCdPoint& ccpOut, const hkpFeatureContactPoint &fcp, const hkpFeaturePointCache& fpp ) const
{
// 	HK_TIME_CODE_BLOCK("faceBVertexAContactPointFromFeaturePointCache",0);
	const int featureIndex = fcp.m_featureIdA - HK_BOXBOX_FACE_B_X;
	ccpOut.m_contactPointId = fcp.m_contactPointId;

	hkVector4 cpPos; cpPos._setTransformedPos( m_wTa, fpp.m_vA );
	ccpOut.m_contact.setPosition(cpPos);
	hkVector4Comparison notFlipped; notFlipped.setNot(fpp.m_normalIsFlipped);
	hkVector4 cpN; cpN.setFlipSign( m_wTb.getRotation().getColumn( featureIndex ), notFlipped );
	ccpOut.m_contact.setSeparatingNormal(cpN, fpp.m_distance);
}

HK_FORCE_INLINE void hkpBoxBoxCollisionDetection::edgeEdgeValidationDataFromFeatureId( hkpFeaturePointCache& fpp, const hkpFeatureContactPoint &fcp ) const
{
	fpp.m_featureIndexA = fcp.m_featureIdA;
	fpp.m_featureIndexB = fcp.m_featureIdB;
}

HK_FORCE_INLINE void hkpBoxBoxCollisionDetection::edgeEdgeContactPointFromFeaturePointCache( hkpProcessCdPoint& ccpOut, const hkpFeatureContactPoint &fcp, const hkpFeaturePointCache& fpp ) const
{
// 	HK_TIME_CODE_BLOCK("edgeEdgeContactPointFromFeaturePointCache",0);
	ccpOut.m_contactPointId = fcp.m_contactPointId;
	hkVector4 cpPos; cpPos._setTransformedPos( m_wTa, fpp.m_vA );
	ccpOut.m_contact.setPosition(cpPos);
	hkVector4 cpN; cpN._setRotatedDir( m_wTa.getRotation(), fpp.m_nA );
	ccpOut.m_contact.setSeparatingNormal(cpN, fpp.m_distance);
}


HK_FORCE_INLINE void hkpBoxBoxCollisionDetection::contactPointFromFeaturePointCache( hkpProcessCdPoint& ccpOut, const hkpFeatureContactPoint &fcp, const hkpFeaturePointCache& fpp ) const
{
	const int sepFeature = fpp.m_featureIndexA;

	if( sepFeature <= HK_BOXBOX_FACE_A_Z )
	{
		faceAVertexBContactPointFromFeaturePointCache( ccpOut, fcp, fpp );
	}
	else if( sepFeature <= HK_BOXBOX_FACE_B_Z )
	{
		faceBVertexAContactPointFromFeaturePointCache( ccpOut, fcp, fpp );
	}
	else  // it's an edge-edge collision
	{
		edgeEdgeContactPointFromFeaturePointCache( ccpOut, fcp, fpp );
	}

}


HK_FORCE_INLINE hkResult hkpBoxBoxCollisionDetection::addPoint( hkpBoxBoxManifold& manifold, const hkpFeaturePointCache& fpp, hkpFeatureContactPoint& fcp) const
{

	// don't want to have a vertex on each box that basically duplicates the other one.
	if( fcp.m_featureIdA <= HK_BOXBOX_FACE_B_Z )
	{
		// make sure there isn't a corresponding worldspace point
		hkVector4 vCandidate;
		if( fcp.m_featureIdA <= HK_BOXBOX_FACE_A_Z )
		{
			vCandidate._setTransformedPos( m_wTb, fpp.m_vB );
		}
		else
		{
			vCandidate._setTransformedPos( m_wTa, fpp.m_vA );
		}

		hkSimdReal tol2 = m_tolerance4.getComponent<0>(); tol2.mul(tol2); tol2.add(hkSimdReal_Eps);
		int i;
		for( i = 0; i < manifold.getNumPoints(); i++ )
		{
			if( (manifold[i].m_featureIdA <= HK_BOXBOX_FACE_B_Z) )
			{
				const hkpProcessCdPoint& ccp = m_result->m_contactPoints[i];
				hkVector4 diff;
				diff.setSub( vCandidate, ccp.m_contact.getPosition() );
				const hkSimdReal delta = diff.lengthSquared<3>();

				// weld tolerance must be less than the smallest extent of the pair of boxes, or
				// it will result in welding to aggressively.  It would be best if we could remove the
				// other point here, instead of the closest, then the constraint on the weld tolerance could
				// be removed.
				if( delta.isLessEqual(tol2) )
				{
					// don't add it, there's already one there.
					return HK_FAILURE;
				}
			}
		}
	}

	// if we have no points left remove the one that is furthest away
	if( manifold.hasNoPointsLeft() )
	{
		
		return HK_FAILURE;
		/*
		int removeCandidate = 0;
		hkReal largestD = -1000.0f;
		int i;
		for( i = 0; i < manifold.getNumPoints(); i++ )
		{
			hkReal d = m_result->m_contactPoints[i].m_contact.getDistance();
			if( d > largestD )
			{
				d = largestD;
				removeCandidate = i;
			}
		}

		removePoint( manifold, removeCandidate );
		*/
	}

	int fcpIndex = manifold.addPoint( m_bodyA, m_bodyB, fcp );
	if( fcpIndex >= 0 )
	{
		hkpProcessCdPoint& ccp = *m_result->reserveContactPoints(1);

		contactPointFromFeaturePointCache( ccp, fcp, fpp );

		// this is to catch deep penetration case where one normal points in the opposite direction of another
		if( manifold.getNumPoints() > 1 )
		{
			HK_ASSERT2(0x66909703,  !m_result->isEmpty(), "box-box: can't cull point with empty results manifold" );
			const hkpProcessCdPoint& existingCcp = m_result->getEnd()[-1];

			if( existingCcp.m_contact.getNormal().dot<3>( ccp.m_contact.getNormal() ).isLessEqualZero() )
			{
				m_result->abortContactPoints(1);
				manifold.removePoint( fcpIndex );
				return HK_FAILURE;
			}
		}

		manifold[fcpIndex].m_contactPointId = m_contactMgr->addContactPoint( m_bodyA, m_bodyB, *m_env, *m_result, HK_NULL, ccp.m_contact );

		if ( manifold[fcpIndex].m_contactPointId == HK_INVALID_CONTACT_POINT )
		{
			m_result->abortContactPoints(1);
			manifold.removePoint( fcpIndex );
		}
		else
		{
			m_result->commitContactPoints(1);
			fcp.m_contactPointId = manifold[fcpIndex].m_contactPointId;
			ccp.m_contactPointId = manifold[fcpIndex].m_contactPointId;

			if( fcp.m_featureIdA <= HK_BOXBOX_FACE_B_Z )
			{
				manifold.m_faceVertexFeatureCount++;
			}
		}
	}
	return HK_SUCCESS;
}

static HK_FORCE_INLINE int getMaxPlaneMask3( hkVector4Parameter vAbs, int& axisOut )
{
	axisOut = vAbs.getIndexOfMaxComponent<3>();
	return 16 * hkVector4Comparison::getMaskForComponent(axisOut);
}


static const hkUint16 cycleMask[7] = {
								hkVector4ComparisonMask::MASK_XZ * 16,
								hkVector4ComparisonMask::MASK_XY * 16,
								hkVector4ComparisonMask::MASK_YZ * 16,
								hkVector4ComparisonMask::MASK_X  * 16,
								hkVector4ComparisonMask::MASK_Y  * 16,
								hkVector4ComparisonMask::MASK_Z  * 16,
								0 };

//!me get this compiling on PlayStation(R)2... it complains about not finding the type
/*
void hkpBoxBoxCollisionDetection::tryToAddPoint( hkpBoxBoxManifold& manifold,
												const hkpFeatureContactPoint fcpTemplate, int planeMask, hkReal closestPointDist,
												tFcnValidationDataFromFeatureId fcnValidationDataFromFeatureId, tFcnIsValid fcnIsValid ) const
{

	int ia;
	for( ia = 0; ia < 7; ia++ )
	{
		const int cMask = cycleMask[ia];
		if( !(cMask  & planeMask) )
		{
			int newFeatureID = cMask ^ fcpTemplate.m_featureIdB;
			hkpFeatureContactPoint fcpTest = fcpTemplate;
			fcpTest.m_featureIdB = newFeatureID | (fcpTest.m_featureIdB & 0xff0f);
			if( !manifold.findInManifold(fcpTest) )
			{
				hkpFeaturePointCache fpp;
				(*this.*fcnValidationDataFromFeatureId)( fpp, fcpTest );

				if((*this.*fcnIsValid)( fpp ))
				{
					// any points that are not farther away than closest point
					// are almost surely invalid
					if( fpp.m_distance >= closestPointDist * 0.999f )
					{
						addPoint( manifold, fpp, fcpTest );
					}
				}
			}

		}
	}
}
*/



HK_FORCE_INLINE void hkpBoxBoxCollisionDetection::tryToAddPointFaceA( hkpBoxBoxManifold& manifold,
												const hkpFeatureContactPoint fcpTemplate, hkUint16 planeMask, hkSimdRealParameter closestPointDist ) const
{
	int ia;
	for( ia = 0; ia < 7; ia++ )
	{
		const hkUint16 cMask = cycleMask[ia];
		if( !(cMask  & planeMask) )
		{
			hkUint16 newFeatureID = cMask ^ fcpTemplate.m_featureIdB;
			hkpFeatureContactPoint fcpTest = fcpTemplate;
			fcpTest.m_featureIdB = newFeatureID | (fcpTest.m_featureIdB & 0xff0f);
			if( !manifold.findInManifold(fcpTest) )
			{
				hkpFeaturePointCache fpp;
				fpp.m_nA.setZero();
				faceAVertexBValidationDataFromFeatureId( fpp, fcpTest );

				if(isValidFaceAVertexB( fpp ))
				{
					// any points that are not farther away than closest point
					// are almost surely invalid
					calcDistanceFaceAVertexB(fpp);
					if( fpp.m_distance.isGreaterEqual(closestPointDist - hkSimdReal_Eps) )
					{
						addPoint( manifold, fpp, fcpTest );
					}
				}
			}
		}
	}
}



HK_FORCE_INLINE void hkpBoxBoxCollisionDetection::tryToAddPointFaceB( hkpBoxBoxManifold& manifold,
												const hkpFeatureContactPoint fcpTemplate, hkUint16 planeMask, hkSimdRealParameter closestPointDist ) const
{
	int ia;
	for( ia = 0; ia < 7; ia++ )
	{
		const hkUint16 cMask = cycleMask[ia];
		if( !(cMask  & planeMask) )
		{
			hkUint16 newFeatureID = cMask ^ fcpTemplate.m_featureIdB;
			hkpFeatureContactPoint fcpTest = fcpTemplate;
			fcpTest.m_featureIdB = newFeatureID | (fcpTest.m_featureIdB & 0xff0f);
			if( !manifold.findInManifold(fcpTest) )
			{
				hkpFeaturePointCache fpp;
				fpp.m_nA.setZero();
				faceBVertexAValidationDataFromFeatureId( fpp, fcpTest );

				if(isValidFaceBVertexA( fpp ))
				{
					// any points that are not farther away than closest point
					// are almost surely invalid
					calcDistanceFaceBVertexA(fpp);
					if( fpp.m_distance.isGreaterEqual(closestPointDist - hkSimdReal_Eps) )
					{
						addPoint( manifold, fpp, fcpTest );
					}
				}
			}

		}
	}
}



void hkpBoxBoxCollisionDetection::addAdditionalEdgeHelper( hkpBoxBoxManifold& manifold, hkpFeatureContactPoint& fcp, hkSimdRealParameter closestPointDist ) const
{

	if(!manifold.findInManifold(fcp))
	{
		hkpFeaturePointCache fpp;
		
		// Fix uninitialized variable warnings
		fpp.m_vA.setZero();
		fpp.m_vB.setZero();
		fpp.m_normalIsFlipped.set<hkVector4ComparisonMask::MASK_NONE>();
		
		edgeEdgeValidationDataFromFeatureId( fpp, fcp );
		if( isValidEdgeEdge( fpp ) )
		{
			if( fpp.m_distance.isGreaterEqual(closestPointDist * hkSimdReal::fromFloat(0.999f)) )
			{
				addPoint( manifold, fpp, fcp );
			}
		}
	}
}



//!me this can most likely be optimized, too many calls to isValidEdgeEdge that probably duplicate work
void hkpBoxBoxCollisionDetection::tryToAddPointOnEdge( hkpBoxBoxManifold& manifold, int edgeA, int edgeB, int nextVertA, int nextVertB, const hkVector4& normalA, const hkVector4& normalB, hkSimdRealParameter closestPointDist ) const
{
	hkpFeatureContactPoint fcp;
	setEdgeFeatureBitSetFromAxisMap( normalA, normalB, edgeA, edgeB, fcp );

	addAdditionalEdgeHelper( manifold, fcp, closestPointDist );

	// check opposite edge by flipping along axis that is orthogonal to edge and face edge is on
	int shiftWidthA;
	shiftWidthA = nextVertA;

	int signFlipA = 1 << ( 4 + shiftWidthA );

	int shiftWidthB;
	shiftWidthB = nextVertB;

	int signFlipB = 1 << ( 4 + shiftWidthB );

	fcp.m_featureIdA ^= signFlipA;

	addAdditionalEdgeHelper( manifold, fcp, closestPointDist );

	fcp.m_featureIdB ^= signFlipB;

	addAdditionalEdgeHelper( manifold, fcp, closestPointDist );

	// flipping A twice resets to original sign
	fcp.m_featureIdA ^= signFlipA;

	addAdditionalEdgeHelper( manifold, fcp, closestPointDist );
}

HK_DISABLE_OPTIMIZATION_VS2008_X64
void hkpBoxBoxCollisionDetection::findAdditionalManifoldPoints( hkpBoxBoxManifold& manifold, hkpFeatureContactPoint fcpATemplate ) const
{
	if( manifold.getNumPoints() < 1 )
	{
		return;
	}

	int planeMaskA, planeMaskB;
	hkVector4 axisMapA, axisMapB;
	int axisA, axisB;

	const int resultIndex = m_result->getNumContactPoints() - manifold.getNumPoints();

	{
		hkVector4 axisMapA_, axisMapB_;
		const hkpProcessCdPoint& ccp = m_result->m_contactPoints[ resultIndex ];
		const hkVector4& normal = ccp.m_contact.getNormal();

		axisMapA._setRotatedInverseDir( m_wTa.getRotation(), normal );
		axisMapB._setRotatedInverseDir( m_wTb.getRotation(), normal );
		axisMapA.setNeg<4>( axisMapA );
		axisMapA_.setAbs( axisMapA );
		axisMapB_.setAbs( axisMapB );
		// bitset with 1 set in place of plane dimension

		//!me posible completeness criteria == best normal dim > some tolerance && 2 vertexface contacts
		planeMaskA = getMaxPlaneMask3( axisMapA_, axisA );
		planeMaskB = getMaxPlaneMask3( axisMapB_, axisB );

		// only check for additional points if the faces are close to coplanar.
		hkSimdReal tol; tol.load<1>(&m_coplanerAngularCosTolerance);
		if( axisMapA_.getComponent(axisA).isLess(tol) | axisMapB_.getComponent(axisB).isLess(tol) )
		{
			return;
		}

	}

	//
	// check for additional face-vertex points
	//

	hkpFeatureContactPoint fcpBTemplate;

	//!me how was this superior method supposed to work??? what if B is rotated?  Seemed to be getting some weird
	// attempts at contact points.  I can see how it works for faceA, just fine.
//	if( fcpATemplate.m_featureIdA <= HK_BOXBOX_FACE_B_Z )
//	{
		// superior check if we our new point is a face-vertex point
		// better contact point ordering and better processing.
//		fcpBTemplate.m_featureIdB = fcpATemplate.m_featureIdB;
//	}
//	else
	{
		const hkVector4Comparison signA = m_dinA.getComponent( axisA ).lessZero();
		const hkVector4Comparison signB = m_dinB.getComponent( axisB ).lessZero();

		fcpATemplate.m_featureIdB = calcFaceFeatureBitSetFromAxisMap( axisMapB, signA );
		fcpBTemplate.m_featureIdB = calcFaceFeatureBitSetFromAxisMap( axisMapA, signB );
	}

	fcpATemplate.m_featureIdA = axisA;
	fcpBTemplate.m_featureIdA = axisB + HK_BOXBOX_FACE_B_X;

	int i;
	hkSimdReal closestPointDist = hkSimdReal_1;	// <todo: fix this dodgy constant>
	for( i = 0; i < manifold.getNumPoints(); i++ )
	{
		const hkSimdReal candidate = m_result->m_contactPoints[ resultIndex + i ].m_contact.getDistanceSimdReal();
		closestPointDist.setMin(closestPointDist, candidate);
	}

	// check for additional vertexes from B on face of A
	tryToAddPointFaceA( manifold, fcpATemplate, planeMaskB, closestPointDist );
//	tryToAddPoint( manifold, fcpATemplate, planeMaskB, closestPointDist, faceAVertexBValidationDataFromFeatureId, isValidFaceAVertexB );


	// check for additional vertexes from A on face of B
	tryToAddPointFaceB(  manifold, fcpBTemplate, planeMaskA, closestPointDist );
//	tryToAddPoint( manifold, fcpATemplate, planeMaskB, closestPointDist, faceBVertexAValidationDataFromFeatureId, isValidFaceBVertexA );

	// edge-edge is expensive so do simple completeness check this first
	if( manifold.m_faceVertexFeatureCount >= 4 )
	{
		manifold.setComplete( true );
		return;
	}

/*
	if( hkpBoxBoxAgent::getAttemptToFindAllEdges() )
	{
		// check for additional edge-edge contact points

		hkpFeatureContactPoint fcpEdgeTemplate;
		int edgeA1Axis = hkBoxBoxUtils::mod3(axisA + 1);
		int edgeA2Axis = hkBoxBoxUtils::mod3(axisA + 2);
		int edgeB1Axis = hkBoxBoxUtils::mod3(axisB + 1);
		int edgeB2Axis = hkBoxBoxUtils::mod3(axisB + 2);

		axisMapB.setNeg4( axisMapB );  // flip sign so it is a normal

		tryToAddPointOnEdge( manifold, edgeA1Axis, edgeB1Axis, edgeA2Axis, edgeB2Axis, axisMapA, axisMapB, closestPointDist );
		tryToAddPointOnEdge( manifold, edgeA1Axis, edgeB2Axis, edgeA2Axis, edgeB1Axis, axisMapA, axisMapB, closestPointDist );
		tryToAddPointOnEdge( manifold, edgeA2Axis, edgeB1Axis, edgeA1Axis, edgeB2Axis, axisMapA, axisMapB, closestPointDist );
		tryToAddPointOnEdge( manifold, edgeA2Axis, edgeB2Axis, edgeA1Axis, edgeB1Axis, axisMapA, axisMapB, closestPointDist );

	}
*/

	checkCompleteness( manifold, planeMaskA, planeMaskB );

}
HK_RESTORE_OPTIMIZATION_VS2008_X64

hkResult hkpBoxBoxCollisionDetection::checkIntersection( const hkVector4& tolerance ) const
{

	// this is used to ignore edge-edge combinations that are close to parallel.
	// should be caught by face collision instead.
	//!me is this a good thing?
	const hkSimdReal HK_EDGE_IGNORE_TOLERANCE = hkSimdReal::fromFloat(1.0f/0.001f); //1.0f/0.20f;  //!me may just need = 0
	const hkSimdReal FLT_MINPS = hkSimdReal::fromFloat(-1e17f);

	// init abs space
	hkRotation aRb_;
	aRb_.getColumn(0).setAbs( m_aTb.getRotation().getColumn<0>() );
	aRb_.getColumn(1).setAbs( m_aTb.getRotation().getColumn<1>() );
	aRb_.getColumn(2).setAbs( m_aTb.getRotation().getColumn<2>() );

	// consider faces of A
	{
		hkVector4 vWidth; vWidth._setRotatedDir( aRb_, m_radiusB );
		hkVector4 vdProj; vdProj.setAbs( m_dinA );
		vWidth.add( m_radiusA );
		vdProj.sub( vWidth );

		HK_ASSERT(0x8229111, tolerance.getW() == hkSimdReal_Max);
		if( vdProj.greater(tolerance).anyIsSet() )
		{
			return HK_FAILURE;
		}
		m_sepDist[HK_BOXBOX_FACE_A_X/4] = vdProj;
	}

	//consider faces of B
	{
		hkVector4 vWidth; vWidth._setRotatedInverseDir( aRb_, m_radiusA );
		hkVector4 vdProj; vdProj.setAbs( m_dinB );
		vWidth.add(m_radiusB);
		vdProj.sub(vWidth);

		HK_ASSERT(0x8229111, tolerance.getW() == hkSimdReal_Max);
		if( vdProj.greater(tolerance).anyIsSet() )
		{
			return HK_FAILURE;
		}
		m_sepDist[HK_BOXBOX_FACE_B_X/4] = vdProj;
	}

	// consider faces derived from an edge of each object
	{
		// A0 % BN

		hkRotation bRa; bRa._setTranspose(m_aTb.getRotation());

		// set up some values with SIMD efficiency for use later
		hkVector4 aRbSquaredC0 = bRa.getColumn<0>();
		hkVector4 aRbSquaredC1 = bRa.getColumn<1>();
		hkVector4 aRbSquaredC2 = bRa.getColumn<2>();
		{
			aRbSquaredC0.mul( aRbSquaredC0 );
			aRbSquaredC1.mul( aRbSquaredC1 );
			aRbSquaredC2.mul( aRbSquaredC2 );
		}

		hkRotation aRb_t; aRb_t._setTranspose(aRb_);
		const hkVector4& c0 = aRb_t.getColumn<0>();
		const hkVector4& c1 = aRb_t.getColumn<1>();
		const hkVector4& c2 = aRb_t.getColumn<2>();

		hkVector4 rByxx; rByxx.setPermutation<hkVectorPermutation::YXXY>(m_radiusB);
		hkVector4 rBzzy; rBzzy.setPermutation<hkVectorPermutation::ZZYY>(m_radiusB);

		{
			hkVector4 vdProj;	setvdProj<1,2>( bRa, vdProj );

			hkVector4 c0zzy; c0zzy.setPermutation<hkVectorPermutation::ZZYY>(c0);
			hkVector4 c0yxx; c0yxx.setPermutation<hkVectorPermutation::YXXY>(c0);

			hkVector4 vWidth;
			vWidth.setMul(c2, m_radiusA.getComponent<1>());
			vWidth.addMul(c1, m_radiusA.getComponent<2>());
			vWidth.addMul(c0zzy, rByxx);
			vWidth.addMul(c0yxx, rBzzy);

			// since one axis is canonical it reduces the cross product to something simple.  (1,0,0)%(x,y,z) = (0,-z,y)
			// We square things so we can get the length after the sqrt.  We are doing 3 axis at a time.
			hkVector4 vXprodLengthInv;
			vXprodLengthInv.setAdd( aRbSquaredC2, aRbSquaredC1 );
			vXprodLengthInv.setMax( vXprodLengthInv, hkVector4::getConstant<HK_QUADREAL_EPS_SQRD>());

			vdProj.sub( vWidth );
			vXprodLengthInv.setSqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>(vXprodLengthInv);
			vdProj.setMul( vXprodLengthInv, vdProj );

			HK_ASSERT(0x8229111, m_tolerance4.getW() == hkSimdReal_Max);
			if( vdProj.greater(m_tolerance4).anyIsSet() ) 
			{
				return HK_FAILURE;
			}

			selectIfGT3( vdProj, FLT_MINPS, vXprodLengthInv, HK_EDGE_IGNORE_TOLERANCE );

			m_sepDist[HK_BOXBOX_EDGE_0_0/4] = vdProj;
		}
		// A1 % BN
		{
			hkVector4 vdProj;	setvdProj<2,0>( bRa, vdProj );

			hkVector4 c1yxx; c1yxx.setPermutation<hkVectorPermutation::YXXY>(c1);
			hkVector4 c1zzy; c1zzy.setPermutation<hkVectorPermutation::ZZYY>(c1);

			hkVector4 vWidth;
			vWidth.setMul(c2, m_radiusA.getComponent<0>());
			vWidth.addMul(c0, m_radiusA.getComponent<2>());
			vWidth.addMul(c1zzy, rByxx);
			vWidth.addMul(c1yxx, rBzzy);

			hkVector4 vXprodLengthInv;
			vXprodLengthInv.setAdd( aRbSquaredC2, aRbSquaredC0 );
			vXprodLengthInv.setMax( vXprodLengthInv, hkVector4::getConstant<HK_QUADREAL_EPS_SQRD>());

			vdProj.sub( vWidth );
			vXprodLengthInv.setSqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>(vXprodLengthInv);
			vdProj.setMul( vXprodLengthInv, vdProj );

			HK_ASSERT(0x8229111, m_tolerance4.getW() == hkSimdReal_Max);
			if( vdProj.greater(m_tolerance4).anyIsSet() )
			{
				return HK_FAILURE;
			}

			selectIfGT3( vdProj, FLT_MINPS, vXprodLengthInv, HK_EDGE_IGNORE_TOLERANCE );

			m_sepDist[HK_BOXBOX_EDGE_1_0/4] = vdProj;
		}

		// A2 % BN
		{
			hkVector4 vdProj;	setvdProj<0,1>( bRa, vdProj );

			hkVector4 c2zxy; c2zxy.setPermutation<hkVectorPermutation::ZXYW>(c2);
			hkVector4 c2yzx; c2yzx.setPermutation<hkVectorPermutation::YZXW>(c2);
			hkVector4 rBzxy; rBzxy.setPermutation<hkVectorPermutation::ZXYW>(m_radiusB);
			hkVector4 rByzx; rByzx.setPermutation<hkVectorPermutation::YZXW>(m_radiusB);

			hkVector4 vWidth;
			vWidth.setMul(c1, m_radiusA.getComponent<0>());
			vWidth.addMul(c0, m_radiusA.getComponent<1>());
			vWidth.addMul(c2zxy, rByzx);
			vWidth.addMul(c2yzx, rBzxy);

			hkVector4 vXprodLengthInv;
			vXprodLengthInv.setAdd( aRbSquaredC1, aRbSquaredC0 );
			vXprodLengthInv.setMax( vXprodLengthInv, hkVector4::getConstant<HK_QUADREAL_EPS_SQRD>());

			vdProj.sub( vWidth );
			vXprodLengthInv.setSqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>(vXprodLengthInv);
			vdProj.setMul( vXprodLengthInv, vdProj );

			HK_ASSERT(0x8229111, m_tolerance4.getW() == hkSimdReal_Max);
			if( vdProj.greater(m_tolerance4).anyIsSet() ) 
			{
				return HK_FAILURE;
			}

			selectIfGT3( vdProj, FLT_MINPS, vXprodLengthInv, HK_EDGE_IGNORE_TOLERANCE );

			m_sepDist[HK_BOXBOX_EDGE_2_0/4] = vdProj;
		}
	}
	return HK_SUCCESS;
}


HK_FORCE_INLINE void hkpBoxBoxCollisionDetection::faceAVertexBValidationDataFromFeatureIndex( hkpFeaturePointCache &fpp, int featureIndex ) const
{
	fpp.m_featureIndexA = featureIndex;
	hkMatrix3 rotT; rotT.setTranspose(m_aTb.getRotation());
	hkVector4 dimA = rotT.getColumn(featureIndex);	// dimA gives us the direction to move

	// normal is axis map, but we must negate it first
	// we also must check which side of object we are on.  Check was initially done in "absolute" space
	hkVector4 tweak;
	tweak.setAll(hkReal(0.00001f));

	const hkVector4Comparison lt0 = m_dinA.getComponent( featureIndex ).lessZero();
	fpp.m_normalIsFlipped = lt0;
	tweak.setFlipSign(tweak,lt0);

	hkVector4Comparison ge0; ge0.setNot(lt0);
	fpp.m_vB.setFlipSign(m_radiusB,ge0);

	// tweak it over ever so slightly to choose proper vertex for parallel faces.
	// corrected for which side we are on
	tweak.setFlipSign(tweak, m_dinB.lessEqualZero());
	dimA.add( tweak );

	fpp.m_vB.setFlipSign(fpp.m_vB, dimA);
	fpp.m_vA._setTransformedPos( m_aTb, fpp.m_vB );

}

HK_FORCE_INLINE void hkpBoxBoxCollisionDetection::faceBVertexAValidationDataFromFeatureIndex( hkpFeaturePointCache &fpp, int featureIndex ) const
{
	fpp.m_featureIndexA = featureIndex + HK_BOXBOX_FACE_B_X;

	hkVector4 dimB = m_aTb.getRotation().getColumn( featureIndex );

	// normal is axis map, but we must negate it first.
	// we also must check which side of object we are on.  Check was initially done in "absolute" space

	hkVector4 tweak; 
	tweak.setAll(hkReal(-0.00001f));

	const hkVector4Comparison lt0 = m_dinB.getComponent( featureIndex ).lessZero();
	fpp.m_normalIsFlipped = lt0;
	fpp.m_vA.setFlipSign(m_radiusA, lt0);
	tweak.setFlipSign(tweak,lt0);

	// tweak it over ever so slightly to choose proper vertex for parallel faces.
	// corrected for which side we are on
	tweak.setFlipSign(tweak, m_dinB.lessEqualZero());
	dimB.add( tweak );

	fpp.m_vA.setFlipSign(fpp.m_vA, dimB);
	fpp.m_vB._setTransformedInversePos( m_aTb, fpp.m_vA );
}


HK_FORCE_INLINE void hkpBoxBoxCollisionDetection::edgeEdgeValidationDataFromFeatureIndex( hkpFeaturePointCache &fpp ) const
{
	const int fIA = fpp.m_featureIndexA;
	const int fIB = fpp.m_featureIndexB;

	hkVector4 edge_A_in_A = hkVector4::getConstant((hkVectorConstant)(HK_QUADREAL_1000 + fIA));
	hkVector4 edge_B_in_A = m_aTb.getRotation().getColumn( fIB );
	hkVector4 normal_in_A; normal_in_A.setCross( edge_A_in_A, edge_B_in_A );
	normal_in_A.setFlipSign(normal_in_A, normal_in_A.dot<3>( m_dinA ).lessZero()); 
	hkVector4 normal_in_B; normal_in_B._setRotatedInverseDir( m_aTb.getRotation(), normal_in_A );

	fpp.m_vA = normal_in_A;
	fpp.m_vB = normal_in_B;
}


int hkpBoxBoxCollisionDetection::findClosestPoint( hkpBoxBoxManifold& manifold, hkpFeatureContactPoint& fcp, hkpFeaturePointCache& fpp ) const
{
	// compute collision info

	// a valid point should be found within 8 "closest" features. so find closest features.
	for( int featuresToConsider = m_maxFeaturesToReject; featuresToConsider > 0; featuresToConsider-- )
	{
		unsigned int sepFeature = 0;

		//
		//	find minimum penetration
		//

		// calc max distance, in case of equality choose last index
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
		{
			hkVector4 bestDist = m_sepDist[0];

			hkIntVector zeroIdx; zeroIdx.setZero();
			hkIntVector idxIdx = hkIntVector::getConstant<HK_QUADINT_0123>();
			hkIntVector maxIdx = idxIdx;

			for( int i = 1; i < 5; i++ )
			{
				idxIdx.setAddS32(idxIdx, hkIntVector::getConstant<HK_QUADINT_4>());

				hkVector4Comparison maxGTbest = m_sepDist[i].greater(bestDist);
				maxIdx.setSelect(maxGTbest, idxIdx, maxIdx);
				bestDist.setMax(m_sepDist[i], bestDist);
			}

			hkVector4 hMax; hMax.setHorizontalMax<3>(bestDist);
			hkVector4Comparison isMax = hMax.equal(bestDist);
			maxIdx.setSelect(isMax, maxIdx, zeroIdx);
			sepFeature = maxIdx.horizontalMaxS32<3>();
		}
#else
		{
			sepFeature = m_sepDist[0].getIndexOfMaxComponent<3>();
			hkSimdReal bestDist = m_sepDist[0].getComponent(sepFeature);

			for( int i = 1; i < 5; i++ )
			{
				int maxIdx = m_sepDist[i].getIndexOfMaxComponent<3>();
				hkSimdReal maxDist = m_sepDist[i].getComponent(maxIdx);

				if (maxDist > bestDist)
				{
					bestDist = maxDist;
					sepFeature = i*4+maxIdx;
				}
			}
		}
#endif
		// calculate contact info for vertex on B against face of A
		if( sepFeature <= HK_BOXBOX_FACE_A_Z )
		{
			const int featureIndex = sepFeature /* - HK_BOXBOX_FACE_A_X */;	// axis

			fcp.m_featureIdA = sepFeature;
			faceAVertexBValidationDataFromFeatureIndex( fpp, featureIndex );

			fcp.m_featureIdB = calcFaceFeatureBitSetFromAxisMap( fpp.m_vB, fpp.m_normalIsFlipped );


			if ( manifold.findInManifold(fcp))
			{
				// this could result in an invalid point being tracked for one frame
				// since refresh happens at the start of the control loop
				return HK_FINDCLOSESTPOINT_POINT_IN_MANIFOLD;
			}


			if (isValidFaceAVertexB( fpp ))
			{
				calcDistanceFaceAVertexB(fpp);
				return HK_FINDCLOSESTPOINT_VALID_POINT;
			}
		}
		// vertex in A, face of B
		else if( sepFeature <= HK_BOXBOX_FACE_B_Z )
		{

			const int featureIndex = sepFeature - HK_BOXBOX_FACE_B_X;
			fcp.m_featureIdA = sepFeature;

			faceBVertexAValidationDataFromFeatureIndex( fpp, featureIndex );

			fcp.m_featureIdB = calcFaceFeatureBitSetFromAxisMap( fpp.m_vA, fpp.m_normalIsFlipped );

			if ( manifold.findInManifold(fcp))
			{
				return HK_FINDCLOSESTPOINT_POINT_IN_MANIFOLD;
			}

			if (isValidFaceBVertexA( fpp ))
			{
				calcDistanceFaceBVertexA(fpp);
				return HK_FINDCLOSESTPOINT_VALID_POINT;
			}

		}
		// edge edge
		else
		{
			const unsigned int fI = sepFeature - HK_BOXBOX_EDGE_0_0;
			fpp.m_featureIndexA = fI/4;
			fpp.m_featureIndexB = fI&3;

			edgeEdgeValidationDataFromFeatureIndex( fpp );
			setEdgeFeatureBitSetFromAxisMap( fpp.m_vA, fpp.m_vB, fpp.m_featureIndexA, fpp.m_featureIndexB, fcp );
			fpp.m_featureIndexA = fcp.m_featureIdA;
			fpp.m_featureIndexB = fcp.m_featureIdB;

			if (manifold.findInManifold(fcp))
			{
				return HK_FINDCLOSESTPOINT_POINT_IN_MANIFOLD;
			}

			bool result = isValidEdgeEdge( fpp );
			if ( result )
			{
				return HK_FINDCLOSESTPOINT_VALID_POINT;
			}

		}
		// swap best feature out so it isn't considered again
		m_sepDist[sepFeature>>2].setComponent(sepFeature&3, hkSimdReal_MinusMax);
	}

	return HK_FINDCLOSESTPOINT_NO_VALID_POINT;
}


// fcp.m_featureIdA  bits 654 are the vertex sign bits for zyx  bits 0123 are the axis direction
// hkpFeaturePointCache::m_nA, m_vLocal, m_distance are all filled out if true returned
bool hkpBoxBoxCollisionDetection::isValidEdgeEdge( hkpFeaturePointCache &fpp ) const
{

	const int axisDimA = fpp.m_featureIndexA & 0xf;
	hkVector4 signsA; getAxisMapFromFeatureBitSet( fpp.m_featureIndexA, signsA );

	hkVector4 vertexA_inA;	vertexA_inA.setMul( signsA, m_radiusA );

 	const int axisDimB = fpp.m_featureIndexB & 0xf;
	hkVector4 signsB; getAxisMapFromFeatureBitSet( fpp.m_featureIndexB, signsB );

	// d12 is the distance vector from the contact point from body A to point on B in A's space
	hkVector4 edgeB_inA, d12;
	{
		// Get vertex B in B's space first
		hkVector4 vertexB_inA;	vertexB_inA.setMul( signsB, m_radiusB );

		// in setEdgeFeatureBitSetFromAxisMap, the axis map always gives you to the "positive" endpoint of the edge
		// so we don't need to multiply by the axis map sign here
		HK_ASSERT2(0x6a01329f,  signsB( axisDimB ) < 0, INTERNAL_ERROR_STR ); //!me 
		edgeB_inA.setMul( m_radiusB.getComponent(axisDimB) * hkSimdReal_2, m_aTb.getRotation().getColumn( axisDimB ));

		// move vertex B into A's space and subtract from to get support vector.
		vertexB_inA._setTransformedPos( m_aTb, vertexB_inA );
		d12.setSub( vertexB_inA, vertexA_inA );
	}

	// this is the line segment - line segment intersection test after having all 0's canceled out
	// due to the fact that our A edge is a canonical axis.  exit early or compute the normal ( unormalized )
	hkVector4 normal_inA;
	{
		// in setEdgeFeatureBitSetFromAxisMap, the axis map always gives you to the "positive" endpoint of the edge
		// but then in setEdgeFeatureBitSetFromAxisMap it is reversed, so it's now the "negative" side
		HK_ASSERT2(0x1b3ad1e9,  signsA( axisDimA ) > 0, INTERNAL_ERROR_STR );  //!me
		const hkSimdReal d1_at_DimA = m_radiusA.getComponent( axisDimA ) * (-hkSimdReal_2);

		const hkSimdReal R = d1_at_DimA * edgeB_inA.getComponent(axisDimA);						//VECTOR_DOT_MACRO(d1,d2);
		const hkSimdReal D1 = d1_at_DimA * d1_at_DimA;											//VECTOR_DOT_MACRO(d1,d1);
		hkSimdReal D2 = m_radiusB.getComponent( axisDimB ); D2 = D2 * D2 * hkSimdReal_4;	//VECTOR_DOT_MACRO(d2,d2);

		const hkSimdReal S1=	d1_at_DimA * d12.getComponent( axisDimA );							//VECTOR_DOT_MACRO(d1,d12);
		const hkSimdReal S2=	edgeB_inA.dot<3>( d12 );								//VECTOR_DOT_MACRO(d2,d12);


		// Step 1, compute D1, D2, R and the denominator.
		// The cases (a), (b) and (c) are covered in Steps 2, 3 and 4 as
		// checks for division by zero.
		hkSimdReal edgeEndpointTolerance; edgeEndpointTolerance.load<1>(&m_edgeEndpointTolerance);

		hkSimdReal denom; denom.setAbs(D1*D2-R*R);

		hkSimdReal t = (S1*D2-S2*R);
		if ( t.isLessEqual(edgeEndpointTolerance*denom) | t.isGreaterEqual((hkSimdReal_1-edgeEndpointTolerance)*denom) )
		{
			return false;
		}

		// now denom is > 0
		t.div(denom);

		const hkSimdReal u = (t*R-S2);
		if ( u.isLessEqual(edgeEndpointTolerance*D2) | u.isGreaterEqual((hkSimdReal_1-edgeEndpointTolerance)*D2) )
		{
			return false;
		}

		const hkSimdReal aa = vertexA_inA.getComponent( axisDimA) + d1_at_DimA * t;
		vertexA_inA.setComponent( axisDimA, aa );

		hkVector4 edgeA_inA;   edgeA_inA.setZero();	edgeA_inA.setComponent( axisDimA, d1_at_DimA * hkSimdReal_Inv2 );
		normal_inA.setCross( edgeA_inA, edgeB_inA );
	}

	// normalize our normal
	{
		const hkSimdReal normalLen = normal_inA.normalizeWithLength<3>();

		if( normalLen.isLess(m_boundaryTolerance) )	// parallel edges
		{
			return false;
		}
	}

	// check that we are within tolerance and update contact point if we are
	{
		// check the direction of the normal
		const hkSimdReal check = normal_inA.dot<3>( signsA );
		normal_inA.setFlipSign(normal_inA, check.lessZero()); 

		const hkSimdReal depthOut = normal_inA.dot<3>( d12 );
		if ( depthOut.isGreater(m_tolerance4.getComponent<0>()) )
		{
			return false;
		}

		fpp.m_vA = vertexA_inA;
		fpp.m_nA.setNeg<4>( normal_inA );
		fpp.m_distance = depthOut;
	}
	return true;
}


// calculate the normal for this feature pair
void hkpBoxBoxCollisionDetection::calcManifoldNormal( hkVector4& manifoldN, const hkpFeatureContactPoint& fcp, hkpFeaturePointCache &fpp, bool isPointValidated ) const
{
	hkVector4 normal = manifoldN;
	const int sepFeature = fpp.m_featureIndexA;
	hkVector4Comparison notFlipped; notFlipped.setNot(fpp.m_normalIsFlipped);

	if( sepFeature <= HK_BOXBOX_FACE_A_Z )
	{
		const int featureIndex = fcp.m_featureIdA;
		normal = hkVector4::getConstant((hkVectorConstant)(HK_QUADREAL_1000 + featureIndex));
		normal.setFlipSign(normal, notFlipped);
	}
	else if( sepFeature <= HK_BOXBOX_FACE_B_Z )
	{
		const int featureIndex = fcp.m_featureIdA - HK_BOXBOX_FACE_B_X;
		normal.setFlipSign( m_aTb.getRotation().getColumn( featureIndex ), notFlipped );
	}
	else  // it's an edge-edge collision
	{
		if( isPointValidated )
		{
			HK_ASSERT(0x7e8be737,  isValidEdgeEdge( fpp ) );
			normal = fpp.m_nA;
		}
	}

	manifoldN.setXYZ(normal);	// We need to preserve the w component.
}


void hkpBoxBoxCollisionDetection::refreshManifold( hkpBoxBoxManifold& manifold, hkSimdReal& minContactPointDistance ) const
{
// 	HK_MONITOR_ADD_VALUE("manifoldsize", manifold.getNumPoints(), HK_MONITOR_TYPE_INT);

	int i = 0;
	while( i < manifold.getNumPoints() )
	{
		const hkpFeatureContactPoint &fcp = manifold[i++];

		const int sepFeature = fcp.m_featureIdA;

		if( sepFeature <= HK_BOXBOX_FACE_A_Z )
		{

			hkpFeaturePointCache fpp;
			faceAVertexBValidationDataFromFeatureId( fpp, fcp );

			if (isValidFaceAVertexB( fpp ))
			{
				hkpProcessCdPoint& ccp = *m_result->reserveContactPoints(1);
				m_result->commitContactPoints(1);
				calcDistanceFaceAVertexB(fpp);
				minContactPointDistance.setMin( minContactPointDistance, fpp.m_distance );
				faceAVertexBContactPointFromFeaturePointCache( ccp, fcp, fpp );

				continue;
			}

		}
		else if( sepFeature <= HK_BOXBOX_FACE_B_Z )
		{

			hkpFeaturePointCache fpp;
			faceBVertexAValidationDataFromFeatureId( fpp, fcp );

			if (isValidFaceBVertexA( fpp ))
			{
				hkpProcessCdPoint& ccp = *m_result->reserveContactPoints(1);
				m_result->commitContactPoints(1);
				calcDistanceFaceBVertexA(fpp);
				minContactPointDistance.setMin( minContactPointDistance, fpp.m_distance );
				faceBVertexAContactPointFromFeaturePointCache( ccp, fcp, fpp );

				continue;
			}

		}
		else  // it's an edge-edge contact
		{

			hkpFeaturePointCache fpp;
			edgeEdgeValidationDataFromFeatureId( fpp, fcp );

			if (isValidEdgeEdge( fpp ))
			{
				minContactPointDistance.setMin( minContactPointDistance, fpp.m_distance );
				hkpProcessCdPoint& ccp = *m_result->reserveContactPoints(1);
				m_result->commitContactPoints(1);
				edgeEdgeContactPointFromFeaturePointCache( ccp, fcp, fpp );

				continue;
			}

		}

		//
		//	kill contact point if it is no longer valid
		//
		removePoint( manifold, --i );

	} // while manifold numCp

#ifdef HK_DEBUG
	debugCheckManifold( manifold, m_result );
#endif
}


hkBool32 hkpBoxBoxCollisionDetection::queryManifoldNormalConsistency( hkpBoxBoxManifold& manifold ) const
{

	if( manifold.getNumPoints() < 2 )
	{
		manifold.clearNormalInitialized();
		return true;
	}

	if( !manifold.isNormalInitialized() )
	{
		hkVector4 manifoldNormalB;	manifoldNormalB._setRotatedInverseDir( m_aTb.getRotation(), manifold.m_manifoldNormalA );
		manifold.setNormalInitialized();
		manifold.m_manifoldNormalB = hkVector4Util::packQuaternionIntoInt32( manifoldNormalB );
		return true;
	}

	hkVector4 manifoldNormalB;
	hkVector4Util::unPackInt32IntoQuaternion( manifold.m_manifoldNormalB, manifoldNormalB);

	// test if manifold has rotated enough to warrant a check for collision normal coherence
	// check manifold normal recorded in A's space to the one recorded in B's space
	hkVector4 manNormalBinA; manNormalBinA._setRotatedDir( m_aTb.getRotation(), manifoldNormalB );

	const hkSimdReal cosDelta = manifold.m_manifoldNormalA.dot<3>( manNormalBinA );
	return cosDelta.isLess(hkSimdReal::fromFloat(m_manifoldConsistencyCheckAngularCosTolerance));
}

void hkpBoxBoxCollisionDetection::checkManifoldNormalConsistency( hkpBoxBoxManifold& manifold ) const
{
	// loop through and check points
	hkVector4 manifoldNormalInWorld; manifoldNormalInWorld._setRotatedDir( m_wTa.getRotation(), manifold.m_manifoldNormalA );

	int iResult = m_result->getNumContactPoints() - manifold.getNumPoints();
	hkSimdReal contactNormalAngularCosTolerance; contactNormalAngularCosTolerance.load<1>(&m_contactNormalAngularCosTolerance);
	int i = 0;
	while( i < manifold.getNumPoints() )
	{
		const hkpProcessCdPoint& ccp = m_result->m_contactPoints[ iResult ];

		const hkSimdReal cosCheck = ccp.m_contact.getNormal().dot<3>( manifoldNormalInWorld );

		if( cosCheck.isLess(contactNormalAngularCosTolerance) )
		{
			m_result->m_contactPoints[iResult] = m_result->getEnd()[-1];
			removePoint( manifold, i );
			m_result->commitContactPoints(-1);
		}
		else
		{
			i++;
			iResult++;
		}

	}

	if( manifold.getNumPoints() < 2 )
	{
		// set not initialized
		manifold.clearNormalInitialized();
	}
	else
	{
		// finish up our manifold normal consistency check by resetting the check for relative rotation
		hkVector4 manifoldNormalB; manifoldNormalB._setRotatedInverseDir( m_aTb.getRotation(), manifold.m_manifoldNormalA );
		manifold.m_manifoldNormalB = hkVector4Util::packQuaternionIntoInt32(manifoldNormalB);
		manifold.setNormalInitialized();
	}
}


//!me just some sanity checks.  Depends on _DEBUG.   Should define my own MICHAELDEBUG or something.
#ifdef HK_DEBUG
void hkpBoxBoxCollisionDetection::debugCheckManifold( hkpBoxBoxManifold& manifold, hkpProcessCollisionOutput* result ) const
{
	for( int i = 0; i < m_result->getNumContactPoints(); i++ )
	{
		const hkpProcessCdPoint& ccp = m_result->m_contactPoints[i];
		hkReal len = ccp.m_contact.getNormal().length<3>().getReal();
		HK_ASSERT(0x7ee63fe7,  len > 0.99f && len < 1.01f );

		//hkVector4 maxRad; maxRad.setMax4( m_radiusA, m_radiusB );
		//hkReal maxExtent =  hkMath::max2( hkMath::max2( maxRad(0), maxRad(1) ), maxRad(2) );

		
		//HK_ASSERT3(0x286ed8b8,  ccp.m_distance < m_tolerance*1.01f && ccp.m_distance > -(m_tolerance*1.01f + maxExtent * 2.0f ), "Penetration depth reported, which is greated the extents of a box" << ccp.m_distance );

	}
}
#endif

//this public version also call checkIntersection internally !
hkBool hkpBoxBoxCollisionDetection::calculateClosestPoint( hkContactPoint& contact ) const
{
	hkpBoxBoxManifold manifold;
	hkpFeatureContactPoint fcp;
	hkpFeaturePointCache fpp;
	fpp.m_nA.setZero();
	fpp.m_normalIsFlipped.set<hkVector4ComparisonMask::MASK_NONE>();

	initWorkVariables();

	if( HK_SUCCESS == checkIntersection( m_tolerance4 )	)
	{

		int validCollision = findClosestPoint( manifold, fcp, fpp );

		if( validCollision == HK_FINDCLOSESTPOINT_VALID_POINT )
		{
			hkpProcessCdPoint ccp;

			contactPointFromFeaturePointCache( ccp, fcp, fpp );

			// face B vertex A and edge edge report points on surface of A, we want them on B
			if( fcp.m_featureIdA > HK_BOXBOX_FACE_A_Z )
			{
				hkVector4 cpPos; cpPos.setSubMul( ccp.m_contact.getPosition(), ccp.m_contact.getNormal(), ccp.m_contact.getDistanceSimdReal() );
				ccp.m_contact.setPosition(cpPos);
			}

			contact.setPositionNormalAndDistance( ccp.m_contact.getPosition(), ccp.m_contact.getNormal(), ccp.m_contact.getDistanceSimdReal() );

			return true;
		}
	}

	return false;
}


void hkpBoxBoxCollisionDetection::calcManifold( hkpBoxBoxManifold& manifold ) const
{
// 	HK_TIMER_BEGIN_LIST("boxbox","init");
	initWorkVariables();

// 	HK_TIMER_SPLIT_LIST("manifold");
	hkSimdReal minContactPointDistance = hkSimdReal_Max;
	refreshManifold( manifold, minContactPointDistance );

// 	HK_TIMER_SPLIT_LIST("consistency");
	hkBool32 requestManifoldNormal = queryManifoldNormalConsistency( manifold );

	// early exit check
	if( !requestManifoldNormal && manifold.isComplete() )
	{
// 		HK_TIMER_END_LIST();
		return;
	}

// 	HK_TIMER_SPLIT_LIST("checkInter");
	{
		hkResult pointsAvailable;
		if( manifold.getNumPoints() )
		{
			hkSimdReal minDist;
			minDist.setSubMul(minContactPointDistance, m_tolerance4.getComponent<0>(), hkSimdReal::fromFloat(0.05f) );
			hkVector4 minTol4; 
			minTol4.setAll( minDist );
			minTol4.setW( hkSimdReal_Max );
			pointsAvailable = checkIntersection( minTol4 );
		}
		else
		{
			pointsAvailable = checkIntersection( m_tolerance4 );
		}

// 		HK_TIMER_SPLIT_LIST("evalInter");

		if( pointsAvailable == HK_SUCCESS )
		{
			hkpFeatureContactPoint fcp;
			hkpFeaturePointCache fpp;
			fpp.m_nA.setZero();
			fpp.m_distance.setZero();
			fpp.m_normalIsFlipped.set<hkVector4ComparisonMask::MASK_NONE>();

// 			HK_TIMER_SPLIT_LIST("closestPoint");
			int validCollision = findClosestPoint( manifold, fcp, fpp );

			if( requestManifoldNormal && validCollision != HK_FINDCLOSESTPOINT_NO_VALID_POINT )
			{
// 				HK_TIMER_SPLIT_LIST("normal");
				hkBool32 manifoldNormalInitialized = manifold.isNormalInitialized();
				calcManifoldNormal( manifold.m_manifoldNormalA, fcp, fpp, ( validCollision == HK_FINDCLOSESTPOINT_VALID_POINT ) );
				if( manifoldNormalInitialized )
				{
					HK_ON_DEBUG( hkSimdReal normalLength = manifold.m_manifoldNormalA.lengthSquared<3>() );
					HK_ASSERT2( 0x541e454e, normalLength > hkSimdReal_Eps, "Manifold uninitialized");
					checkManifoldNormalConsistency( manifold );
				}
			}

			if( validCollision == HK_FINDCLOSESTPOINT_VALID_POINT )
			{
// 				HK_TIMER_SPLIT_LIST("addPoint");
				hkResult result = addPoint( manifold, fpp, fcp );

				if ( result == HK_SUCCESS)
				{
					// try to find more points once we have added a new one
// 					HK_TIMER_SPLIT_LIST("morePoints");
					findAdditionalManifoldPoints( manifold, fcp );
				}
			}
		}
	}

// 	HK_TIMER_END_LIST();

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
