/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COLLIDE2_GSK_UTIL_H
#define HKNP_COLLIDE2_GSK_UTIL_H

#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Math/Vector/hk4xVector2.h>

HK_COMPILE_TIME_ASSERT( hkVector4ComparisonMask::MASK_X <= sizeof(hkVector4) && hkVector4ComparisonMask::MASK_W <= sizeof(hkVector4));
class hkMonitorStream;

class hknpGskUtil
{
public:

	static void HK_CALL debugDisplayManifold( hknpManifold &manifold, const hkTransform& trA, const hkVector4* nearestOnAinA, const hkTransform& trB, const hkVector4* nearestOnBinB, hkVector4Parameter planeBinA );



	/// select from a list of input planes, shift them and extract the .w component
	static HK_FORCE_INLINE void selectAndTransform4Verts( const hkVector4* planesIn, hkUint8 vertsIndizes[4], const hkTransform* HK_RESTRICT trA, hkVector4Parameter shift, hkVector4* HK_RESTRICT vertsOut, hkVector4* HK_RESTRICT distancesOut )
	{
		hkVector4 position0 = planesIn[vertsIndizes[0]];
		hkVector4 position1 = planesIn[vertsIndizes[1]];
		hkVector4 position2 = planesIn[vertsIndizes[2]];
		hkVector4 position3 = planesIn[vertsIndizes[3]];

		hkMatrix4 rotPos; rotPos.setRows(position0, position1, position2, position3 ); // we only need the w, but the compiler should optimize the rest away
		hkVector4 distances = rotPos.getColumn<3>();

		//hkVector4 minDistance;  minDistance.setAll( normalInA.getSimdAt(3) );
		//distances.setMax( distances, minDistance );	// this clips the distances and should be removed

		position0.add( shift );
		position1.add( shift );
		position2.add( shift );
		position3.add( shift );

		position0._setTransformedPos( *trA, position0 );
		position1._setTransformedPos( *trA, position1 );
		position2._setTransformedPos( *trA, position2 );
		position3._setTransformedPos( *trA, position3 );

		*distancesOut	= distances;

		vertsOut[0] = position0;
		vertsOut[1] = position1;
		vertsOut[2] = position2;
		vertsOut[3] = position3;
	}

	static HK_FORCE_INLINE void HK_CALL findBestQuadImpl(hkVector4Parameter normal, int numContacts, const hkVector4* HK_RESTRICT contacts, hkVector4* quad)
	{
		// start from point 0 and search most distant point
		hkVector4 a = contacts[0];
		{
			hkVector4 v0 = contacts[1];
			hkVector4 v1 = contacts[2];
			hkVector4 bestSql0; bestSql0.setAll( v0.distanceToSquared( a ) );
			hkVector4 bestSql1; bestSql1.setAll( v1.distanceToSquared( a ) );
			int i=3;
			do {
				hkVector4 p0 = contacts[i];
				hkVector4 p1 = contacts[i+1];
				hkVector4 sql0; sql0.setAll( p0.distanceToSquared( a ) );
				hkVector4 sql1; sql1.setAll( p1.distanceToSquared( a ) );
				hkVector4Comparison isGreater0 = sql0.greater(bestSql0);
				hkVector4Comparison isGreater1 = sql1.greater(bestSql1);
				v0.setSelect( isGreater0, p0, v0 );
				v1.setSelect( isGreater1, p1, v1 );
				bestSql0.setSelect( isGreater0, sql0, bestSql0 );
				bestSql1.setSelect( isGreater1, sql1, bestSql1 );
				i+= 2;
			}
			while ( i < numContacts );
			a.setSelect( bestSql0.greater(bestSql1), v0, v1 );
			quad[0] = a;
		}

		// search the most distant point from point 0
		hkVector4 b;
		{
			hkVector4 v0 = a;
			hkVector4 v1 = a;
			hkVector4 bestSql0; bestSql0.setZero( );
			hkVector4 bestSql1; bestSql1.setZero();
			int i=0;
			do {
				hkVector4 p0 = contacts[i];
				hkVector4 p1 = contacts[i+1];
				hkVector4 sql0; sql0.setAll( p0.distanceToSquared( a ) );
				hkVector4 sql1; sql1.setAll( p1.distanceToSquared( a ) );
				hkVector4Comparison isGreater0 = sql0.greater(bestSql0);
				hkVector4Comparison isGreater1 = sql1.greater(bestSql1);
				v0.setSelect( isGreater0, p0, v0 );
				v1.setSelect( isGreater1, p1, v1 );
				bestSql0.setSelect( isGreater0, sql0, bestSql0 );
				bestSql1.setSelect( isGreater1, sql1, bestSql1 );
				i+= 2;
			}
			while ( i < numContacts );
			b.setSelect( bestSql0.greater(bestSql1), v0, v1 );
			quad[1] = b;
		}

		// Search largest triangle
		hkVector4 c;
		hkVector4 ab; 		ab.setSub(b, a);
		{
			hkVector4 v0 = b;
			hkVector4 v1 = b;
			{
				hkVector4 bestSql0; bestSql0.setZero();
				hkVector4 bestSql1; bestSql1.setZero();
				int i=0;
				do {
					hkVector4 p0 = contacts[i];
					hkVector4 p1 = contacts[i+1];
					hkVector4	ac0; ac0.setSub( a, p0 );
					hkVector4	ac1; ac1.setSub( a, p1 );
					hkVector4	crs0; crs0.setCross(ab,ac0);
					hkVector4	crs1; crs1.setCross(ab,ac1);
					hkVector4	sql0; sql0.setAll( crs0.lengthSquared<3>());
					hkVector4	sql1; sql1.setAll( crs1.lengthSquared<3>());

					hkVector4Comparison isGreater0 = sql0.greater(bestSql0);
					hkVector4Comparison isGreater1 = sql1.greater(bestSql1);
					v0.setSelect(  isGreater0, p0, v0 );
					v1.setSelect(  isGreater1, p1, v1 );
					bestSql0.setSelect( isGreater0, sql0, bestSql0 );
					bestSql1.setSelect( isGreater1, sql1, bestSql1 );
					i+= 2;
				}
				while ( i < numContacts );
				c.setSelect( bestSql0.greater(bestSql1), v0, v1 );
			}
			quad[2] = c;		//
		}

		/* Build planes							*/
		hkVector4 d;
		{
			hkVector4	bc; bc.setSub(c,b);
			hkVector4	ca; ca.setSub(a,c);
			hkVector4	n; n.setCross(ab,bc);
			hkTransform	planes;
			planes.getColumn(0).setCross(ab,n);
			planes.getColumn(1).setCross(bc,n);
			planes.getColumn(2).setCross(ca,n);
			planes.getColumn(3).setZero();

			hkVector4 offsets; hkVector4Util::dot3_3vs3( planes.getColumn(0),a,  planes.getColumn(1),b,  planes.getColumn(2),c, offsets );

			planes.getRotation()._setTranspose( planes.getRotation() );
			planes.getColumn(3).setNeg<4>( offsets );

			/* Pick farthest away					*/
			hkVector4 v0 = contacts[0];
			hkVector4 v1 = contacts[1];
			hkVector4 bestDist0; bestDist0.setZero();
			hkVector4 bestDist1; bestDist1.setZero();
			int i=0;
			do {
				hkVector4  p0 = contacts[i];
				hkVector4  p1 = contacts[i+1];
				hkVector4  dist0;  dist0._setTransformedPos( planes, p0 );
				hkVector4  dist1;  dist1._setTransformedPos( planes, p1 );
				hkVector4  vDist0; vDist0.setHorizontalMax<4>( dist0 );
				hkVector4  vDist1; vDist1.setHorizontalMax<4>( dist1 );

				hkVector4Comparison isGreater0 = vDist0.greater(bestDist0);
				hkVector4Comparison isGreater1 = vDist1.greater(bestDist1);
				v0.setSelect(  isGreater0, p0, v0 );
				v1.setSelect(  isGreater1, p1, v1 );
				bestDist0.setSelect( isGreater0, vDist0, bestDist0 );
				bestDist1.setSelect( isGreater1, vDist1, bestDist1 );
				i+= 2;
			}
			while ( i < numContacts );
			d.setSelect( bestDist0.greater(bestDist1), v0, v1 );
			quad[3] = d;		//
		}
	}

	HK_FORCE_INLINE static int findBestQuad( int numContacts, hkVector4 * HK_RESTRICT compressedVerts, hkMonitorStream& timerStream, hkVector4Parameter normalInA,
		hkUint8* HK_RESTRICT bestQuadOut, hkBool32& forceIncludeGskPointOut )
	{
		if(numContacts <= 4)
		{
			int index0 = compressedVerts[0].getInt16W();
			bestQuadOut[0] = hkUint8(index0);
			bestQuadOut[2] = hkUint8(index0);
			bestQuadOut[3] = hkUint8(index0);
			if(numContacts>2)	// 3 or 4
			{
				bestQuadOut[1] = hkUint8(compressedVerts[1].getInt16W());
				bestQuadOut[2] = hkUint8(compressedVerts[2].getInt16W());
				bestQuadOut[3] = hkUint8(compressedVerts[numContacts-1].getInt16W());
			}
			else				// 1 or 2
			{
				bestQuadOut[1] = hkUint8(compressedVerts[numContacts-1].getInt16W());
				//				forceIncludeGskPointOut = true;
			}
		}
		else
		{
			hkVector4 bestQuad[4];
			findBestQuadImpl(normalInA, numContacts, compressedVerts, bestQuad );
			int i0 = bestQuad[0].getInt16W();
			int i1 = bestQuad[1].getInt16W();
			int i2 = bestQuad[2].getInt16W();
			int i3 = bestQuad[3].getInt16W();
			bestQuadOut[0] = hkUint8(i0);
			bestQuadOut[1] = hkUint8(i1);
			bestQuadOut[2] = hkUint8(i2);
			bestQuadOut[3] = hkUint8(i3);
			numContacts = 4;
		}
		return numContacts;
	}

	static HK_FORCE_INLINE int HK_CALL findBestQuad2D_isFlat(
		const hk4xVector2& gradientDir, int numContacts,
		const hkReal* HK_RESTRICT contacts_u, const hkReal* HK_RESTRICT contacts_v, const hkVector4Comparison * activePoints,
		HK_PAD_ON_SPU(int) * HK_RESTRICT quadOut)
		{
		hkIntVector zero; zero.setZero();
		hkIntVector maxGradientDirIndices; maxGradientDirIndices.splatImmediate32<1>(); maxGradientDirIndices.setSubU32(zero, maxGradientDirIndices); // init with -1
		hkIntVector minGradientDirIndices; // no need to initialize, since we always should find a point
		hkIntVector maxOrthoDirIndices;
		hkIntVector minOrthoDirIndices;
		hkVector4 maxGradientDirDots; maxGradientDirDots.setAll(-hkSimdReal_Max);
		hkVector4 minGradientDirDots; minGradientDirDots.setAll(hkSimdReal_Max);
		hkVector4 maxOrthoDirDots = maxGradientDirDots;
		hkVector4 minOrthoDirDots = minGradientDirDots;

		hkIntVector curIndices = hkIntVector::getConstant<HK_QUADINT_0123>();

		int i_div4 = 0;
		for(int i=0; i<numContacts; i += 4)
		{
			hk4xVector2 contacts4; contacts4.m_x.load<4>( &contacts_u[i] ); contacts4.m_y.load<4>( &contacts_v[i] );
			const hkVector4Comparison& activePoints4 = activePoints[i_div4];
			hkVector4 dotsGradientDir;	hk4xVector2::dot(gradientDir, contacts4, dotsGradientDir);
			hkVector4 dotsOrthoDir;		hk4xVector2::dotPerpendicular(gradientDir, contacts4, dotsOrthoDir);

			{
				hkVector4Comparison cmpGradientDirMax = dotsGradientDir.greater(maxGradientDirDots);
				cmpGradientDirMax.setAnd(cmpGradientDirMax, activePoints4);
				maxGradientDirIndices.setSelect(cmpGradientDirMax, curIndices, maxGradientDirIndices);
				maxGradientDirDots.   setSelect(cmpGradientDirMax, dotsGradientDir, maxGradientDirDots);
			}

			{
				hkVector4Comparison cmpGradientDirMin = dotsGradientDir.less(minGradientDirDots);
				cmpGradientDirMin.setAnd(cmpGradientDirMin, activePoints4);
				minGradientDirIndices.setSelect(cmpGradientDirMin, curIndices, minGradientDirIndices);
				minGradientDirDots.   setSelect(cmpGradientDirMin, dotsGradientDir, minGradientDirDots);
			}

			{
				hkVector4Comparison cmpOrthoDirMax = dotsOrthoDir.greater(maxOrthoDirDots);
				cmpOrthoDirMax.setAnd(cmpOrthoDirMax, activePoints4);
				maxOrthoDirIndices.setSelect(cmpOrthoDirMax, curIndices, maxOrthoDirIndices);
				maxOrthoDirDots.setSelect(cmpOrthoDirMax, dotsOrthoDir, maxOrthoDirDots);
			}

			{
				hkVector4Comparison cmpOrthoDirMin = dotsOrthoDir.less(minOrthoDirDots);
				cmpOrthoDirMin.setAnd(cmpOrthoDirMin, activePoints4);
				minOrthoDirIndices.setSelect(cmpOrthoDirMin, curIndices, minOrthoDirIndices);
				minOrthoDirDots.   setSelect(cmpOrthoDirMin, dotsOrthoDir, minOrthoDirDots);
			}

			curIndices.setAddU32(curIndices, hkIntVector::getConstant<HK_QUADINT_4>());
			i_div4++;
		}

		hkVector4 negMinGradientDirDots; negMinGradientDirDots.setNeg<4>(minGradientDirDots);
		int maxGradientDirIndex = maxGradientDirIndices.getComponentAtVectorMax(maxGradientDirDots);
		int minGradientDirIndex = minGradientDirIndices.getComponentAtVectorMax(negMinGradientDirDots); // effectively getComponentAtVectorMin(minGradientDirDots)

		hkVector4 negMinOrthoDirDots; negMinOrthoDirDots.setNeg<4>(minOrthoDirDots);
		int maxOrthoDirIndex = maxOrthoDirIndices.getComponentAtVectorMax(maxOrthoDirDots);
		int minOrthoDirIndex = minOrthoDirIndices.getComponentAtVectorMax(negMinOrthoDirDots); // effectively getComponentAtVectorMin(minOrthoDirDots)

		quadOut[0] = maxGradientDirIndex;
		quadOut[1] = minGradientDirIndex;
		quadOut[2] = maxOrthoDirIndex;
		quadOut[3] = minOrthoDirIndex;
		if ( maxGradientDirIndex >= 0 )
		{
			return 4;
		}
		return 0;
	}

	static HK_FORCE_INLINE int HK_CALL findBestQuad2D_isNotFlat(
		const hk4xVector2& gradientDir, int numContacts,
		const hk4xVector2* HK_RESTRICT contacts, const hkVector4Comparison * activePoints,
		HK_PAD_ON_SPU(int)* HK_RESTRICT quadOut)
	{
		hkIntVector maxGradientDirIndices; // init with -1
		{
			hkIntVector zero; zero.setZero();
			maxGradientDirIndices.splatImmediate32<1>();
			maxGradientDirIndices.setSubU32(zero, maxGradientDirIndices);
		}
		hkIntVector minGradientDirIndices;
		hkVector4 maxGradientDirDots; maxGradientDirDots.setAll(-hkSimdReal_Max);
		hkVector4 minGradientDirDots; minGradientDirDots.setAll(hkSimdReal_Max);

		hkIntVector curIndices = hkIntVector::getConstant<HK_QUADINT_0123>();
		int i_div4 = 0;
		for(int i=0; i<numContacts; i += 4)
		{
			hk4xVector2 contacts4 = contacts[i_div4];
			hkVector4Comparison activePoints4 = activePoints[i_div4];
			hkVector4 dotsGradientDir;	hk4xVector2::dot(gradientDir, contacts4, dotsGradientDir);

			hkVector4Comparison cmpGradientDirMax = dotsGradientDir.greater(maxGradientDirDots);
			cmpGradientDirMax.setAnd(cmpGradientDirMax, activePoints4);

			maxGradientDirIndices.setSelect(cmpGradientDirMax, curIndices, maxGradientDirIndices);
			maxGradientDirDots.   setSelect(cmpGradientDirMax, dotsGradientDir, maxGradientDirDots);

			hkVector4Comparison cmpGradientDirMin = dotsGradientDir.less(minGradientDirDots);
			cmpGradientDirMin.setAnd(cmpGradientDirMin, activePoints4);
			minGradientDirIndices.setSelect(cmpGradientDirMin, curIndices, minGradientDirIndices);
			minGradientDirDots.setSelect(cmpGradientDirMin, dotsGradientDir, minGradientDirDots);

			curIndices.setAddU32(curIndices, hkIntVector::getConstant<HK_QUADINT_4>());
			i_div4++;
		}
		maxGradientDirIndices.broadcastComponentAtVectorMax( maxGradientDirDots );

		hkVector4 negMinGradientDirDots; negMinGradientDirDots.setNeg<4>(minGradientDirDots);
		minGradientDirIndices.broadcastComponentAtVectorMax( negMinGradientDirDots ); // effectively getComponentAtVectorMin(minGradientDirDots)

		int maxGradientDirIndex = maxGradientDirIndices.getU32(0);
		int minGradientDirIndex = minGradientDirIndices.getU32(0); // effectively getComponentAtVectorMin(minGradientDirDots)

		if ( HK_VERY_UNLIKELY(maxGradientDirIndex < 0) )
		{
			return 0;
		}
		quadOut[0] = (hkUint8) maxGradientDirIndex;
		quadOut[2] = (hkUint8) minGradientDirIndex;

		int arrayIndex = maxGradientDirIndex >> 2;
		int componentIndex = maxGradientDirIndex & 0x3;
		hkSimdReal u; u.load<1>( (&contacts[arrayIndex].m_x(0)) + componentIndex);
		hkSimdReal v; v.load<1>( (&contacts[arrayIndex].m_y(0)) + componentIndex);
		hk4xVector2 p0; p0.m_x.setAll(u); p0.m_y.setAll(v);

		hkIntVector maxGradientDirIndices_neighbor1 = maxGradientDirIndices;	// initialize with best points
		hkIntVector maxGradientDirIndices_neighbor2 = maxGradientDirIndices;

		hkVector4 maxGradientDirDots_neighbor1; maxGradientDirDots_neighbor1.setAll(-hkSimdReal_Max);
		hkVector4 maxGradientDirDots_neighbor2; maxGradientDirDots_neighbor2.setAll(-hkSimdReal_Max);

		curIndices = hkIntVector::getConstant<HK_QUADINT_0123>();
		i_div4 = 0;
		for(int i=0; i<numContacts; i += 4)
		{
			hk4xVector2 contacts4; contacts4.setSub(contacts[i_div4], p0);
			hkVector4Comparison activePoints4 = activePoints[i_div4];
			hkVector4 dotsGradientDir;	hk4xVector2::dot(gradientDir, contacts4, dotsGradientDir);
			hkVector4 dotsOrthoDir;		hk4xVector2::dotPerpendicular(gradientDir, contacts4, dotsOrthoDir);

			{
				hkVector4Comparison cmpGradientDirMax_neighbor1 = dotsGradientDir.greater(maxGradientDirDots_neighbor1);
				cmpGradientDirMax_neighbor1.setAnd(cmpGradientDirMax_neighbor1, activePoints4);
				hkVector4Comparison isRight = dotsOrthoDir.greaterZero();
				cmpGradientDirMax_neighbor1.setAnd(cmpGradientDirMax_neighbor1, isRight);

				maxGradientDirIndices_neighbor1.setSelect(cmpGradientDirMax_neighbor1, curIndices, maxGradientDirIndices_neighbor1);
				maxGradientDirDots_neighbor1.   setSelect(cmpGradientDirMax_neighbor1, dotsGradientDir, maxGradientDirDots_neighbor1);
			}
			{
				hkVector4Comparison cmpGradientDirMax_neighbor2 = dotsGradientDir.greater(maxGradientDirDots_neighbor2);
				cmpGradientDirMax_neighbor2.setAnd(cmpGradientDirMax_neighbor2, activePoints4);
				hkVector4Comparison isLeft = dotsOrthoDir.lessZero();
				cmpGradientDirMax_neighbor2.setAnd(cmpGradientDirMax_neighbor2, isLeft);

				maxGradientDirIndices_neighbor2.setSelect(cmpGradientDirMax_neighbor2, curIndices, maxGradientDirIndices_neighbor2);
				maxGradientDirDots_neighbor2.   setSelect(cmpGradientDirMax_neighbor2, dotsGradientDir, maxGradientDirDots_neighbor2);
			}

			curIndices.setAddU32(curIndices, hkIntVector::getConstant<HK_QUADINT_4>());
			i_div4++;
		}

		int maxGradientDirIndex_neighbor1 = maxGradientDirIndices_neighbor1.getComponentAtVectorMax(maxGradientDirDots_neighbor1);
		int maxGradientDirIndex_neighbor2 = maxGradientDirIndices_neighbor2.getComponentAtVectorMax(maxGradientDirDots_neighbor2);

		quadOut[1] = (hkUint8) maxGradientDirIndex_neighbor1;
		quadOut[3] = (hkUint8) maxGradientDirIndex_neighbor2;
		return 4;
	}

	static HK_FORCE_INLINE void HK_CALL getVertexAt( const hk4xVector2* HK_RESTRICT contacts, int index, hk4xVector2& pOut )
	{
		int arrayIndex = index >> 2;
		int componentIndex = index & 0x3;
		hkSimdReal u; u.load<1>( (&contacts[arrayIndex].m_x(0)) + componentIndex);
		hkSimdReal v; v.load<1>( (&contacts[arrayIndex].m_y(0)) + componentIndex);
		pOut.m_x.setAll(u); pOut.m_y.setAll(v);
	}

	static HK_FORCE_INLINE hkBool32 vertexIsValid( const hkVector4Comparison * activePoints, int index )
	{
		if ( hkSizeOf(hkVector4Comparison)  == 16)
		{
			hkUint32 mask = 0x80000000;
			return mask & (((hkUint32*)activePoints)[index]);
		}
		else
		{
			int comp = index>>2;
			int flag = index&3;
			int mask = 1<<flag;
			return mask & activePoints[comp].getMask();
		}
	}

	static HK_FORCE_INLINE int HK_CALL findBestQuad2D_isNotFlat2(
		const hkReal* HK_RESTRICT contacts_u, const hkReal* HK_RESTRICT contacts_v, int numContacts, const hkVector4Comparison * activePoints, const hkReal* distances,
		hkSimdRealParameter collisionAccuracy,
		HK_PAD_ON_SPU(int)* const HK_RESTRICT quadOut)
	{
		hkIntVector maxGradientDirIndices; // init with -1
		{
			hkIntVector zero; zero.setZero();
			maxGradientDirIndices.splatImmediate32<1>();
			maxGradientDirIndices.setSubU32(zero, maxGradientDirIndices);
		}

		hkVector4 maxGradient; maxGradient.setAll(-hkSimdReal_Max);

		// find the point with the smallest distance gradient
		hkIntVector curIndices = hkIntVector::getConstant<HK_QUADINT_0123>();
		hk4xVector2 point0; point0.m_x.setZero(); point0.m_y.setZero();
		{
			int i=0;
			do
			{
				hk4xVector2 contacts4; contacts4.m_x.load<4>(&contacts_u[i]); contacts4.m_y.load<4>(&contacts_v[i]);
				hkVector4 distance4; distance4.load<4>( &distances[i] );
				hkVector4Comparison activePoints4 = activePoints[i>>2];

				hkVector4 negDistance; negDistance.setNeg<4>( distance4 );
				hkVector4Comparison currentPointBetter = negDistance.greater(maxGradient);
				currentPointBetter.setAnd(currentPointBetter, activePoints4);

				maxGradientDirIndices.setSelect(currentPointBetter, curIndices,      maxGradientDirIndices);
				maxGradient.   setSelect(currentPointBetter, negDistance, maxGradient);
				point0.setSelect( currentPointBetter, contacts4, point0 );

				curIndices.setAddU32(curIndices, hkIntVector::getConstant<HK_QUADINT_4>());
				i += 4;
			}
			while( i < numContacts );
		}
		hkVector4Util::setXAtVectorMax( maxGradient, maxGradientDirIndices, point0.m_x, point0.m_y );
		point0.m_x.broadcast<0>();
		point0.m_y.broadcast<0>();
		maxGradientDirIndices.store<1>(&HK_PADSPU_REF(quadOut[0]));
		maxGradientDirIndices.setBroadcast<0>(maxGradientDirIndices);


		// find the point which is furthest away from point 0 (and not the biggest distance)
		hkIntVector otherSideIndices; otherSideIndices.setZero();
		hkVector4 otherSideDots; otherSideDots.setAll(-hkSimdReal_Max);
		curIndices = hkIntVector::getConstant<HK_QUADINT_0123>();
		hk4xVector2 point1; point1.m_x.setZero(); point1.m_y.setZero();
		{
			int i=0;
			do
			{
				hk4xVector2 contacts4; contacts4.m_x.load<4>(&contacts_u[i]); contacts4.m_y.load<4>(&contacts_v[i]);
				hkVector4 distance4; distance4.load<4>( &distances[i] );
				hk4xVector2 diff; diff.setSub( contacts4, point0 );
				hkVector4Comparison activePoints4 = activePoints[i>>2];
				hkVector4 distance;	hk4xVector2::dot(diff, diff, distance);

				hkVector4Comparison currentPointBetter = distance.greater(otherSideDots);
				currentPointBetter.setAnd(currentPointBetter, activePoints4);
				otherSideIndices.setSelect(currentPointBetter, curIndices, otherSideIndices);
				otherSideDots.setSelect   (currentPointBetter, distance, otherSideDots);
				point1.setSelect( currentPointBetter, contacts4, point1 );

				curIndices.setAddU32(curIndices, hkIntVector::getConstant<HK_QUADINT_4>());
				i += 4;
			}
			while( i < numContacts );
		}

		if ( quadOut[0] < 0 )
		{
			return 0;
		}

		hkVector4Util::setXAtVectorMax( otherSideDots, otherSideIndices, point1.m_x, point1.m_y );

		point1.m_x.broadcast<0>();
		point1.m_y.broadcast<0>();
		otherSideIndices.store<1>(&HK_PADSPU_REF(quadOut[1]));

		// build the new direction
		hk4xVector2 p0MinusP1;	p0MinusP1.setSub( point0, point1 );

		hkIntVector maxGradientDirIndices_neighbor1 = maxGradientDirIndices;	// initialize with best points
		hkIntVector maxGradientDirIndices_neighbor2 = maxGradientDirIndices;

		hkVector4 maxGradientDirDots_neighbor1; maxGradientDirDots_neighbor1.setZero();
		hkVector4 maxGradientDirDots_neighbor2; maxGradientDirDots_neighbor2.setZero();

		curIndices = hkIntVector::getConstant<HK_QUADINT_0123>();
		hkVector4 collisionAccuracy4; collisionAccuracy4.setAll( collisionAccuracy * hkSimdReal_2 );
		hkSimdReal closestDistance; closestDistance.load<1>( &distances[ quadOut[0] ]);
		hkVector4 closestDistance4; closestDistance4.setAll( closestDistance );

		{	// maximize/minimize:   ( distance to diagonal ) / ( extraContactDistance + collisionAccuracy )
			int i = 0;
			do
			{
				hk4xVector2 contacts4; contacts4.m_x.load<4>(&contacts_u[i]); contacts4.m_y.load<4>(&contacts_v[i]);
				contacts4.setSub( contacts4, point0 );

				hkVector4 distance4; distance4.load<4>( &distances[i] );
				distance4.sub( closestDistance4 );
				distance4.add( collisionAccuracy4 );
				hkVector4 invDist4; invDist4.setReciprocal<HK_ACC_12_BIT, HK_DIV_IGNORE>( distance4 );

				const hkVector4Comparison& activePoints4 = activePoints[i>>2];
				hkVector4 dotsOrthoDir;		hk4xVector2::dotPerpendicular(p0MinusP1, contacts4, dotsOrthoDir);
				dotsOrthoDir.mul(invDist4);
				{
					hkVector4Comparison cmpGradientDirMax_neighbor1 = dotsOrthoDir.greater(maxGradientDirDots_neighbor1);
					cmpGradientDirMax_neighbor1.setAnd(cmpGradientDirMax_neighbor1, activePoints4);

					maxGradientDirIndices_neighbor1.setSelect(cmpGradientDirMax_neighbor1, curIndices, maxGradientDirIndices_neighbor1);
					maxGradientDirDots_neighbor1.   setSelect(cmpGradientDirMax_neighbor1, dotsOrthoDir, maxGradientDirDots_neighbor1);
				}
				{
					hkVector4Comparison cmpGradientDirMax_neighbor2 = dotsOrthoDir.less(maxGradientDirDots_neighbor2);
					cmpGradientDirMax_neighbor2.setAnd(cmpGradientDirMax_neighbor2, activePoints4);

					maxGradientDirIndices_neighbor2.setSelect(cmpGradientDirMax_neighbor2, curIndices,   maxGradientDirIndices_neighbor2);
					maxGradientDirDots_neighbor2.   setSelect(cmpGradientDirMax_neighbor2, dotsOrthoDir, maxGradientDirDots_neighbor2);
				}

				curIndices.setAddU32(curIndices, hkIntVector::getConstant<HK_QUADINT_4>());
				i += 4;
			}
			while( i < numContacts );
		}

		maxGradientDirIndices_neighbor1.broadcastComponentAtVectorMax(maxGradientDirDots_neighbor1);
		maxGradientDirDots_neighbor2.setNeg<4>(maxGradientDirDots_neighbor2);
		maxGradientDirIndices_neighbor2.broadcastComponentAtVectorMax(maxGradientDirDots_neighbor2);

		maxGradientDirIndices_neighbor1.store<1>(&HK_PADSPU_REF(quadOut[2]));
		maxGradientDirIndices_neighbor2.store<1>(&HK_PADSPU_REF(quadOut[3]));
		if (quadOut[1] == quadOut[0])
		{
			return 1;
		}

		// if we have less than 4 points, simply find another random not used yet and valid point.
		int numHits = 4;
		{
			if (quadOut[2] == quadOut[3] || quadOut[0] == quadOut[3] || quadOut[1] == quadOut[3] )
			{
				numHits = 3;
			}
			if (quadOut[0] == quadOut[2] || quadOut[1] == quadOut[2] )
			{
				quadOut[2] = quadOut[3];
				numHits--;
			}
			if ( numHits < 4 )
			{
				// slow try to find another point
				for (int i =0; i < numContacts; i++ )
				{
					if ( vertexIsValid(activePoints, i) )
					{
						// check id
						if ( i == quadOut[0] || i == quadOut[1] || i == quadOut[2] )
						{
							continue;
						}

						// check floating point equality
						hkReal u = contacts_u[i];
						hkReal v = contacts_v[i];
						if (	(u==contacts_u[ quadOut[0] ] && v==contacts_v[ quadOut[0] ])
							||	(u==contacts_u[ quadOut[1] ] && v==contacts_v[ quadOut[1] ])
							||	(u==contacts_u[ quadOut[2] ] && v==contacts_v[ quadOut[2] ])
							)
						{
							continue;
						}
						quadOut[ numHits++ ] = i;
						if ( numHits == 4 )
						{
							break;
						}
					}
				}
			}
		}

		return numHits;
	}
};

#endif // HKNP_COLLIDE2_GSK_UTIL_H

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
