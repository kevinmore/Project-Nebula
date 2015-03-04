/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

#include <Geometry/Internal/Algorithms/Intersect/hkcdIntersectRayAabb.h>
#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastTriangle.h>
#include <Geometry/Internal/Algorithms/Distance/hkcdDistanceAabbAabb.h>

#include <Physics/Physics/Collide/Query/hknpQueryAabbNmpUtil.h>

namespace hknpHeightFieldShapeUtils
{

	static HK_FORCE_INLINE hknpShapeKey shapeKeyFromCoordinates(int bitsPerX, int x, int z, int triIndex)
	{
		HK_ASSERT(0x14482424, triIndex==0 ||  triIndex==1);
		HK_ASSERT(0x14482423, x>=0 && z>=0 && x<32768 && z<32768);
		return hknpShapeKey( (z << (1+bitsPerX)) + (x << 1) + triIndex );
	}


	static HK_FORCE_INLINE void coordinatesFromShapeKey(hknpShapeKey key, int bitsPerX,  int* xOut, int* zOut, int* triIndexOut)
	{
		hkUint32 keyInt = (hkUint32)key;
		*triIndexOut = keyInt & 1;
		*xOut = (keyInt>>1) & ((1<<(bitsPerX))-1);
		*zOut = keyInt>>(1+bitsPerX);
	}


	static HK_FORCE_INLINE void fillQueryResult(
		hknpCollisionResult* resultOut, hknpCollisionQueryType::Enum queryType, hkVector4Parameter pos, hkVector4Parameter normal,
		const hknpRayCastQuery& ray, const hknpShapeQueryInfo& shapeInfo,
		hkSimdRealParameter hitFraction, hknpShapeKey shapeKey, int bitsPerKey, const hknpQueryFilterData& shapeFilterData, hkInt32 hitResult )
	{
		resultOut->m_queryType = queryType;

		resultOut->m_position	. setTransformedPos( *shapeInfo.m_shapeToWorld, pos );
		resultOut->m_normal		. setRotatedDir( shapeInfo.m_shapeToWorld->getRotation(), normal );
		resultOut->m_fraction	= hitFraction.getReal();

		resultOut->m_queryBodyInfo.m_shapeMaterialId = ray.m_filterData.m_materialId;
		resultOut->m_queryBodyInfo.m_shapeCollisionFilterInfo = ray.m_filterData.m_collisionFilterInfo;
		resultOut->m_queryBodyInfo.m_shapeUserData = ray.m_filterData.m_userData;

		resultOut->m_hitBodyInfo.m_bodyId = ( shapeInfo.m_body ? shapeInfo.m_body->m_id : hknpBodyId::INVALID );
		resultOut->m_hitBodyInfo.m_shapeKey = shapeKey;
		resultOut->m_hitBodyInfo.m_shapeMaterialId = shapeFilterData.m_materialId;
		resultOut->m_hitBodyInfo.m_shapeCollisionFilterInfo = shapeFilterData.m_collisionFilterInfo;
		resultOut->m_hitBodyInfo.m_shapeUserData = shapeFilterData.m_userData;

		resultOut->m_hitResult = hitResult;
	}


	static HK_FORCE_INLINE void rayCastFindStartCell(
		QuadTreeWalkerStackElement& startOut, hkVector4Parameter from, hkVector4Parameter to, int sizeX, int sizeZ, hkReal minHeight, hkReal maxHeight )
	{
		// When we descend the quadtree it is like checking the bits of the integer positions (msb first).
		// So if we now how many bits are constant along the ray (both x and z) we can skip directly to that cell in the tree
		int x1 = hkMath::clamp((int)from(0), 0, sizeX-1);
		int z1 = hkMath::clamp((int)from(2), 0, sizeZ-1);
		int x2 = hkMath::clamp((int)to(0), 0, sizeX-1);
		int z2 = hkMath::clamp((int)to(2), 0, sizeZ-1);
		hkUint32 leadingEqualBitsX = hkMath::countLeadingZeros(x1 ^ x2);
		hkUint32 leadingEqualBitsZ = hkMath::countLeadingZeros(z1 ^ z2);
		int level = 32 - hkMath::min2(leadingEqualBitsX, leadingEqualBitsZ);
		int x = x1>>level;
		int z = z1>>level;
		int gridSize = 1<<level;
		startOut.m_aabb.m_min.set(hkReal(x*gridSize), minHeight, hkReal(z*gridSize));
		startOut.m_aabb.m_max.set(hkReal((x+1)*gridSize), maxHeight, hkReal((z+1)*gridSize));
		startOut.m_level = level;
		startOut.m_x = x;
		startOut.m_z = z;
	}

	HK_FORCE_INLINE static void createRotationMasks(
		const hkcdRay& ray, int& rotOffsetOut, hkVector4Comparison& mask0Out, hkVector4Comparison& mask1Out, hkVector4Comparison& mask2Out, hkVector4Comparison& mask3Out )
	{
		hkVector4Comparison comp = ray.getDirection().lessZero();
		int signs = comp.getMask() & hkVector4ComparisonMask::MASK_XZ;

		const hkVector4Comparison::Mask minMasks[] = {
			hkVector4ComparisonMask::MASK_NONE,
			hkVector4ComparisonMask::MASK_X,
			hkVector4ComparisonMask::MASK_XZ,
			hkVector4ComparisonMask::MASK_Z,
		};

		rotOffsetOut=0;
		switch (signs)
		{
		case hkVector4ComparisonMask::MASK_NONE:
			{
				rotOffsetOut = 0;
				break;
			}
		case hkVector4ComparisonMask::MASK_X:
			{
				rotOffsetOut = 1;
				break;
			}
		case hkVector4ComparisonMask::MASK_Z:
			{
				rotOffsetOut = 3;
				break;
			}
		case hkVector4ComparisonMask::MASK_XZ:
			{
				rotOffsetOut = 2;
				break;
			}
		}

		mask0Out.set(minMasks[ rotOffsetOut		]);
		mask1Out.set(minMasks[(rotOffsetOut+1)&0x3]);
		mask2Out.set(minMasks[(rotOffsetOut+2)&0x3]);
		mask3Out.set(minMasks[(rotOffsetOut+3)&0x3]);
	}


	static void HK_FORCE_INLINE descendQuadTreeWithRay(
		QuadTreeWalkerStackElement* stack, int* stackSize,
		const hkAabb& aabb, hkVector4Parameter minY, hkVector4Parameter maxY,
		int x, int z, int nextlevel,
		const hkcdRay& ray, int rotOffset, hkSimdRealParameter earlyOutHitFraction,
		hkVector4ComparisonParameter mask0, hkVector4ComparisonParameter mask1, hkVector4ComparisonParameter mask2, hkVector4ComparisonParameter mask3 )
	{
		// Divide the aabb into four sub AABBs. In the case where both ray.m_dir.x and ray.m_dir.z are positive this corresponds to:
		// Z ^	  3 | 2
		//   |	  --|--
		//   |	  0 | 1
		//   |
		//    ------> X
		// The ray order will always be 0<(1/3)<2
		(*stackSize)--;

		hkVector4 center;
		aabb.getCenter(center);

		hkAabb aabb0;
		aabb0.m_min.setSelect(mask0, center, aabb.m_min);
		aabb0.m_max.setSelect(mask0, aabb.m_max, center);
		aabb0.m_min.setComponent<1>(minY.getComponent<0>());
		aabb0.m_max.setComponent<1>(maxY.getComponent<0>());

		hkAabb aabb1;
		aabb1.m_min.setSelect(mask1, center, aabb.m_min);
		aabb1.m_max.setSelect(mask1, aabb.m_max, center);
		aabb1.m_min.setComponent<1>(minY.getComponent<1>());
		aabb1.m_max.setComponent<1>(maxY.getComponent<1>());

		hkAabb aabb2;
		aabb2.m_min.setSelect(mask2, center, aabb.m_min);
		aabb2.m_max.setSelect(mask2, aabb.m_max, center);
		aabb2.m_min.setComponent<1>(minY.getComponent<2>());
		aabb2.m_max.setComponent<1>(maxY.getComponent<2>());

		hkAabb aabb3;
		aabb3.m_min.setSelect(mask3, center, aabb.m_min);
		aabb3.m_max.setSelect(mask3, aabb.m_max, center);
		aabb3.m_min.setComponent<1>(minY.getComponent<3>());
		aabb3.m_max.setComponent<1>(maxY.getComponent<3>());

		// Check all AABBs for ray collision
		int hits = hkcdIntersectRayBundleAabb(ray, aabb0, aabb1, aabb2, aabb3, earlyOutHitFraction);

		// We want to visit the AABBs in ray-order so we reverse the order when we put them on the stack (2 -> (1,3) -> 0)
		if (hits&hkVector4ComparisonMask::MASK_Z)
		{
			QuadTreeWalkerStackElement& newElement = stack[(*stackSize)++];
			newElement.m_level = nextlevel;
			newElement.m_x = 2*x+(((rotOffset+3)&0x2)>>1);
			newElement.m_z = 2*z+(((rotOffset+2)&0x2)>>1);
			newElement.m_aabb = aabb2;
		}
		if (hits&hkVector4ComparisonMask::MASK_Y)
		{
			QuadTreeWalkerStackElement& newElement = stack[(*stackSize)++];
			newElement.m_level = nextlevel;
			newElement.m_x = 2*x+(((rotOffset+2)&0x2)>>1);
			newElement.m_z = 2*z+(((rotOffset+1)&0x2)>>1);
			newElement.m_aabb = aabb1;
		}
		else if (hits&hkVector4ComparisonMask::MASK_W) // A ray cannot hit both aabb1 and aabb3
		{
			QuadTreeWalkerStackElement& newElement = stack[(*stackSize)++];
			newElement.m_level = nextlevel;
			newElement.m_x = 2*x+(((rotOffset+4)&0x2)>>1);
			newElement.m_z = 2*z+(((rotOffset+3)&0x2)>>1);
			newElement.m_aabb = aabb3;
		}
		if (hits&hkVector4ComparisonMask::MASK_X)
		{
			QuadTreeWalkerStackElement& newElement = stack[(*stackSize)++];
			newElement.m_level = nextlevel;
			newElement.m_x = 2*x+(((rotOffset+1)&0x2)>>1);
			newElement.m_z = 2*z+(((rotOffset+0)&0x2)>>1);
			newElement.m_aabb = aabb0;
		}
	}


	static HK_FORCE_INLINE void sortByX(hkVector4* A, hkVector4* B)
	{
		hkVector4 a = *A;
		hkVector4 b = *B;
		hkVector4Comparison comp = a.getComponent<0>().less(b.getComponent<0>());
		A->setSelect(comp, b, a);
		B->setSelect(comp, a, b);
	}


	// Like descendQuadTreeWithRay, but extended ray (with an aabb halfextent)
	template<bool SCALED>
	static void HK_FORCE_INLINE descendQuadTreeWithExtendedRay(
		QuadTreeWalkerStackElement* stack, int* stackSize,
		const hkAabb& aabb, hkVector4Parameter minY, hkVector4Parameter maxY,
		int x, int z, int nextlevel,
		const hkcdRay& ray,  hkVector4Parameter rayHalfExtent, hkSimdRealParameter earlyOutHitFraction,
		hkVector4Parameter intToFloatScale, const hkTransform& heightfieldToWorld )
	{
		(*stackSize)--;

		hkVector4 center;
		aabb.getCenter(center);

		hkVector4Comparison mask[4];
		mask[0].set(hkVector4ComparisonMask::MASK_NONE);
		mask[1].set(hkVector4ComparisonMask::MASK_X);
		mask[2].set(hkVector4ComparisonMask::MASK_XZ);
		mask[3].set(hkVector4ComparisonMask::MASK_Z);

		hkAabb subAabb[4];
		hkVector4 fractions[4];
		for (int i=0; i<4; i++)
		{
			subAabb[i].m_min.setSelect(mask[i], center, aabb.m_min);
			subAabb[i].m_max.setSelect(mask[i], aabb.m_max, center);
			subAabb[i].m_min.setComponent<1>(minY.getComponent(i));
			subAabb[i].m_max.setComponent<1>(maxY.getComponent(i));

			hkAabb expAabb;
			if ( SCALED )
			{
				// If the heightfield is 'externally' scaled we need to work in world space, so that the check against
				// the EarlyOutHitDistance (which is in world space and stored in the collector) works correctly.
				// In that case the ray and the rayHalfExtent will be in world space as well. Also the 'external' scale
				// will be baked into heightfieldToWorld already.
				hkAabb aabbWs;
				aabbWs.m_min.setMul( subAabb[i].m_min, intToFloatScale );
				aabbWs.m_max.setMul( subAabb[i].m_max, intToFloatScale );
				hkAabbUtil::calcAabb( heightfieldToWorld, aabbWs, expAabb );
				expAabb.m_max.add( rayHalfExtent );
				expAabb.m_min.sub( rayHalfExtent );
			}
			else
			{
				expAabb.m_max.setAdd(subAabb[i].m_max, rayHalfExtent);
				expAabb.m_min.setSub(subAabb[i].m_min, rayHalfExtent);
			}
			hkSimdReal fraction = earlyOutHitFraction;
			hkBool32 b = hkcdIntersectRayAabb(ray, expAabb, &fraction);
			fractions[i].setAll(fraction);
			fractions[i].setInt24W(b ? i : 8); // 8 arbitrary number larger than 4
		}

		//Sort them using a simple sorting network
		sortByX(&fractions[0],&fractions[2]);
		sortByX(&fractions[1],&fractions[3]);
		sortByX(&fractions[0],&fractions[1]);
		sortByX(&fractions[2],&fractions[3]);
		sortByX(&fractions[1],&fractions[2]);

		for (int i=0; i<4; i++)
		{
			int index = fractions[i].getInt16W();
			if (index>=4) continue;
			QuadTreeWalkerStackElement& newElement = stack[(*stackSize)++];
			newElement.m_level = nextlevel;
			newElement.m_x = 2*x+(((index+1)&0x2)>>1);
			newElement.m_z = 2*z+((index&0x2)>>1);
			newElement.m_aabb = subAabb[index];
		}
	}


	template<bool USE_NMP>
	static void HK_FORCE_INLINE descendQuadTreeWithAabb(
		QuadTreeWalkerStackElement* stack, int* stackSize,
		const hkAabb& aabb, hkVector4Parameter minY, hkVector4Parameter maxY,
		int x, int z, int nextlevel, const hkAabb& intAabb,
		hknpQueryAabbNmp* HK_RESTRICT nmpInOut )
	{
		(*stackSize)--;

		hkVector4 center;
		aabb.getCenter(center);

		hkVector4Comparison mask0; mask0.set(hkVector4ComparisonMask::MASK_NONE);
		hkVector4Comparison mask1; mask1.set(hkVector4ComparisonMask::MASK_X);
		hkVector4Comparison mask2; mask2.set(hkVector4ComparisonMask::MASK_XZ);
		hkVector4Comparison mask3; mask3.set(hkVector4ComparisonMask::MASK_Z);

		hkAabb aabb0;
		aabb0.m_min.setSelect(mask0, center, aabb.m_min);
		aabb0.m_max.setSelect(mask0, aabb.m_max, center);
		aabb0.m_min.setComponent<1>(minY.getComponent<0>());
		aabb0.m_max.setComponent<1>(maxY.getComponent<0>());

		hkAabb aabb1;
		aabb1.m_min.setSelect(mask1, center, aabb.m_min);
		aabb1.m_max.setSelect(mask1, aabb.m_max, center);
		aabb1.m_min.setComponent<1>(minY.getComponent<1>());
		aabb1.m_max.setComponent<1>(maxY.getComponent<1>());

		hkAabb aabb2;
		aabb2.m_min.setSelect(mask2, center, aabb.m_min);
		aabb2.m_max.setSelect(mask2, aabb.m_max, center);
		aabb2.m_min.setComponent<1>(minY.getComponent<2>());
		aabb2.m_max.setComponent<1>(maxY.getComponent<2>());

		hkAabb aabb3;
		aabb3.m_min.setSelect(mask3, center, aabb.m_min);
		aabb3.m_max.setSelect(mask3, aabb.m_max, center);
		aabb3.m_min.setComponent<1>(minY.getComponent<3>());
		aabb3.m_max.setComponent<1>(maxY.getComponent<3>());

		if ( USE_NMP )
		{
			if ( hknpQueryAabbNmpUtil::checkOverlapWithNmp(intAabb, aabb0, nmpInOut) )
			{
				QuadTreeWalkerStackElement& newElement = stack[(*stackSize)++];
				newElement.m_level = nextlevel;
				newElement.m_x = 2*x+0;
				newElement.m_z = 2*z+0;
				newElement.m_aabb = aabb0;
			}
			if ( hknpQueryAabbNmpUtil::checkOverlapWithNmp(intAabb, aabb1, nmpInOut) )
			{
				QuadTreeWalkerStackElement& newElement = stack[(*stackSize)++];
				newElement.m_level = nextlevel;
				newElement.m_x = 2*x+1;
				newElement.m_z = 2*z+0;
				newElement.m_aabb = aabb1;
			}
			if ( hknpQueryAabbNmpUtil::checkOverlapWithNmp(intAabb, aabb2, nmpInOut) )
			{
				QuadTreeWalkerStackElement& newElement = stack[(*stackSize)++];
				newElement.m_level = nextlevel;
				newElement.m_x = 2*x+1;
				newElement.m_z = 2*z+1;
				newElement.m_aabb = aabb2;
			}
			if ( hknpQueryAabbNmpUtil::checkOverlapWithNmp(intAabb, aabb3, nmpInOut) )
			{
				QuadTreeWalkerStackElement& newElement = stack[(*stackSize)++];
				newElement.m_level = nextlevel;
				newElement.m_x = 2*x+0;
				newElement.m_z = 2*z+1;
				newElement.m_aabb = aabb3;
			}
		}
		else
		{
			if ( intAabb.overlaps(aabb0) )
			{
				QuadTreeWalkerStackElement& newElement = stack[(*stackSize)++];
				newElement.m_level = nextlevel;
				newElement.m_x = 2*x+0;
				newElement.m_z = 2*z+0;
				newElement.m_aabb = aabb0;
			}
			if ( intAabb.overlaps(aabb1) )
			{
				QuadTreeWalkerStackElement& newElement = stack[(*stackSize)++];
				newElement.m_level = nextlevel;
				newElement.m_x = 2*x+1;
				newElement.m_z = 2*z+0;
				newElement.m_aabb = aabb1;
			}
			if ( intAabb.overlaps(aabb2) )
			{
				QuadTreeWalkerStackElement& newElement = stack[(*stackSize)++];
				newElement.m_level = nextlevel;
				newElement.m_x = 2*x+1;
				newElement.m_z = 2*z+1;
				newElement.m_aabb = aabb2;
			}
			if ( intAabb.overlaps(aabb3) )
			{
				QuadTreeWalkerStackElement& newElement = stack[(*stackSize)++];
				newElement.m_level = nextlevel;
				newElement.m_x = 2*x+0;
				newElement.m_z = 2*z+1;
				newElement.m_aabb = aabb3;
			}
		}
	}


	template<bool SCALED>
	static void HK_FORCE_INLINE descendQuadTreeWithClosestPoint(
		QuadTreeWalkerStackElement* stack, int* stackSize,
		const hkAabb& aabb, hkVector4Parameter minY, hkVector4Parameter maxY,
		int x, int z, int nextlevel, hkSimdRealParameter maxSquaredDist, hkVector4Parameter cvxHalfExtent, hkVector4Parameter cvxCenter, hkVector4Parameter intToFloatScale, const hkTransform& heightfieldToWorld )
	{
		(*stackSize)--;

		hkVector4 center;
		aabb.getCenter(center);

		hkVector4Comparison mask[4];
		mask[0].set(hkVector4ComparisonMask::MASK_NONE);
		mask[1].set(hkVector4ComparisonMask::MASK_X);
		mask[2].set(hkVector4ComparisonMask::MASK_XZ);
		mask[3].set(hkVector4ComparisonMask::MASK_Z);

		hkAabb subAabb[4];
		hkVector4 distSquared[4];
		for (int i=0; i<4; i++)
		{
			subAabb[i].m_min.setSelect(mask[i], center, aabb.m_min);
			subAabb[i].m_max.setSelect(mask[i], aabb.m_max, center);
			subAabb[i].m_min.setComponent<1>(minY.getComponent(i));
			subAabb[i].m_max.setComponent<1>(maxY.getComponent(i));
			hkAabb scaledAabb;
			scaledAabb.m_min.setMul(subAabb[i].m_min, intToFloatScale);
			scaledAabb.m_max.setMul(subAabb[i].m_max, intToFloatScale);
			if ( SCALED )
			{
				// If the heightfield is 'externally' scaled we need to work in world space, so that the check against
				// the EarlyOutHitDistance (which is in world space and stored in the collector) works correctly.
				// In that case cvxCenter and cvxHalfExtent will be in world space as well. Also the 'external' scale
				// will then either be baked into intToFloatScale or heightfieldToWorld already.
				hkAabb scaledAabbWS; hkAabbUtil::calcAabb( heightfieldToWorld, scaledAabb, scaledAabbWS );
				distSquared[i].setAll(hkcdAabbAabbDistanceSquared(scaledAabbWS, cvxCenter, cvxHalfExtent));
			}
			else
			{
				distSquared[i].setAll(hkcdAabbAabbDistanceSquared(scaledAabb, cvxCenter, cvxHalfExtent));
			}
			distSquared[i].setInt24W(i);
		}

		//Sort them using a simple sorting network
		sortByX(&distSquared[0],&distSquared[2]);
		sortByX(&distSquared[1],&distSquared[3]);
		sortByX(&distSquared[0],&distSquared[1]);
		sortByX(&distSquared[2],&distSquared[3]);
		sortByX(&distSquared[1],&distSquared[2]);

		for (int i=0; i<4; i++)
		{
			if (distSquared[i].getComponent<0>()>maxSquaredDist) continue;
			int index = distSquared[i].getInt16W();
			QuadTreeWalkerStackElement& newElement = stack[(*stackSize)++];
			newElement.m_level = nextlevel;
			newElement.m_x = 2*x+(((index+1)&0x2)>>1);
			newElement.m_z = 2*z+((index&0x2)>>1);
			newElement.m_aabb = subAabb[index];
		}
	}

	static HK_FORCE_INLINE void rayTriangleQuadCheckHelper(
		const hkcdRay& ray, hkAabb& aabb, hkVector4Parameter heights, hkVector4ComparisonParameter flipSelect,
		hkVector4Parameter floatToIntScale, hkcdRayQueryFlags::Enum flags, hkSimdRealParameter earlyOutFraction, hkSimdReal *earlyOutHitFractionOut0, hkSimdReal *earlyOutHitFractionOut1,
		hkVector4* normalOut0, hkInt32* hit0, hkVector4* normalOut1, hkInt32* hit1 )
	{
		*hit0 = hkcdRayCastResult::createMiss();
		*hit1 = hkcdRayCastResult::createMiss();
		{
			// Calculate a tight AABB (using the heights) and early out if the ray doesn't hit it
			aabb.m_min.setComponent<1>(heights.horizontalMin<4>());
			aabb.m_max.setComponent<1>(heights.horizontalMax<4>());
			hkSimdReal earlyOut = earlyOutFraction; // make a copy, since we don't want to override it
			if (!hkcdIntersectRayAabb(ray, aabb, &earlyOut))
			{
				return;
			}
		}

		hkVector4Comparison disableBackFaces;
		hkVector4Comparison disableFrontFaces;
		hkcdRayQueryFlags::extractFlag(flags, hkcdRayQueryFlags::DISABLE_BACK_FACING_TRIANGLE_HITS, disableBackFaces);
		hkcdRayQueryFlags::extractFlag(flags, hkcdRayQueryFlags::DISABLE_FRONT_FACING_TRIANGLE_HITS, disableFrontFaces);

		// Calculate the four corner vertices
		hkVector4 c0; c0 = aabb.m_min;
		c0.setComponent<1>(heights.getComponent<0>());

		hkVector4 c1; c1 = aabb.m_min;
		c1.setComponent<1>(heights.getComponent<1>());
		c1.setComponent<0>(aabb.m_max.getComponent<0>());

		hkVector4 c2; c2 = aabb.m_max;
		c2.setComponent<1>(heights.getComponent<3>());

		hkVector4 c3; c3 = aabb.m_min;
		c3.setComponent<1>(heights.getComponent<2>());
		c3.setComponent<2>(aabb.m_max.getComponent<2>());


		// Now that we have the vertices do the two triangle checks
		hkVector4 cFlip; cFlip.setSelect(flipSelect, c2, c1);
		hkVector4 normal;
		*earlyOutHitFractionOut0 = earlyOutFraction;
		if (hkcdSegmentTriangleIntersect( ray, c3, c0, cFlip, normal, *earlyOutHitFractionOut0 ))
		{
			const hkVector4Comparison isBackFace = normal.dot<3>(ray.getDirection()).greaterZero();
			hkVector4Comparison disableFace;
			disableFace.setSelect( isBackFace, disableBackFaces, disableFrontFaces );
			if (!disableFace.allAreSet())
			{
				normalOut0->setFlipSign( normal, isBackFace );
				*hit0 = hkcdRayCastResult::createHit(isBackFace);
			}
		}

		cFlip.setSelect(flipSelect, c0, c3);
		*earlyOutHitFractionOut1 = earlyOutFraction;
		if (hkcdSegmentTriangleIntersect( ray, c1, c2, cFlip, normal, *earlyOutHitFractionOut1 ))
		{
			const hkVector4Comparison isBackFace = normal.dot<3>(ray.getDirection()).greaterZero();
			hkVector4Comparison disableFace;
			disableFace.setSelect( isBackFace, disableBackFaces, disableFrontFaces );
			if (!disableFace.allAreSet())
			{
				normalOut1->setFlipSign( normal, isBackFace );
				*hit1 = hkcdRayCastResult::createHit(isBackFace);
			}
		}
	}

	static hkVector4 HK_FORCE_INLINE calcFractionXZ(int baseX, int baseZ, hkVector4Parameter hitPos)
	{
		hkIntVector intHitPos; intHitPos.set(baseX, 0, baseZ, 0);
		hkVector4 rounded;   intHitPos.convertU32ToF32( rounded ) ;
		hkVector4 res;
		res.setSub(hitPos, rounded);
		HK_ASSERT(0x65dc23ee, res(0)>-0.01f && res(0)<1.01f && res(2)>-0.01f && res(2)<1.01f);
		return res;
	}

	static hkVector4 HK_FORCE_INLINE calcFractionXZWithScale(int baseX, int baseZ, hkVector4Parameter hitPos, hkVector4Parameter baseScale, hkVector4Parameter invBaseScale)
	{
		hkIntVector intHitPos; intHitPos.set(baseX, 0, baseZ, 0);
		hkVector4 rounded;   intHitPos.convertU32ToF32( rounded ) ;
		rounded.mul(baseScale);
		hkVector4 res;
		res.setSub(hitPos, rounded);
		res.mul(invBaseScale);
		HK_ASSERT(0x65dc233e, res(0)>-0.01f && res(0)<1.01f && res(2)>-0.01f && res(2)<1.01f);
		return res;
	}

	template <typename T, typename CacheT>
	void HK_FORCE_INLINE QuadCollector<T, CacheT>::collect4Heights(int x, int z, hkVector4* heightOut, hknpShapeTag* shapeTagOut, hkBool32 *triangleFlipOut)
	{
		if (!m_cache.get(x,z, heightOut, shapeTagOut, triangleFlipOut))
		{
			m_obj->getQuadInfoAt(x, z, heightOut, shapeTagOut, triangleFlipOut);
			m_cache.put(x, z, *heightOut, *shapeTagOut, *triangleFlipOut);
		}
	}

	template <typename T, typename CacheT>
	void HK_FORCE_INLINE QuadCollector<T, CacheT>::collect16Heights(int x, int z, hkVector4* heightsOut, hknpShapeTag* shapeTagOut, hkBool32 *triangleFlipOut)
	{
		hkVector4Comparison mask; mask.set<hkVector4ComparisonMask::MASK_XY>();
		hkVector4 samples[2];
		{
			collect4Heights(x-1, z-1, &samples[0], shapeTagOut, triangleFlipOut);
			collect4Heights(x+1, z-1, &samples[1], shapeTagOut, triangleFlipOut);
			{
				hkVector4 samples1swp; samples1swp.setPermutation<hkVectorPermutation::ZWXY>(samples[1]);
				heightsOut[0].setSelect(mask, samples[0], samples1swp);
				hkVector4 samples0swp; samples0swp.setPermutation<hkVectorPermutation::ZWXY>(samples[0]);
				heightsOut[1].setSelect(mask, samples0swp, samples[1]);
			}
		}
		{
			collect4Heights(x-1, z+1, &samples[0], shapeTagOut, triangleFlipOut);
			collect4Heights(x+1, z+1, &samples[1], shapeTagOut, triangleFlipOut);
			{
				hkVector4 samples1swp; samples1swp.setPermutation<hkVectorPermutation::ZWXY>(samples[1]);
				heightsOut[2].setSelect(mask, samples[0], samples1swp);
				hkVector4 samples0swp; samples0swp.setPermutation<hkVectorPermutation::ZWXY>(samples[0]);
				heightsOut[3].setSelect(mask, samples0swp, samples[1]);
			}
		}
		// Need to get the right shapeTag and triangleFlip so we need to call again (arghh, all these heights returned are redundant....)
		collect4Heights(x, z, &samples[0], shapeTagOut, triangleFlipOut);
	}

	// For the triangle evaluator
	// Given a position in [0,1] X [0,1] and the four corner heights it returns the height in w component
	// and the height differentiated w.r.t. x,z in x,z components.
	static HK_FORCE_INLINE hkVector4 evalQuadTriangles(hkVector4Parameter fracPos, hkVector4Parameter samp, hkVector4ComparisonParameter flipMask)
	{
		HK_ASSERT(0x14492424, fracPos(0)>-HK_REAL_EPSILON  && fracPos(0)<(1.0f+HK_REAL_EPSILON));
		HK_ASSERT(0x14492425, fracPos(2)>-HK_REAL_EPSILON  && fracPos(2)<(1.0f+HK_REAL_EPSILON));

		hkVector4 permSamp; permSamp.setPermutation<hkVectorPermutation::ZWXY>(samp);
		hkVector4 flippedSamples; flippedSamples.setSelect(flipMask, samp, permSamp);

		hkSimdReal u = fracPos.getComponent<0>();
		hkSimdReal v;
		{
			hkVector4 oneMinusFrac; oneMinusFrac.setSub(hkVector4::getConstant<HK_QUADREAL_1>(), fracPos);
			hkVector4 flippedFrac; flippedFrac.setSelect(flipMask, fracPos, oneMinusFrac);
			v = flippedFrac.getComponent<2>();
		}

		hkVector4Comparison cmp = u.greater(v);
		hkVector4 Ds;
		{
			hkVector4 c1; c1.set(
				flippedSamples.getComponent<1>()-flippedSamples.getComponent<0>(),
				flippedSamples.getComponent<3>()-flippedSamples.getComponent<1>(),
				hkSimdReal::getConstant<HK_QUADREAL_0>(),
				hkSimdReal::getConstant<HK_QUADREAL_0>());

			hkVector4 c2; c2.set(
				flippedSamples.getComponent<3>()-flippedSamples.getComponent<2>(),
				flippedSamples.getComponent<2>()-flippedSamples.getComponent<0>(),
				hkSimdReal::getConstant<HK_QUADREAL_0>(),
				hkSimdReal::getConstant<HK_QUADREAL_0>());
			Ds.setSelect(cmp, c1,c2);
		}
		hkSimdReal dhdx = Ds.getComponent<0>();
		hkSimdReal dhdz = Ds.getComponent<1>();
		hkSimdReal height = flippedSamples.getComponent<0>() + dhdx*u + dhdz*v;
		dhdz.setSelect(flipMask, dhdz, -dhdz);

		hkVector4 res;
		res.set(dhdx,  hkSimdReal::getConstant<HK_QUADREAL_0>(), dhdz, height);
		return res;
	}

	template<typename T, typename CacheT>
	HK_FORCE_INLINE TriangleEvaluator<T,CacheT>::TriangleEvaluator( const T* obj )
	:	QuadCollector<T, CacheT>(obj)
	{
	}

	template<typename T, typename CacheT>
	void HK_FORCE_INLINE TriangleEvaluator<T,CacheT>::gatherHeightsAndInfo(int x, int z)
	{
		QuadCollector<T, CacheT>::collect4Heights(x, z, &m_heights, &m_shapeTag, &m_triangleFlip);
	}

	template<typename T, typename CacheT>
	hkVector4 HK_FORCE_INLINE TriangleEvaluator<T,CacheT>::evalHeightAndDiffHeight(hkVector4Parameter fracPos)
	{
		hkVector4Comparison flipSelect; flipSelect.set(m_triangleFlip ? hkVector4ComparisonMask::MASK_XYZW : hkVector4ComparisonMask::MASK_NONE);
		return evalQuadTriangles(fracPos, m_heights, flipSelect);
	}


	template<int N>
	HK_FORCE_INLINE void HeightCacheT<N>::clear()
	{
		HK_ASSERT2( 0xf051d14a, (sizeof(m_elements) & 0xf) == 0, "memSet16 requires size a multiple of 16 bytes" );
		static HK_ALIGN16( int clearBits[4] ) = { 0x40000000, 0x40000000,0x40000000,0x40000000};
		hkString::memSet16(m_elements, clearBits, sizeof(m_elements)>>4);
	}

	template<int N>
	HK_FORCE_INLINE bool HeightCacheT<N>::get(int x, int z, hkVector4 *outHeight, hknpShapeTag *shapeTag, hkBool32 *triangleFlip ) const
	{
		int xm = x&IndexMask;
		int zm = z&IndexMask;
		const CacheElement* elm = &m_elements[xm][zm];
		if (elm->m_x!=x || elm->m_z!=z) return false;
		*outHeight = m_heights[xm][zm];
		*shapeTag = m_shapeTags[xm][zm];
		*triangleFlip = m_triangleFlips[xm][zm];
		return true;
	}

	template<int N>
	HK_FORCE_INLINE void HeightCacheT<N>::put(int x, int z, hkVector4Parameter height, hknpShapeTag shapeTag, hkBool32 triangleFlip )
	{
		int xm = x&IndexMask;
		int zm = z&IndexMask;
		CacheElement* elm = &m_elements[xm][zm];
		elm->m_x = x;
		elm->m_z = z;
		m_heights[xm][zm] = height;
		m_shapeTags[xm][zm] = shapeTag;
		m_triangleFlips[xm][zm] = triangleFlip;
	}


	HK_FORCE_INLINE void NoCacheT::clear()
	{
	}


	HK_FORCE_INLINE bool NoCacheT::get(int x, int z, hkVector4 *outHeight, hknpShapeTag *shapeTag, hkBool32 *triangleFlip) const
	{
		return false;
	}


	HK_FORCE_INLINE void NoCacheT::put(int x, int z, hkVector4Parameter height, hknpShapeTag shapeTag, hkBool32 triangleFlip)
	{
	}


	template <typename T, typename CacheT>
	HK_FORCE_INLINE QuadCollector<T, CacheT>::QuadCollector(const T* obj)
	:	m_obj(obj)
	{
		m_cache.clear();
	}

} //namespace hknpHeightFieldShapeUtils

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
