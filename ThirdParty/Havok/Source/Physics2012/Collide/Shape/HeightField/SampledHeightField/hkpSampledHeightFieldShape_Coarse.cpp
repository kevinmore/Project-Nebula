/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/HeightField/SampledHeightField/hkpSampledHeightFieldShape.h>
#include <Physics2012/Collide/Shape/Query/hkpRayHitCollector.h>
#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastTriangle.h>
#include <Geometry/Internal/Algorithms/Intersect/hkcdIntersectRayAabb.h>
#include <Common/Base/Config/hkOptionalComponent.h>


void HK_CALL hkpSampledHeightField_registerCoarseTreeRayCastFunction()
{
	// Use coarse min max tree algorithm only.
	// NOTE: This expects the tree to be available for all height fields.
	hkpSampledHeightFieldShape::s_rayCastFunc = (&hkpSampledHeightFieldShape::castRayCoarseTree);
#ifndef HK_PLATFORM_SPU
	hkpSampledHeightFieldShape::s_sphereCastFunc = (&hkpSampledHeightFieldShape::castSphereCoarseTree); 
#endif
}
HK_OPTIONAL_COMPONENT_DEFINE_MANUAL(hkpSampledHeightFieldShape_CoarseCast, hkpSampledHeightField_registerCoarseTreeRayCastFunction);


//
// Ray and sphere cast internal function pointer interface
//

void hkpSampledHeightFieldShape::castRayCoarseTree( const hkpShapeRayCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector ) const
{
	HK_OPTIONAL_COMPONENT_MARK_USED(hkpSampledHeightFieldShape_CoarseCast);
	hkVector4 from; from.setMul( input.m_from, m_floatToIntScale );
	hkVector4 to;     to.setMul( input.m_to, m_floatToIntScale );
	castRayCoarseTreeInternal( from, to, cdBody, collector );
}

// Sphere casting is not available on SPU
#ifndef HK_PLATFORM_SPU
void hkpSampledHeightFieldShape::castSphereCoarseTree( const hkpSphereCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector ) const
{
	HK_OPTIONAL_COMPONENT_MARK_USED(hkpSampledHeightFieldShape_CoarseCast);
	// Convert to local height field space
	hkVector4 from; from.setMul(input.m_from, m_floatToIntScale );
	hkVector4 to;     to.setMul(input.m_to, m_floatToIntScale );

	// Find the height if we are inside the grid
	hkReal distToSurface = hkReal(0);
	hkIntVector fromPosF;
	fromPosF.setConvertF32toS32(from);
	int xPos = fromPosF.getU32<0>();
	int zPos = fromPosF.getU32<2>();
	if (hkUint32(xPos)<hkUint32(m_xRes) && hkUint32(zPos)<hkUint32(m_zRes)) 
	{
		hkVector4 normal;
		hkReal height;
		int triangleId;
		_getHeightAndNormalAt(xPos, zPos, from(0)-hkReal(xPos), from(2)-hkReal(zPos), normal, height, triangleId);
		distToSurface = height-from(1);
	}

	// Move the ray up to allow for maxExtraPenetration
	hkReal deltaUp = hkMath::max2(distToSurface,hkReal(0)) + input.m_maxExtraPenetration;
	hkVector4 deltaUpV; deltaUpV.set(0.0f, deltaUp, 0.0f);
	from.add(deltaUpV);
	to.add(deltaUpV);
	castRayCoarseTreeInternal( from, to, cdBody, collector );
}
#endif

// Helper functions to buildCoarseTree
 HK_FORCE_INLINE static hkSimdReal calcNextMinHeight(const hkArray<hkVector4>& v, int xRes, int zRes, int x, int z)
{
	if (x>=xRes || z>=zRes) 
	{
		return hkSimdReal_Max;
	}
	return v[2*(x*zRes + z)].horizontalMin<4>();
}

HK_FORCE_INLINE static hkSimdReal calcNextMaxHeight(const hkArray<hkVector4>& v, int xRes, int zRes, int x, int z)
{
	if (x>=xRes || z>=zRes) 
	{
		return hkSimdReal_MinusMax;
	}
	return v[2*(x*zRes + z)+1].horizontalMax<4>();
}


HK_FORCE_INLINE void hkpSampledHeightFieldShape::calcMinMax(int coarseness, int x, int z, hkReal& minheightOut, hkReal& maxheightOut) const
{
	int grid = (1<<coarseness);
	minheightOut = HK_REAL_MAX; 
	maxheightOut = -HK_REAL_MAX; 
	for (int ix=x*grid, endx=hkMath::min2((x+1)*grid+1, m_xRes); ix<endx; ix++) 
	{
		for (int iz=z*grid, endz=hkMath::min2((z+1)*grid+1, m_zRes); iz<endz; iz++) 
		{
			hkReal f = getHeightAt(ix,iz);
			minheightOut = hkMath::min2(minheightOut,f);
			maxheightOut = hkMath::max2(maxheightOut,f);
		}
	}
}

void hkpSampledHeightFieldShape::buildCoarseMinMaxTree(int coarseness)
{
	m_coarseness = coarseness;
	m_coarseTreeData.clearAndDeallocate();
	if (m_coarseness<=0) {
		return;
	}

	unsigned int s = hkMath::max2(m_xRes, m_zRes)-1;
	int toplevel = 1;
	while ( s>1 )
	{
		s = s>>1;
		toplevel++;
	}
	int coarseLevelsNeeded = toplevel-coarseness;
	m_coarseTreeData.reserveExactly(coarseLevelsNeeded);

	// Lowest level
	int grid = (1<<coarseness);
	int xGridRes = (m_xRes+grid-1)/grid;
	int zGridRes = (m_zRes+grid-1)/grid;
	{
		CoarseMinMaxLevel& lowerstLevel = m_coarseTreeData.expandOne();
		lowerstLevel.m_xRes = (xGridRes+1)/2;
		lowerstLevel.m_zRes = (zGridRes+1)/2;
		lowerstLevel.m_minMaxData.reserveExactly(2*m_coarseTreeData[0].m_xRes*m_coarseTreeData[0].m_zRes);
		for (int x=0; x<lowerstLevel.m_xRes; x++)
		{
			for (int z=0; z<lowerstLevel.m_zRes; z++)
			{
				hkReal min00, min01, min10, min11;
				hkReal max00, max01, max10, max11;
				calcMinMax(coarseness, 2*x  , 2*z  , min00, max00);
				calcMinMax(coarseness, 2*x  , 2*z+1, min01, max01);
				calcMinMax(coarseness, 2*x+1, 2*z  , min10, max10);
				calcMinMax(coarseness, 2*x+1, 2*z+1, min11, max11);
				hkVector4 vmin; vmin.set(min00, min10, min11, min01);
				hkVector4 vmax; vmax.set(max00, max10, max11, max01);
				lowerstLevel.m_minMaxData.pushBackUnchecked(vmin);
				lowerstLevel.m_minMaxData.pushBackUnchecked(vmax);
			}
		}
	}

	for (int level=1; level<coarseLevelsNeeded; level++) 
	{
		int xResPrev = m_coarseTreeData[level-1].m_xRes;
		int zResPrev = m_coarseTreeData[level-1].m_zRes;
		const hkArray<hkVector4>& prevMinMax = m_coarseTreeData[level-1].m_minMaxData;

		CoarseMinMaxLevel& newlevel = m_coarseTreeData.expandOne();
		newlevel.m_xRes = (xResPrev+1)/2;
		newlevel.m_zRes = (zResPrev+1)/2;
		newlevel.m_minMaxData.reserveExactly(2*newlevel.m_xRes*newlevel.m_zRes);
		for (int x=0; x<m_coarseTreeData[level].m_xRes; x++)
		{
			for (int z=0; z<newlevel.m_zRes; z++)
			{
				hkVector4 vmin; vmin.set(
					calcNextMinHeight(prevMinMax, xResPrev, zResPrev, 2*x  , 2*z  ),
					calcNextMinHeight(prevMinMax, xResPrev, zResPrev, 2*x+1, 2*z  ),
					calcNextMinHeight(prevMinMax, xResPrev, zResPrev, 2*x+1, 2*z+1),
					calcNextMinHeight(prevMinMax, xResPrev, zResPrev, 2*x  , 2*z+1)
				);
				hkVector4 vmax; vmax.set(
					calcNextMaxHeight(prevMinMax, xResPrev, zResPrev, 2*x  , 2*z  ),
					calcNextMaxHeight(prevMinMax, xResPrev, zResPrev, 2*x+1, 2*z  ),
					calcNextMaxHeight(prevMinMax, xResPrev, zResPrev, 2*x+1, 2*z+1),
					calcNextMaxHeight(prevMinMax, xResPrev, zResPrev, 2*x  , 2*z+1)
				);
				newlevel.m_minMaxData.pushBackUnchecked(vmin);
				newlevel.m_minMaxData.pushBackUnchecked(vmax);
			}
		}
	}
}

void hkpSampledHeightFieldShape::setMinMaxTreeCoarseness(int coarseness)
{
	m_coarseness = coarseness;
}



HK_FORCE_INLINE static void createRotationMasks(const hkcdRay& ray, int& rotOffsetOut, hkVector4Comparison& mask0Out, hkVector4Comparison& mask1Out, hkVector4Comparison& mask2Out, hkVector4Comparison& mask3Out) 
{
	hkVector4Comparison comp = ray.m_direction.lessZero();
	int signs = comp.getMask<hkVector4ComparisonMask::MASK_XZ>();
	
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


HK_FORCE_INLINE void hkpSampledHeightFieldShape::rayTriangleQuadCheck(const hkcdRay& ray, hkAabb& aabb, int x, int z, const hkVector4Comparison& flipSelect,
												  const hkpCdBody& cdBody, hkpRayHitCollector& collector, hkSimdReal *earlyOutHitFractionInOut) const 
{
	hkSimdReal h0 = hkSimdReal::fromFloat(getHeightAt(x  , z));
	hkSimdReal h1 = hkSimdReal::fromFloat(getHeightAt(x+1, z));
	hkSimdReal h2 = hkSimdReal::fromFloat(getHeightAt(x+1, z+1));
	hkSimdReal h3 = hkSimdReal::fromFloat(getHeightAt(x  , z+1));

	{
		// Calculate a tight aabb (using the heights) and early out if the ray doesn't hit it
		hkSimdReal minh0; minh0.setMin(h0,h1);
		hkSimdReal minh1; minh1.setMin(h2,h3);
		hkSimdReal maxh0; maxh0.setMax(h0,h1);
		hkSimdReal maxh1; maxh1.setMax(h2,h3);
		hkSimdReal minh; minh.setMin(minh0,minh1);
		hkSimdReal maxh; maxh.setMax(maxh0,maxh1);
		aabb.m_min.setComponent<1>(minh);
		aabb.m_max.setComponent<1>(maxh);
		hkSimdReal earlyOut = *earlyOutHitFractionInOut; // make a copy, since we don't want to override it 
		if (!hkcdIntersectRayAabb(ray, aabb, &earlyOut))
		{
			return;
		}
	}

	// Calculate the four corner vertices
	hkVector4 c0; c0 = aabb.m_min;
	c0.setComponent<1>(h0);
	
	hkVector4 c1; c1 = aabb.m_min;
	c1.setComponent<1>(h1);
	c1.setComponent<0>(aabb.m_max.getComponent<0>());
	
	hkVector4 c2; c2 = aabb.m_max;
	c2.setComponent<1>(h2);
	
	hkVector4 c3; c3 = aabb.m_min;
	c3.setComponent<1>(h3);
	c3.setComponent<2>(aabb.m_max.getComponent<2>());


	// Now that we have the vertices do the two triangle checks
	hkpShapeRayCastOutput output;
	hkVector4 cFlip; 
	cFlip.setSelect(flipSelect, c2, c1);
	if (hkcdSegmentTriangleIntersect( ray, c3, c0, cFlip, output.m_normal, *earlyOutHitFractionInOut ))
	{
		output.m_normal.mul( m_floatToIntScale ); 
		output.m_normal.normalize<3>();
		output.m_extraInfo = (x << 1) + (z << 16) + 0;
		earlyOutHitFractionInOut->store<1>((hkReal*)&output.m_hitFraction);
		collector.addRayHit( cdBody, output );
	}

	cFlip.setSelect(flipSelect, c0, c3);
	if (hkcdSegmentTriangleIntersect( ray, c1, c2, cFlip, output.m_normal, *earlyOutHitFractionInOut ))
	{
		output.m_normal.mul( m_floatToIntScale ); 
		output.m_normal.normalize<3>();
		output.m_extraInfo = (x << 1) + (z << 16) + 1;
		earlyOutHitFractionInOut->store<1>((hkReal*)&output.m_hitFraction);
		collector.addRayHit( cdBody, output );
	}
}

void hkpSampledHeightFieldShape::findStartCell(AABBStackElement& startOut, const hkVector4& from, const hkVector4& to) const
{
	// When we descend the quadtree it is like checking the bits of the integer positions (msb first).
	// So if we now how many bits are constant along the ray (both x and z) we can skip directly to that cell in the tree
	int x1 = hkMath::clamp((int)from(0), 0, m_xRes-2);
	int z1 = hkMath::clamp((int)from(2), 0, m_zRes-2);
	int x2 = hkMath::clamp((int)to(0), 0, m_xRes-2);
	int z2 = hkMath::clamp((int)to(2), 0, m_zRes-2);
	hkUint32 leadingEqualBitsX = hkMath::countLeadingZeros(x1 ^ x2);
	hkUint32 leadingEqualBitsZ = hkMath::countLeadingZeros(z1 ^ z2);
	int level = 32 - hkMath::min2(leadingEqualBitsX, leadingEqualBitsZ);
	int x = x1>>level;
	int z = z1>>level;
	int gridSize = 1<<level;
	startOut.aabb.m_min.set(hkReal(x*gridSize), m_raycastMinY, hkReal(z*gridSize)); 
	startOut.aabb.m_max.set(hkReal((x+1)*gridSize), m_raycastMaxY, hkReal((z+1)*gridSize)); 
	startOut.level = level;
	startOut.x = x;
	startOut.z = z;
}

void hkpSampledHeightFieldShape::castRayCoarseTreeInternal(const hkVector4& from, const hkVector4& to, const hkpCdBody& cdBody, hkpRayHitCollector& collector) const
{
	HK_TIME_CODE_BLOCK("rcHeightFieldCoarse", HK_NULL);

	hkcdRay ray;
	ray.setEndPoints(from, to);

	// We maintain an explicit stack of AABBs we need to visit. Start with the top-level one
	const int STACKSIZE=32; //Max stack usage is 2*ceil(log2(max(m_resX,m_resZ)))+1
	AABBStackElement stack[STACKSIZE];
	int stackSize = 1;
	findStartCell(stack[0], from, to);

	hkSimdReal earlyOutHitFraction = hkSimdReal::fromFloat(collector.m_earlyOutHitFraction);

	// When colliding with triangles we need to respect getTriangleFlip(). Create a flip mask we can use:
	hkVector4Comparison flipSelect; 
	if (getTriangleFlip())
		flipSelect.set<hkVector4ComparisonMask::MASK_XYZW>();
	else
		flipSelect.set<hkVector4ComparisonMask::MASK_NONE>();

	// We want to visit the sub AABBs in ray-order so we build these masks:
	hkVector4Comparison mask0, mask1, mask2, mask3;
	int rotOffset; // rotOffset Is used to calculate the integer coordinates when we descend the tree
	createRotationMasks(ray, rotOffset, mask0, mask1, mask2, mask3);

	while (stackSize) 
	{
		stackSize--;
		int level = stack[stackSize].level;
		int x = stack[stackSize].x;
		int z = stack[stackSize].z;
		hkAabb aabb = stack[stackSize].aabb;

		if (level==0) 
		{
			// Lowest level is the one with the triangles. Check collision against the pair
			if (x>=m_xRes-1) continue; // Can happen for for heightfield sizes that are not a power of two
			if (z>=m_zRes-1) continue;
			rayTriangleQuadCheck(ray, aabb, x, z, flipSelect, cdBody, collector, &earlyOutHitFraction);
			continue;
		}

		hkVector4 minY; 
		hkVector4 maxY; 
		minY.setAll(aabb.m_min.getComponent<1>());
		maxY.setAll(aabb.m_max.getComponent<1>());
		int nextlevel = level-1;
		if (nextlevel>=m_coarseness) 
		{
			getCoarseMinMax(nextlevel, x, z, minY, maxY);
			
			// Permute the min-max values to match the ray-order
			switch(rotOffset) 
			{
			case 1:
				minY.setPermutation<hkVectorPermutation::YZWX>(minY);
				maxY.setPermutation<hkVectorPermutation::YZWX>(maxY);
				break;
			case 2:
				minY.setPermutation<hkVectorPermutation::ZWXY>(minY);
				maxY.setPermutation<hkVectorPermutation::ZWXY>(maxY);
				break;
			case 3:
				minY.setPermutation<hkVectorPermutation::WXYZ>(minY);
				maxY.setPermutation<hkVectorPermutation::WXYZ>(maxY);
				break;
			}
		}

		// Now divide the aabb into four sub AABBs. In the case where both ray.m_dir.x and ray.m_dir.z are positive this corresponds to:
		// Z ^	  3 | 2
		//   |	  --|--
		//   |	  0 | 1
		//   |
		//    ------> X
		// The ray order will always be 0<(1/3)<2

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
			stack[stackSize].level = nextlevel;
			stack[stackSize].x = 2*x+(((rotOffset+3)&0x2)>>1);
			stack[stackSize].z = 2*z+(((rotOffset+2)&0x2)>>1);
			stack[stackSize].aabb = aabb2;
			stackSize++;
		}
		if (hits&hkVector4ComparisonMask::MASK_Y) 
		{
			stack[stackSize].level = nextlevel;
			stack[stackSize].x = 2*x+(((rotOffset+2)&0x2)>>1);
			stack[stackSize].z = 2*z+(((rotOffset+1)&0x2)>>1);
			stack[stackSize].aabb = aabb1;
			stackSize++;
		}
		else if (hits&hkVector4ComparisonMask::MASK_W) // A ray cannot hit both aabb1 and aabb3 
		{
			stack[stackSize].level = nextlevel;
			stack[stackSize].x = 2*x+(((rotOffset+4)&0x2)>>1);
			stack[stackSize].z = 2*z+(((rotOffset+3)&0x2)>>1);
			stack[stackSize].aabb = aabb3;
			stackSize++;
		}
		if (hits&hkVector4ComparisonMask::MASK_X) 
		{
			stack[stackSize].level = nextlevel;
			stack[stackSize].x = 2*x+(((rotOffset+1)&0x2)>>1);
			stack[stackSize].z = 2*z+(((rotOffset+0)&0x2)>>1);
			stack[stackSize].aabb = aabb0;
			stackSize++;
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
