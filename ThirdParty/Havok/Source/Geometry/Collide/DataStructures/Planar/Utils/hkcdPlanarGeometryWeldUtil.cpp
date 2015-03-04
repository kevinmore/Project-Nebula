/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/DataStructures/Planar/Utils/hkcdPlanarGeometryWeldUtil.h>
#include <Geometry/Collide/DataStructures/Planar/Predicates/hkcdPlanarGeometryPredicates.h>
#include <Geometry/Collide/DataStructures/Planar/Memory/hkcdPlanarGeometryPlanesCollection.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>

namespace hkcdPlaneWeldUtilImpl
{
	//
	//	Compares 2 hkIntVectors by x, y, z in that order

	HK_FORCE_INLINE int HK_CALL vectorLess(hkIntVectorParameter ivA, hkIntVectorParameter ivB)
	{
		const hkVector4Comparison cmpL = ivA.compareLessThanS32(ivB);
		const hkVector4Comparison cmpE = ivA.compareEqualS32(ivB);

		const int code = ((cmpL.getMask() << 2) & 0x1C) | (cmpE.getMask() & 3);
		return (0xFAF8FAF0 >> code) & 1;
	}

	//
	//	Computes an approximate normalized direction for the the plane

	static void HK_CALL approximateNormal(const hkcdPlanarGeometryWeldUtil::Plane& plane, hkIntVector& planeNormalOut)
	{
		// Compute maximum absolute component
		hkInt64Vector4 vNrm;		plane.getExactNormal(vNrm);
		hkInt64Vector4 vAbsNrm;		vAbsNrm.setAbs(vNrm);
		const int axisZ				= vAbsNrm.getIndexOfMaxComponent<3>();

		// We want to divide the other two components by the maximum component so we get back two sub-unitary fractions.
		// We'll use the fractions to encode direction
		const int axisX = ((1 << axisZ) & 3);
		const int axisY = ((1 << axisX) & 3);

		// Get components
		hkUint64 dx = (hkUint64)vAbsNrm.getComponent(axisX);
		hkUint64 dy = (hkUint64)vAbsNrm.getComponent(axisY);
		hkUint64 dz = (hkUint64)vAbsNrm.getComponent(axisZ);

		// Reduce first by the greatest common divisor
		const hkUint64 gcdxz = hkMath::greatestCommonDivisor<hkUint64>(dx, dz);
		const hkUint64 gcdyz = hkMath::greatestCommonDivisor<hkUint64>(dy, dz);

		// Divide by the gcd, to get the unreductible directions
		hkInt64Vector4 vTop;	vTop.set(dx, dz, dy, dz);
		hkInt64Vector4 vBot;	vBot.set(gcdxz, gcdxz, gcdyz, gcdyz);
		vTop.setUnsignedDiv(vTop, vBot);

		// We know the components of vNrm are stored on 52 bits (including the sign bit), so we can shift them by 12 and get an approximate int value direction
		// We don't care about accuracy at this stage, as we are only required to return an approximate direction. The final tests will be done using the exact predicates anyway!
		const int numBitsShl = 64 - hkcdPlanarGeometryPrimitives::NumBitsPlaneNormal::NUM_BITS;
		vBot.setPermutation<hkVectorPermutation::YYWW>(vTop);	// [topY, topY, topW, topW]
		vTop.setShiftLeft<numBitsShl>(vTop);					// [topX << 12, *, topZ << 12, *]
		vTop.setUnsignedDiv(vTop, vBot);

		hkIntVector vDir;
		vDir.setComponent(axisX, (int)vTop.getComponent<0>());
		vDir.setComponent(axisY, (int)vTop.getComponent<2>());
		vDir.setComponent(axisZ, (1 << numBitsShl));

		// Set proper signs. Since we divide by z, z should always end-up positive!
		vAbsNrm.setAll(vNrm.getComponent(axisZ));
		hkVector4fComparison signsf;	signsf.setXor(vNrm.lessZero(), vAbsNrm.lessZero());
#ifdef HK_REAL_IS_DOUBLE
		hkVector4Comparison signs;		signs.set(signsf.getMask());
#else
		const hkVector4Comparison signs	= signsf;
#endif
		planeNormalOut.setFlipSignS32(vDir, signs);
	}
}

//
//	Welds the given set of planes

void HK_CALL hkcdPlanarGeometryWeldUtil::weldPlanes(const hkArray<Plane>& planesIn, hkArray<Plane>& weldedPlanesOut, hkArray<int>& remapTableOut)
{
	typedef hkcdPlanarGeometryPredicates::Coplanarity		Coplanarity;

	const int numAllPlanes	= planesIn.getSize();
	const int numBounds		= hkcdPlanarGeometryPlanesCollection::NUM_BOUNDS;
	HK_ASSERT(0x57249e55, numAllPlanes >= numBounds);

#ifdef HK_DEBUG

	// Sanity check. We expect the boundary planes to be present only once!!
	for (int bi = 0; bi < numBounds; bi++)
	{
		const Plane& boundaryPlane = planesIn[bi];

		for (int k = numBounds; k < numAllPlanes; k++)
		{
			if ( hkcdPlanarGeometryPredicates::coplanar(boundaryPlane, planesIn[k]) == hkcdPlanarGeometryPredicates::COINCIDENT )
			{
				HK_ASSERT(0xa140e62, false);
			}
		}
	}

#endif

	// Add the boundary planes
	remapTableOut.setSize(numAllPlanes, -1);
	weldedPlanesOut.setSize(numBounds);
	for (int k = numBounds - 1; k >= 0; k--)
	{
		weldedPlanesOut[k]	= planesIn[k];
		remapTableOut[k]	= k;
	}

	// We determine the approximate plane normal and only test planes with identical approximate normals
	hkArray<hkIntVector> normals;	normals.setSize(numAllPlanes + 1);
	for (int k = numAllPlanes - 1; k >= numBounds; k--)
	{
		hkIntVector v;	hkcdPlaneWeldUtilImpl::approximateNormal(planesIn[k], v);
		v.setComponent<3>(k);	// Save plane index in the .w component
		normals[k] = v;
	}

	// Sort normals by x, y, z so we can detect identical candidates
	hkAlgorithm::explicitStackQuickSort(&normals[numBounds], numAllPlanes - numBounds, hkcdPlaneWeldUtilImpl::vectorLess);
	//hkAlgorithm::quickSort(&normals[numBounds], numAllPlanes - numBounds, hkcdPlaneWeldUtilImpl::vectorLess);
	normals[numAllPlanes].setAll(-1);	// Add the end marker!

	// Try to collapse directions in the same group
	{
		hkIntVector prevNrmGroup;	prevNrmGroup.setAll(-1);
		hkVector4Comparison mXYZ;	mXYZ.set<hkVector4Comparison::MASK_XYZ>();
		int prevGroupStartIdx		= numBounds;

		for (int k = numBounds; k <= numAllPlanes; k++)
		{
			const hkIntVector crtPlane	= normals[k];
			hkVector4Comparison cmp		= prevNrmGroup.compareEqualS32(crtPlane);	// [px == cx, py == cy, pz == cz, *]
			cmp.setAndNot(mXYZ, cmp);												// [px != cx, py != cy, pz != cz, 0]

			if ( cmp.anyIsSet() )
			{
				// A new group has started!
				// Process previous group
				if ( k - prevGroupStartIdx )
				{
					const int startNewPlaneIdx = weldedPlanesOut.getSize();

					// For each old plane, try to weld to the new ones
					for (int i = prevGroupStartIdx; i < k; i++)
					{
						const int oldPlaneIdx = normals[i].getComponent<3>();
						const Plane& pOld = planesIn[oldPlaneIdx];

						// Compare against all new
						int j = weldedPlanesOut.getSize() - 1;
						for (; j >= startNewPlaneIdx; j--)
						{
							const Plane& pNew				= weldedPlanesOut[j];
							const Coplanarity coplanarRes	= hkcdPlanarGeometryPredicates::coplanar(pOld, pNew);

							if ( coplanarRes != hkcdPlanarGeometryPredicates::NOT_COPLANAR )
							{
								const int newJ = (coplanarRes == hkcdPlanarGeometryPredicates::COINCIDENT) ? j : (hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG | j);
								remapTableOut[oldPlaneIdx] = newJ;
								break;
							}
						}

						// If nothing was found, add new edge
						if ( j < startNewPlaneIdx )
						{
							remapTableOut[oldPlaneIdx] = weldedPlanesOut.getSize();
							weldedPlanesOut.pushBack(pOld);
						}
					}
				}

				// Initialize stuff for the new group
				prevGroupStartIdx	= k;
				prevNrmGroup		= crtPlane;
			}
		}
	}
}

//
//	Computes a mapping between two plane sets. The first plane set provided is assumed to be included in the second plane set

void HK_CALL hkcdPlanarGeometryWeldUtil::computeMappingBetweenPlaneSets(const hkArray<Plane>& planesIncluded, const hkArray<Plane>& allPlanes, hkArray<int>& remapTableOut, int invalidPlaneValue)
{
	typedef hkcdPlanarGeometryPredicates::Coplanarity		Coplanarity;

	// We determine the approximate plane normal and only test planes with identical approximate normals
	const int numPlanesIn		= planesIncluded.getSize();
	const int numPlanesAll		= allPlanes.getSize();
	hkArray<hkIntVector> normalsIn, normalsAll;	
	normalsIn.setSize(numPlanesIn);
	normalsAll.setSize(numPlanesAll);
	remapTableOut.setSize(numPlanesIn);

	// Compute and sort normals by x, y, z so we can detect identical candidates
	for (int k = numPlanesIn - 1; k >= 0; k--)
	{
		hkIntVector v;	hkcdPlaneWeldUtilImpl::approximateNormal(planesIncluded[k], v);
		v.setComponent<3>(k);	// Save plane index in the .w component
		normalsIn[k] = v;
	}
	for (int k = numPlanesAll - 1; k >= 0; k--)
	{
 		hkIntVector v;	hkcdPlaneWeldUtilImpl::approximateNormal(allPlanes[k], v);
		v.setComponent<3>(k);	// Save plane index in the .w component
		normalsAll[k] = v;
	}
	
	hkAlgorithm::explicitStackQuickSort(normalsIn.begin(), numPlanesIn, hkcdPlaneWeldUtilImpl::vectorLess);
	hkAlgorithm::explicitStackQuickSort(normalsAll.begin(), numPlanesAll, hkcdPlaneWeldUtilImpl::vectorLess);

	// Sweep the two list to compute mapping
	int inInd;
	hkVector4Comparison mXYZ;	mXYZ.set<hkVector4Comparison::MASK_XYZ>();

	for (inInd = 0 ; inInd < numPlanesIn ; inInd++)
	{
		// Choose plane id value as minimum
		hkIntVector& currNormal = normalsIn[inInd];
		const int pInIdx		= normalsIn[inInd].getComponent<3>();
		const Plane& pIn 		= planesIncluded[pInIdx];

		// Find the indice in the general plane set that has the same normal value
		int allInd = hkAlgorithm::binarySearch(currNormal, normalsAll.begin(), normalsAll.getSize(), hkcdPlaneWeldUtilImpl::vectorLess);
		if ( allInd == -1 )
		{
			// plane not ound in the result table
			remapTableOut[pInIdx] = invalidPlaneValue;
			continue;
		}

		// make sure we are at the actual start of the group
		hkVector4Comparison cmp;
		while ( allInd > 0 )
		{
			cmp	= currNormal.compareEqualS32(normalsAll[allInd - 1]);	// [px == cx, py == cy, pz == cz, *]
			cmp.setAndNot(mXYZ, cmp);									// [px != cx, py != cy, pz != cz, 0]
			if ( cmp.anyIsSet() ) break;
			allInd--;
		}
		
		// Find plane with same exact normal and orientation
		bool found = false;
		while ( allInd < numPlanesAll && !found )
		{
			// Check if we are still on the same group of planes
			cmp	= currNormal.compareEqualS32(normalsAll[allInd]);	// [px == cx, py == cy, pz == cz, *]
			cmp.setAndNot(mXYZ, cmp);								// [px != cx, py != cy, pz != cz, 0]
			if ( cmp.anyIsSet() ) break;

			const int pAllIdx		= normalsAll[allInd].getComponent<3>();
			const Plane& pAll 		= allPlanes[pAllIdx];

			const Coplanarity coplanarRes	= hkcdPlanarGeometryPredicates::coplanar(pIn, pAll);
			if ( coplanarRes != hkcdPlanarGeometryPredicates::NOT_COPLANAR )
			{
				// We found the equivalent plane in the all set, save it into the remap table
				remapTableOut[pInIdx] = ( coplanarRes == hkcdPlanarGeometryPredicates::COINCIDENT ) ? pAllIdx : (hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG | pAllIdx);
				found = true;
			}
			allInd++;
		} 

		// No equivalent plane
		if ( !found )
		{
			remapTableOut[pInIdx] = invalidPlaneValue;
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
