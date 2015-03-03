/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/DataStructures/Planar/Memory/hkcdPlanarGeometryPlanesCollection.h>
#include <Geometry/Collide/DataStructures/Planar/Utils/hkcdPlanarGeometryWeldUtil.h>

//
//	Empty constructor

hkcdPlanarGeometryPlanesCollection::hkcdPlanarGeometryPlanesCollection()
:	hkReferencedObject()
,	m_cache(HK_NULL)
{
	m_offsetAndScale.setZero();

	// Add boundary planes
	createBoundaryPlanes();
}

//
//	Default constructor

hkcdPlanarGeometryPlanesCollection::hkcdPlanarGeometryPlanesCollection(const hkAabb& origMaxAabb, int initialPlaneCapacity)
:	hkReferencedObject()
,	m_cache(HK_NULL)
{
	// Compute factors for float to fixed precision conversion
	hkAabb maxAabb				= origMaxAabb;
	maxAabb.expandBy(hkSimdReal_1);									// We want to be able to add / subtract 1 from the bounds and still get a valid fixed precision coordinate
	hkVector4 vExtents;			maxAabb.getExtents(vExtents);
	hkSimdReal maxExtent;		maxExtent.setAdd(vExtents.horizontalMax<3>(), hkSimdReal_Inv2);	// Round to nearest

	// We want the smallest power of 2 larger than the max extent
	hkInt32 intValue;			maxExtent.storeSaturateInt32(&intValue);
	intValue++;	
	if ( intValue < 1 )			intValue = 1;
	const int logValue			= 32 - hkMath::countLeadingZeros(intValue);

	HK_ASSERT(0x7803bf08, logValue <= (hkcdPlanarGeometryPrimitives::NumBitsVertex::NUM_BITS - 1));
	hkSimdReal intFromFloatScale;	intFromFloatScale.setFromInt32(1 << (hkcdPlanarGeometryPrimitives::NumBitsVertex::NUM_BITS - 1 - logValue));

	// Store conversion offset and scale
	m_offsetAndScale.setXYZ_W(maxAabb.m_min, intFromFloatScale);

	// Pre-allocate planes
	if ( initialPlaneCapacity )
	{	
		m_planes.reserve(initialPlaneCapacity + 6);
	}

	// Add boundary planes
	createBoundaryPlanes();
}

//
//	Copy constructor

hkcdPlanarGeometryPlanesCollection::hkcdPlanarGeometryPlanesCollection(const hkcdPlanarGeometryPlanesCollection& other)
:	hkReferencedObject()
,	m_offsetAndScale(other.m_offsetAndScale)
,	m_cache(HK_NULL)
{
	m_planes.append(other.m_planes);
}

//
//	Destructor

hkcdPlanarGeometryPlanesCollection::~hkcdPlanarGeometryPlanesCollection()
{
	clearCaches();
}

//
//	Appends the world boundary planes to the collection. These 6 planes are used by various other
//	algorithms and will always be present in the first 6 slots of any planes collection.

void hkcdPlanarGeometryPlanesCollection::createBoundaryPlanes()
{
	HK_ASSERT(0x64886180, m_planes.isEmpty());

	// Build the min and max vertices
	const int iMin = 0;
	const int iMax = (1 << (hkcdPlanarGeometryPrimitives::NumBitsVertex::NUM_BITS - 1)) - 1;

	hkInt64Vector4 iNrm;
	hkSimdInt<128> iOffset;
	Plane p;

	// Create 6 boundary planes, one for each axis
	{
		iNrm.set(+1, 0, 0);			iOffset.setFromInt32(-iMax);	p.setExactEquation(iNrm, iOffset);	
		addPlane(p);

		iNrm.set(-1, 0, 0);			iOffset.setFromInt32(+iMin);	p.setExactEquation(iNrm, iOffset);
		addPlane(p);

		iNrm.set(0, +1, 0);			iOffset.setFromInt32(-iMax);	p.setExactEquation(iNrm, iOffset);
		addPlane(p);

		iNrm.set(0, -1, 0);			iOffset.setFromInt32(+iMin);	p.setExactEquation(iNrm, iOffset);
		addPlane(p);

		iNrm.set(0, 0, +1);			iOffset.setFromInt32(-iMax);	p.setExactEquation(iNrm, iOffset);
		addPlane(p);

		iNrm.set(0, 0, -1);			iOffset.setFromInt32(+iMin);	p.setExactEquation(iNrm, iOffset);
		addPlane(p);
	}
}

//
//	Creates a new collection containing the welded planes of both source collections

hkcdPlanarGeometryPlanesCollection* HK_CALL hkcdPlanarGeometryPlanesCollection::createMergedCollection(	const hkcdPlanarGeometryPlanesCollection* planesColA,
																										const hkcdPlanarGeometryPlanesCollection* planesColB,
																										hkArray<int>* planesRemapTableOut)
{
	// Create the result
	hkcdPlanarGeometryPlanesCollection* planesColAB = new hkcdPlanarGeometryPlanesCollection();
	{
		const hkVector4 vOffset = planesColA->getPositionOffset();
		const hkSimdReal vScale	= planesColA->getPositionScale();
		HK_ASSERT(0x20e13b42,	vOffset.allEqual<3>(planesColB->getPositionOffset(), hkSimdReal_Eps) && \
								vScale.approxEqual(planesColB->getPositionScale(), hkSimdReal_Eps));

		planesColAB->setPositionOffset(vOffset);
		planesColAB->setPositionScale(vScale);
	}

	// Merge planes
	hkArray<int> planesRemapTable;
	if ( !planesRemapTableOut )
	{
		planesRemapTableOut = &planesRemapTable;
	}

	const int numPlanesA = planesColA->getNumPlanes();				HK_ASSERT(0x2ebbb536, numPlanesA >= NUM_BOUNDS);
	const int numPlanesB = planesColB->getNumPlanes();				HK_ASSERT(0x5152b9d, numPlanesB >= NUM_BOUNDS);
	const hkArray<Plane>& planesA	= planesColA->m_planes;
	const hkArray<Plane>& planesB	= planesColB->m_planes;
	hkArray<Plane>& planesAB		= planesColAB->m_planes;

	// Keep only one set of boundary planes
	planesAB.reserve(numPlanesA + numPlanesB - NUM_BOUNDS);
	planesAB.setSize(0);
	planesAB.append(planesA);
	if ( numPlanesB - NUM_BOUNDS )
	{
		planesAB.append(&planesB[NUM_BOUNDS], numPlanesB - NUM_BOUNDS);
	}

	// Weld planes from both geometries
	hkArray<Plane> weldedPlanesAB;
	if ( planesRemapTableOut )
	{
		planesRemapTableOut->reserve(numPlanesA + numPlanesB);
	}
	hkcdPlanarGeometryWeldUtil::weldPlanes(planesAB, weldedPlanesAB, *planesRemapTableOut);

	// Fix-up the remap table
	if ( planesRemapTableOut )
	{
		HK_ASSERT(0x3da4f873, planesRemapTableOut->getSize() == numPlanesA + numPlanesB - NUM_BOUNDS);
		planesRemapTableOut->expandAt(numPlanesA, NUM_BOUNDS);

		// Remapped boundary planes of B
		for (int k = 0; k < NUM_BOUNDS; k++)
		{
			(*planesRemapTableOut)[numPlanesA + k] = k;
		}
	}

	// Build a new geometry		
	planesAB.swap(weldedPlanesAB);

// 	hkStringBuf str;
// 	str.printf("That's a new plane collection with %d planes", planesAB.getSize());
// 	HK_REPORT(str);

	return planesColAB;
}

//
//	Computes a mapping between two plane collection. The first plane set provided is assumed to be included in the second plane collection.

void HK_CALL hkcdPlanarGeometryPlanesCollection::computeMappingBetweenPlaneSets(const hkcdPlanarGeometryPlanesCollection* planedIncluded, const hkcdPlanarGeometryPlanesCollection* planesGlobal, hkArray<int>& remapTableOut)
{
	hkcdPlanarGeometryWeldUtil::computeMappingBetweenPlaneSets(planedIncluded->m_planes, planesGlobal->m_planes, remapTableOut, (int)(PlaneId::invalid().valueUnchecked()));
}

//
//	Computes the maximum number of bits used by the plane equations

void hkcdPlanarGeometryPlanesCollection::computeMaxNumUsedBits(int& numBitsNormal, int& numBitsOffset) const
{
	const int numPlanes = m_planes.getSize();
	hkIntVector msbMax;	msbMax.setZero();
	hkIntVector cv;		cv.set(63, 63, 63, 127);

	for (int pi = numPlanes - 1; pi >= NUM_BOUNDS; pi--)
	{
		const Plane& p		= m_planes[pi];
		hkInt64Vector4 iN;	p.getExactNormal(iN);
		iN.setAbs(iN);
		hkSimdInt<128> iO;	p.getExactOffset(iO);
		iO.setAbs(iO);

		hkIntVector msb;	iN.countLeadingZeros<3>(msb);
		msb.setComponent<3>(iO.countLeadingZeros());		
		msb.setSubS32(cv, msb);
		msbMax.setMaxS32(msbMax, msb);
	}

	// Write output
	numBitsNormal = msbMax.horizontalMaxS32<3>();
	numBitsOffset = msbMax.getComponent<3>();
}

//
//	Helper functions

namespace hkcdPlanarGeometryImpl
{
	//
	//	Computes a discretized quaternion

	static HK_FORCE_INLINE void HK_CALL discretizeRot(hkQuaternionParameter qIn, int numBitsRot, hkIntVector& qOut)
	{
		const int N		= 1 << numBitsRot;
		hkVector4 x;	x.setAll((hkReal)N / 1.1f);
						x.mul(qIn.m_vec);
						x.add(hkVector4::getConstant<HK_QUADREAL_INV_2>());

		qOut.setConvertF32toS32(x);

		// Test
#ifdef HK_DEBUG
		hkIntVector vv;		vv.setAbsS32(qOut);
		hkIntVector vvv;	vvv.setAll(N);
		HK_ASSERT(0x3fd39b5, vv.compareLessThanS32(vvv).allAreSet());
#endif
	}

	//
	//	Computes a rotation matrix from the given quaternion

	static HK_FORCE_INLINE int HK_CALL computeRotationMatrix(	hkIntVectorParameter qIn, 
																/*hkIntVector& col0Out, hkIntVector& col1Out, hkIntVector& col2Out,*/
																hkIntVector& row0Out, hkIntVector& row1Out, hkIntVector& row2Out)
	{
		const int x = qIn.getComponent<0>();
		const int y = qIn.getComponent<1>();
		const int z = qIn.getComponent<2>();
		const int w = qIn.getComponent<3>();

		const int xx = x * x,	yy = y * y,		zz = z * z,		ww = w * w;
		const int xy = x * y,	xz = x * z,		xw = x * w;
		const int yz = y * z,	yw = y * w;
		const int zw = z * w;

		row0Out.set((xx + ww) - (yy + zz),	(xy - zw) << 1,			(xz + yw) << 1,			0);
		row1Out.set((xy + zw) << 1,			(yy + ww) - (xx + zz),	(yz - xw) << 1,			0);
		row2Out.set((xz - yw) << 1,			(yz + xw) << 1,			(zz + ww) - (xx + yy),	0);

		// Compute Transpose[R].R. Should be a scaled unit matrix.
		const int det = (xx + yy + zz + ww);
#ifdef HK_DEBUG
		{
			hkIntVector c0;		c0.set((xx + ww) - (yy + zz),	(xy + zw) << 1,			(xz - yw) << 1,			0);
			hkIntVector c1;		c1.set((xy - zw) << 1,			(yy + ww) - (xx + zz),	(yz + xw) << 1,			0);
			hkIntVector c2;		c2.set((xz + yw) << 1,			(yz - xw) << 1,			(zz + ww) - (xx + yy),	0);

			const hkInt64 m00 = c0.dot<3>(c0);
			const hkInt64 m11 = c1.dot<3>(c1);
			const hkInt64 m22 = c2.dot<3>(c2);
			HK_ASSERT(0x3e9bbc18, (m00 == m11) && (m11 == m22) && (m00 == (hkInt64)det * (hkInt64)det));

			const hkInt64 m01 = c0.dot<3>(c1);
			const hkInt64 m02 = c0.dot<3>(c2);
			const hkInt64 m12 = c1.dot<3>(c2);
			HK_ASSERT(0x1514c0fb, (m01 == 0) && (m02 == 0) && (m12 == 0));
		}
#endif

		return det;
	}
}

//
//	Applies the given transform on the planes. Note that this will lose precision!

void hkcdPlanarGeometryPlanesCollection::applyTransform(const hkQTransform& transform, int numBitsTransform, bool simplifyEquations, int startPlaneIdx, int endPlaneIdx)
{
	// Discretize rotation quaternion
	hkIntVector iRot;
	hkcdPlanarGeometryImpl::discretizeRot(transform.m_rotation, numBitsTransform, iRot);

	// Compute rotation matrix
	hkIntVector rX, rY, rZ;
	const int rDet = hkcdPlanarGeometryImpl::computeRotationMatrix(iRot, rX, rY, rZ);

	// Compute translation delta. Rotation takes place around the real world's origin, so we need to rotate around that as well
	hkIntVector t;	convertWorldDirection(transform.m_translation, t);
	hkIntVector c;	convertWorldPosition(hkVector4::getConstant<HK_QUADREAL_0>(), c);

	// Transform planes
	startPlaneIdx = hkMath::max2(startPlaneIdx, NUM_BOUNDS);
	const int numPlanes = ( endPlaneIdx < NUM_BOUNDS ) ? m_planes.getSize() : hkMath::min2(m_planes.getSize(), endPlaneIdx + 1);
	for (int pi = startPlaneIdx ; pi < numPlanes; pi++)
	{
		Plane& p			= m_planes[pi];
		hkInt64Vector4 iN;	p.getExactNormal(iN);
		hkSimdInt<128> iO;	p.getExactOffset(iO);
							iO.setAdd(iO, iN.dot<3>(c));

		// Rotate normal & offset
		iN.set(iN.dot_64<3>(rX), iN.dot_64<3>(rY), iN.dot_64<3>(rZ), 0);

		// Add translation
		hkIntVector ct;		ct.setAddS32(c, t);
		hkSimdInt<128> dO	= iN.dot<3>(ct);
		iO.setMul(iO, rDet);
		iO.setSub(iO, dO);

		// Check for Normal overflow
#if 0
		{
			hkSimdInt<128> absX;	absX.setAbs(nx);
			hkSimdInt<128> absY;	absX.setAbs(ny);
			hkSimdInt<128> absZ;	absZ.setAbs(nz);
			const int msbX		= 127 - hkMath::countLeadingZeros<hkUint64>(absX);
			const int msbY		= 127 - hkMath::countLeadingZeros<hkUint64>(absY);
			const int msbZ		= 127 - hkMath::countLeadingZeros<hkUint64>(absZ);
			const int msbXYZ	= hkMath::max2(hkMath::max2(msbX, msbY), msbZ);
			HK_ASSERT(0x1f181269, msbXYZ <= hkcdPlanarGeometryPrimitives::NumBitsPlaneNormal::NUM_BITS - 2);
		}

		// Check for Offset overflow
		{
			hkSimdInt<128> absO;	absO.setAbs(iO);
			const int msbW			= 127 - absO.countLeadingZeros();
			HK_ASSERT(0x117dc11, msbW <= hkcdPlanarGeometryPrimitives::NumBitsPlaneOffset::NUM_BITS - 2);
		}
#endif

		// Set new plane equation
		p.setExactEquation(iN, iO, simplifyEquations);

	}
}

//
//	Appends the planes of the given collection to this collection. Optionally returns the array of plane Ids for the merged planes.
//	Appending planes can preserve the cache, since the cached planes do not change.

void hkcdPlanarGeometryPlanesCollection::append(const hkcdPlanarGeometryPlanesCollection& src, hkArray<PlaneId>* appendedPlaneIdsOut)
{
	const hkArray<Plane>& srcPlanes	= src.m_planes;
	const int numSrcPlanes			= srcPlanes.getSize();
	
	// Do not append the boundary planes!
	const int numOldplanes	= m_planes.getSize();
	const int numNewPlanes	= numSrcPlanes - NUM_BOUNDS;
	Plane* newPlanes		= m_planes.expandBy(numNewPlanes);

	PlaneId* idsOut = HK_NULL;
	if ( appendedPlaneIdsOut )
	{
		idsOut = appendedPlaneIdsOut->expandBy(numNewPlanes);
	}

	for (int k = 0; k < numNewPlanes; k++)
	{
		const Plane p	= srcPlanes[k + NUM_BOUNDS];
		newPlanes[k]	= p;

		if ( idsOut )
		{
			idsOut[k] = PlaneId(numOldplanes + k);
		}
	}
}

//
//	Welds the planes

void hkcdPlanarGeometryPlanesCollection::weldPlanes(hkArray<int>* planesRemapTableOut)
{
	hkArray<Plane> weldedPlanes;
	hkArray<int> remapTable;
	if ( !planesRemapTableOut )
	{
		planesRemapTableOut = &remapTable;
	}

	hkcdPlanarGeometryWeldUtil::weldPlanes(m_planes, weldedPlanes, *planesRemapTableOut);

	m_planes.swap(weldedPlanes);
	clearCaches();
}

//
//	Removes all planes set in the bit-field.

void hkcdPlanarGeometryPlanesCollection::removePlanes(const hkBitField& planesToRemove, hkArray<int>* planeRemapOut)
{
	hkArray<int> localPlaneRemap;
	if ( !planeRemapOut )
	{
		planeRemapOut = &localPlaneRemap;
	}

	const int maxNumPlanes		= m_planes.getSize();
	hkArray<Plane> newPlanes;	newPlanes.reserve(maxNumPlanes);
	planeRemapOut->setSize(maxNumPlanes, -1);

	// Make sure the boundary planes are preserved
	for (int k = hkcdPlanarGeometryPlanesCollection::NUM_BOUNDS - 1; k >= 0; k--)
	{
		HK_ASSERT(0x2f020323, !planesToRemove.get(k));
	}

	// Add all planes that were requested
	for (int k = 0; k < planesToRemove.getSize(); k++)
	{
		if ( !planesToRemove.get(k) )
		{
			HK_ASSERT(0x7a7bae2a, (*planeRemapOut)[k] < 0);
			(*planeRemapOut)[k] = newPlanes.getSize();
			newPlanes.pushBack(m_planes[k]);
		}
	}

	// Replace with new planes
	newPlanes.optimizeCapacity(0, true);
	m_planes.swap(newPlanes);
	clearCaches();
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
