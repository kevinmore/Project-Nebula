/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

#if !defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE hknpCapsuleShape::hknpCapsuleShape( hkVector4Parameter posA, hkVector4Parameter posB, hkReal radius )
	: hknpConvexPolytopeShape( 8, 6, 24, radius, HKNP_CAPSULE_BASE_SIZE )
{
	init( posA, posB );
}

#endif


HK_FORCE_INLINE void hknpCapsuleShape::init( hkVector4Parameter a, hkVector4Parameter b )
{
	m_flags.orWith( hknpShape::SUPPORTS_BPLANE_COLLISIONS );
	m_a = a;
	m_b = b;

	hkVector4 dir; dir.setSub(b, a);
	hkSimdReal length = dir.normalizeWithLength<3>();
	hkVector4 center; center.setInterpolate(a, b, hkSimdReal_Inv2);
	hkVector4 dir2, dir3; hkVector4Util::calculatePerpendicularNormalizedVectors<false>(dir, dir2, dir3);
	hkTransform transform; transform.getRotation().setCols(dir, dir2, dir3);
	transform.setTranslation(center);

	const hkReal sizeFraction = 0.01f;
	const hkReal coreRadius = sizeFraction * m_convexRadius;
	m_convexRadius -= coreRadius;

	// Calculate vertices
	//	 5 -- 4
	//  /|   /|    Y
	// 1____0 |	   |__X
	// | 7--|-6	  /
	// |/   |/	 Z
	// 3____2
	//
	hkReal halfHeight = length.getReal() * 0.5f + coreRadius;
	hkAabb aabb; aabb.m_max.set(halfHeight, coreRadius, coreRadius);
	aabb.m_min.setNeg<4>(aabb.m_max);
	hkVector4* HK_RESTRICT vertices = getVertices();
	for (int i = 0; i < 8; i++)
	{
		hkVector4 vertex;
		hkAabbUtil::getVertex(aabb, i, vertex);
		vertices[i].setTransformedPos(transform, vertex);
		vertices[i].setInt24W(i);
	}

	// Calculate planes
	
	{
		// Calculate and transform planes
		hkVector4 planesTmp[6];
		planesTmp[0].set(0, -1, 0, -coreRadius);
		planesTmp[1].set(0, 0, 1, -coreRadius);
		planesTmp[2].set(0, 0, -1, -coreRadius);
		planesTmp[3].set(0, 1, 0, -coreRadius);
		planesTmp[4].set(-1, 0, 0, -halfHeight);
		planesTmp[5].set(1, 0, 0, -halfHeight);
		hkVector4* HK_RESTRICT planes = getPlanes();
		hkVector4Util::transformPlaneEquations(transform, planesTmp, 6, planes);

		// Set padding planes
		planes[6].set(0, 0, 0, -HK_REAL_MAX);
		planes[7].set(0, 0, 0, -HK_REAL_MAX);
	}

	// Setup faces
	
	Face* HK_RESTRICT faces = getFaces();
	for (int i = 0; i < 6; ++i)
	{
		faces[i].m_minHalfAngle = 128;
		faces[i].m_numIndices = 4;
		faces[i].m_firstIndex = hkUint16(i << 2);
	}

	// Setup indices
	
	VertexIndex* HK_RESTRICT indices = getIndices();
	indices[0] = 7;  indices[1]  = 6; indices[2]  = 2; indices[3]  = 3;
	indices[4] = 3;  indices[5]  = 2; indices[6]  = 0; indices[7]  = 1;
	indices[8] = 7;  indices[9]  = 5; indices[10] = 4; indices[11] = 6;
	indices[12] = 1; indices[13] = 0; indices[14] = 4; indices[15] = 5;
	indices[16] = 1; indices[17] = 5; indices[18] = 7; indices[19] = 3;
	indices[20] = 2; indices[21] = 6; indices[22] = 4; indices[23] = 0;
}


HK_FORCE_INLINE hknpCapsuleShape* hknpCapsuleShape::createInPlace(hkVector4Parameter posA, hkVector4Parameter posB,
																  hkReal radius, hkUint8* buffer, int bufferSize)
{
#ifndef HK_ALIGN_RELAX_CHECKS
	HK_ASSERT2(0x15249213, !(hkUlong(buffer) & 0xF), "Shape buffer must be 16 byte aligned");
#endif
	HK_ON_DEBUG(const int shapeSize = calcConvexPolytopeShapeSize(8, 6, 24, HKNP_CAPSULE_BASE_SIZE));
	HK_ASSERT2(0x30a392a2, shapeSize <= bufferSize, "Shape too large to fit in buffer");

#if !defined(HK_PLATFORM_SPU)
	// Construct in-place
	hknpCapsuleShape* capsule= new (buffer) hknpCapsuleShape(posA, posB, radius);
#else
	// Initialize manually to avoid creating the v-table
	hknpCapsuleShape* capsule = reinterpret_cast<hknpCapsuleShape*>(buffer);
	capsule->hknpShape::init(hknpCollisionDispatchType::CONVEX);
	capsule->hknpConvexShape::init(8, radius, HKNP_CAPSULE_BASE_SIZE);
	capsule->hknpConvexPolytopeShape::init(8, 6, 24, HKNP_CAPSULE_BASE_SIZE);
	capsule->init(posA, posB);

	hknpShapeVirtualTableUtil::patchVirtualTableWithType<hknpShapeType::CAPSULE>(capsule);
#endif

	capsule->m_memSizeAndFlags = 0;
	return capsule;
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
