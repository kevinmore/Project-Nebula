/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#if !defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE hknpSphereShape::hknpSphereShape( hkVector4Parameter center, hkReal radius )
	: hknpConvexShape( 4, radius, HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeof(hknpSphereShape)) )
{
	init( center );
}

#endif

HK_FORCE_INLINE void hknpSphereShape::init(hkVector4Parameter center)
{
	m_flags.orWith( USE_SINGLE_POINT_MANIFOLD );

	hkVector4* HK_RESTRICT vertices = getVertices();
	vertices[0] = center;
	vertices[1] = center;
	vertices[2] = center;
	vertices[3] = center;
	vertices[0].setInt24W(0);
	vertices[1].setInt24W(0);
	vertices[2].setInt24W(0);
	vertices[3].setInt24W(0);
}

HK_FORCE_INLINE hknpSphereShape* hknpSphereShape::createInPlace( hkVector4Parameter center, hkReal radius,
																 hkUint8* buffer, int bufferSize)
{
#if !defined(HK_ALIGN_RELAX_CHECKS)
	HK_ASSERT2(0x15249213, !(hkUlong(buffer) & 0xF), "Shape buffer must be 16 byte aligned");
#endif
	HK_ON_DEBUG(const int shapeSize = calcConvexShapeSize(4, HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeof(hknpSphereShape))));
	HK_ASSERT2(0x30a392a2, shapeSize <= bufferSize, "Shape too large to fit in buffer");

#if !defined(HK_PLATFORM_SPU)
	// Construct in-place
	hknpSphereShape* sphere = new (buffer) hknpSphereShape( center, radius );
#else
	// Initialize manually to avoid creating the v-table
	hknpSphereShape* sphere = reinterpret_cast<hknpSphereShape*>(buffer);
	sphere->hknpShape::init(hknpCollisionDispatchType::CONVEX);
	sphere->hknpConvexShape::init(4, radius, HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeof(hknpSphereShape)));
	sphere->init(center);

	hknpShapeVirtualTableUtil::patchVirtualTableWithType<hknpShapeType::SPHERE>(sphere);
#endif

	sphere->m_memSizeAndFlags = 0;
	return sphere;
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
