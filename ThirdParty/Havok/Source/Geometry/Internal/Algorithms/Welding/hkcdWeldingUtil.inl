/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

HK_FORCE_INLINE void hkcdWeldingUtil::_applyModifiedNormal(hkVector4Parameter newNormal, hkSimdRealParameter radiusA, hkcdManifold4* HK_RESTRICT manifoldBInOut )
{
	// Point on Aw
	hkVector4* HK_RESTRICT pOnBs = manifoldBInOut->m_positions;
	HK_ASSERT( 0xf0345465, newNormal.isNormalized<3>() && newNormal.isOk<4>() );
	hkVector4 planeB = manifoldBInOut->m_normal;

	hkSimdReal product = manifoldBInOut->m_normal.dot<3>(newNormal);
	hkVector4 projDirection; projDirection.setAddMul(planeB, newNormal, -product);

	// Ensure distance is positive
	product.setAbs(product);

	hkVector4 radiusCorrectedDistance; radiusCorrectedDistance.setAdd( manifoldBInOut->m_distances, radiusA );

	// Move points on B to their projection points (relating to the point on A and the welded normal)
 	pOnBs[0].setAddMul( pOnBs[0], projDirection, radiusCorrectedDistance.getComponent<0>() );
 	pOnBs[1].setAddMul( pOnBs[1], projDirection, radiusCorrectedDistance.getComponent<1>() );
 	pOnBs[2].setAddMul( pOnBs[2], projDirection, radiusCorrectedDistance.getComponent<2>() );
 	pOnBs[3].setAddMul( pOnBs[3], projDirection, radiusCorrectedDistance.getComponent<3>() );

	// Scale distances to projected length
	radiusCorrectedDistance.mul( product );
	hkVector4 newDistances; newDistances.setSub( radiusCorrectedDistance, radiusA );

	manifoldBInOut->m_distances = newDistances;
	manifoldBInOut->m_normal = newNormal;
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
