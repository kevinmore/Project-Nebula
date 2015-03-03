/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */



inline hkpBoxBoxCollisionDetection::hkpBoxBoxCollisionDetection(
		const hkpCdBody& bodyA, const hkpCdBody& bodyB,
		const hkpProcessCollisionInput* env, hkpContactMgr* mgr,	hkpProcessCollisionOutput* result,
		const hkTransform& aTb, 
		const hkTransform& wTa, hkVector4Parameter radiusA, 
		const hkTransform& wTb, hkVector4Parameter radiusB, hkSimdRealParameter tolerance )
	
: 	m_bodyA( bodyA ), 
	m_bodyB( bodyB ), 
	m_env( env ),
	m_contactMgr( mgr ),
	m_result( result ),
	m_wTa(wTa), 
	m_wTb(wTb), 
	m_aTb(aTb),
	m_radiusA(radiusA), 
	m_radiusB(radiusB)
{
	m_boundaryTolerance.setFromFloat(hkReal(0.01f));
	m_tolerance4.setAll( tolerance );
	m_tolerance4.setW(hkSimdReal_Max);
	m_keepRadiusA.setAdd( m_tolerance4, m_radiusA );
	m_keepRadiusB.setAdd( m_tolerance4, m_radiusB );
}
 
template <int edgeNext, int edgeNextNext>
inline void hkpBoxBoxCollisionDetection::setvdProj( const hkRotation& bRa, hkVector4& vdproj ) const
{
	hkVector4 a; a.setMul( m_dinA.getComponent<edgeNextNext>(), bRa.getColumn<edgeNext>() );
	hkVector4 b; b.setMul( m_dinA.getComponent<edgeNext>(), bRa.getColumn<edgeNextNext>() );
	vdproj.setSub( a,b );
	vdproj.setAbs( vdproj );
}

void hkpBoxBoxCollisionDetection::initWorkVariables() const
{
	m_dinA = m_aTb.getTranslation();
	m_dinB._setRotatedInverseDir( m_aTb.getRotation(), m_dinA );
}


inline hkBool hkpBoxBoxCollisionDetection::getPenetrations()
{
	initWorkVariables();
	return HK_SUCCESS == checkIntersection( m_tolerance4 );
}


HK_FORCE_INLINE hkBool32 hkpBoxBoxCollisionDetection::isValidFaceAVertexB( const hkpFeaturePointCache& fpp ) const
{
	hkVector4 vertexB_inA;
	vertexB_inA.setAbs( fpp.m_vA );

	return vertexB_inA.allLess<3>( m_keepRadiusA );
}

HK_FORCE_INLINE void hkpBoxBoxCollisionDetection::calcDistanceFaceAVertexB( hkpFeaturePointCache& fpp ) const
{
	const int featureIndex = fpp.m_featureIndexA;
	hkVector4 dist; 
	dist.setFlipSign(fpp.m_vA, fpp.m_normalIsFlipped);
	dist.sub(m_radiusA);
	fpp.m_distance = dist.getComponent(featureIndex);
}

HK_FORCE_INLINE hkBool32 hkpBoxBoxCollisionDetection::isValidFaceBVertexA( const hkpFeaturePointCache& fpp ) const
{
	hkVector4 vertexA_inB;
	vertexA_inB.setAbs( fpp.m_vB );

	return vertexA_inB.allLess<3>( m_keepRadiusB );
}

HK_FORCE_INLINE void hkpBoxBoxCollisionDetection::calcDistanceFaceBVertexA( hkpFeaturePointCache& fpp ) const
{
	const int featureIndex = fpp.m_featureIndexA - HK_BOXBOX_FACE_B_X;
	hkVector4Comparison notFlipped; notFlipped.setNot(fpp.m_normalIsFlipped);
	hkVector4 dist; 
	dist.setFlipSign(fpp.m_vB, notFlipped);
	dist.sub(m_radiusB);
	fpp.m_distance = dist.getComponent(featureIndex);
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
